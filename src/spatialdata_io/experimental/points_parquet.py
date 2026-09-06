"""Rewrite a Points element into regular-grid row groups.

The output keeps every canonical column and adds two render-oriented ones:

``display_xy``
    ``fixed_size_list<uint32>[2]`` of integer level-0 pixel coordinates. The Arrow child
    buffer is therefore already ``[x0, y0, x1, y1, ...]`` -- directly usable as a deck.gl
    binary ``getPosition`` attribute with no interleaving step in the browser.
``feature_code``
    Small unsigned integer into the :class:`FeatureCatalog`.

Physical row order changes (rows are grouped by tile), but no row is added, dropped or
altered, and the DataFrame index is preserved so the reordering is fully traceable.

Row groups are written one-per-logical-tile *including empty tiles*, so a client can find
a tile's data from the tile formula alone, with no lookup table and no reliance on Parquet
statistics. Output is split across several files because a reader must fetch a whole
footer before reading any row group, and footer size grows with row-group count.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.regular_grid import (
    DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    RegularGrid,
)

__all__ = [
    "POSITION_COLUMN",
    "FEATURE_COLUMN",
    "DisplayTransform",
    "write_points_regular_grid",
]

#: Column holding interleaved integer pixel positions.
POSITION_COLUMN = "display_xy"
#: Column holding the integer feature code.
FEATURE_COLUMN = "feature_code"

#: Largest representable display coordinate.
_UINT32_MAX = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class DisplayTransform:
    """The affine mapping from canonical element coordinates to display pixels.

    Recorded in the manifest so a client can reproduce the mapping, and so the profile can
    be invalidated if the underlying transform changes.
    """

    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    coordinate_system: str
    rounding: str = "nearest"

    @classmethod
    def from_element(cls, element: Any, coordinate_system: str) -> DisplayTransform:
        """Derive the transform from an element's SpatialData coordinate transformations."""
        from spatialdata.transformations import get_transformation

        t = get_transformation(element, coordinate_system)
        affine = np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
        return cls(
            matrix=((affine[0, 0], affine[0, 1], affine[0, 2]), (affine[1, 0], affine[1, 1], affine[1, 2])),
            coordinate_system=coordinate_system,
        )

    def apply(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map canonical coordinates to (unrounded) display pixel coordinates."""
        (a, b, c), (d, e, f) = self.matrix
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return a * x + b * y + c, d * x + e * y + f

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize the transform for the manifest, so a client can reproduce the mapping."""
        return {
            "coordinate_space": "image-pixel",
            "coordinate_system": self.coordinate_system,
            "affine_matrix": [list(self.matrix[0]), list(self.matrix[1])],
            "rounding": self.rounding,
        }


def _to_display_pixels(
    x: NDArray[Any], y: NDArray[Any], transform: DisplayTransform
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Transform and round to non-negative integer pixels, validating the declared dtype."""
    px, py = transform.apply(x, y)

    for name, v in (("x", px), ("y", py)):
        if not np.isfinite(v).all():
            raise ValueError(f"display {name} contains non-finite values after transform")

    rx = np.rint(px)
    ry = np.rint(py)

    for name, v in (("x", rx), ("y", ry)):
        lo, hi = float(v.min()), float(v.max())
        if lo < 0:
            raise ValueError(
                f"display {name} has negative values (min {lo}). display_xy is unsigned; "
                f"shift the grid origin or fix the coordinate transform."
            )
        if hi > _UINT32_MAX:
            raise ValueError(f"display {name} max {hi} exceeds uint32 range")

    return rx.astype(np.uint32), ry.astype(np.uint32)


def _interleaved_positions(px: NDArray[np.uint32], py: NDArray[np.uint32]) -> pa.FixedSizeListArray:
    """Build ``fixed_size_list<uint32>[2]`` whose child buffer is ``[x0,y0,x1,y1,...]``."""
    flat = np.empty(px.size * 2, dtype=np.uint32)
    flat[0::2] = px
    flat[1::2] = py
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), 2)


#: Internal column carrying the tile assignment through the streaming spill files.
_TILE_ID = "__tile_id"


def _prepare_table(
    df: pd.DataFrame,
    *,
    transform: DisplayTransform,
    catalog: FeatureCatalog,
    feature_key: str,
    grid: RegularGrid,
    categories: Any | None,
    render_only: bool = False,
) -> tuple[pa.Table, NDArray[np.int64]]:
    """Build one chunk's output table and return it with its tile assignment.

    With ``render_only`` the table holds just the render columns, for the standalone file
    a viewer reads. Otherwise it is the canonical columns, tile-ordered but otherwise
    untouched.

    ``categories`` pins the categorical dictionary so that every chunk converts to an
    identical Arrow schema; without it, partitions observing different feature subsets
    would produce incompatible dictionary types and could not be written to one file.
    """
    px, py = _to_display_pixels(df["x"].to_numpy(), df["y"].to_numpy(), transform)
    tile_ids = grid.assign(px, py)

    if render_only:
        # Only the render columns. A viewer reads every column of this file, which is
        # why no column projection is needed -- and parquet-wasm's projection is broken
        # anyway (any `columns` argument corrupts the IPC stream it emits).
        table = pa.table(
            {
                POSITION_COLUMN: _interleaved_positions(px, py),
                FEATURE_COLUMN: pa.array(catalog.encode(df[feature_key])),
            }
        )
        return table, tile_ids

    # The canonical element keeps only its own columns. The render columns live in a
    # separate file, for two reasons: a nested Arrow column cannot survive dask's parquet
    # round-trip (SpatialData.write() either fails or silently returns it as a string),
    # and a standalone render file means a viewer reads every column of it, so no column
    # projection is needed -- which matters because parquet-wasm's projection is broken.
    #
    # Any render columns left by an earlier version are dropped, so re-tiling a store
    # written before this change cleans it up rather than preserving them.
    stale = [c for c in (POSITION_COLUMN, FEATURE_COLUMN, _TILE_ID) if c in df.columns]
    if df.attrs or stale or categories is not None:
        df = df.copy(deep=False)
        # The transform is not JSON-serializable; spatialdata's own points writer drops it
        # the same way. It is persisted in the element's zarr attributes, not the parquet.
        df.attrs = {}
        if stale:
            df = df.drop(columns=stale)
        if categories is not None and isinstance(df[feature_key].dtype, pd.CategoricalDtype):
            df[feature_key] = df[feature_key].cat.set_categories(categories)

    return pa.Table.from_pandas(df, preserve_index=True), tile_ids


def _sorted_by_tile(table: pa.Table, tile_ids: NDArray[np.int64]) -> tuple[pa.Table, NDArray[np.int64]]:
    """Group rows by tile. The sort is stable, so the rewrite is deterministic."""
    order = np.argsort(tile_ids, kind="stable")
    return table.take(pa.array(order)), tile_ids[order]


def _write_tile_row_groups(
    writer: pq.ParquetWriter,
    table: pa.Table,
    sorted_tile_ids: NDArray[np.int64],
    tile_range: range,
    schema: pa.Schema,
) -> None:
    """Write one row group per tile in ``tile_range``, empty tiles included.

    Empty tiles must still occupy a row group, since that is what makes
    ``row_group_index == tile_id`` hold without a lookup table.
    """
    offsets = np.searchsorted(sorted_tile_ids, np.array([*tile_range, tile_range.stop]), side="left")
    for i in range(len(tile_range)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        writer.write_table(table.slice(start, end - start) if end > start else schema.empty_table())


def _iter_chunks(points: Any) -> Any:
    """Yield the element one partition at a time, or once if it is already in memory."""
    if hasattr(points, "npartitions"):
        for i in range(points.npartitions):
            yield points.partitions[i].compute()
    else:
        yield points


def _known_categories(points: Any, feature_key: str) -> Any | None:
    """Return the element's full category list, so every chunk shares one dictionary."""
    col = points[feature_key]
    if not hasattr(col, "cat"):
        return None
    try:
        return list(col.cat.categories)
    except (NotImplementedError, AttributeError):
        return list(col.cat.as_known().cat.categories)


def write_points_regular_grid(
    points: Any,
    output_dir: str | Path,
    *,
    catalog: FeatureCatalog,
    grid: RegularGrid | None = None,
    display_transform: DisplayTransform | None = None,
    coordinate_system: str = "global",
    feature_key: str = "feature_name",
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    compression: str = "zstd",
    streaming: bool | None = None,
    render_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a Points element as regular-grid row groups.

    Parameters
    ----------
    points
        A SpatialData Points element (dask DataFrame) or a pandas DataFrame.
    output_dir
        Directory to write the chunk files into. Written atomically: a temporary sibling
        directory is populated first and swapped in only on success.
    catalog
        Feature vocabulary providing ``feature_code``.
    grid
        The tile grid. If ``None``, the smallest grid covering the data is derived using
        ``tile_size_px``.
    coordinate_system
        SpatialData coordinate system defining display pixel space.
    feature_key
        Column holding the feature name.
    tile_size_px
        Tile edge length, used only when ``grid`` is ``None``.
    max_row_groups_per_file
        Row groups per chunk file.
    compression
        Parquet compression codec.
    overwrite
        Replace ``output_dir`` if it exists.

    Returns
    -------
    The manifest fragment describing the written files.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")

    columns = list(points.columns)
    if feature_key not in columns:
        raise ValueError(f"feature column {feature_key!r} not found; have {columns}")
    for axis in ("x", "y"):
        if axis not in columns:
            raise ValueError(f"points element has no {axis!r} column; have {columns}")

    partitioned = hasattr(points, "npartitions") and points.npartitions > 1
    if streaming is None:
        streaming = partitioned
    if streaming and not partitioned:
        raise ValueError(
            "streaming requires a partitioned (dask) points element; an in-memory frame cannot be read incrementally"
        )

    # When called as a SpatialData ``points_writer`` hook the element arrives with its
    # transformations already stripped from attrs, so the caller must supply the transform.
    transform = display_transform or DisplayTransform.from_element(points, coordinate_system)

    if grid is None:
        grid = _derive_grid(points, transform, tile_size_px)

    if streaming:
        return _write_streaming(
            points,
            output_dir,
            catalog=catalog,
            grid=grid,
            transform=transform,
            feature_key=feature_key,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            render_only=render_only,
        )

    df = points.compute() if hasattr(points, "compute") else points
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")

    table, tile_ids = _prepare_table(
        df,
        transform=transform,
        catalog=catalog,
        feature_key=feature_key,
        grid=grid,
        categories=None,
        render_only=render_only,
    )
    table, sorted_tile_ids = _sorted_by_tile(table, tile_ids)

    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    filenames = grid.chunk_filenames(max_row_groups_per_file)
    schema = table.schema.with_metadata(
        {
            **(table.schema.metadata or {}),
            b"profile": b"grid_files_v1",
            b"storage_mode": b"row_groups_chunked",
            b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
            b"tile_grid": json.dumps(grid.to_manifest_dict()).encode(),
        }
    )

    try:
        for file_index, name in enumerate(filenames):
            lo = file_index * max_row_groups_per_file
            tile_range = range(lo, min(lo + max_row_groups_per_file, grid.num_tiles))
            with pq.ParquetWriter(
                staging / name,
                schema,
                compression=compression,
                # Statistics are dead weight here: the tile formula is the spatial index,
                # so no client consults per-column-chunk min/max, and they inflate the
                # footer the browser must download before its first read.
                write_statistics=False,
            ) as writer:
                _write_tile_row_groups(writer, table, sorted_tile_ids, tile_range, schema)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    return _manifest_fragment(
        output_dir,
        filenames,
        grid,
        transform,
        catalog,
        max_row_groups_per_file,
        int(table.num_rows),
        render_only=render_only,
    )


def _derive_grid(points: Any, transform: DisplayTransform, tile_size_px: float) -> RegularGrid:
    """Find the grid covering the element, reading only the coordinate columns."""
    xmax = points["x"].max()
    ymax = points["y"].max()
    if hasattr(xmax, "compute"):
        xmax, ymax = xmax.compute(), ymax.compute()
    px, py = transform.apply(np.array([float(xmax)]), np.array([float(ymax)]))
    return RegularGrid.from_bounds(0, 0, float(np.rint(px[0])), float(np.rint(py[0])), tile_size_px)


def _manifest_fragment(
    output_dir: Path,
    filenames: list[str],
    grid: RegularGrid,
    transform: DisplayTransform,
    catalog: FeatureCatalog,
    max_row_groups_per_file: int,
    n_rows: int,
    render_only: bool = False,
) -> dict[str, Any]:
    fragment: dict[str, Any] = {
        "directory": str(output_dir.name),
        "files": filenames,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": grid.num_tiles,
        "position_column": POSITION_COLUMN,
        "position_encoding": "fixed_size_list",
        "position_dtype": "uint32",
        "position_size": 2,
        "feature_column": FEATURE_COLUMN,
        "n_rows": n_rows,
        "tile_grid": grid.to_manifest_dict(),
        "display_transform": transform.to_manifest_dict(),
        "feature_catalog": catalog.to_manifest_dict(),
        "render_only": render_only,
    }
    return fragment


def _write_streaming(
    points: Any,
    output_dir: Path,
    *,
    catalog: FeatureCatalog,
    grid: RegularGrid,
    transform: DisplayTransform,
    feature_key: str,
    max_row_groups_per_file: int,
    compression: str,
    render_only: bool = False,
) -> dict[str, Any]:
    """Write the tiled output without holding the whole element in memory.

    Grouping rows by tile is a global sort -- a row at the end of the input can belong to
    the first tile -- so streaming the read alone is not enough. This makes two passes:

    1. Stream the input a partition at a time, and spill each row into a temporary file
       chosen by its *destination chunk file*.
    2. Sort each spill file independently and write its chunk.

    Peak memory is then one input partition plus one spill file, rather than the dataset.
    """
    filenames = grid.chunk_filenames(max_row_groups_per_file)
    categories = _known_categories(points, feature_key)

    staging = output_dir.with_name(output_dir.name + ".tmp")
    spill = output_dir.with_name(output_dir.name + ".spill")
    for path in (staging, spill):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir(parents=True)
    spill.mkdir(parents=True)

    schema: pa.Schema | None = None
    n_rows = 0
    try:
        # -- pass 1: spill by destination file ---------------------------------
        spill_writers: dict[int, pq.ParquetWriter] = {}
        for chunk in _iter_chunks(points):
            if len(chunk) == 0:
                continue
            table, tile_ids = _prepare_table(
                chunk,
                transform=transform,
                catalog=catalog,
                feature_key=feature_key,
                grid=grid,
                categories=categories,
                render_only=render_only,
            )
            n_rows += table.num_rows
            if schema is None:
                schema = table.schema
            table = table.append_column(_TILE_ID, pa.array(tile_ids))

            buckets = tile_ids // max_row_groups_per_file
            for bucket in np.unique(buckets):
                rows = np.flatnonzero(buckets == bucket)
                part = table.take(pa.array(rows))
                if bucket not in spill_writers:
                    spill_writers[int(bucket)] = pq.ParquetWriter(
                        spill / f"{int(bucket)}.parquet", table.schema, compression="zstd", write_statistics=False
                    )
                spill_writers[int(bucket)].write_table(part)
        for writer in spill_writers.values():
            writer.close()

        if schema is None:
            raise ValueError("points element is empty; nothing to tile")
        schema = schema.with_metadata(
            {
                **(schema.metadata or {}),
                b"profile": b"grid_files_v1",
                b"storage_mode": b"row_groups_chunked",
                b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
                b"tile_grid": json.dumps(grid.to_manifest_dict()).encode(),
            }
        )

        # -- pass 2: sort each spill file and write its chunk -------------------
        for file_index, name in enumerate(filenames):
            lo = file_index * max_row_groups_per_file
            tile_range = range(lo, min(lo + max_row_groups_per_file, grid.num_tiles))
            spill_path = spill / f"{file_index}.parquet"

            if spill_path.exists():
                table = pq.read_table(spill_path)
                tile_ids = table[_TILE_ID].to_numpy()
                table = table.drop_columns([_TILE_ID]).cast(schema)
                table, sorted_tile_ids = _sorted_by_tile(table, tile_ids)
            else:
                # No row landed in this chunk's tiles; it is still written, as all-empty
                # row groups, so the global row-group numbering stays contiguous.
                table, sorted_tile_ids = schema.empty_table(), np.empty(0, dtype=np.int64)

            with pq.ParquetWriter(staging / name, schema, compression=compression, write_statistics=False) as writer:
                _write_tile_row_groups(writer, table, sorted_tile_ids, tile_range, schema)

            del table
            spill_path.unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(spill, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(spill, ignore_errors=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    return _manifest_fragment(
        output_dir, filenames, grid, transform, catalog, max_row_groups_per_file, n_rows, render_only=render_only
    )
