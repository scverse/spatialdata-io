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

    df = points.compute() if hasattr(points, "compute") else points
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    if feature_key not in df.columns:
        raise ValueError(f"feature column {feature_key!r} not found; have {list(df.columns)}")
    for axis in ("x", "y"):
        if axis not in df.columns:
            raise ValueError(f"points element has no {axis!r} column; have {list(df.columns)}")

    # When called as a SpatialData ``points_writer`` hook the element arrives with its
    # transformations already stripped from attrs, so the caller must supply the transform.
    transform = display_transform or DisplayTransform.from_element(points, coordinate_system)
    px, py = _to_display_pixels(df["x"].to_numpy(), df["y"].to_numpy(), transform)

    if grid is None:
        grid = RegularGrid.from_bounds(0, 0, float(px.max()), float(py.max()), tile_size_px)

    tile_ids = grid.assign(px, py)
    codes = catalog.encode(df[feature_key])

    # Keep every canonical column and index; append the two render columns.
    # The transform lives in .attrs and is not JSON-serializable, so drop it before the
    # Arrow conversion exactly as spatialdata's own points writer does -- it is persisted
    # in the element's zarr attributes, not in the parquet file.
    stale = [c for c in (POSITION_COLUMN, FEATURE_COLUMN) if c in df.columns]
    if df.attrs or stale:
        df = df.copy(deep=False)
        df.attrs = {}
        # Re-running the optimizer on an already-optimized element must replace the render
        # columns, not append duplicates. A duplicated name makes the file unreadable by
        # column projection ("Multiple matches for FieldRef"), and pandas round-trips
        # fixed_size_list back as a variable-length list, so the stale copy is also the
        # wrong Arrow type. Both are recomputed below from the canonical coordinates.
        if stale:
            df = df.drop(columns=stale)
    table = pa.Table.from_pandas(df, preserve_index=True)
    table = table.append_column(POSITION_COLUMN, _interleaved_positions(px, py))
    table = table.append_column(FEATURE_COLUMN, pa.array(codes))

    # Stable sort keeps the original relative order inside a tile, so the rewrite is
    # deterministic and diffable.
    order = np.argsort(tile_ids, kind="stable")
    table = table.take(pa.array(order))
    sorted_tile_ids = tile_ids[order]

    # Row-group boundaries: offsets[t]..offsets[t+1] is tile t's slice.
    offsets = np.searchsorted(sorted_tile_ids, np.arange(grid.num_tiles + 1), side="left")

    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    filenames = grid.chunk_filenames(max_row_groups_per_file)
    schema = table.schema.with_metadata(
        {
            **(table.schema.metadata or {}),
            b"profile": b"celldega_regular_grid_v1",
            b"storage_mode": b"row_groups_chunked",
            b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
            b"tile_grid": json.dumps(grid.to_manifest_dict()).encode(),
        }
    )

    try:
        writer: pq.ParquetWriter | None = None
        current_file = -1
        for tile_id in range(grid.num_tiles):
            file_index, _ = grid.chunk_location(tile_id, max_row_groups_per_file)
            if file_index != current_file:
                if writer is not None:
                    writer.close()
                writer = pq.ParquetWriter(
                    staging / filenames[file_index],
                    schema,
                    compression=compression,
                    # Statistics are dead weight here: the tile formula is the spatial
                    # index, so no client ever consults per-column-chunk min/max, and
                    # they inflate the footer the browser must download up front.
                    write_statistics=False,
                )
                current_file = file_index

            start, end = int(offsets[tile_id]), int(offsets[tile_id + 1])
            assert writer is not None
            # Empty tiles are written as zero-row row groups so that
            # row_group_index == tile_id holds without a lookup table.
            writer.write_table(table.slice(start, end - start) if end > start else schema.empty_table())
        if writer is not None:
            writer.close()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    return {
        "directory": str(output_dir.name),
        "files": filenames,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": grid.num_tiles,
        "position_column": POSITION_COLUMN,
        "position_encoding": "fixed_size_list",
        "position_dtype": "uint32",
        "position_size": 2,
        "feature_column": FEATURE_COLUMN,
        # Projected by the client, so canonical coordinates, ids and QC columns are never
        # decoded or transferred during ordinary rendering.
        "columns": [POSITION_COLUMN, FEATURE_COLUMN],
        "n_rows": int(table.num_rows),
        "tile_grid": grid.to_manifest_dict(),
        "display_transform": transform.to_manifest_dict(),
        "feature_catalog": catalog.to_manifest_dict(),
    }
