"""Rewrite a Shapes element into regular-grid row groups.

Adds two render-oriented columns beside the canonical geometry:

``display_geometry``
    ``list<list<fixed_size_list<uint32>[2]>>`` -- polygon -> rings -> interleaved integer
    pixel vertices. The nesting is chosen so a client can lift deck.gl's ``getPolygon``
    straight out of the flat coordinate child buffer and ``startIndices`` out of the list
    offsets, with no WKB parsing and no per-vertex JavaScript objects.
``cell_code``
    Positional index into the annotating table, so cells can be coloured from a
    cell-by-gene vector without a string join in the browser.

``display_geometry`` is explicitly a *lossy display* representation: only the exterior
ring is kept, and for a MultiPolygon only its largest part. The canonical geometry column
is written through unchanged, and the GeoParquet metadata is preserved so the file is
still readable by :func:`geopandas.read_parquet` and by SpatialData itself.

Each cell is assigned to exactly one tile, by its centroid in display pixel space. A
polygon whose outline crosses into a neighbouring tile is *not* duplicated -- duplicating
would inflate the file and make cell counts wrong.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from numpy.typing import NDArray

from spatialdata_io.experimental.points_parquet import DisplayTransform, _to_display_pixels
from spatialdata_io.experimental.regular_grid import (
    DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    RegularGrid,
)

__all__ = ["GEOMETRY_COLUMN", "CELL_CODE_COLUMN", "write_shapes_regular_grid"]

#: Column holding the nested integer-pixel display polygons.
GEOMETRY_COLUMN = "display_geometry"
#: Column holding the positional cell index.
CELL_CODE_COLUMN = "cell_code"


def _largest_polygon(geom: Any) -> Any:
    """Reduce a MultiPolygon to its largest part, matching Celldega's existing behaviour."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _exterior_only(geometry: Any) -> NDArray[Any]:
    """Return an array of single-ring polygons: largest part, exterior ring only."""
    geoms = np.asarray([_largest_polygon(g) for g in geometry], dtype=object)
    rings = shapely.get_exterior_ring(geoms)
    return shapely.polygons(rings)


def _display_geometry_array(
    geometry: Any, transform: DisplayTransform
) -> tuple[pa.ListArray, NDArray[np.uint32], NDArray[np.uint32]]:
    """Build the nested display-geometry array and the per-cell display centroids.

    Returns the Arrow array plus the integer pixel centroid coordinates used for tiling.
    """
    simple = _exterior_only(geometry)

    # to_ragged_array gives exactly the buffers the nested Arrow layout needs:
    # a flat (N, 2) coordinate array plus ring and polygon offsets.
    _, coords, offsets = shapely.to_ragged_array(simple)
    ring_offsets, polygon_offsets = offsets

    px, py = _to_display_pixels(coords[:, 0], coords[:, 1], transform)
    flat = np.empty(px.size * 2, dtype=np.uint32)
    flat[0::2] = px
    flat[1::2] = py

    vertices = pa.FixedSizeListArray.from_arrays(pa.array(flat), 2)
    rings = pa.ListArray.from_arrays(pa.array(ring_offsets, type=pa.int32()), vertices)
    polygons = pa.ListArray.from_arrays(pa.array(polygon_offsets, type=pa.int32()), rings)

    centroids = shapely.centroid(simple)
    cx, cy = _to_display_pixels(shapely.get_x(centroids), shapely.get_y(centroids), transform)
    return polygons, cx, cy


def _canonical_geoparquet_table(shapes: Any) -> pa.Table:
    """Convert a GeoDataFrame to Arrow while keeping the GeoParquet ``geo`` metadata.

    ``GeoDataFrame.to_arrow()`` emits GeoArrow *extension* metadata but not the GeoParquet
    ``geo`` schema key, which is added by ``to_parquet()``. Without that key the rewritten
    file is no longer readable by :func:`geopandas.read_parquet` or by SpatialData. We
    therefore reuse the same conversion ``to_parquet`` performs, rather than writing the
    file once just to recover its metadata.
    """
    try:
        from geopandas.io.arrow import _geopandas_to_arrow
    except ImportError as exc:  # pragma: no cover - depends on geopandas internals
        raise RuntimeError(
            "could not access geopandas' arrow conversion; a geopandas version with "
            "geopandas.io.arrow._geopandas_to_arrow is required to preserve GeoParquet metadata"
        ) from exc

    table = _geopandas_to_arrow(shapes, index=None, geometry_encoding="WKB")
    if b"geo" not in (table.schema.metadata or {}):  # pragma: no cover - defensive
        raise RuntimeError("geopandas did not produce GeoParquet 'geo' metadata")
    return table


def write_shapes_regular_grid(
    shapes: Any,
    output_path: str | Path,
    *,
    grid: RegularGrid,
    display_transform: DisplayTransform | None = None,
    coordinate_system: str = "global",
    cell_index: Any | None = None,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    compression: str = "snappy",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a Shapes element as regular-grid row groups.

    Parameters
    ----------
    shapes
        A SpatialData Shapes element (GeoDataFrame).
    output_path
        Destination. A single ``shapes.parquet`` file when it fits in one chunk (which is
        what SpatialData's reader expects), otherwise a directory of chunk files.
    grid
        The tile grid; must be the same grid used for the points element.
    display_transform
        Transform to display pixel space. Derived from the element when omitted.
    coordinate_system
        Coordinate system used when deriving the transform.
    cell_index
        Index defining ``cell_code`` order, normally the annotating table's ``obs_names``.
        Defaults to the shapes' own index order.
    max_row_groups_per_file
        Row groups per chunk file.
    compression
        Parquet compression codec.
    overwrite
        Replace an existing output.

    Returns
    -------
    The manifest fragment describing the written file(s).
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")

    transform = display_transform or DisplayTransform.from_element(shapes, coordinate_system)
    display, cx, cy = _display_geometry_array(shapes.geometry, transform)

    table = _canonical_geoparquet_table(shapes)

    if cell_index is None:
        codes = np.arange(len(shapes), dtype=np.uint32)
    else:
        positions = {k: i for i, k in enumerate(cell_index)}
        missing = [k for k in shapes.index if k not in positions]
        if missing:
            raise ValueError(
                f"{len(missing)} shape(s) are absent from cell_index (e.g. {missing[:3]}); "
                f"cell_code would be undefined. Pass the table's obs_names for these cells."
            )
        codes = np.fromiter((positions[k] for k in shapes.index), dtype=np.uint32, count=len(shapes))

    table = table.append_column(GEOMETRY_COLUMN, display)
    table = table.append_column(CELL_CODE_COLUMN, pa.array(codes))

    tile_ids = grid.assign(cx, cy)
    order = np.argsort(tile_ids, kind="stable")
    table = table.take(pa.array(order))
    offsets = np.searchsorted(tile_ids[order], np.arange(grid.num_tiles + 1), side="left")

    n_files = grid.num_files(max_row_groups_per_file)
    single_file = n_files == 1
    filenames = ["shapes.parquet"] if single_file else grid.chunk_filenames(max_row_groups_per_file)

    schema = table.schema.with_metadata(
        {
            **(table.schema.metadata or {}),
            b"profile": b"celldega_regular_grid_v1",
            b"storage_mode": b"row_groups_chunked",
            b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
            b"tile_grid": json.dumps(grid.to_manifest_dict()).encode(),
        }
    )

    staging = output_path.with_name(output_path.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging) if staging.is_dir() else staging.unlink()

    try:
        if single_file:
            staging.parent.mkdir(parents=True, exist_ok=True)
            targets = [staging]
        else:
            staging.mkdir(parents=True)
            targets = [staging / f for f in filenames]

        writer: pq.ParquetWriter | None = None
        current = -1
        for tile_id in range(grid.num_tiles):
            file_index = 0 if single_file else grid.chunk_location(tile_id, max_row_groups_per_file)[0]
            if file_index != current:
                if writer is not None:
                    writer.close()
                writer = pq.ParquetWriter(targets[file_index], schema, compression=compression, write_statistics=False)
                current = file_index
            start, end = int(offsets[tile_id]), int(offsets[tile_id + 1])
            assert writer is not None
            writer.write_table(table.slice(start, end - start) if end > start else schema.empty_table())
        if writer is not None:
            writer.close()
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging) if staging.is_dir() else staging.unlink()
        raise

    if output_path.exists():
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()
    staging.rename(output_path)

    fragment: dict[str, Any] = {
        "geometry_column": GEOMETRY_COLUMN,
        "cell_id_column": CELL_CODE_COLUMN,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": grid.num_tiles,
        "n_shapes": int(table.num_rows),
        "geometry_is_lossy": True,
        "geometry_note": "exterior ring of the largest polygon part; canonical geometry retained",
        "tile_grid": grid.to_manifest_dict(),
    }
    if single_file:
        fragment["path"] = output_path.name
    else:
        fragment["directory"] = output_path.name
        fragment["files"] = filenames
    return fragment
