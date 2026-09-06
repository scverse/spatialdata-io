"""Tests for the regular-grid Shapes rewrite.

The fixture deliberately includes a polygon whose outline crosses a tile boundary while
its centroid does not, since assigning by centroid (rather than by overlap) is what keeps
each cell in exactly one row group.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import MultiPolygon, Polygon

from spatialdata_io.experimental.points_parquet import DisplayTransform
from spatialdata_io.experimental.regular_grid import RegularGrid
from spatialdata_io.experimental.shapes_parquet import (
    CELL_CODE_COLUMN,
    GEOMETRY_COLUMN,
    write_shapes_regular_grid,
)

# Same 2x3 grid of 10 px tiles as the points fixture; canonical coords are half of pixels.
GRID = RegularGrid(origin_x=0.0, origin_y=0.0, tile_size_px=10.0, num_tiles_x=2, num_tiles_y=3)
XFORM = DisplayTransform(matrix=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)), coordinate_system="global")


def _square(cx: float, cy: float, half: float) -> Polygon:
    """A square in *canonical* coords, centred on (cx, cy)."""
    return Polygon([(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])


#: name -> (geometry, expected tile id). Canonical coords; pixels are 2x.
SHAPES_SPEC = {
    "cell-a": (_square(2.5, 2.5, 1.0), 0),  # centroid px (5,5) -> tile (0,0)
    "cell-b": (_square(2.0, 7.5, 1.0), 1),  # centroid px (4,15) -> tile (0,1)
    "cell-c": (_square(2.0, 12.5, 1.0), 2),  # centroid px (4,25) -> tile (0,2)
    # centroid px (8,4) -> tile (0,0), but the outline reaches x=12px, crossing into tile_x=1
    "cell-crossing": (_square(4.0, 2.0, 2.0), 0),
    "cell-e": (_square(7.5, 12.5, 1.0), 5),  # centroid px (15,25) -> tile (1,2)
}


@pytest.fixture
def shapes() -> gpd.GeoDataFrame:
    from spatialdata.models import ShapesModel

    gdf = gpd.GeoDataFrame(
        {"area": [g.area for g, _ in SHAPES_SPEC.values()]},
        geometry=[g for g, _ in SHAPES_SPEC.values()],
        index=list(SHAPES_SPEC),
    )
    return ShapesModel.parse(gdf)


@pytest.fixture
def written(tmp_path: Path, shapes: gpd.GeoDataFrame) -> tuple[Path, dict]:
    out = tmp_path / "shapes.parquet"
    manifest = write_shapes_regular_grid(shapes, out, grid=GRID, display_transform=XFORM)
    return out, manifest


# -- layout -------------------------------------------------------------------


def test_single_file_when_grid_fits_one_chunk(written: tuple[Path, dict]) -> None:
    out, manifest = written
    assert out.is_file()
    assert manifest["path"] == "shapes.parquet"
    assert pq.ParquetFile(out).metadata.num_row_groups == GRID.num_tiles


def test_each_cell_is_in_its_centroid_tile(written: tuple[Path, dict]) -> None:
    out, _ = written
    f = pq.ParquetFile(out)
    for tile_id in range(GRID.num_tiles):
        expected = sorted(n for n, (_, t) in SHAPES_SPEC.items() if t == tile_id)
        rg = f.read_row_group(tile_id, columns=[CELL_CODE_COLUMN])
        got = sorted(list(SHAPES_SPEC)[c] for c in rg[CELL_CODE_COLUMN].to_pylist())
        assert got == expected, f"tile {tile_id}"


def test_tile_crossing_polygon_is_not_duplicated(written: tuple[Path, dict]) -> None:
    """Its outline spans two tiles but it must appear exactly once, in its centroid's tile."""
    out, _ = written
    table = pq.read_table(out)
    codes = table[CELL_CODE_COLUMN].to_pylist()
    crossing = list(SHAPES_SPEC).index("cell-crossing")
    assert codes.count(crossing) == 1
    assert table.num_rows == len(SHAPES_SPEC)


def test_every_cell_appears_exactly_once(written: tuple[Path, dict]) -> None:
    out, _ = written
    codes = pq.read_table(out)[CELL_CODE_COLUMN].to_pylist()
    assert sorted(codes) == list(range(len(SHAPES_SPEC)))


def test_empty_tile_is_zero_rows(written: tuple[Path, dict]) -> None:
    out, _ = written
    # tiles 3 and 4 hold no cells
    f = pq.ParquetFile(out)
    for tile_id in (3, 4):
        assert f.metadata.row_group(tile_id).num_rows == 0


# -- canonical preservation ---------------------------------------------------


def test_canonical_geometry_is_unchanged(written: tuple[Path, dict], shapes: gpd.GeoDataFrame) -> None:
    out, _ = written
    back = gpd.read_parquet(out)
    for name, (geom, _) in SHAPES_SPEC.items():
        assert back.loc[name].geometry.equals(geom), name


def test_output_is_still_valid_geoparquet(written: tuple[Path, dict]) -> None:
    out, _ = written
    assert b"geo" in (pq.ParquetFile(out).schema_arrow.metadata or {})
    back = gpd.read_parquet(out)
    assert isinstance(back, gpd.GeoDataFrame)
    assert back.geometry.name == "geometry"


def test_non_geometry_columns_survive(written: tuple[Path, dict], shapes: gpd.GeoDataFrame) -> None:
    out, _ = written
    back = gpd.read_parquet(out)
    for name in SHAPES_SPEC:
        assert back.loc[name, "area"] == pytest.approx(shapes.loc[name, "area"])


# -- display geometry ---------------------------------------------------------


def test_display_geometry_has_the_nested_layout(written: tuple[Path, dict]) -> None:
    """polygon -> rings -> interleaved uint32 pairs, as get_polygon_data.js walks it."""
    out, _ = written
    t = pq.ParquetFile(out).schema_arrow.field(GEOMETRY_COLUMN).type
    assert pa.types.is_list(t)  # polygon level
    assert pa.types.is_list(t.value_type)  # ring level
    assert t.value_type.value_type == pa.list_(pa.uint32(), 2)  # interleaved vertices


def test_display_geometry_offsets_resolve_like_the_js_reader(written: tuple[Path, dict]) -> None:
    """Mirror of getPolygonDataFromChunk: polygon offset -> ring offset -> coord index."""
    out, _ = written
    col = pq.read_table(out)[GEOMETRY_COLUMN].combine_chunks()
    polygon_offsets = col.offsets.to_numpy()
    rings = col.values
    ring_offsets = rings.offsets.to_numpy()
    flat = rings.values.values.to_numpy(zero_copy_only=False)

    start_indices = ring_offsets[polygon_offsets]
    assert len(start_indices) == len(col) + 1
    # First polygon's first vertex, read the way the browser would.
    first = flat[2 * start_indices[0] : 2 * start_indices[0] + 2]
    assert first.tolist() == col.to_pylist()[0][0][0]


def test_display_geometry_is_exterior_ring_only(written: tuple[Path, dict]) -> None:
    out, _ = written
    for polygon in pq.read_table(out)[GEOMETRY_COLUMN].to_pylist():
        assert len(polygon) == 1, "expected exactly one ring per display polygon"


def test_display_vertices_match_the_transform(written: tuple[Path, dict]) -> None:
    out, _ = written
    table = pq.read_table(out)
    codes = table[CELL_CODE_COLUMN].to_pylist()
    geoms = table[GEOMETRY_COLUMN].to_pylist()
    names = list(SHAPES_SPEC)
    for code, poly in zip(codes, geoms):
        canonical = SHAPES_SPEC[names[code]][0]
        expected = [[int(round(x * 2)), int(round(y * 2))] for x, y in canonical.exterior.coords]
        assert [list(v) for v in poly[0]] == expected


def test_multipolygon_reduces_to_largest_part(tmp_path: Path) -> None:
    from spatialdata.models import ShapesModel

    big, small = _square(2.5, 2.5, 2.0), _square(8.0, 13.0, 0.5)
    gdf = gpd.GeoDataFrame(geometry=[MultiPolygon([big, small])], index=["multi"])
    out = tmp_path / "m.parquet"
    write_shapes_regular_grid(ShapesModel.parse(gdf), out, grid=GRID, display_transform=XFORM)
    poly = pq.read_table(out)[GEOMETRY_COLUMN].to_pylist()[0]
    got = {tuple(v) for v in poly[0]}
    assert got == {(int(round(x * 2)), int(round(y * 2))) for x, y in big.exterior.coords}


# -- cell codes ---------------------------------------------------------------


def test_cell_codes_follow_the_table_order(tmp_path: Path, shapes: gpd.GeoDataFrame) -> None:
    """cell_code must index the annotating table, not the shapes' own row order."""
    reversed_index = list(SHAPES_SPEC)[::-1]
    out = tmp_path / "c.parquet"
    write_shapes_regular_grid(shapes, out, grid=GRID, display_transform=XFORM, cell_index=reversed_index)
    back = gpd.read_parquet(out)
    for name in SHAPES_SPEC:
        assert back.loc[name, CELL_CODE_COLUMN] == reversed_index.index(name)


def test_cell_missing_from_index_is_reported(tmp_path: Path, shapes: gpd.GeoDataFrame) -> None:
    with pytest.raises(ValueError, match="absent from cell_index"):
        write_shapes_regular_grid(
            shapes, tmp_path / "x.parquet", grid=GRID, display_transform=XFORM, cell_index=["cell-a"]
        )


def test_overwrite_guard(written: tuple[Path, dict], shapes: gpd.GeoDataFrame) -> None:
    out, _ = written
    with pytest.raises(FileExistsError):
        write_shapes_regular_grid(shapes, out, grid=GRID, display_transform=XFORM)
