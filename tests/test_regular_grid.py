"""Tests for the deterministic regular-grid tiling used by the visualization profile.

These pin the *normative* parts of the profile: the tile formula, half-open boundary
semantics, x-major tile numbering, and the row-group/file numbering. A change that
breaks one of these invalidates every store already written with the profile.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialdata_io.experimental.regular_grid import (
    DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    RegularGrid,
)


@pytest.fixture
def grid_2x3() -> RegularGrid:
    """The synthetic fixture grid: 2 tile columns x 3 tile rows, 10 px tiles, origin at 0."""
    return RegularGrid(origin_x=0.0, origin_y=0.0, tile_size_px=10.0, num_tiles_x=2, num_tiles_y=3)


# -- construction -------------------------------------------------------------


def test_from_bounds_covers_extent() -> None:
    grid = RegularGrid.from_bounds(0, 0, 25000, 20000, tile_size_px=250)
    assert (grid.num_tiles_x, grid.num_tiles_y) == (100, 80)
    assert grid.num_tiles == 8000
    assert grid.x_max >= 25000 and grid.y_max >= 20000


def test_from_bounds_rounds_up_partial_tiles() -> None:
    # 105 px of data at 50 px tiles needs 3 columns, not 2.
    grid = RegularGrid.from_bounds(0, 0, 105, 40, tile_size_px=50)
    assert (grid.num_tiles_x, grid.num_tiles_y) == (3, 1)


def test_from_bounds_degenerate_extent_still_valid() -> None:
    grid = RegularGrid.from_bounds(5, 5, 5, 5, tile_size_px=10)
    assert grid.num_tiles == 1


def test_from_bounds_nonzero_origin() -> None:
    grid = RegularGrid.from_bounds(100, 200, 140, 260, tile_size_px=20)
    assert (grid.origin_x, grid.origin_y) == (100.0, 200.0)
    assert (grid.num_tiles_x, grid.num_tiles_y) == (2, 3)
    # A point at the origin lands in tile (0, 0), not somewhere negative.
    assert grid.assign(np.array([100.0]), np.array([200.0]))[0] == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonpositive_tile_size(bad: float) -> None:
    with pytest.raises(ValueError, match="tile_size_px must be positive"):
        RegularGrid(0, 0, bad, 2, 3)


def test_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        RegularGrid(0, 0, 10, 0, 3)


def test_from_bounds_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="invalid bounds"):
        RegularGrid.from_bounds(10, 0, 0, 10, tile_size_px=5)


# -- tile assignment ----------------------------------------------------------


def test_tile_id_is_x_major(grid_2x3: RegularGrid) -> None:
    """tile_id = tile_x * num_tiles_y + tile_y, matching Celldega's RowGroupTileReader."""
    expected = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5}
    for (tx, ty), tid in expected.items():
        assert int(grid_2x3.tile_id(np.array(tx), np.array(ty))) == tid


def test_tile_ids_cover_every_tile_exactly_once(grid_2x3: RegularGrid) -> None:
    ids = [
        int(grid_2x3.tile_id(np.array(tx), np.array(ty)))
        for tx in range(grid_2x3.num_tiles_x)
        for ty in range(grid_2x3.num_tiles_y)
    ]
    assert sorted(ids) == list(range(grid_2x3.num_tiles))


def test_boundary_is_half_open(grid_2x3: RegularGrid) -> None:
    """A point exactly on an internal tile edge belongs to the *upper* tile."""
    tx, ty = grid_2x3.tile_xy(np.array([9.999, 10.0]), np.array([0.0, 0.0]))
    assert tx.tolist() == [0, 1]
    assert ty.tolist() == [0, 0]


def test_upper_edge_is_clamped_into_last_tile(grid_2x3: RegularGrid) -> None:
    """Points exactly on x_max / y_max stay in the grid rather than falling off it."""
    tx, ty = grid_2x3.tile_xy(np.array([20.0]), np.array([30.0]))
    assert (int(tx[0]), int(ty[0])) == (1, 2)
    assert int(grid_2x3.assign(np.array([20.0]), np.array([30.0]))[0]) == 5


def test_out_of_range_raises_rather_than_clamping(grid_2x3: RegularGrid) -> None:
    with pytest.raises(ValueError, match="outside the grid"):
        grid_2x3.tile_xy(np.array([-1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="outside the grid"):
        grid_2x3.tile_xy(np.array([100.0]), np.array([0.0]))


def test_assign_is_vectorized_and_order_preserving(grid_2x3: RegularGrid) -> None:
    x = np.array([0.0, 15.0, 5.0, 19.0])
    y = np.array([0.0, 25.0, 12.0, 5.0])
    got = grid_2x3.assign(x, y)
    assert got.tolist() == [0, 5, 1, 3]


def test_assign_empty_input(grid_2x3: RegularGrid) -> None:
    got = grid_2x3.assign(np.array([]), np.array([]))
    assert got.shape == (0,)


def test_mismatched_shapes_raise(grid_2x3: RegularGrid) -> None:
    with pytest.raises(ValueError, match="same shape"):
        grid_2x3.tile_xy(np.array([1.0, 2.0]), np.array([1.0]))


def test_tile_bounds_are_contiguous(grid_2x3: RegularGrid) -> None:
    _, _, x_max_0, _ = grid_2x3.tile_bounds(0, 0)
    x_min_1, _, _, _ = grid_2x3.tile_bounds(1, 0)
    assert x_max_0 == x_min_1


def test_tile_bounds_rejects_out_of_grid(grid_2x3: RegularGrid) -> None:
    with pytest.raises(ValueError, match="outside the grid"):
        grid_2x3.tile_bounds(2, 0)


def test_every_tile_center_maps_back_to_its_own_tile(grid_2x3: RegularGrid) -> None:
    """Round-trip: tile -> center point -> tile."""
    for tx in range(grid_2x3.num_tiles_x):
        for ty in range(grid_2x3.num_tiles_y):
            x0, y0, x1, y1 = grid_2x3.tile_bounds(tx, ty)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            got_x, got_y = grid_2x3.tile_xy(np.array([cx]), np.array([cy]))
            assert (int(got_x[0]), int(got_y[0])) == (tx, ty)


# -- row-group and file numbering ---------------------------------------------


def test_chunk_location_formula() -> None:
    grid = RegularGrid.from_bounds(0, 0, 25000, 20000, tile_size_px=250)
    assert grid.chunk_location(0, 400) == (0, 0)
    assert grid.chunk_location(399, 400) == (0, 399)
    assert grid.chunk_location(400, 400) == (1, 0)
    assert grid.chunk_location(7999, 400) == (19, 399)


def test_num_files_rounds_up() -> None:
    grid = RegularGrid.from_bounds(0, 0, 100, 100, tile_size_px=10)  # 100 tiles
    assert grid.num_files(400) == 1
    assert grid.num_files(40) == 3  # 100 / 40 -> 3 files
    assert grid.num_files(100) == 1


def test_default_max_row_groups_matches_celldega() -> None:
    assert DEFAULT_MAX_ROW_GROUPS_PER_FILE == 400


def test_chunk_filenames_are_zero_padded_for_lexicographic_order() -> None:
    """dask globs and sorts lexicographically; Celldega indexes by position. Padding satisfies both."""
    grid = RegularGrid.from_bounds(0, 0, 25000, 20000, tile_size_px=250)  # 8000 tiles -> 20 files
    names = grid.chunk_filenames(400)
    assert len(names) == 20
    assert names[0] == "chunk_00.parquet"
    assert names[10] == "chunk_10.parquet"
    assert names == sorted(names), "lexicographic order must equal numeric order"


def test_chunk_filenames_single_file_unpadded() -> None:
    grid = RegularGrid.from_bounds(0, 0, 100, 100, tile_size_px=50)  # 4 tiles -> 1 file
    assert grid.chunk_filenames(400) == ["chunk_0.parquet"]


def test_every_tile_maps_to_a_listed_file() -> None:
    grid = RegularGrid.from_bounds(0, 0, 25000, 20000, tile_size_px=250)
    names = grid.chunk_filenames(400)
    for tid in range(grid.num_tiles):
        file_index, local = grid.chunk_location(tid, 400)
        assert 0 <= file_index < len(names)
        assert 0 <= local < 400


# -- manifest round-trip ------------------------------------------------------


def test_manifest_round_trip(grid_2x3: RegularGrid) -> None:
    assert RegularGrid.from_manifest_dict(grid_2x3.to_manifest_dict()) == grid_2x3


def test_manifest_uses_celldega_key_names(grid_2x3: RegularGrid) -> None:
    d = grid_2x3.to_manifest_dict()
    assert set(d) == {"num_tiles_x", "num_tiles_y", "tile_size", "x_min", "y_min", "x_max", "y_max"}


def test_manifest_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        RegularGrid.from_manifest_dict({"num_tiles_x": 2, "num_tiles_y": 3})


# -- conformance with Celldega's reader ---------------------------------------


def test_matches_celldega_row_group_index_formula() -> None:
    """Mirror of RowGroupTileReader.computeRowGroupIndex / computeChunkLocation.

    Celldega is the reference client; this reimplements its JS formulas independently so
    that a divergence in either codebase fails here rather than in the browser.
    """
    grid = RegularGrid.from_bounds(0, 0, 25000, 20000, tile_size_px=250)
    max_rg = 400
    rng = np.random.default_rng(0)
    for tx, ty in zip(
        rng.integers(0, grid.num_tiles_x, 200),
        rng.integers(0, grid.num_tiles_y, 200),
        strict=True,
    ):
        js_row_group = int(tx) * grid.num_tiles_y + int(ty)  # computeRowGroupIndex
        js_file = js_row_group // max_rg  # computeChunkLocation
        js_local = js_row_group % max_rg
        tid = int(grid.tile_id(np.array(tx), np.array(ty)))
        assert tid == js_row_group
        assert grid.chunk_location(tid, max_rg) == (js_file, js_local)


def test_meta_gene_index_is_unnamed_so_it_serializes_as_index_level_0(tmp_path) -> None:
    """A client finds the gene list only under '__index_level_0__'.

    pandas writes a *named* index as a column of that name, which the client does not
    look for, leaving the gene list empty and the viewer with no transcript controls.
    """
    import pyarrow.parquet as pq

    from spatialdata_io.experimental.feature_catalog import FeatureCatalog

    catalog = FeatureCatalog(names=("GENEA", "GENEB", "NegControlProbe_1"), n_genes=2)
    path = tmp_path / "meta_gene.parquet"
    catalog.to_frame().to_parquet(path)

    names = pq.read_table(path).schema.names
    assert "__index_level_0__" in names
    assert "color" in names
    for stat in ("mean", "std", "max", "non-zero"):
        assert stat in names
