"""Tests for the regular-grid Points rewrite.

The synthetic fixture is a 2x3 tile grid deliberately containing the awkward cases:
an empty tile, a point exactly on a tile boundary, several points sharing one tile,
and both genes and a non-gene control feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.points_parquet import (
    FEATURE_COLUMN,
    POSITION_COLUMN,
    write_points_regular_grid,
)
from spatialdata_io.experimental.regular_grid import RegularGrid

# Display pixel space is 20 x 30 px => a 2 x 3 grid of 10 px tiles.
# Canonical (micron) coordinates are half of the pixel values, i.e. Scale([2, 2]).
TILE_SIZE = 10.0
GRID = RegularGrid(origin_x=0.0, origin_y=0.0, tile_size_px=TILE_SIZE, num_tiles_x=2, num_tiles_y=3)

#: (pixel_x, pixel_y, feature, expected_tile_id). Tile 4 is intentionally absent.
POINTS_SPEC = [
    (1.0, 1.0, "GENEA", 0),
    (2.0, 3.0, "GENEB", 0),
    (5.0, 5.0, "GENEA", 0),  # three points share tile 0
    (3.0, 12.0, "GENEB", 1),
    (4.0, 25.0, "GENEA", 2),
    (10.0, 2.0, "GENEB", 3),  # exactly on the x tile boundary -> upper tile
    (15.0, 28.0, "NegControlProbe_00042", 5),
]
VAR_NAMES = ["GENEA", "GENEB"]


@pytest.fixture
def catalog() -> FeatureCatalog:
    return FeatureCatalog.from_features_and_table([s[2] for s in POINTS_SPEC], VAR_NAMES)


@pytest.fixture
def points() -> pd.DataFrame:
    """A pandas Points-like frame with canonical micron coords and extra annotations."""
    px = np.array([s[0] for s in POINTS_SPEC])
    py = np.array([s[1] for s in POINTS_SPEC])
    df = pd.DataFrame(
        {
            "x": px / 2.0,
            "y": py / 2.0,
            "z": np.linspace(0.0, 1.0, len(POINTS_SPEC)),
            "feature_name": pd.Categorical([s[2] for s in POINTS_SPEC]),
            "cell_id": [f"cell-{i}" for i in range(len(POINTS_SPEC))],
            "transcript_id": np.arange(100, 100 + len(POINTS_SPEC), dtype=np.uint64),
            "qv": np.linspace(20.0, 40.0, len(POINTS_SPEC)).astype(np.float32),
        }
    )
    from spatialdata.models import PointsModel

    return PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "z"}, feature_key="feature_name")


@pytest.fixture
def written(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> tuple[Path, dict]:
    from spatialdata.transformations import Scale, set_transformation

    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), "global")
    out = tmp_path / "points.parquet"
    manifest = write_points_regular_grid(points, out, catalog=catalog, grid=GRID)
    return out, manifest


def _read_all(directory: Path, manifest: dict) -> pa.Table:
    return pa.concat_tables([pq.read_table(directory / f) for f in manifest["files"]])


# -- row-group layout ---------------------------------------------------------


def test_row_group_count_equals_tile_count(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    total = sum(pq.ParquetFile(directory / f).metadata.num_row_groups for f in manifest["files"])
    assert total == GRID.num_tiles == 6
    assert manifest["total_row_groups"] == 6


def test_each_tile_row_group_holds_exactly_its_points(written: tuple[Path, dict]) -> None:
    """The core contract: row_group_index == tile_id, with no lookup table."""
    directory, manifest = written
    expected: dict[int, list[tuple[float, float]]] = {}
    for pxv, pyv, _, tid in POINTS_SPEC:
        expected.setdefault(tid, []).append((pxv, pyv))

    for tile_id in range(GRID.num_tiles):
        file_index, local = GRID.chunk_location(tile_id, manifest["max_row_groups_per_file"])
        f = pq.ParquetFile(directory / manifest["files"][file_index])
        rg = f.read_row_group(local, columns=[POSITION_COLUMN])
        got = [tuple(v) for v in rg[POSITION_COLUMN].to_pylist()]
        assert sorted(got) == sorted(expected.get(tile_id, [])), f"tile {tile_id}"


def test_empty_tile_is_a_zero_row_row_group(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    file_index, local = GRID.chunk_location(4, manifest["max_row_groups_per_file"])
    f = pq.ParquetFile(directory / manifest["files"][file_index])
    assert f.metadata.row_group(local).num_rows == 0
    assert f.read_row_group(local).num_rows == 0


def test_boundary_point_goes_to_the_upper_tile(written: tuple[Path, dict]) -> None:
    """A point at exactly x=10 belongs to tile_x=1 under half-open bounds."""
    directory, manifest = written
    f = pq.ParquetFile(directory / manifest["files"][0])
    rg = f.read_row_group(3, columns=[POSITION_COLUMN])
    assert [tuple(v) for v in rg[POSITION_COLUMN].to_pylist()] == [(10, 2)]


# -- canonical data preservation ----------------------------------------------


def test_no_row_lost_or_duplicated(written: tuple[Path, dict], points: pd.DataFrame) -> None:
    directory, manifest = written
    table = _read_all(directory, manifest)
    original = points.compute() if hasattr(points, "compute") else points
    assert table.num_rows == len(POINTS_SPEC)
    assert sorted(table["transcript_id"].to_pylist()) == sorted(original["transcript_id"].tolist())


def test_canonical_columns_are_unchanged(written: tuple[Path, dict], points: pd.DataFrame) -> None:
    """Values must survive the reorder exactly; only row order may differ."""
    directory, manifest = written
    got = _read_all(directory, manifest).to_pandas()
    original = points.compute() if hasattr(points, "compute") else points

    merged = got.set_index("transcript_id").loc[original["transcript_id"].to_numpy()]
    for col in ("x", "y", "z", "cell_id", "qv"):
        np.testing.assert_array_equal(
            merged[col].to_numpy(), original[col].to_numpy(), err_msg=f"column {col} changed"
        )
    assert list(merged["feature_name"].astype(str)) == list(original["feature_name"].astype(str))


def test_index_is_preserved(written: tuple[Path, dict], points: pd.DataFrame) -> None:
    directory, manifest = written
    got = _read_all(directory, manifest).to_pandas()
    original = points.compute() if hasattr(points, "compute") else points
    assert sorted(got.index.tolist()) == sorted(original.index.tolist())


def test_row_order_is_grouped_by_tile(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    table = _read_all(directory, manifest)
    xs = np.array([v[0] for v in table[POSITION_COLUMN].to_pylist()])
    ys = np.array([v[1] for v in table[POSITION_COLUMN].to_pylist()])
    tile_ids = GRID.assign(xs, ys)
    assert (np.diff(tile_ids) >= 0).all(), "rows are not grouped by tile"


# -- render columns -----------------------------------------------------------


def test_display_xy_is_fixed_size_list_uint32(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    field = pq.ParquetFile(directory / manifest["files"][0]).schema_arrow.field(POSITION_COLUMN)
    assert field.type == pa.list_(pa.uint32(), 2)
    assert manifest["position_dtype"] == "uint32"
    assert manifest["position_encoding"] == "fixed_size_list"


def test_display_xy_child_buffer_is_interleaved(written: tuple[Path, dict]) -> None:
    """The flat child buffer must be [x0,y0,x1,y1,...] so deck.gl can consume it directly."""
    directory, manifest = written
    col = _read_all(directory, manifest)[POSITION_COLUMN].combine_chunks()
    flat = col.values.to_numpy(zero_copy_only=False)
    pairs = [tuple(v) for v in col.to_pylist()]
    assert flat[0::2].tolist() == [p[0] for p in pairs]
    assert flat[1::2].tolist() == [p[1] for p in pairs]


def test_display_xy_matches_the_transform(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    table = _read_all(directory, manifest)
    got = {int(t): tuple(v) for t, v in zip(table["transcript_id"].to_pylist(), table[POSITION_COLUMN].to_pylist())}
    for i, (pxv, pyv, _, _) in enumerate(POINTS_SPEC):
        assert got[100 + i] == (int(pxv), int(pyv))


def test_feature_codes_match_catalog(written: tuple[Path, dict], catalog: FeatureCatalog) -> None:
    directory, manifest = written
    table = _read_all(directory, manifest)
    for name, code in zip(table["feature_name"].to_pylist(), table[FEATURE_COLUMN].to_pylist()):
        assert catalog.names[code] == name


def test_control_feature_is_coded_above_every_gene(written: tuple[Path, dict], catalog: FeatureCatalog) -> None:
    directory, manifest = written
    table = _read_all(directory, manifest).to_pandas()
    control = table[table["feature_name"].astype(str).str.startswith("NegControl")]
    assert len(control) == 1
    assert int(control[FEATURE_COLUMN].iloc[0]) >= catalog.n_genes


# -- column projection --------------------------------------------------------


def test_column_projection_reads_only_render_columns(written: tuple[Path, dict]) -> None:
    """Celldega projects these two columns; canonical columns must not be required."""
    directory, manifest = written
    f = pq.ParquetFile(directory / manifest["files"][0])
    rg = f.read_row_group(0, columns=[POSITION_COLUMN, FEATURE_COLUMN])
    assert rg.column_names == [POSITION_COLUMN, FEATURE_COLUMN]
    assert rg.num_rows == 3


# -- file layout --------------------------------------------------------------


def test_multi_file_split_and_padding(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> None:
    from spatialdata.transformations import Scale, set_transformation

    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), "global")
    out = tmp_path / "multi.parquet"
    manifest = write_points_regular_grid(points, out, catalog=catalog, grid=GRID, max_row_groups_per_file=2)

    assert manifest["files"] == ["chunk_0.parquet", "chunk_1.parquet", "chunk_2.parquet"]
    assert [pq.ParquetFile(out / f).metadata.num_row_groups for f in manifest["files"]] == [2, 2, 2]
    assert sum(pq.ParquetFile(out / f).metadata.num_rows for f in manifest["files"]) == len(POINTS_SPEC)


def test_statistics_are_disabled(written: tuple[Path, dict]) -> None:
    """Footer weight matters in the browser and the tile formula is the spatial index."""
    directory, manifest = written
    rg = pq.ParquetFile(directory / manifest["files"][0]).metadata.row_group(0)
    assert not rg.column(0).is_stats_set


def test_overwrite_guard(written: tuple[Path, dict], points: pd.DataFrame, catalog: FeatureCatalog) -> None:
    directory, _ = written
    with pytest.raises(FileExistsError):
        write_points_regular_grid(points, directory, catalog=catalog, grid=GRID)


def test_failure_leaves_no_partial_output(tmp_path: Path, points: pd.DataFrame) -> None:
    """A mid-write error must not leave a half-rewritten directory behind."""
    from spatialdata.transformations import Scale, set_transformation

    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), "global")
    out = tmp_path / "boom.parquet"
    bad = FeatureCatalog(names=("GENEA",), n_genes=1)  # missing GENEB -> encode() raises
    with pytest.raises(ValueError, match="not in the catalog"):
        write_points_regular_grid(points, out, catalog=bad, grid=GRID)
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()


# -- validation ---------------------------------------------------------------


def test_negative_display_coordinates_are_rejected(tmp_path: Path, catalog: FeatureCatalog) -> None:
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Scale, set_transformation

    df = pd.DataFrame({"x": [-5.0, 1.0], "y": [1.0, 1.0], "feature_name": pd.Categorical(["GENEA", "GENEB"])})
    p = PointsModel.parse(df, coordinates={"x": "x", "y": "y"}, feature_key="feature_name")
    set_transformation(p, Scale([2.0, 2.0], axes=("x", "y")), "global")
    with pytest.raises(ValueError, match="negative values"):
        write_points_regular_grid(p, tmp_path / "neg.parquet", catalog=catalog, grid=GRID)


def test_missing_feature_column_is_reported(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> None:
    with pytest.raises(ValueError, match="feature column 'nope' not found"):
        write_points_regular_grid(points, tmp_path / "x.parquet", catalog=catalog, grid=GRID, feature_key="nope")


def test_rewrite_is_idempotent(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> None:
    """Re-optimizing an already-optimized element replaces the render columns, not appends them.

    A duplicated column name makes projected reads fail outright, and pandas round-trips
    fixed_size_list back as a variable-length list, so a stale copy is also mistyped.
    """
    from spatialdata.transformations import Scale, set_transformation

    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), "global")
    out = tmp_path / "idem.parquet"
    m1 = write_points_regular_grid(points, out, catalog=catalog, grid=GRID)
    first = _read_all(out, m1)

    # Feed the written result back in, the way read_zarr would hand it back: a dask
    # frame that already carries display_xy/feature_code.
    import dask.dataframe as dd

    again = dd.from_pandas(first.to_pandas(), npartitions=1)
    # Attach the transform the way read_zarr does, rather than via set_transformation,
    # which requires an element that already carries one.
    again.attrs["transform"] = {"global": Scale([2.0, 2.0], axes=("x", "y"))}
    assert POSITION_COLUMN in again.columns  # precondition: the stale columns are present
    m2 = write_points_regular_grid(again, out, catalog=catalog, grid=GRID, overwrite=True)
    second = _read_all(out, m2)

    assert second.column_names.count(POSITION_COLUMN) == 1
    assert second.column_names.count(FEATURE_COLUMN) == 1
    assert second.schema.field(POSITION_COLUMN).type == pa.list_(pa.uint32(), 2)
    assert second.num_rows == first.num_rows
    assert second[POSITION_COLUMN].to_pylist() == first[POSITION_COLUMN].to_pylist()
