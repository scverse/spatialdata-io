"""Tests for the regular-grid Points rewrite.

Two outputs are produced from one element and both are tested here:

* the **canonical** file, re-ordered into tile row groups but carrying only its own
  columns, so it still round-trips through ``SpatialData.write()``;
* the **render** file, holding only ``display_xy`` and ``feature_code``, which a viewer
  reads in full -- no column projection, because parquet-wasm's projection corrupts the
  IPC stream it emits.

The synthetic fixture is a 2x3 tile grid deliberately containing the awkward cases: an
empty tile, a point exactly on a tile boundary, several points sharing one tile, and both
genes and a non-gene control feature.
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
    """A Points element with canonical micron coords and extra annotations."""
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
    from spatialdata.transformations import Scale, set_transformation

    element = PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "z"}, feature_key="feature_name")
    set_transformation(element, Scale([2.0, 2.0], axes=("x", "y")), "global")
    return element


@pytest.fixture
def written(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> tuple[Path, dict]:
    """The canonical element: tile-ordered, carrying only its own columns."""
    out = tmp_path / "points.parquet"
    return out, write_points_regular_grid(points, out, catalog=catalog, grid=GRID)


@pytest.fixture
def rendered(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> tuple[Path, dict]:
    """The standalone render file: display_xy and feature_code only."""
    out = tmp_path / "trx"
    return out, write_points_regular_grid(points, out, catalog=catalog, grid=GRID, render_only=True)


def _read_all(directory: Path, manifest: dict) -> pa.Table:
    return pa.concat_tables([pq.read_table(directory / f) for f in manifest["files"]])


def _row_group(directory: Path, manifest: dict, tile_id: int) -> pa.Table:
    file_index, local = GRID.chunk_location(tile_id, manifest["max_row_groups_per_file"])
    return pq.ParquetFile(directory / manifest["files"][file_index]).read_row_group(local)


# -- row-group layout ---------------------------------------------------------


@pytest.mark.parametrize("fixture", ["written", "rendered"])
def test_row_group_count_equals_tile_count(fixture: str, request: pytest.FixtureRequest) -> None:
    directory, manifest = request.getfixturevalue(fixture)
    total = sum(pq.ParquetFile(directory / f).metadata.num_row_groups for f in manifest["files"])
    assert total == GRID.num_tiles == 6
    assert manifest["total_row_groups"] == 6


@pytest.mark.parametrize("fixture", ["written", "rendered"])
def test_each_tile_row_group_holds_exactly_its_points(fixture: str, request: pytest.FixtureRequest) -> None:
    """The core contract: row_group_index == tile_id, with no lookup table."""
    directory, manifest = request.getfixturevalue(fixture)
    expected: dict[int, int] = {}
    for *_, tid in POINTS_SPEC:
        expected[tid] = expected.get(tid, 0) + 1
    for tile_id in range(GRID.num_tiles):
        got = _row_group(directory, manifest, tile_id).num_rows
        assert got == expected.get(tile_id, 0), f"tile {tile_id}"


@pytest.mark.parametrize("fixture", ["written", "rendered"])
def test_empty_tile_is_a_zero_row_row_group(fixture: str, request: pytest.FixtureRequest) -> None:
    directory, manifest = request.getfixturevalue(fixture)
    file_index, local = GRID.chunk_location(4, manifest["max_row_groups_per_file"])
    md = pq.ParquetFile(directory / manifest["files"][file_index]).metadata
    assert md.row_group(local).num_rows == 0


def test_boundary_point_goes_to_the_upper_tile(rendered: tuple[Path, dict]) -> None:
    """A point at exactly x=10 belongs to tile_x=1 under half-open bounds."""
    directory, manifest = rendered
    rg = _row_group(directory, manifest, 3)
    assert [tuple(v) for v in rg[POSITION_COLUMN].to_pylist()] == [(10, 2)]


# -- the canonical element ----------------------------------------------------


def test_canonical_file_has_no_render_columns(written: tuple[Path, dict]) -> None:
    """A nested Arrow column cannot survive dask's parquet round-trip, so it stays out."""
    directory, manifest = written
    names = _read_all(directory, manifest).column_names
    assert POSITION_COLUMN not in names
    assert FEATURE_COLUMN not in names
    assert manifest["render_only"] is False


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
        np.testing.assert_array_equal(merged[col].to_numpy(), original[col].to_numpy(), err_msg=col)
    assert list(merged["feature_name"].astype(str)) == list(original["feature_name"].astype(str))


def test_index_is_preserved(written: tuple[Path, dict], points: pd.DataFrame) -> None:
    directory, manifest = written
    got = _read_all(directory, manifest).to_pandas()
    original = points.compute() if hasattr(points, "compute") else points
    assert sorted(got.index.tolist()) == sorted(original.index.tolist())


def test_canonical_rows_are_grouped_by_tile(written: tuple[Path, dict]) -> None:
    """The reordering is the point: it is what makes spatial subsetting cheap in Python."""
    directory, manifest = written
    df = _read_all(directory, manifest).to_pandas()
    tile_ids = GRID.assign(df["x"].to_numpy() * 2, df["y"].to_numpy() * 2)
    assert (np.diff(tile_ids) >= 0).all()


# -- the render file ----------------------------------------------------------


def test_render_file_holds_only_the_render_columns(rendered: tuple[Path, dict]) -> None:
    """A viewer reads every column of this file, which is why no projection is needed."""
    directory, manifest = rendered
    assert _read_all(directory, manifest).column_names == [POSITION_COLUMN, FEATURE_COLUMN]
    assert manifest["render_only"] is True


def test_display_xy_is_fixed_size_list_uint32(rendered: tuple[Path, dict]) -> None:
    directory, manifest = rendered
    field = pq.ParquetFile(directory / manifest["files"][0]).schema_arrow.field(POSITION_COLUMN)
    assert field.type == pa.list_(pa.uint32(), 2)
    assert manifest["position_dtype"] == "uint32"


def test_display_xy_child_buffer_is_interleaved(rendered: tuple[Path, dict]) -> None:
    """The flat child buffer must be [x0,y0,x1,y1,...] so deck.gl can consume it directly."""
    directory, manifest = rendered
    col = _read_all(directory, manifest)[POSITION_COLUMN].combine_chunks()
    flat = col.values.to_numpy(zero_copy_only=False)
    pairs = [tuple(v) for v in col.to_pylist()]
    assert flat[0::2].tolist() == [p[0] for p in pairs]
    assert flat[1::2].tolist() == [p[1] for p in pairs]


def test_display_xy_matches_the_transform(rendered: tuple[Path, dict]) -> None:
    directory, manifest = rendered
    got = sorted(tuple(v) for v in _read_all(directory, manifest)[POSITION_COLUMN].to_pylist())
    assert got == sorted((int(x), int(y)) for x, y, _, _ in POINTS_SPEC)


def _codes_by_position(directory: Path, manifest: dict) -> dict[tuple[int, int], int]:
    table = _read_all(directory, manifest)
    return {tuple(v): c for v, c in zip(table[POSITION_COLUMN].to_pylist(), table[FEATURE_COLUMN].to_pylist())}


def test_feature_codes_match_catalog(rendered: tuple[Path, dict], catalog: FeatureCatalog) -> None:
    codes = _codes_by_position(*rendered)
    for x, y, feature, _ in POINTS_SPEC:
        assert catalog.names[codes[(int(x), int(y))]] == feature


def test_control_feature_is_coded_above_every_gene(rendered: tuple[Path, dict], catalog: FeatureCatalog) -> None:
    codes = _codes_by_position(*rendered)
    assert codes[(15, 28)] >= catalog.n_genes


def test_render_file_is_smaller_than_canonical(written: tuple[Path, dict], rendered: tuple[Path, dict]) -> None:
    """The render file is what crosses the wire, so it should be a fraction of canonical."""
    canonical = sum(f.stat().st_size for f in written[0].glob("*.parquet"))
    render = sum(f.stat().st_size for f in rendered[0].glob("*.parquet"))
    assert render < canonical


# -- file layout --------------------------------------------------------------


def test_multi_file_split_and_padding(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> None:
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
    out = tmp_path / "boom.parquet"
    bad = FeatureCatalog(names=("GENEA",), n_genes=1)  # missing GENEB -> encode() raises
    with pytest.raises(ValueError, match="not in the catalog"):
        write_points_regular_grid(points, out, catalog=bad, grid=GRID, render_only=True)
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


# -- streaming ----------------------------------------------------------------


def _as_partitioned(frame: pd.DataFrame, npartitions: int = 3):
    import dask.dataframe as dd
    from spatialdata.transformations import Scale

    pdf = frame.compute() if hasattr(frame, "compute") else frame
    out = dd.from_pandas(pdf, npartitions=npartitions)
    out.attrs["transform"] = {"global": Scale([2.0, 2.0], axes=("x", "y"))}
    return out


@pytest.mark.parametrize("render_only", [False, True])
def test_streaming_matches_in_memory(
    tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog, render_only: bool
) -> None:
    """The two write paths must be interchangeable, or large datasets would diverge.

    Grouping by tile is a global sort, so the streaming path spills rows into
    per-output-file buckets and sorts each independently. That must land every row in the
    same row group as the single-pass path.
    """
    in_memory = tmp_path / "mem"
    m1 = write_points_regular_grid(
        points, in_memory, catalog=catalog, grid=GRID, streaming=False, render_only=render_only
    )
    streamed = tmp_path / "stream"
    m2 = write_points_regular_grid(
        _as_partitioned(points),
        streamed,
        catalog=catalog,
        grid=GRID,
        streaming=True,
        max_row_groups_per_file=2,
        render_only=render_only,
    )

    assert m1["total_row_groups"] == m2["total_row_groups"] == GRID.num_tiles
    assert m1["n_rows"] == m2["n_rows"] == len(POINTS_SPEC)
    for tile_id in range(GRID.num_tiles):
        assert _row_group(in_memory, m1, tile_id).num_rows == _row_group(streamed, m2, tile_id).num_rows


def test_streaming_preserves_canonical_columns(tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog) -> None:
    original = points.compute() if hasattr(points, "compute") else points
    out = tmp_path / "s.parquet"
    manifest = write_points_regular_grid(_as_partitioned(points), out, catalog=catalog, grid=GRID, streaming=True)

    got = _read_all(out, manifest).to_pandas().set_index("transcript_id").loc[original["transcript_id"].to_numpy()]
    for col in ("x", "y", "z", "cell_id", "qv"):
        np.testing.assert_array_equal(got[col].to_numpy(), original[col].to_numpy(), err_msg=col)


def test_streaming_requires_a_partitioned_element(
    tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog
) -> None:
    with pytest.raises(ValueError, match="streaming requires a partitioned"):
        write_points_regular_grid(points.compute(), tmp_path / "x.parquet", catalog=catalog, grid=GRID, streaming=True)


def test_retiling_drops_render_columns_left_by_an_older_writer(
    tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog
) -> None:
    """Stores written before the split carry render columns in the canonical element."""
    out = tmp_path / "idem.parquet"
    first = write_points_regular_grid(points, out, catalog=catalog, grid=GRID)
    stale = _read_all(out, first).to_pandas()
    stale[POSITION_COLUMN] = [[1, 2]] * len(stale)
    stale[FEATURE_COLUMN] = np.zeros(len(stale), dtype=np.uint16)

    again = write_points_regular_grid(
        _as_partitioned(stale, npartitions=1), out, catalog=catalog, grid=GRID, overwrite=True
    )
    names = _read_all(out, again).column_names
    assert POSITION_COLUMN not in names
    assert FEATURE_COLUMN not in names


@pytest.mark.parametrize("render_only", [False, True])
def test_manifest_reports_which_flavour_was_written(
    tmp_path: Path, points: pd.DataFrame, catalog: FeatureCatalog, render_only: bool
) -> None:
    """Both write paths must report it; the streaming path previously always said False."""
    in_memory = write_points_regular_grid(
        points, tmp_path / "mem", catalog=catalog, grid=GRID, streaming=False, render_only=render_only
    )
    streamed = write_points_regular_grid(
        _as_partitioned(points), tmp_path / "str", catalog=catalog, grid=GRID,
        streaming=True, render_only=render_only,
    )
    assert in_memory["render_only"] is render_only
    assert streamed["render_only"] is render_only
