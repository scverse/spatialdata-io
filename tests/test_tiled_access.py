"""Tests for the profile manifest and the opt-in tiling entry points."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
from anndata import AnnData
from shapely.geometry import Polygon

from spatialdata_io.experimental.manifest import (
    MANIFEST_FILENAME,
    PROFILE_NAME,
    build_manifest,
    validate_manifest,
    write_manifest,
)
from spatialdata_io.experimental.regular_grid import RegularGrid
from spatialdata_io.experimental.tiled_access import add_spatial_tiling

GRID = RegularGrid(origin_x=0.0, origin_y=0.0, tile_size_px=10.0, num_tiles_x=2, num_tiles_y=3)


# -- manifest -----------------------------------------------------------------


def _transcripts_entry(**over):
    entry = {
        "directory": "trx",
        "files": ["chunk_0.parquet"],
        "max_row_groups_per_file": 400,
        "total_row_groups": GRID.num_tiles,
        "position_column": "display_xy",
        "feature_column": "feature_code",
    }
    entry.update(over)
    return entry


def test_manifest_uses_celldega_keys() -> None:
    m = build_manifest(grid=GRID, transcripts=_transcripts_entry())
    for key in ("technology", "use_row_groups", "tile_grid", "row_group_files", "image_info"):
        assert key in m
    assert m["use_row_groups"] is True
    assert m["profile"] == PROFILE_NAME
    assert m["tile_grid"]["num_tiles_x"] == 2


def test_manifest_omits_absent_elements() -> None:
    m = build_manifest(grid=GRID, transcripts=_transcripts_entry())
    assert "cell_segmentation" not in m["row_group_files"]
    assert "cbg" not in m["row_group_files"]
    # images is always present so the reader's loop has something to iterate
    assert m["row_group_files"]["images"] == {}


def test_validate_rejects_row_group_count_mismatch() -> None:
    m = build_manifest(grid=GRID, transcripts=_transcripts_entry(total_row_groups=5))
    with pytest.raises(ValueError, match="does not match the tile grid"):
        validate_manifest(m)


def test_validate_rejects_wrong_file_count() -> None:
    m = build_manifest(
        grid=GRID, transcripts=_transcripts_entry(max_row_groups_per_file=2, files=["chunk_0.parquet"])
    )
    with pytest.raises(ValueError, match="lists 1 file"):
        validate_manifest(m)


def test_validate_rejects_entry_without_files_or_path() -> None:
    m = build_manifest(grid=GRID, transcripts={"total_row_groups": 6, "max_row_groups_per_file": 400})
    with pytest.raises(ValueError, match="neither 'files' nor 'path'"):
        validate_manifest(m)


def test_validate_rejects_empty_cbg_mapping() -> None:
    m = build_manifest(grid=GRID, cbg={"directory": "cbg", "files": ["c.parquet"], "gene_to_row_group": {}})
    with pytest.raises(ValueError, match="gene_to_row_group is empty"):
        validate_manifest(m)


def test_validate_rejects_cbg_row_group_out_of_range() -> None:
    m = build_manifest(
        grid=GRID,
        cbg={
            "directory": "cbg",
            "files": ["c.parquet"],
            "max_row_groups_per_file": 2,
            "total_row_groups": 3,
            "gene_to_row_group": {"A": 0, "B": 1, "C": 9},
        },
    )
    with pytest.raises(ValueError, match="references row group 9"):
        validate_manifest(m)


def test_validate_checks_files_exist(tmp_path: Path) -> None:
    m = build_manifest(grid=GRID, transcripts=_transcripts_entry())
    with pytest.raises(ValueError, match="does not exist"):
        validate_manifest(m, base_path=tmp_path)


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    m = build_manifest(grid=GRID, transcripts=_transcripts_entry())
    path = write_manifest(m, tmp_path)
    assert path.name == MANIFEST_FILENAME
    assert json.loads(path.read_text()) == m


# -- end-to-end tiling --------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A minimal but complete SpatialData store: points, shapes and an annotating table."""
    from spatialdata import SpatialData
    from spatialdata.models import PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import Scale, set_transformation

    rng = np.random.default_rng(0)
    n = 200
    cells = [f"cell-{i}" for i in range(12)]
    genes = ["GENEA", "GENEB", "GENEC"]

    pts = pd.DataFrame(
        {
            "x": rng.uniform(0, 9.9, n),
            "y": rng.uniform(0, 14.9, n),
            "feature_name": pd.Categorical(rng.choice([*genes, "NegControlProbe_0001"], n)),
            "cell_id": rng.choice(cells, n),
        }
    )
    points = PointsModel.parse(pts, coordinates={"x": "x", "y": "y"}, feature_key="feature_name")
    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), "global")

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)])
            for x, y in zip(rng.uniform(0.5, 9, 12), rng.uniform(0.5, 14, 12))
        ],
        index=cells,
    )
    shapes = ShapesModel.parse(gdf)
    set_transformation(shapes, Scale([2.0, 2.0], axes=("x", "y")), "global")

    obs = pd.DataFrame({"region": pd.Categorical(["cell_boundaries"] * 12), "instance_id": range(12)}, index=cells)
    table = TableModel.parse(
        AnnData(X=sp.csr_matrix(rng.integers(0, 5, (12, 3)).astype(np.float32)), obs=obs,
                var=pd.DataFrame(index=genes)),
        region="cell_boundaries", region_key="region", instance_key="instance_id",
    )

    path = tmp_path / "s.zarr"
    SpatialData(points={"transcripts": points}, shapes={"cell_boundaries": shapes},
                tables={"table": table}).write(path)
    return path


def test_add_spatial_tiling_produces_a_valid_profile(store: Path) -> None:
    manifest = add_spatial_tiling(store, tile_size_px=10.0)
    profile = store / "visualization" / PROFILE_NAME
    assert (profile / MANIFEST_FILENAME).exists()
    assert (profile / "meta_gene.parquet").exists()
    validate_manifest(manifest, base_path=profile)


def test_tiled_store_still_reads_with_read_zarr(store: Path) -> None:
    import spatialdata

    before = spatialdata.read_zarr(store)
    n_points, n_shapes = len(before.points["transcripts"]), len(before.shapes["cell_boundaries"])

    add_spatial_tiling(store, tile_size_px=10.0)

    after = spatialdata.read_zarr(store)
    assert len(after.points["transcripts"]) == n_points
    assert len(after.shapes["cell_boundaries"]) == n_shapes
    assert "display_xy" in after.points["transcripts"].columns
    assert "display_geometry" in after.shapes["cell_boundaries"].columns
    # canonical columns survive untouched
    assert {"x", "y", "feature_name", "cell_id"} <= set(after.points["transcripts"].columns)


def test_manifest_paths_resolve_from_the_profile_directory(store: Path) -> None:
    """Celldega is pointed at the profile dir, so relative paths must resolve from there."""
    manifest = add_spatial_tiling(store, tile_size_px=10.0)
    profile = store / "visualization" / PROFILE_NAME
    trx = manifest["row_group_files"]["transcripts"]
    for f in trx["files"]:
        assert (profile / trx["directory"] / f).resolve().exists()


def test_row_group_count_matches_grid(store: Path) -> None:
    manifest = add_spatial_tiling(store, tile_size_px=10.0)
    grid = RegularGrid.from_manifest_dict(manifest["tile_grid"])
    profile = store / "visualization" / PROFILE_NAME
    trx = manifest["row_group_files"]["transcripts"]
    total = sum(pq.ParquetFile(profile / trx["directory"] / f).metadata.num_row_groups for f in trx["files"])
    assert total == grid.num_tiles == manifest["row_group_files"]["transcripts"]["total_row_groups"]


def test_cbg_covers_every_gene(store: Path) -> None:
    manifest = add_spatial_tiling(store, tile_size_px=10.0)
    cbg = manifest["row_group_files"]["cbg"]
    assert set(cbg["gene_to_row_group"]) == {"GENEA", "GENEB", "GENEC"}
    assert "NegControlProbe_0001" not in cbg["gene_to_row_group"]


def test_tiling_is_rerunnable(store: Path) -> None:
    """The two-call workflow means users will re-run this; it must not accumulate state."""
    first = add_spatial_tiling(store, tile_size_px=10.0)
    second = add_spatial_tiling(store, tile_size_px=10.0)
    assert first["tile_grid"] == second["tile_grid"]
    assert first["row_group_files"]["transcripts"]["total_row_groups"] == (
        second["row_group_files"]["transcripts"]["total_row_groups"]
    )

    import spatialdata

    after = spatialdata.read_zarr(store)
    cols = list(after.points["transcripts"].columns)
    assert cols.count("display_xy") == 1
    assert cols.count("feature_code") == 1


def test_missing_element_is_reported(store: Path) -> None:
    with pytest.raises(ValueError, match="points element 'nope' not found"):
        add_spatial_tiling(store, points_element="nope")


def test_cbg_can_be_skipped(store: Path) -> None:
    manifest = add_spatial_tiling(store, tile_size_px=10.0, include_cbg=False)
    assert "cbg" not in manifest["row_group_files"]
