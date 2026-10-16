from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from spatialdata.transformations import Identity, Translation, get_transformation
from tests._akoya_data import write_fcs

from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers._utils._cell_tables import build_cells, read_codex_table

if TYPE_CHECKING:
    from pathlib import Path


def _cells(n: int = 5, region: int = 1, markers: tuple[str, ...] = ("cyc001_ch001", "cyc001_ch002")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, n + 1, dtype=float),
            "region": float(region),
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 80, n),
            "size": rng.uniform(10, 100, n),
            **{marker: rng.uniform(0, 1000, n) for marker in markers},
        }
    )


def test_read_codex_fcs_marker_names_from_pns(tmp_path: Path) -> None:
    cells = _cells(markers=("cyc001_ch001", "cyc001_ch002", "cyc002_ch001"))
    path = tmp_path / "reg001_compensated.fcs"
    write_fcs(path, cells, {"cyc001_ch001": "DAPI", "cyc001_ch002": "CD3", "cyc002_ch001": "DAPI"})

    table = read_codex_table(path)

    assert list(table.var.index) == ["DAPI", "CD3", "DAPI (2)"]
    assert list(table.var["cycle"]) == [1, 1, 2]
    assert list(table.var["channel"]) == [1, 2, 1]
    assert list(table.var["source_column"]) == ["cyc001_ch001", "cyc001_ch002", "cyc002_ch001"]
    assert table.X.dtype == np.float32
    np.testing.assert_allclose(table.X, cells[["cyc001_ch001", "cyc001_ch002", "cyc002_ch001"]], rtol=1e-6)
    assert "cyc001_ch001" not in table.obs.columns
    assert list(table.obs.index) == ["1", "2", "3", "4", "5"]
    assert list(table.region.unique()) == ["reg001"]
    assert table.instance_id.dtype == np.int64
    np.testing.assert_allclose(table.centroids_px, cells[["x", "y"]], rtol=1e-6)
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(cells["size"] / np.pi), rtol=1e-6)


def test_read_codex_fcs_without_pns_keeps_column_names(tmp_path: Path) -> None:
    path = tmp_path / "reg001.fcs"
    write_fcs(path, _cells())

    assert list(read_codex_table(path).var.index) == ["cyc001_ch001", "cyc001_ch002"]


def test_read_codex_csv_colon_header(tmp_path: Path) -> None:
    path = tmp_path / "reg002.csv"
    path.write_text(
        "cell_id:cell_id,region:region,x:x,y:y,size:size,cyc001_ch001:DAPI1,cyc001_ch002:CD3\n"
        "1,2,10,20,30,100.5,7\n"
        "2,2,11,21,31,200.5,8\n"
    )

    table = read_codex_table(path)

    assert list(table.var.index) == ["DAPI1", "CD3"]
    assert list(table.var["source_column"]) == ["cyc001_ch001:DAPI1", "cyc001_ch002:CD3"]
    np.testing.assert_array_equal(table.X, np.array([[100.5, 7], [200.5, 8]], dtype=np.float32))
    assert list(table.obs.columns) == ["cell_id", "region", "x", "y", "size"]
    assert list(table.region) == ["reg002", "reg002"]
    np.testing.assert_array_equal(table.centroids_px, [[10, 20], [11, 21]])


def test_read_codex_csv_marker_name_columns(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,x,y,tile_num,CD3,CD8,label\n1,1.0,2.0,1,5.0,6.0,T\n")

    table = read_codex_table(path, region="Kidney")

    assert list(table.var.index) == ["CD3", "CD8"]
    assert "cycle" not in table.var.columns
    assert list(table.obs.columns) == ["cell_id", "x", "y", "tile_num", "label"]
    assert list(table.region) == ["Kidney"]
    assert table.radius_px is None


def test_read_codex_table_errors(tmp_path: Path) -> None:
    no_xy = tmp_path / "a.csv"
    no_xy.write_text("cell_id,cyc001_ch001\n1,2.0\n")
    with pytest.raises(ValueError, match="'x' and 'y'"):
        read_codex_table(no_xy)
    no_region = tmp_path / "b.csv"
    no_region.write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,2.0\n")
    with pytest.raises(ValueError, match="region"):
        read_codex_table(no_region)


def test_build_cells_single_table_and_circles(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(n=3, region=1))
    write_fcs(tmp_path / "reg002.fcs", _cells(n=4, region=2))
    tables = [read_codex_table(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]
    translation = Translation([10, 0], axes=("x", "y"))

    with pytest.warns(UserWarning, match=r"No segmentation mask found for region.*circles") as record:
        anndatas, shapes = build_cells(tables, {}, {}, {"reg001": Identity(), "reg002": translation})

    assert len(record) == 1
    assert list(anndatas) == ["table"]
    table = anndatas["table"]
    assert table.n_obs == 7
    assert table.obs_names.is_unique
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert table.uns["spatialdata_attrs"]["region"] == ["reg001_cells", "reg002_cells"]
    assert table.obs["cell_id"].dtype == np.int64
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"])[:3], tables[0].centroids_px)
    assert sorted(shapes) == ["reg001_cells", "reg002_cells"]
    assert list(shapes["reg002_cells"].index) == [1, 2, 3, 4]
    assert get_transformation(shapes["reg002_cells"], "global") == translation


def test_build_cells_different_panels_split_tables(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(region=1, markers=("cyc001_ch001",)))
    write_fcs(tmp_path / "reg002.fcs", _cells(region=2, markers=("cyc001_ch001", "cyc001_ch002")))
    tables = [read_codex_table(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]

    with pytest.warns(UserWarning, match="circles"):
        anndatas, _ = build_cells(tables, {}, {}, {})

    assert sorted(anndatas) == ["reg001_table", "reg002_table"]
    assert anndatas["reg002_table"].n_vars == 2
    assert anndatas["reg002_table"].uns["spatialdata_attrs"]["region"] == ["reg002_cells"]


def test_dedupe_matches_akoya_convention(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": "X", "cyc001_ch002": "X"})

    assert list(read_codex_table(tmp_path / "reg001.fcs").var.index) == dedupe_channel_names(["X", "X"])


def test_read_codex_fcs_blank_pns_falls_back_to_pnn(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": " ", "cyc001_ch002": "CD3"})

    assert list(read_codex_table(tmp_path / "reg001.fcs").var.index) == ["cyc001_ch001", "CD3"]


@pytest.mark.parametrize("different_panels", [False, True])
def test_build_cells_overlapping_regions_raise(tmp_path: Path, different_panels: bool) -> None:
    write_fcs(tmp_path / "a.fcs", _cells(region=1))
    markers = ("cyc001_ch001",) if different_panels else ("cyc001_ch001", "cyc001_ch002")
    write_fcs(tmp_path / "b.fcs", _cells(region=1, markers=markers))
    tables = [read_codex_table(tmp_path / name) for name in ("a.fcs", "b.fcs")]

    with pytest.raises(ValueError, match="Several cell tables"):
        build_cells(tables, {}, {}, {})
