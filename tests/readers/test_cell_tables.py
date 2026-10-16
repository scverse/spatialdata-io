from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata.models import get_axes_names
from spatialdata.transformations import Identity, Translation, get_transformation
from tests._akoya_data import write_fcs

from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers._utils._cell_tables import (
    build_cells,
    detect_table_format,
    read_cell_table,
    read_codex_table,
    read_labels,
    resolve_per_region,
)

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


_QUPATH_HEADER = (
    "Image\tObject ID\tName\tClass\tCentroid X µm\tCentroid Y µm\tCell: Area µm^2\t"
    "CD3: Nucleus: Mean\tCD3: Cell: Mean\tCD3: Cell: Median\tCD8: Nucleus: Mean"
)


def _write_qupath(path: Path, image: str = "S_Scan1.qptiff") -> None:
    rows = [
        f"{image}\t0a1b\tcell\tTumor\t10\t5\t3.14159\t1\t2\t3\t4",
        f"{image}\t0a1c\tcell\tStroma\t20\t15\t12.56637\t5\t6\t7\t8",
    ]
    path.write_text("\n".join([_QUPATH_HEADER, *rows]) + "\n", encoding="utf-8")


def _write_mcquant(path: Path, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cells = pd.DataFrame(
        {
            "CellID": np.arange(1, n + 1),
            "DAPI": rng.uniform(0, 100, n),
            "CD3": rng.uniform(0, 100, n),
            "X_centroid": rng.uniform(0, 60, n),
            "Y_centroid": rng.uniform(0, 40, n),
            "Area": rng.uniform(10, 50, n),
            "Eccentricity": rng.uniform(0, 1, n),
        }
    )
    cells.to_csv(path, index=False)
    return cells


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("cell_id,region,x,y,size,cyc001_ch001:DAPI1\n", "codex"),
        (_QUPATH_HEADER + "\n", "qupath"),
        ("CellID,DAPI,CD3,X_centroid,Y_centroid,Area\n", "mcquant"),
        ("ID,DAPI,CD3\n", "sprm"),
        ("Path\tSample Name\tCell ID\tCell X Position\tCell Y Position\n", "inform"),
        ("label,cx,cy,CD3\n", "generic"),
    ],
)
def test_detect_formats(tmp_path: Path, header: str, expected: str) -> None:
    path = tmp_path / "cells.csv"
    path.write_text(header + "\n", encoding="utf-8")

    assert detect_table_format(path) == expected


def test_detect_fcs_is_codex(tmp_path: Path) -> None:
    write_fcs(tmp_path / "cells.fcs", _cells())

    assert detect_table_format(tmp_path / "cells.fcs") == "codex"


def test_qupath_um_to_px(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tsv"
    _write_qupath(path)

    table = read_cell_table(path, "qupath", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    np.testing.assert_allclose(table.centroids_px, [[20, 10], [40, 30]])
    assert list(table.var.index) == ["CD3", "CD8"]
    assert list(table.var["compartment"]) == ["Cell", "Nucleus"]
    assert list(table.var["source_column"]) == ["CD3: Cell: Mean", "CD8: Nucleus: Mean"]
    np.testing.assert_array_equal(table.X, np.array([[2, 4], [6, 8]], dtype=np.float32))
    # the other measurements, including the unused compartment, stay in obs
    assert {"Class", "Object ID", "CD3: Nucleus: Mean", "CD3: Cell: Median"} <= set(table.obs.columns)
    # Object IDs are UUIDs, so the instance ids are row numbers
    assert list(table.instance_id) == [1, 2]
    assert list(table.region) == ["S_Scan1.qptiff", "S_Scan1.qptiff"]
    # areas of pi and 4 pi µm^2 at 0.5 µm/px are circles of radius 2 and 4 px
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, [2, 4], rtol=1e-5)


def test_qupath_without_pixel_size_raises(tmp_path: Path) -> None:
    path = tmp_path / "measurements.tsv"
    _write_qupath(path)

    with pytest.raises(ValueError, match="pixel size"):
        read_cell_table(path, "qupath", pixel_size_um=None, default_region="S", cells_table_kwargs={})


def test_mcquant_columns(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    cells = _write_mcquant(path)

    table = read_cell_table(path, "auto", pixel_size_um=None, default_region="S", cells_table_kwargs={})

    assert list(table.var.index) == ["DAPI", "CD3"]
    np.testing.assert_allclose(table.X, cells[["DAPI", "CD3"]], rtol=1e-6)
    np.testing.assert_allclose(table.centroids_px, cells[["X_centroid", "Y_centroid"]])
    assert list(table.instance_id) == [1, 2, 3, 4, 5]
    assert list(table.region.unique()) == ["S"]
    assert "Eccentricity" in table.obs.columns
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(cells["Area"] / np.pi))


def test_mcquant_without_region_raises(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    _write_mcquant(path)

    with pytest.raises(ValueError, match="needs a region column"):
        read_cell_table(path, "mcquant", pixel_size_um=None, default_region=None, cells_table_kwargs={})


def test_generic_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("label,cx,cy,sample,CD3,CD8,score\n7,2,4,A,1,2,0.5\n9,6,8,B,3,4,0.7\n")
    kwargs = {
        "x": "cx",
        "y": "cy",
        "instance_id": "label",
        "region_column": "sample",
        "markers": "CD.*",
        "coordinates_unit": "um",
    }

    table = read_cell_table(path, "generic", pixel_size_um=2.0, default_region=None, cells_table_kwargs=kwargs)

    np.testing.assert_allclose(table.centroids_px, [[1, 2], [3, 4]])
    assert list(table.var.index) == ["CD3", "CD8"]
    assert list(table.instance_id) == [7, 9]
    assert list(table.region) == ["A", "B"]
    assert "score" in table.obs.columns
    assert table.radius_px is None

    listed = read_cell_table(
        path,
        "generic",
        pixel_size_um=None,
        default_region="S",
        cells_table_kwargs={"x": "cx", "y": "cy", "markers": ["CD8"]},
    )
    assert list(listed.var.index) == ["CD8"]
    assert list(listed.instance_id) == [1, 2]


def test_generic_bad_kwargs_raise(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("cx,cy,CD3\n1,2,3\n")

    with pytest.raises(ValueError, match="'x'"):
        read_cell_table(path, "generic", pixel_size_um=None, default_region="S", cells_table_kwargs={})
    with pytest.raises(ValueError, match="unknown"):
        read_cell_table(path, "generic", pixel_size_um=None, default_region="S", cells_table_kwargs={"xx": "cx"})
    with pytest.raises(ValueError, match="nope"):
        read_cell_table(
            path, "generic", pixel_size_um=None, default_region="S", cells_table_kwargs={"x": "cx", "y": "nope"}
        )


@pytest.mark.parametrize("fmt", ["sprm", "inform"])
def test_sprm_inform_not_implemented(tmp_path: Path, fmt: str) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("ID,DAPI\n1,2\n")

    with pytest.raises(NotImplementedError, match="later release"):
        read_cell_table(path, fmt, pixel_size_um=None, default_region="S", cells_table_kwargs={})


def test_unknown_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("ID,DAPI\n1,2\n")

    with pytest.raises(ValueError, match="qupath"):
        read_cell_table(path, "excel", pixel_size_um=None, default_region="S", cells_table_kwargs={})


def test_resolve_per_region(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.touch()
    b.touch()

    assert resolve_per_region(None, ["A"]) == {}
    assert resolve_per_region(str(a), ["A"]) == {"A": a}
    assert resolve_per_region(a, ["A", "B"]) == {"*": a}
    assert resolve_per_region({"B": str(b), "A": a}, ["A", "B"]) == {"A": a, "B": b}
    with pytest.raises(ValueError, match="'C'"):
        resolve_per_region({"C": a}, ["A", "B"])
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        resolve_per_region(tmp_path / "missing.csv", ["A"])


def test_read_labels_lazy_with_transformation(tmp_path: Path) -> None:
    mask = np.zeros((40, 60), dtype=np.uint32)
    mask[5:10, 5:10] = 1
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, mask)
    translation = Translation([3, 4], axes=("x", "y"))

    labels = read_labels(path, {"scale_factors": [2]}, translation)

    level0 = labels["scale0"]["image"]
    assert get_axes_names(labels) == ("y", "x")
    np.testing.assert_array_equal(level0.values, mask)
    assert get_transformation(labels, "global") == translation


@pytest.mark.parametrize(("mask_channel", "expected"), [(None, 0), (1, 1), ("nuclei", 1)])
def test_read_labels_ome_channels(tmp_path: Path, mask_channel: int | str | None, expected: int) -> None:
    masks = np.stack([np.full((16, 24), i + 1, dtype=np.int32) for i in range(2)])
    path = tmp_path / "mask.ome.tif"
    tifffile.imwrite(path, masks, metadata={"axes": "CYX", "Channel": {"Name": ["cells", "nuclei"]}})

    labels = read_labels(path, {"scale_factors": None}, Identity(), mask_channel=mask_channel)

    np.testing.assert_array_equal(labels.values, masks[expected])


def test_read_labels_errors(tmp_path: Path) -> None:
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="integer"):
        read_labels(path, {}, Identity())

    tifffile.imwrite(path, np.zeros((2, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="channel"):
        read_labels(path, {}, Identity(), mask_channel=5)


def test_read_labels_pyramidal_ome_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.repeat(np.arange(64, dtype=np.uint16), 48).reshape(64, 48)
    path = tmp_path / "mask.ome.tif"
    with tifffile.TiffWriter(path) as tw:
        tw.write(mask, subifds=1, tile=(16, 16), compression="zlib", metadata={"axes": "YX"})
        tw.write(mask[::2, ::2], subfiletype=1, tile=(16, 16), compression="zlib")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("full page decode during read")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    labels = read_labels(path, {"scale_factors": None}, Identity())

    assert labels.shape == (64, 48)
    np.testing.assert_array_equal(labels.values, mask)
