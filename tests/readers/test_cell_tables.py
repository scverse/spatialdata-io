from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import shapely
import tifffile
from spatialdata import get_extent, read_zarr
from spatialdata.models import Labels2DModel, get_axes_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation
from tests._akoya_data import forbid_pixel_reads, write_fcs, write_qptiff

from spatialdata_io import phenocycler
from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers._utils._cell_tables import (
    build_cells,
    detect_table_format,
    read_cell_table,
    read_labels,
    resolve_per_region,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils._cell_tables import CellTable


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


def _read_codex(path: Path, region: str | None = None) -> CellTable:
    return read_cell_table(path, "codex", pixel_size_um=None, default_region=region, cells_table_kwargs={})


def test_read_codex_fcs_marker_names_from_pns(tmp_path: Path) -> None:
    cells = _cells(markers=("cyc001_ch001", "cyc001_ch002", "cyc002_ch001"))
    path = tmp_path / "reg001_compensated.fcs"
    write_fcs(path, cells, {"cyc001_ch001": "DAPI", "cyc001_ch002": "CD3", "cyc002_ch001": "DAPI"})

    table = _read_codex(path)

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
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, cells[["x", "y"]], rtol=1e-6)
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(cells["size"] / np.pi), rtol=1e-6)


def test_read_codex_fcs_without_pns_keeps_column_names(tmp_path: Path) -> None:
    path = tmp_path / "reg001.fcs"
    write_fcs(path, _cells())

    assert list(_read_codex(path).var.index) == ["cyc001_ch001", "cyc001_ch002"]


def test_read_codex_csv_colon_header(tmp_path: Path) -> None:
    path = tmp_path / "reg002.csv"
    path.write_text(
        "cell_id:cell_id,region:region,x:x,y:y,size:size,cyc001_ch001:DAPI1,cyc001_ch002:CD3\n"
        "1,2,10,20,30,100.5,7\n"
        "2,2,11,21,31,200.5,8\n"
    )

    table = _read_codex(path)

    assert list(table.var.index) == ["DAPI1", "CD3"]
    assert list(table.var["source_column"]) == ["cyc001_ch001:DAPI1", "cyc001_ch002:CD3"]
    np.testing.assert_array_equal(table.X, np.array([[100.5, 7], [200.5, 8]], dtype=np.float32))
    # the columns read as the instance ids and regions are replaced by the keys of the table element
    assert list(table.obs.columns) == ["x", "y", "size"]
    assert list(table.region) == ["reg002", "reg002"]
    np.testing.assert_array_equal(table.centroids_px, [[10, 20], [11, 21]])


def test_read_codex_csv_marker_name_columns(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,x,y,tile_num,CD3,CD8,label\n1,1.0,2.0,1,5.0,6.0,T\n")

    table = _read_codex(path, region="Kidney")

    assert list(table.var.index) == ["CD3", "CD8"]
    assert "cycle" not in table.var.columns
    assert list(table.obs.columns) == ["x", "y", "tile_num", "label"]
    assert list(table.region) == ["Kidney"]
    assert table.radius_px is None


def test_read_codex_table_errors(tmp_path: Path) -> None:
    no_xy = tmp_path / "a.csv"
    no_xy.write_text("cell_id,cyc001_ch001\n1,2.0\n")
    with pytest.raises(ValueError, match="'x' and 'y'"):
        _read_codex(no_xy)
    no_region = tmp_path / "b.csv"
    no_region.write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,2.0\n")
    with pytest.raises(ValueError, match="region"):
        _read_codex(no_region)


@pytest.mark.parametrize("sep", ["\t", ", "])
def test_read_codex_table_delimiters_like_detection(tmp_path: Path, sep: str) -> None:
    path = tmp_path / "cells.tsv"
    header = ["cell_id", "region", "x", "y", "size", "cyc001_ch001:DAPI"]
    path.write_text("\n".join(sep.join(row) for row in [header, ["1", "2", "1.0", "2.0", "3.0", "4.0"]]) + "\n")

    assert detect_table_format(path) == "codex"
    table = read_cell_table(path, "auto", pixel_size_um=None, default_region=None, cells_table_kwargs={})

    assert list(table.var.index) == ["DAPI"]
    assert list(table.region) == ["reg002"]
    np.testing.assert_array_equal(table.centroids_px, [[1, 2]])


def test_read_codex_table_cells_table_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,label,Sample,x,y,CD3,CD8\n1,7,A,1.0,2.0,3.0,4.0\n2,8,B,5.0,6.0,7.0,8.0\n")
    kwargs = {"region_column": "Sample", "instance_id": "label", "markers": ["CD8"]}

    table = read_cell_table(path, "codex", pixel_size_um=None, default_region="A", cells_table_kwargs=kwargs)

    assert list(table.region) == ["A", "B"]
    assert list(table.instance_id) == [7, 8]
    assert list(table.var.index) == ["CD8"]
    assert "CD3" in table.obs.columns


@pytest.mark.parametrize(
    ("fmt", "header", "markers"),
    [
        ("mcquant", "CellID,DNA,CD3,DNA,X_centroid,Y_centroid,Area", ["DNA", "CD3", "DNA (2)"]),
        ("sprm", "ID,DAPI,Blank,CD3,Blank", ["DAPI", "Blank", "CD3", "Blank (2)"]),
        ("codex", "cell_id,x,y,size,DAPI,CD3,DAPI", ["DAPI", "CD3", "DAPI (2)"]),
        ("generic", "cx,cy,DAPI,CD3,DAPI", ["DAPI", "CD3", "DAPI (2)"]),
    ],
)
def test_repeated_column_names_are_deduplicated(tmp_path: Path, fmt: str, header: str, markers: list[str]) -> None:
    path = tmp_path / "cells.csv"
    n = header.count(",") + 1
    path.write_text(f"{header}\n{','.join(str(i + 1) for i in range(n))}\n", encoding="utf-8")
    kwargs = {"x": "cx", "y": "cy"} if fmt == "generic" else {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # SPRM without centroids
        table = read_cell_table(path, fmt, pixel_size_um=None, default_region="S", cells_table_kwargs=kwargs)

    assert list(table.var.index) == markers
    assert table.X.shape == (1, len(markers))


@pytest.mark.parametrize("cell_id", ["", "1.5", "a"])
def test_missing_or_fractional_instance_ids_raise(tmp_path: Path, cell_id: str) -> None:
    path = tmp_path / "cells.csv"
    path.write_text(f"CellID,CD3,X_centroid,Y_centroid\n1,1,1,1\n{cell_id},1,2,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"1 cell\(s\) whose instance id in 'CellID' is missing or not an integer"):
        read_cell_table(path, "mcquant", pixel_size_um=None, default_region="S", cells_table_kwargs={})


def test_build_cells_single_table_and_circles(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(n=3, region=1))
    write_fcs(tmp_path / "reg002.fcs", _cells(n=4, region=2))
    tables = [_read_codex(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]
    translation = Translation([10, 0], axes=("x", "y"))

    with pytest.warns(UserWarning, match=r"No segmentation mask found for region.*circles") as record:
        anndatas, shapes = build_cells(
            tables, {}, {}, {"reg001": {"reg001": Identity()}, "reg002": {"reg002": translation}}
        )

    assert len(record) == 1
    assert list(anndatas) == ["table"]
    table = anndatas["table"]
    assert table.n_obs == 7
    assert table.obs_names.is_unique
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert table.uns["spatialdata_attrs"]["region"] == ["reg001_cells", "reg002_cells"]
    assert table.obs["cell_id"].dtype == np.int64
    assert tables[0].centroids_px is not None
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"])[:3], tables[0].centroids_px)
    assert sorted(shapes) == ["reg001_cells", "reg002_cells"]
    assert list(shapes["reg002_cells"].index) == [1, 2, 3, 4]
    assert get_transformation(shapes["reg002_cells"], get_all=True) == {"reg002": translation}


def test_build_cells_different_panels_split_tables(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(region=1, markers=("cyc001_ch001",)))
    write_fcs(tmp_path / "reg002.fcs", _cells(region=2, markers=("cyc001_ch001", "cyc001_ch002")))
    tables = [_read_codex(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]

    with pytest.warns(UserWarning, match="circles"):
        anndatas, _ = build_cells(tables, {}, {}, {})

    assert sorted(anndatas) == ["reg001_table", "reg002_table"]
    assert anndatas["reg002_table"].n_vars == 2
    assert anndatas["reg002_table"].uns["spatialdata_attrs"]["region"] == ["reg002_cells"]


def test_dedupe_matches_akoya_convention(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": "X", "cyc001_ch002": "X"})

    assert list(_read_codex(tmp_path / "reg001.fcs").var.index) == dedupe_channel_names(["X", "X"])


def test_read_codex_fcs_blank_pns_falls_back_to_pnn(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": " ", "cyc001_ch002": "CD3"})

    assert list(_read_codex(tmp_path / "reg001.fcs").var.index) == ["cyc001_ch001", "CD3"]


@pytest.mark.parametrize("different_panels", [False, True])
def test_build_cells_overlapping_regions_raise(tmp_path: Path, different_panels: bool) -> None:
    write_fcs(tmp_path / "a.fcs", _cells(region=1))
    markers = ("cyc001_ch001",) if different_panels else ("cyc001_ch001", "cyc001_ch002")
    write_fcs(tmp_path / "b.fcs", _cells(region=1, markers=markers))
    tables = [_read_codex(tmp_path / name) for name in ("a.fcs", "b.fcs")]

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


def _write_mcquant(path: Path, n: int = 5, extent: tuple[float, float] = (60, 40)) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cells = pd.DataFrame(
        {
            "CellID": np.arange(1, n + 1),
            "DAPI": rng.uniform(0, 100, n),
            "CD3": rng.uniform(0, 100, n),
            "X_centroid": rng.uniform(0, extent[0], n),
            "Y_centroid": rng.uniform(0, extent[1], n),
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

    assert table.centroids_px is not None
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
    assert table.centroids_px is not None
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

    assert table.centroids_px is not None
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


def _write_sprm(directory: Path, *, centers: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    means = pd.DataFrame({"ID": [1, 2, 4], "DAPI": rng.uniform(0, 1, 3), "CD3": rng.uniform(0, 1, 3)})
    means.to_csv(directory / "reg1.ome.tiff-cell_channel_mean.csv", index=False)
    if centers:
        # more cells than in the means table, in another order
        pd.DataFrame({"ID": [4, 3, 2, 1], "x": [40.0, 30.0, 20.0, 10.0], "y": [4.0, 3.0, 2.0, 1.0]}).to_csv(
            directory / "reg1.ome.tiff-cell_centers.csv", index=False
        )
    return means


def test_sprm_adapter(tmp_path: Path) -> None:
    means = _write_sprm(tmp_path)
    path = tmp_path / "reg1.ome.tiff-cell_channel_mean.csv"

    assert detect_table_format(path) == "sprm"
    table = read_cell_table(path, "auto", pixel_size_um=None, default_region="reg1", cells_table_kwargs={})

    assert list(table.var.index) == ["DAPI", "CD3"]
    np.testing.assert_allclose(table.X, means[["DAPI", "CD3"]], rtol=1e-6)
    assert list(table.instance_id) == [1, 2, 4]
    assert not table.row_instance_ids
    assert list(table.region.unique()) == ["reg1"]
    assert table.centroids_px is not None
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[10, 1], [20, 2], [40, 4]])
    assert table.radius_px is None


def test_sprm_without_centers_warns_and_omits_spatial(tmp_path: Path) -> None:
    _write_sprm(tmp_path, centers=False)
    path = tmp_path / "reg1.ome.tiff-cell_channel_mean.csv"

    with pytest.warns(UserWarning, match="cell_centers"):
        table = read_cell_table(path, "sprm", pixel_size_um=None, default_region="reg1", cells_table_kwargs={})
    assert table.centroids_px is None

    mask = np.zeros((6, 6), dtype=np.uint16)
    mask[1:3, 1:3], mask[4, 4] = 1, 2
    labels = Labels2DModel.parse(mask, dims=("y", "x"))
    sdata_tables, shapes = build_cells([table], {}, {"reg1": labels}, {})
    assert "spatial" not in sdata_tables["table"].obsm
    assert shapes == {}
    with pytest.raises(ValueError, match="centroids"):
        build_cells([table], {}, {}, {})


def test_sprm_centers_missing_cells_raise(tmp_path: Path) -> None:
    _write_sprm(tmp_path)
    pd.DataFrame({"ID": [1, 2], "x": [1.0, 2.0], "y": [1.0, 2.0]}).to_csv(
        tmp_path / "reg1.ome.tiff-cell_centers.csv", index=False
    )

    with pytest.raises(ValueError, match="cell_centers"):
        read_cell_table(
            tmp_path / "reg1.ome.tiff-cell_channel_mean.csv",
            "sprm",
            pixel_size_um=None,
            default_region="reg1",
            cells_table_kwargs={},
        )


def _write_inform(path: Path, unit: str | None, area_unit: str = "square microns") -> None:
    suffix = "" if unit is None else f" ({unit})"
    header = [
        "Sample Name",
        "Cell ID",
        f"Cell X Position{suffix}",
        f"Cell Y Position{suffix}",
        f"Entire Cell Area ({area_unit})",
        "Entire Cell DAPI Mean (Normalized Counts, Total Weighting)",
        "Entire Cell CD8 (Opal 520) Mean (Normalized Counts, Total Weighting)",
        "Nucleus DAPI Mean (Normalized Counts, Total Weighting)",
        "Phenotype",
    ]
    rows = [
        ["Scan1.im3", "1", "100", "50", "314.159", "1.5", "2.5", "9", "CD8+"],
        ["Scan1.im3", "2", "200", "80", "78.54", "3.5", "4.5", "9", "other"],
    ]
    path.write_text("\n".join("\t".join(row) for row in [header, *rows]) + "\n", encoding="utf-8")


def test_inform_adapter_microns(tmp_path: Path) -> None:
    path = tmp_path / "Scan1_cell_seg_data.txt"
    _write_inform(path, "microns")

    assert detect_table_format(path) == "inform"
    table = read_cell_table(path, "inform", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[200, 100], [400, 160]])
    assert list(table.var.index) == ["DAPI", "CD8 (Opal 520)"]
    np.testing.assert_allclose(table.X, [[1.5, 2.5], [3.5, 4.5]])
    assert list(table.instance_id) == [1, 2]
    assert not table.row_instance_ids
    assert list(table.region) == ["Scan1.im3"] * 2
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(np.array([314.159, 78.54]) / 0.25 / np.pi))
    assert "Phenotype" in table.obs.columns
    with pytest.raises(ValueError, match="pixel size"):
        read_cell_table(path, "inform", pixel_size_um=None, default_region=None, cells_table_kwargs={})


def test_inform_adapter_pixels(tmp_path: Path) -> None:
    path = tmp_path / "Scan1_cell_seg_data.txt"
    _write_inform(path, "pixels", area_unit="pixels")

    table = read_cell_table(path, "auto", pixel_size_um=None, default_region="S", cells_table_kwargs={})

    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[100, 50], [200, 80]])
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(np.array([314.159, 78.54]) / np.pi))


def test_inform_without_unit_is_pixels_and_renames_cell_id(tmp_path: Path) -> None:
    path = tmp_path / "Scan1_cell_seg_data.txt"
    _write_inform(path, None, area_unit="pixels")

    table = read_cell_table(path, "inform", pixel_size_um=None, default_region=None, cells_table_kwargs={})
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[100, 50], [200, 80]])
    assert table.radius_px is not None

    # mixed units (px positions, µm areas) cannot give a radius
    _write_inform(path, "pixels")
    table = read_cell_table(path, "inform", pixel_size_um=None, default_region=None, cells_table_kwargs={})
    assert table.radius_px is None
    elements, _ = build_cells([table], {}, {}, {})
    assert "source_Cell ID" in elements["table"].obs.columns or "source_Cell_ID" in elements["table"].obs.columns


# the header of the inForm micron variant (github.com/lilab-stanford/auto-cell): no unit after `Cell X Position`,
# trailing spaces after the marker columns, CRLF line ends
_INFORM_MICRON_HEADER = [
    "Sample Name",
    "Tissue Category",
    "Cell ID",
    "Total Cells",
    "Tissue Category Area (square microns)",
    "Cell Density (per square mm)",
    "Cell X Position",
    "Cell Y Position",
    "Distance from Process Region Edge (microns)",
    "Nucleus PD1 (PPD520) Mean ",
    "Entire Cell Area (square microns)",
    "Entire Cell PD1 (PPD520) Mean ",
    "Entire Cell PPDDAPI Mean ",
    "Lab ID",
    "Slide ID",
]


def test_inform_micron_variant_without_position_unit(tmp_path: Path) -> None:
    path = tmp_path / "sample_cell_seg_data.txt"
    rows = [
        ['"TMA_Core[1,1,A]_[4768,30518].im3"', "STROMA", "1", "", "", "", "100", "50", "#N/A", "0", "314.159", "1.5"],
        ['"TMA_Core[1,1,A]_[4768,30518].im3"', "TUMOR", "2", "", "", "", "200", "80", "#N/A", "0", "78.54", "3.5"],
    ]
    rows = [[*row, "2.5" if i == 0 else "4.5", "", ""] for i, row in enumerate(rows)]
    path.write_text("\r\n".join("\t".join(row) for row in [_INFORM_MICRON_HEADER, *rows]), encoding="utf-8")

    assert detect_table_format(path) == "inform"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    # the other columns are in microns, so the positions are too
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[200, 100], [400, 160]])
    assert list(table.var.index) == ["PD1 (PPD520)", "PPDDAPI"]
    np.testing.assert_allclose(table.X, [[1.5, 2.5], [3.5, 4.5]])
    assert list(table.region) == ["TMA_Core[1,1,A]_[4768,30518].im3"] * 2
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(np.array([314.159, 78.54]) / 0.25 / np.pi))
    assert "Nucleus PD1 (PPD520) Mean" in table.obs.columns


def test_inform_position_unit_from_other_columns(tmp_path: Path) -> None:
    # the phenoptr sample (inForm 2.3) names the unit of its areas and distances only, here pixels
    path = tmp_path / "S_cell_seg_data.txt"
    header = ["Sample Name", "Cell ID", "Tissue Category Area (pixels)", "Cell X Position", "Cell Y Position"]
    path.write_text("\t".join(header) + "\nS.im3\t1\t5\t10\t20\n", encoding="utf-8")

    table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[10, 20]])

    # columns in both units leave the unit of the positions open
    header[2] = "Distance from Process Region Edge (microns)"
    path.write_text("\t".join([*header, "Entire Cell Area (pixels)"]) + "\nS.im3\t1\t5\t10\t20\t3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinates_unit"):
        read_cell_table(path, "inform", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})
    table = read_cell_table(
        path, "inform", pixel_size_um=0.5, default_region=None, cells_table_kwargs={"coordinates_unit": "um"}
    )
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[20, 40]])


# R-exported QuPath tables (dryad doi:10.5061/dryad.hqbzkh1v1): `make.names` turns `Centroid X µm` into
# `Centroid.X.um`; the coordinates and the expression are split into two files
_R_COORDS_HEADER = "Object.ID,Centroid.X.um,Centroid.Y.um,core"


def _square(x: float, y: float, size: float = 8) -> dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]]}


def test_qupath_r_exported_coordinates_with_geojson(tmp_path: Path) -> None:
    path = tmp_path / "C-7_coords.csv"
    # µm centroids at 0.5 µm/px: the centres of the 8 px squares at (0, 0) and (20, 20)
    path.write_text(f"{_R_COORDS_HEADER}\n0000aa8d-e87f,2,2,C-7\n0001e8f3-07bc,12,12,C-7\n", encoding="utf-8")
    annotation = {"objectType": "annotation", "isLocked": True}
    features = [
        {"type": "Feature", "id": "c28880d8", "geometry": _square(0, 0, 2270), "properties": annotation},
        *(
            {
                "type": "Feature",
                "id": object_id,
                "geometry": _square(x, x),
                "nucleusGeometry": _square(x + 2, x + 2, 4),
                "properties": {"objectType": "cell"},
            }
            # in another order than the table; the last cell was filtered out of the table
            for object_id, x in [("0001e8f3-07bc", 20), ("0000aa8d-e87f", 0), ("ffff0000-0000", 40)]
        ),
    ]
    geojson = tmp_path / "C-7.geojson"
    geojson.write_text(json.dumps(features), encoding="utf-8")

    assert detect_table_format(path) == "qupath"
    with pytest.warns(UserWarning, match=r"1 of the 3 cell features of .*C-7.geojson have no row"):
        table = read_cell_table(
            path, "auto", pixel_size_um=0.5, default_region="C-7", cells_table_kwargs={"geojson": geojson}
        )

    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[4, 4], [24, 24]])
    assert table.polygons_px is not None
    assert table.polygons_px[0].equals(shapely.box(0, 0, 8, 8))
    assert table.polygons_px[1].equals(shapely.box(20, 20, 28, 28))
    assert list(table.instance_id) == [1, 2]
    assert list(table.obs["Object.ID"]) == ["0000aa8d-e87f", "0001e8f3-07bc"]


def test_qupath_r_exported_marker_columns(tmp_path: Path) -> None:
    # the expression file joined to the coordinates on `Object.ID`: `<marker>_<compartment>_Mean`, and the
    # `make.names` form of `CD3: Cell: Mean`
    path = tmp_path / "C-7.csv"
    header = f"{_R_COORDS_HEADER},DAPI_Nucleus_Mean,HLA.A_Cytoplasm_Mean,Pan.Cytokeratin_Cytoplasm_Mean,CD3..Cell..Mean"
    path.write_text(f"{header}\nab,2,2,C-7,1,2,3,4\n", encoding="utf-8")

    table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region="C-7", cells_table_kwargs={})

    assert list(table.var.index) == ["DAPI", "HLA.A", "Pan.Cytokeratin", "CD3"]
    assert list(table.var["compartment"]) == ["Nucleus", "Cytoplasm", "Cytoplasm", "Cell"]
    np.testing.assert_allclose(table.X, [[1, 2, 3, 4]])


@pytest.mark.parametrize("area", ["Cell..Area.µm.2", "Cell..Area.um.2"])
def test_qupath_r_exported_area(tmp_path: Path, area: str) -> None:
    # `make.names` of `Cell: Area µm^2`; areas of pi and 4 pi µm^2 at 0.5 µm/px are circles of radius 2 and 4 px
    path = tmp_path / "cells.csv"
    path.write_text(f"{_R_COORDS_HEADER},{area}\nab,2,2,C-7,3.14159\ncd,12,12,C-7,12.56637\n", encoding="utf-8")

    table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region="C-7", cells_table_kwargs={})

    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, [2, 4], rtol=1e-5)

    # the unit of R-form measurements gives that of `X`/`Y` too
    path.write_text(f"Object.ID,X,Y,{area}\nab,2,2,3.14159\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="reading them in um"):
        table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region="C-7", cells_table_kwargs={})
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[4, 4]])
    assert table.radius_px is not None


def test_qupath_expression_without_centroids_raises(tmp_path: Path) -> None:
    path = tmp_path / "C-7_expression.csv"
    path.write_text("Object.ID,DAPI_Nucleus_Mean,Ki67_Nucleus_Mean\nab,1,2\n", encoding="utf-8")

    assert detect_table_format(path) == "qupath"
    with pytest.raises(ValueError, match="no centroid columns"):
        read_cell_table(path, "auto", pixel_size_um=0.5, default_region="C-7", cells_table_kwargs={})


# a QuPath detection export with `X`/`Y` centroids and bare marker names (zenodo.org/records/15514605)
_QUPATH_XY_HEADER = (
    "Image,Object ID,Parent,X,Y,Nucleus: Area µm^2,Nucleus: Length µm,Cell: Area µm^2,Nucleus/Cell area ratio,"
    "DAPI: Nucleus: Median,CD44,HLA-DR-AF647,CD45RO"
)


def test_qupath_xy_centroids_and_bare_markers(tmp_path: Path) -> None:
    path = tmp_path / "measurement_TA4369.csv"
    rows = [
        "STardist,74fcc0e7-d660,A-1,10,5,7.7,11.5,3.14159,0.13,20.5,1.5,34.2,4.3",
        "STardist,9b2e5c11-0a3f,A-2,20,15,8.1,12.0,12.56637,0.2,21,2.5,35.2,5.3",
    ]
    path.write_text("\r\n".join([_QUPATH_XY_HEADER, *rows]) + "\r\n", encoding="utf-8")

    assert detect_table_format(path) == "qupath"
    with (
        pytest.warns(UserWarning, match=r"reading the 3 other numeric columns \['CD44', 'HLA-DR-AF647', 'CD45RO'\]"),
        pytest.warns(UserWarning, match=r"'X' and 'Y' .* name no unit; reading them in um.*coordinates_unit"),
    ):
        table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    # the other measurements are in µm, so the centroids are too
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[20, 10], [40, 30]])
    assert list(table.var.index) == ["CD44", "HLA-DR-AF647", "CD45RO"]
    np.testing.assert_allclose(table.X, [[1.5, 34.2, 4.3], [2.5, 35.2, 5.3]], rtol=1e-6)
    assert {"Parent", "DAPI: Nucleus: Median", "Nucleus/Cell area ratio", "Object ID"} <= set(table.obs.columns)
    assert list(table.region) == ["STardist"] * 2
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, [2, 4], rtol=1e-5)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = read_cell_table(
            path,
            "qupath",
            pixel_size_um=0.5,
            default_region=None,
            cells_table_kwargs={"region_column": "Parent", "markers": ["CD44", "CD45RO"], "coordinates_unit": "um"},
        )
    assert list(table.region) == ["A-1", "A-2"]
    assert list(table.var.index) == ["CD44", "CD45RO"]


@pytest.mark.parametrize("r_names", [False, True])
def test_qupath_bare_markers_exclude_morphology(tmp_path: Path, r_names: bool) -> None:
    path = tmp_path / "detections.csv"
    header = [
        "Image",
        "Object ID",
        "Centroid X µm",
        "Centroid Y µm",
        "Area µm^2",
        "Perimeter µm",
        "Max diameter µm",
        "Circularity",
        "Solidity",
        "Eccentricity",
        "Detection probability",
        "CD44",
        "CD45RO",
    ]
    if r_names:
        header = [name.replace(" ", ".").replace("µm^2", "um.2").replace("µm", "um") for name in header]
    path.write_text(",".join(header) + "\nS,u1,1,2,3,4,5,0.9,0.8,0.5,0.99,1.5,4.3\n", encoding="utf-8")

    with pytest.warns(UserWarning, match=r"reading the 2 other numeric columns \['CD44', 'CD45RO'\]"):
        table = read_cell_table(path, "qupath", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    assert list(table.var.index) == ["CD44", "CD45RO"]
    assert {header[4], header[7], header[10]} <= set(table.obs.columns)


@pytest.mark.parametrize(
    ("intensities", "compartments"),
    [
        # built-in cell detection (all versions)
        (["Nucleus: DAPI mean", "Nucleus: DAPI std dev", "Cytoplasm: CD3 mean", "Cell: CD3 mean"], ["Nucleus", "Cell"]),
        # cells of QuPath >= 0.6
        (
            ["Nucleus: DAPI: Mean", "Nucleus: DAPI: Median", "Cell: CD3: Mean", "Nucleus: CD3: Mean"],
            ["Nucleus", "Cell"],
        ),
        # the same in R's `make.names` form
        (["Nucleus..DAPI.mean", "Cytoplasm..CD3.mean", "Cell..CD3.mean"], ["Nucleus", "Cell"]),
        # detections without compartments
        (["DAPI: Mean", "DAPI: Median", "DAPI: Std.Dev.", "CD3: Mean", "CD3: Max"], None),
    ],
)
def test_qupath_intensity_column_forms(tmp_path: Path, intensities: list[str], compartments: list[str] | None) -> None:
    path = tmp_path / "cells.csv"
    header = ["Image", "Object ID", "Name", "Classification", "Centroid X µm", "Centroid Y µm", *intensities]
    rows = [f"S,u{i},,,{i},{i},{','.join(str(i + j) for j in range(len(intensities)))}" for i in range(2)]
    path.write_text("\n".join([",".join(header), *rows]) + "\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    assert list(table.var.index) == ["DAPI", "CD3"]
    if compartments is None:
        assert "compartment" not in table.var.columns
    else:
        assert list(table.var["compartment"]) == compartments
    assert {"Name", "Classification"} <= set(table.obs.columns)


@pytest.mark.parametrize("r_names", [False, True])
def test_qupath_bare_markers_exclude_empty_text_and_measurements(tmp_path: Path, r_names: bool) -> None:
    path = tmp_path / "cells.csv"
    header = ["Image", "Object ID", "Name", "Classification", "Centroid X µm", "Centroid Y µm", "DAPI: Median", "CD44"]
    if r_names:
        header = [name.replace(": ", "..").replace(" ", ".").replace("µm", "um") for name in header]
    path.write_text(",".join(header) + "\nS,u1,,,1,2,3,4\nS,u2,,,1,2,3,4\n", encoding="utf-8")

    with pytest.warns(UserWarning, match=r"reading the 1 other numeric columns \['CD44'\]"):
        table = read_cell_table(path, "qupath", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    assert list(table.var.index) == ["CD44"]


def test_qupath_xy_centroids_need_a_unit(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("Object ID,X,Y,CD3: Cell: Mean\nab,10,5,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coordinates_unit"):
        read_cell_table(path, "auto", pixel_size_um=0.5, default_region="S", cells_table_kwargs={})
    table = read_cell_table(
        path, "auto", pixel_size_um=0.5, default_region="S", cells_table_kwargs={"coordinates_unit": "px"}
    )
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[10, 5]])


def test_cells_outside_the_image_warn(tmp_path: Path) -> None:
    # inForm positions in microns relative to the slide, not to the field of the image
    write_qptiff(tmp_path / "Scan1.qptiff", biomarkers=["DAPI"], shape=(64, 80))
    _write_inform(tmp_path / "Scan1_cell_seg_data.txt", "microns")

    with pytest.warns(UserWarning, match=r"2 of the 2 cells of region 'Scan1' lie outside its image \(80 x 64 px\)"):
        phenocycler(
            tmp_path, cells_table=tmp_path / "Scan1_cell_seg_data.txt", image_models_kwargs={"scale_factors": [2]}
        )


def _write_inform_seg_maps(
    path: Path, shape: tuple[int, int] = (16, 24), first: str = "TissueClassMap", compartments: int = 3
) -> np.ndarray:
    # inForm `*_binary_seg_maps.tif`: the tissue class map, then one page per compartment, each with its XML
    descriptions = [
        f'<?xml version="1.0" encoding="utf-16"?>\n<{first}>\n <Version>1</Version>\n</{first}>',
        *(
            '<?xml version="1.0" encoding="utf-16"?>\n<SegmentationImage>\n <Version>1</Version>\n'
            f" <CompartmentType>{compartment}</CompartmentType>\n</SegmentationImage>"
            for compartment in ("Nucleus", "Cytoplasm", "Membrane")[:compartments]
        ),
    ]
    pages = np.stack([np.full(shape, i, dtype=np.uint16) for i in range(len(descriptions))])
    with tifffile.TiffWriter(path) as tw:
        for page, description in zip(pages, descriptions, strict=True):
            tw.write(page, description=description, metadata=None)
    return pages


@pytest.mark.parametrize(("mask_channel", "expected"), [(None, 1), ("Cytoplasm", 2), (0, 0)])
def test_read_labels_inform_seg_maps(tmp_path: Path, mask_channel: int | str | None, expected: int) -> None:
    path = tmp_path / "S_binary_seg_maps.tif"
    pages = _write_inform_seg_maps(path)

    labels = read_labels(path, {"scale_factors": None}, {"global": Identity()}, mask_channel=mask_channel)

    # by default the nuclei, which hold the `Cell ID` of every cell, not the tissue class map
    np.testing.assert_array_equal(labels.values, pages[expected])
    with pytest.raises(ValueError, match=r"'Tissue', 'Nucleus', 'Cytoplasm', 'Membrane'"):
        read_labels(path, {}, {"global": Identity()}, mask_channel="cells")


def test_read_labels_inform_seg_maps_other_pages(tmp_path: Path) -> None:
    # pages are named by their XML root tag; the nuclei are the default without a tissue class map too
    path = tmp_path / "S_binary_seg_maps.tif"
    pages = _write_inform_seg_maps(path, first="ProcessRegionMap", compartments=2)

    labels = read_labels(path, {"scale_factors": None}, {"global": Identity()})

    np.testing.assert_array_equal(labels.values, pages[1])
    with pytest.raises(ValueError, match=r"\['ProcessRegionMap', 'Nucleus', 'Cytoplasm'\]"):
        read_labels(path, {}, {"global": Identity()}, mask_channel="Membrane")


@pytest.mark.parametrize(
    ("header", "x", "y"), [("object_id,x,y,CD3", "x", "y"), ("Object Id,XMin,XMax,cx,cy,CD3", "cx", "cy")]
)
def test_object_id_lookalikes_stay_generic(tmp_path: Path, header: str, x: str, y: str) -> None:
    # only QuPath's own `Object ID` (or R's `Object.ID`) makes a QuPath table, e.g. not HALO's `Object Id`
    path = tmp_path / "cells.csv"
    row = "7,10,5,2" if header.startswith("object_id") else "7,0,20,10,5,2"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    assert detect_table_format(path) == "generic"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = read_cell_table(
            path, "auto", pixel_size_um=0.5, default_region="S", cells_table_kwargs={"x": x, "y": y}
        )

    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[10, 5]])
    assert "CD3" in table.var.index


def test_auto_with_centroid_kwargs_is_generic(tmp_path: Path) -> None:
    # user-given centroid columns win over the detected format
    path = tmp_path / "measurements.tsv"
    _write_qupath(path)

    table = read_cell_table(
        path,
        "auto",
        pixel_size_um=0.5,
        default_region=None,
        cells_table_kwargs={"x": "Centroid X µm", "y": "Centroid Y µm", "region_column": "Image", "markers": "CD.*"},
    )

    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, [[10, 5], [20, 15]])


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

    labels = read_labels(path, {"scale_factors": [2]}, {"global": translation})

    level0 = labels["scale0"]["image"]
    assert get_axes_names(labels) == ("y", "x")
    np.testing.assert_array_equal(level0.values, mask)
    assert get_transformation(labels, "global") == translation


@pytest.mark.parametrize(("mask_channel", "expected"), [(None, 0), (1, 1), ("nuclei", 1)])
def test_read_labels_ome_channels(tmp_path: Path, mask_channel: int | str | None, expected: int) -> None:
    masks = np.stack([np.full((16, 24), i + 1, dtype=np.int32) for i in range(2)])
    path = tmp_path / "mask.ome.tif"
    tifffile.imwrite(path, masks, metadata={"axes": "CYX", "Channel": {"Name": ["cells", "nuclei"]}})

    labels = read_labels(path, {"scale_factors": None}, {"global": Identity()}, mask_channel=mask_channel)

    np.testing.assert_array_equal(labels.values, masks[expected])


def test_read_labels_errors(tmp_path: Path) -> None:
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="integer"):
        read_labels(path, {}, {"global": Identity()})

    tifffile.imwrite(path, np.zeros((2, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="channel"):
        read_labels(path, {}, {"global": Identity()}, mask_channel=5)


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_labels_pyramidal_ome_is_lazy(tmp_path: Path, compression: str | None) -> None:
    mask = np.repeat(np.arange(64, dtype=np.uint16), 48).reshape(64, 48)
    path = tmp_path / "mask.ome.tif"
    tile = (16, 16) if compression else None
    with tifffile.TiffWriter(path) as tw:
        tw.write(mask, subifds=1, tile=tile, compression=compression, metadata={"axes": "YX"})
        tw.write(mask[::2, ::2], subfiletype=1, tile=tile, compression=compression)

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        labels = read_labels(path, {"scale_factors": None}, {"global": Identity()})

    assert labels.shape == (64, 48)
    np.testing.assert_array_equal(labels.values, mask)


@pytest.mark.parametrize(("mask_shape", "factor"), [((32, 40), 2), ((22, 27), 3)])
def test_mask_integer_downsample_autoscaled(tmp_path: Path, mask_shape: tuple[int, int], factor: int) -> None:
    # a mask downsampled by an integer factor (exactly, or with its shape rounded up) is scaled onto the image,
    # before the stage translation of its region; the full-resolution mask of the other region is unaffected
    image_shape = (64, 80)
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"], shape=image_shape, stage_position_cm=(1.0, 2.0))
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI"], shape=image_shape, stage_position_cm=(1.01, 2.0))
    tifffile.imwrite(tmp_path / "A_mask.tif", np.zeros(image_shape, dtype=np.uint32))
    mask = np.zeros(mask_shape, dtype=np.uint32)
    mask[2:6, 2:6], mask[10:14, 8:12] = 1, 2
    tifffile.imwrite(tmp_path / "B_mask.tif", mask)
    # the centroids are in mask pixels, inside the mask
    cells = _write_mcquant(tmp_path / "B.csv", n=2, extent=(mask_shape[1], mask_shape[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(
            tmp_path,
            cells_table={"B": tmp_path / "B.csv"},
            labels={"A": tmp_path / "A_mask.tif", "B": tmp_path / "B_mask.tif"},
            image_models_kwargs={"scale_factors": [2]},
        )

    assert sdata.coordinate_systems == ["global"]
    assert get_transformation(sdata.labels["A_labels"], get_all=True) == get_transformation(
        sdata.images["A_image"], get_all=True
    )
    image_transformation = get_transformation(sdata.images["B_image"], "global")
    assert isinstance(image_transformation, Translation)
    labels_transformation = get_transformation(sdata.labels["B_labels"], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations == [Scale([factor, factor], axes=("y", "x")), image_transformation]
    image_extent = get_extent(sdata.images["B_image"], coordinate_system="global")
    labels_extent = get_extent(sdata.labels["B_labels"], coordinate_system="global")
    for axis, image_size, mask_size in zip(("y", "x"), image_shape, mask_shape, strict=True):
        assert labels_extent[axis][0] == pytest.approx(image_extent[axis][0])
        # a mask rounded up overhangs the image by less than one mask pixel
        expected_end = image_extent[axis][0] + mask_size * factor
        assert labels_extent[axis][1] == pytest.approx(expected_end)
        if image_size % mask_size == 0:
            assert labels_extent[axis][1] == pytest.approx(image_extent[axis][1])
    # mcquant centroids of the mask are in mask pixels, the table's are in image pixels; obs keeps them as read
    table = sdata.tables["table"]
    np.testing.assert_allclose(table.obsm["spatial"], cells[["X_centroid", "Y_centroid"]].to_numpy() * factor)
    np.testing.assert_allclose(table.obs["X_centroid"], cells["X_centroid"])

    # writing may reorder the axes of the translation, so compare the composed affine maps
    sdata.write(tmp_path / "x.zarr")
    read = get_transformation(read_zarr(tmp_path / "x.zarr").labels["B_labels"], "global")
    assert isinstance(read, Sequence)
    np.testing.assert_allclose(
        read.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        labels_transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )


def test_mask_downsample_keeps_image_pixel_centroids(tmp_path: Path) -> None:
    # inForm centroids are in image pixels, so a downsampled mask does not scale them
    write_qptiff(tmp_path / "Scan1.qptiff", biomarkers=["DAPI"], shape=(128, 256))
    tifffile.imwrite(tmp_path / "mask.tif", np.zeros((64, 128), dtype=np.uint32))
    _write_inform(tmp_path / "Scan1_cell_seg_data.txt", "pixels", area_unit="pixels")

    sdata = phenocycler(
        tmp_path,
        cells_table=tmp_path / "Scan1_cell_seg_data.txt",
        labels=tmp_path / "mask.tif",
        image_models_kwargs={"scale_factors": [2]},
    )

    labels_transformation = get_transformation(sdata.labels[next(iter(sdata.labels))], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations[0] == Scale([2, 2], axes=("y", "x"))
    np.testing.assert_allclose(sdata.tables["table"].obsm["spatial"], [[100, 50], [200, 80]])


@pytest.mark.parametrize(("space", "factor"), [(None, 1), ("image", 1), ("mask", 2)])
def test_generic_px_table_coordinates_space(tmp_path: Path, space: str | None, factor: int) -> None:
    # generic pixel centroids are in image pixels unless `coordinates_space="mask"`
    write_qptiff(tmp_path / "S.qptiff", biomarkers=["DAPI"], shape=(64, 80))
    mask = np.zeros((32, 40), dtype=np.uint32)
    mask[2:4, 2:4], mask[5:7, 5:7] = 1, 2
    tifffile.imwrite(tmp_path / "mask.tif", mask)
    (tmp_path / "cells.csv").write_text("label,cx,cy,CD3\n1,6,6,1\n2,12,12,2\n")
    kwargs = {"x": "cx", "y": "cy", "instance_id": "label"}
    if space is not None:
        kwargs["coordinates_space"] = space

    sdata = phenocycler(
        tmp_path,
        cells_table=tmp_path / "cells.csv",
        labels=tmp_path / "mask.tif",
        image_models_kwargs={"scale_factors": [2]},
        cells_table_kwargs=kwargs,
    )

    np.testing.assert_allclose(sdata.tables["table"].obsm["spatial"], np.array([[6, 6], [12, 12]]) * factor)


@pytest.mark.parametrize(
    ("fmt", "kwargs", "match"),
    [
        ("generic", {"coordinates_space": "labels"}, "must be 'image' or 'mask'"),
        ("generic", {"coordinates_space": "mask", "coordinates_unit": "um"}, "in µm"),
        ("mcquant", {"coordinates_space": "image"}, "only used with generic"),
    ],
)
def test_bad_coordinates_space_raises(tmp_path: Path, fmt: str, kwargs: dict[str, str], match: str) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("CellID,CD3,X_centroid,Y_centroid\n1,1,6,6\n")
    kwargs = {"x": "X_centroid", "y": "Y_centroid", **kwargs} if fmt == "generic" else kwargs

    with pytest.raises(ValueError, match=match):
        read_cell_table(path, fmt, pixel_size_um=0.5, default_region="S", cells_table_kwargs=kwargs)


def test_mask_non_integer_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"], shape=(64, 80))
    tifffile.imwrite(tmp_path / "mask.tif", np.zeros((48, 60), dtype=np.uint32))
    cells = _write_mcquant(tmp_path / "cells.csv", n=2)

    with pytest.warns(UserWarning, match=r"\(48, 60\).*\(64, 80\).*not align"):
        sdata = phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.csv",
            labels=tmp_path / "mask.tif",
            image_models_kwargs={"scale_factors": [2]},
        )

    assert get_transformation(sdata.labels["S_labels"], "global") == Identity()
    np.testing.assert_allclose(sdata.tables["table"].obsm["spatial"], cells[["X_centroid", "Y_centroid"]].to_numpy())
