"""Tests for the CODEX reader on synthetic CODEX Processor/Uploader trees and CIM qptiff files."""

from __future__ import annotations

import json
import shutil
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, Sequence, get_transformation

from spatialdata_io import codex, phenocycler
from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers.codex import _discover_legacy
from tests._akoya_data import (
    forbid_pixel_reads,
    write_fcs,
    write_legacy_processor_tree,
    write_qptiff,
)

if TYPE_CHECKING:
    from pathlib import Path

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def _codex(path: Path | str, **kwargs):  # type: ignore[no-untyped-def]
    """`codex()` with fast pyramids, ignoring the warning that cells without a mask are circles."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "No segmentation mask found", UserWarning)
        return codex(path, image_models_kwargs=_FAST, labels_models_kwargs=_FAST, **kwargs)


@pytest.mark.parametrize("layout", ["per_marker", "hyperstack"])
def test_codex_images_are_lazy_and_named(tmp_path: Path, layout: Literal["per_marker", "hyperstack"]) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, layout=layout)

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        sdata = _codex(tmp_path)

    assert sorted(sdata.images) == ["reg001_image", "reg002_image"]
    for region in ("reg001", "reg002"):
        assert get_channel_names(sdata.images[f"{region}_image"]) == dedupe_channel_names(data["markers"])
        expected = data["images"][region]
        np.testing.assert_array_equal(_level0(sdata, f"{region}_image").values, expected.reshape(-1, 128, 160))
        attrs = sdata.attrs["akoya"]["regions"][region]
        assert attrs["generation"] == "legacy_processor"
        assert [(c["cycle"], c["channel"]) for c in attrs["channels"][3:5]] == [(1, 4), (2, 1)]
    assert sdata.attrs["spatialdata_io_reader"] == "codex"


@pytest.mark.parametrize(
    ("labels", "names_txt", "expected"),
    [
        # ImageJ labels win over channelNames.txt
        (["L0", "L1", "L2", "L3"], "o0\no1\no2\no3\n", ["L0", "L1", "L2", "L3"]),
        # ImageJ writes an empty label for every unlabelled slice; such labels do not replace channelNames.txt
        (["cyc1.tif", "", "", ""], "DAPI\nCD3\nDAPI2\nCD8\n", ["DAPI", "CD3", "DAPI2", "CD8"]),
        (None, "DAPI\nIFNµ\nCD3\nCD8\n".encode("utf-16"), ["DAPI", "IFNµ", "CD3", "CD8"]),
        (None, "DAPI\nIFNµ\nCD3\nCD8\n".encode("cp1252"), ["DAPI", "IFNµ", "CD3", "CD8"]),
        # a channelNames.txt of another length is ignored with a warning
        (None, "DAPI\nCD3\n", ["cyc001_ch001", "cyc001_ch002", "cyc002_ch001", "cyc002_ch002"]),
    ],
    ids=["imagej_labels", "blank_imagej_labels", "utf16", "cp1252", "wrong_length"],
)
def test_codex_hyperstack_channel_names(
    tmp_path: Path, labels: list[str] | None, names_txt: str | bytes, expected: list[str]
) -> None:
    metadata = {"axes": "TCYX"} if labels is None else {"axes": "TCYX", "Labels": labels}
    tifffile.imwrite(
        tmp_path / "reg001_montage.tif", np.zeros((2, 2, 16, 16), np.uint16), imagej=True, metadata=metadata
    )
    names = tmp_path / "channelNames.txt"
    names.write_bytes(names_txt) if isinstance(names_txt, bytes) else names.write_text(names_txt)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sdata = _codex(tmp_path)

    assert get_channel_names(sdata.images["reg001_image"]) == expected
    assert any("channelNames.txt" in str(w.message) for w in record) == (expected[0] == "cyc001_ch001")


@pytest.mark.parametrize("from_subdirectory", [False, True])
def test_codex_sidecars_case_insensitive_and_above_path(tmp_path: Path, from_subdirectory: bool) -> None:
    # SPACEc and the HuBMAP codex-pipeline write `channelnames.txt`; older exports `Experiment.json`
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack", fcs=False, csv=False)
    (tmp_path / "channelNames.txt").rename(tmp_path / "channelnames.txt")
    (tmp_path / "experiment.json").unlink()
    (tmp_path / "Experiment.json").write_text(json.dumps({"xyResolution": 377.5}))

    sdata = _codex(tmp_path / "processed_2020-01-01" / "stitched" if from_subdirectory else tmp_path)

    assert get_channel_names(sdata.images["reg001_image"]) == dedupe_channel_names(data["markers"])
    assert sdata.attrs["akoya"]["regions"]["reg001"]["pixel_size_um"] == pytest.approx(0.3775)


def test_codex_regions_channel_names_and_positional_call(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=2, fcs=False, csv=False)

    sdata = _codex(tmp_path, regions=["reg002"], channel_names=["a", "b"])

    assert list(sdata.images) == ["reg002_image"]
    assert get_channel_names(sdata.images["reg002_image"]) == ["a", "b"]
    assert sdata.coordinate_systems == ["global"]
    # `codex(path, fcs, imread_kwargs)` is the signature of earlier releases; `imread_kwargs` is ignored
    with pytest.warns(UserWarning, match="imread_kwargs"):
        assert sorted(codex(tmp_path, True, {"nframes": 1}).images) == ["reg001_image", "reg002_image"]
    for kwargs, error in (({"regions": ["reg003"]}, "reg003"), ({"channel_names": ["a"]}, "channel_names")):
        with pytest.raises(ValueError, match=error):
            codex(tmp_path, **kwargs)
    with pytest.raises(FileNotFoundError, match="reg"):
        codex(tmp_path / "processed_2020-01-01" / "segm")


def test_codex_and_phenocycler_name_qptiffs_alike(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "exp_reg001.qptiff", biomarkers=["DAPI"], dtype="uint8")
    write_qptiff(tmp_path / "exp_reg001.er.qptiff", biomarkers=["DAPI"], dtype="uint16")
    write_qptiff(tmp_path / "Tonsil_Scan1.er.qptiff", biomarkers=["DAPI"], dtype="uint16")
    write_qptiff(tmp_path / "Preg12_Scan1.qptiff", biomarkers=["DAPI"])

    for reader in (codex, phenocycler):
        sdata = reader(tmp_path, image_models_kwargs=_FAST)
        assert sorted(sdata.images) == ["Preg12_image", "Tonsil_image", "reg001_image"]
        assert sdata.attrs["akoya"]["regions"]["reg001"]["path"].endswith("exp_reg001.er.qptiff")
        assert list(reader(tmp_path / "exp_reg001.qptiff").images) == ["reg001_image"]

    write_qptiff(tmp_path / "other_reg1.qptiff", biomarkers=["DAPI"])
    for reader in (codex, phenocycler):
        with pytest.raises(ValueError, match=r"Several qptiff files map to the same region\(s\) \['reg001'\]"):
            reader(tmp_path)


def test_codex_region_ids_and_ignored_files(tmp_path: Path) -> None:
    for name in ("reg001_montage.tif", "exp_reg002.tif", "1_reg003.tif", "Treg2.tif", "coreg1.tif"):
        tifffile.imwrite(tmp_path / name, np.zeros((1, 4, 4), dtype=np.uint8))
    # raw acquisition and intermediate folders, macOS AppleDouble files and a link back up the tree
    for raw in ("cyc001_reg004/1_00001_Z001_CH1.tif", "tiles/reg005_X01_Y01.tif", "__MACOSX/reg006.tif"):
        (tmp_path / raw).parent.mkdir()
        tifffile.imwrite(tmp_path / raw, np.zeros((4, 4), dtype=np.uint8))
    (tmp_path / "._reg007.tif").write_bytes(b"AppleDouble")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)

    assert sorted(_discover_legacy(tmp_path, fcs=True).images) == ["reg001", "reg002", "reg003"]


def test_codex_discovers_tables(tmp_path: Path) -> None:
    cells = pd.DataFrame({"cell_id": [1.0, 2.0], "x": [1.0, 2.0], "y": [3.0, 4.0], "cyc001_ch001": [5.0, 6.0]})
    tifffile.imwrite(tmp_path / "Kidney.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    cells.to_csv(tmp_path / "Kidney.csv", index=False)
    # not cell tables: sidecar CSVs (also named after a region), flow cytometry tubes and broken FCS files
    (tmp_path / "flow").mkdir()
    write_fcs(tmp_path / "flow" / "tube1.fcs", pd.DataFrame({"FSC-A": [1.0], "SSC-A": [2.0]}))
    (tmp_path / "flow" / "broken.fcs").write_bytes(b"not an fcs file")
    # flowio raises an EOFError and a KeyError for these
    write_fcs(tmp_path / "flow" / "reg001.fcs", cells)
    content = (tmp_path / "flow" / "reg001.fcs").read_bytes()
    (tmp_path / "flow" / "reg001.fcs").write_bytes(content[:100])
    (tmp_path / "flow" / "reg002.fcs").write_bytes(content.replace(b"$DATATYPE", b"$DATATYPX"))
    (tmp_path / "reg001_antibodies.csv").write_text("Antibody,Clone\nCD3,UCHT1\n")
    (tmp_path / "xy.csv").write_text("x,y\n1.0,2.0\n")
    assert _discover_legacy(tmp_path, fcs=True).tables == {"*": tmp_path / "Kidney.csv"}

    # a CODEX gate export next to the table: the reader cannot tell which one holds the cells
    write_fcs(tmp_path / "flow" / "CD3pos.fcs", cells.iloc[:1])
    with pytest.raises(ValueError, match="cannot choose"):
        _discover_legacy(tmp_path, fcs=True)

    # compensated beats uncompensated before the file format; `fcs` picks the format of one table
    fcs_dir = tmp_path / "exp" / "segm" / "segm-1" / "fcs"
    for kind in ("compensated", "uncompensated"):
        (fcs_dir / kind).mkdir(parents=True)
    cells.to_csv(fcs_dir / "compensated" / "reg001_compensated.csv", index=False)
    write_fcs(fcs_dir / "compensated" / "reg001_compensated.fcs", cells)
    write_fcs(fcs_dir / "uncompensated" / "reg001_uncompensated.fcs", cells)
    for fcs, suffix in ((True, ".fcs"), (False, ".csv")):
        assert _discover_legacy(tmp_path / "exp", fcs=fcs).tables == {
            "reg001": fcs_dir / f"compensated/reg001_compensated{suffix}"
        }

    # two segmentation runs of one region are not resolved by their path
    segm_10 = tmp_path / "exp" / "segm" / "segm-10" / "fcs" / "compensated"
    segm_10.mkdir(parents=True)
    cells.to_csv(segm_10 / "reg001_compensated.csv", index=False)
    with pytest.raises(ValueError, match=r"segm-1/.*segm-10/|segm-10/.*segm-1/"):
        _discover_legacy(tmp_path / "exp", fcs=True)


def test_codex_tables_of_several_regions(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, n_cells=50)
    shutil.rmtree(tmp_path / "processed_2020-01-01" / "stitched" / "reg002")

    with pytest.warns(UserWarning, match=r"No segmentation mask found for region\(s\) \['reg001', 'reg002'\]"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    assert table.n_obs == 100
    assert isinstance(table.X, np.ndarray)
    assert table.X.dtype == np.float32
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert list(table.var_names) == dedupe_channel_names(data["markers"])
    reg002 = table[table.obs["region"] == "reg002_cells"]
    np.testing.assert_allclose(reg002.obsm["spatial"], data["cells"][["x", "y"]], rtol=1e-6)
    radius = sdata.shapes["reg002_cells"]["radius"].to_numpy()
    np.testing.assert_allclose(radius, np.sqrt(data["cells"]["size"] / np.pi), rtol=1e-6)
    assert sdata.attrs["akoya"]["cell_tables"]["reg002"].endswith("reg002_compensated.fcs")
    # regions without stage positions, and a region without an image, have their own coordinate systems
    assert sorted(sdata.coordinate_systems) == ["reg001", "reg002"]
    for name in ("reg001_image", "reg001_cells"):
        assert get_transformation(sdata[name], get_all=True) == {"reg001": Identity()}
    assert get_transformation(sdata["reg002_cells"], get_all=True) == {"reg002": Identity()}
    assert list(_codex(tmp_path, regions=["reg002"]).shapes) == ["reg002_cells"]

    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")
    np.testing.assert_array_equal(_level0(read, "reg001_image").values, _level0(sdata, "reg001_image").values)
    assert get_transformation(read["reg002_cells"], get_all=True) == {"reg002": Identity()}
    assert read.tables["table"].obs.index.name is None
    assert read.tables["table"].obs["cell_id"].dtype == np.int64
    assert read.attrs["akoya"] == sdata.attrs["akoya"]
    assert read.attrs["spatialdata_io_reader"] == "codex"


def test_codex_tables_with_different_panels_or_formats(tmp_path: Path) -> None:
    columns = {"cell_id": [1.0], "x": [1.0], "y": [2.0]}
    write_fcs(tmp_path / "reg001.fcs", pd.DataFrame(columns | {"cyc001_ch001": [3.0]}), {"cyc001_ch001": "DAPI"})
    pd.DataFrame(columns | {"cyc001_ch001:DAPI": [3.0], "cyc001_ch002:CD3": [4.0]}).to_csv(
        tmp_path / "reg002.csv", index=False
    )

    sdata = _codex(tmp_path)

    assert sorted(sdata.tables) == ["reg001_table", "reg002_table"]
    assert list(sdata.tables["reg002_table"].var_names) == ["DAPI", "CD3"]
    assert sdata.tables["reg001_table"].uns["spatialdata_attrs"]["region"] == ["reg001_cells"]

    # the CSV and FCS tables of one panel share a table; their source columns differ. With `fcs=False`, a region with
    # only an FCS table is read from it
    pd.DataFrame(columns | {"cyc001_ch001:DAPI": [3.0]}).to_csv(tmp_path / "reg002.csv", index=False)
    with pytest.warns(UserWarning, match="no CSV cell table was found"):
        sdata = _codex(tmp_path, fcs=False)
    assert list(sdata.tables["table"].var_names) == ["DAPI"]
    assert "source_column" not in sdata.tables["table"].var


@pytest.mark.parametrize(
    ("images", "rows", "regions", "expected"),
    [
        # the region is the file name without its image extension, `.ome` included
        (["Kidney.ome.tif"], ["1,1,1,2"], None, ["Kidney_cells"]),
        # region values are matched after `sanitize_name`, like the region ids
        (
            ["Tonsil 1.tif", "Tonsil 2.tif"],
            ["1,Tonsil 1,1,2", "1,Tonsil 2,3,4"],
            None,
            ["Tonsil_1_cells", "Tonsil_2_cells"],
        ),
        # a region without an image is still a region
        (["reg001.tif"], ["1,1,1,2", "1,2,3,4"], ["reg002"], ["reg002_cells"]),
    ],
)
def test_codex_regionless_table_regions(
    tmp_path: Path, images: list[str], rows: list[str], regions: list[str] | None, expected: list[str]
) -> None:
    for name in images:
        tifffile.imwrite(tmp_path / name, np.zeros((2, 16, 16), dtype=np.uint16))
    (tmp_path / "cells.csv").write_text("cell_id,region,x,y,cyc001_ch001\n" + "".join(f"{row},5\n" for row in rows))

    sdata = _codex(tmp_path, regions=regions)

    assert list(sdata.tables["table"].obs["region"].cat.categories) == expected
    assert sorted(sdata.shapes) == expected


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        ("two_images", "cannot be assigned to regions"),
        ("duplicate_ids", r"reg001_compensated.csv has 1 duplicate instance id"),
        ("overlapping", r"Several cell tables contain the cells of region\(s\) \['reg001'\]"),
        ("other_region", "has no cells of region 'reg002'"),
    ],
)
def test_codex_table_errors(tmp_path: Path, setup: str, match: str) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=1, fcs=False, n_cells=3)
    table = next(tmp_path.rglob("reg002_compensated.csv" if setup == "other_region" else "reg001_compensated.csv"))
    if setup == "two_images":
        table.rename(tmp_path / "cells.csv")
        data["cells"].drop(columns="region").to_csv(tmp_path / "cells.csv", index=False)
    elif setup == "duplicate_ids":
        data["cells"].assign(cell_id=[1.0, 1.0, 2.0], region=1.0).to_csv(table, index=False)
    elif setup == "overlapping":
        (tmp_path / "all_cells.csv").write_text("cell_id,region,x,y,cyc001_ch001\n1,1,1.0,2.0,3.0\n")
    else:
        # the table of reg002 whose region column names reg001
        data["cells"].assign(region=1.0).to_csv(table, index=False)

    with pytest.raises(ValueError, match=match):
        codex(tmp_path, fcs=False)


@pytest.mark.parametrize(
    ("image", "mask", "region"),
    [
        ("expr/reg001_expr.ome.tiff", "mask/reg001_mask.ome.tiff", "reg001"),
        ("Kidney sample.tif", "Kidney sample_labels.tif", "Kidney_sample"),
        ("Tonsil_Scan1.er.qptiff", "Tonsil_Scan1_mask.tif", "Tonsil"),
        ("Tonsil_Scan1.er.qptiff", "Tonsil_mask.tif", "Tonsil"),
    ],
)
def test_codex_discovers_the_mask_of_a_region(tmp_path: Path, image: str, mask: str, region: str) -> None:
    for name in (image, mask):
        (tmp_path / name).parent.mkdir(exist_ok=True)
    if image.endswith("qptiff"):
        write_qptiff(tmp_path / image, biomarkers=["DAPI"], shape=(16, 16))
    else:
        tifffile.imwrite(tmp_path / image, np.zeros((2, 16, 16), dtype=np.uint16))
    tifffile.imwrite(tmp_path / mask, np.arange(256, dtype=np.uint16).reshape(16, 16))
    (tmp_path / "cells.csv").write_text("cell_id,x,y,cyc001_ch001\n1,0.0,0.0,1.0\n2,1.0,0.0,2.0\n")

    sdata = _codex(tmp_path)

    assert list(sdata.images) == [f"{region}_image"]
    assert list(sdata.labels) == [f"{region}_labels"]
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == [f"{region}_labels"]
    assert sdata.attrs["akoya"]["labels"] == {region: str(tmp_path / mask)}


def test_codex_mask_discovery_errors_and_explicit_labels(tmp_path: Path) -> None:
    for region in ("reg001", "reg002"):
        tifffile.imwrite(tmp_path / f"{region}_montage.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "Other_mask.tif", np.zeros((16, 16), dtype=np.uint16))
    with pytest.warns(UserWarning, match="Ignoring the mask .*Other_mask.tif"):
        assert not _codex(tmp_path).labels

    for name in ("reg002_mask.tif", "reg002_labels.tif"):
        tifffile.imwrite(tmp_path / name, np.ones((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="several masks for region 'reg002'"):
        _codex(tmp_path)

    # given labels disable mask discovery and are not read as images
    tifffile.imwrite(tmp_path / "reg001_x_mask.tif", np.ones((16, 16), dtype=np.uint16))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdata = _codex(tmp_path, labels={"reg001": tmp_path / "reg001_x_mask.tif"})
    assert sorted(sdata.images) == ["reg001_image", "reg002_image"]
    assert list(sdata.labels) == ["reg001_labels"]


def test_codex_explicit_table_and_downsampled_mask(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, n_cells=50)
    (tmp_path / "mine.csv").write_text("cell_id,x,y,cyc001_ch001:CD3\n1,1.0,4.0,1.0\n2,3.0,6.0,2.0\n")
    mask = np.zeros((64, 80), dtype=np.uint16)
    mask[2, 0], mask[3, 1] = 1, 2
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdata = _codex(
            tmp_path, cells_table={"reg002": tmp_path / "mine.csv"}, labels={"reg002": tmp_path / "mask.tif"}
        )

    table = sdata.tables["table"]
    assert list(table.var_names) == ["CD3"]
    assert table.uns["spatialdata_attrs"]["region"] == ["reg002_labels"]
    assert not sdata.shapes
    assert sdata.attrs["akoya"]["cell_tables"] == {"reg002": str(tmp_path / "mine.csv")}
    # the mask is scaled onto the image; the centroids stay in image pixels
    labels_transformation = get_transformation(sdata["reg002_labels"], "reg002")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations[0] == Scale([2, 2], axes=("y", "x"))
    np.testing.assert_allclose(table.obsm["spatial"], [[1, 4], [3, 6]])
    # an explicit table of an unselected region does not bring the discovered tables back
    assert not _codex(tmp_path, regions=["reg001"], cells_table={"reg002": tmp_path / "mine.csv"}).tables


@pytest.mark.parametrize("fcs", [True, False])
def test_codex_old_flat_layout_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcs: bool) -> None:
    # the original reader read the image (and the CSV) relative to the working directory, and ignored `fcs`
    data = write_legacy_processor_tree(tmp_path / "data", layout="old_flat", fcs=fcs)
    monkeypatch.chdir(tmp_path)

    sdata = _codex(tmp_path / "data", fcs=fcs)

    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data["images"]["reg001"][:1])
    table = sdata.tables["table"]
    assert list(table.obs["region"].cat.categories) == ["reg001_cells"]
    np.testing.assert_allclose(table.X, data["cells"].filter(regex="^cyc").to_numpy(), rtol=1e-6)
