"""Tests for the CODEX reader on synthetic CODEX Processor/Uploader trees and CIM qptiff files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata.models import get_channel_names

from spatialdata_io import codex
from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers.codex import discover_legacy, read_legacy_image
from tests._akoya_data import cim_experiment_v4_xml, write_legacy_processor_tree, write_qptiff

if TYPE_CHECKING:
    from pathlib import Path

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def test_codex_per_marker_images(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker")

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["reg001_image", "reg002_image"]
    for region in ("reg001", "reg002"):
        image = sdata.images[f"{region}_image"]
        assert get_channel_names(image) == dedupe_channel_names(data["markers"])
        np.testing.assert_array_equal(_level0(sdata, f"{region}_image").values, data["images"][region])
        attrs = sdata.attrs["akoya"]["regions"][region]
        assert attrs["generation"] == "legacy_processor"
        assert attrs["pixel_size_um"] is None
        assert [(c["cycle"], c["channel"]) for c in attrs["channels"][:5]] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]
    assert sdata.attrs["spatialdata_io_reader"] == "codex"


def test_codex_hyperstack_channel_names_txt(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack")

    sdata = codex(str(tmp_path), image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    image = _level0(sdata, "reg001_image")
    assert image.shape[0] == 8
    assert get_channel_names(sdata.images["reg001_image"]) == dedupe_channel_names(data["markers"])
    np.testing.assert_array_equal(image.values, data["images"]["reg001"].reshape(8, *image.shape[1:]))


def test_codex_hyperstack_bad_channel_names_warns(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack")
    (tmp_path / "channelNames.txt").write_text("\n".join(data["markers"][:7]) + "\n")

    with pytest.warns(UserWarning, match="channelNames.txt"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    names = get_channel_names(sdata.images["reg001_image"])
    assert names[:5] == ["cyc001_ch001", "cyc001_ch002", "cyc001_ch003", "cyc001_ch004", "cyc002_ch001"]
    assert len(names) == 8


def test_codex_hyperstack_imagej_labels_win(tmp_path: Path) -> None:
    labels = [f"L{i}" for i in range(6)]
    tifffile.imwrite(
        tmp_path / "reg001_X01_Y01_Z01.tif",
        np.zeros((3, 2, 32, 48), dtype=np.uint16),
        imagej=True,
        metadata={"axes": "ZCYX", "Labels": labels},
    )
    (tmp_path / "channelNames.txt").write_text("\n".join(f"other{i}" for i in range(6)))

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert get_channel_names(sdata.images["reg001_image"]) == labels


def test_codex_cim_qptiff(tmp_path: Path) -> None:
    path = tmp_path / "experiment_reg001.qptiff"
    data = write_qptiff(
        path,
        biomarkers=["DAPI", "CD3", "DAPI", "CD20"],
        software="Codex Instrument Manager 1.30.0.12",
        scan_profile=cim_experiment_v4_xml([["DAPI", "CD3"], ["DAPI", "CD20"]]),
        label_page=False,
    )

    for source in (tmp_path, path):
        sdata = codex(source, image_models_kwargs=_FAST)
        assert list(sdata.images) == ["reg001_image"]
        assert get_channel_names(sdata.images["reg001_image"]) == ["DAPI", "CD3", "DAPI (2)", "CD20"]
        np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data)
        assert sdata.attrs["akoya"]["regions"]["reg001"]["generation"] == "cim_qptiff"


def test_codex_old_flat_layout_image(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, layout="old_flat")

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    # only the first plane of the stack is written to disk
    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data["images"]["reg001"][:1])


def test_codex_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reg"):
        codex(str(tmp_path))


def test_codex_ignores_raw_tiles(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    expected = sorted(codex(tmp_path, image_models_kwargs=_FAST).images)
    processed = tmp_path / "processed_2020-01-01"
    plane = np.zeros((16, 16), dtype=np.uint16)
    for raw in (
        tmp_path / "cyc001_reg001" / "1_00001_Z001_CH1.tif",
        tmp_path / "cyc001_reg002" / "reg002_X01_Y01_Z01.tif",
        processed / "tiles" / "reg002_X01_Y01" / "reg002_X01_Y01_t001_z001_c001.tif",
        processed / "bestFocus" / "reg002_X01_Y01_Z08.tif",
    ):
        raw.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(raw, plane)

    assert sorted(codex(tmp_path, image_models_kwargs=_FAST).images) == expected


def test_codex_regions_channel_names_and_pixel_size(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=2, layout="per_marker")
    (tmp_path / "experiment.json").write_text(json.dumps({"xyResolution": 377.5, "numChannels": 2}))

    sdata = codex(tmp_path, regions=["reg002"], channel_names=["a", "b"], image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg002_image"]
    assert get_channel_names(sdata.images["reg002_image"]) == ["a", "b"]
    region = sdata.attrs["akoya"]["regions"]["reg002"]
    assert region["pixel_size_um"] == pytest.approx(0.3775)
    assert region["raw"]["experiment_json"]["numChannels"] == 2
    with pytest.raises(ValueError, match="reg003"):
        codex(tmp_path, regions=["reg003"])
    with pytest.raises(ValueError, match="channel_names"):
        codex(tmp_path, channel_names=["a"])


def test_discover_legacy_tables_prefer_requested_format(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker", fcs=True, csv=True)

    assert {region: path.suffix for region, path in discover_legacy(tmp_path, fcs=True).tables.items()} == {
        "reg001": ".fcs",
        "reg002": ".fcs",
    }
    assert {region: path.suffix for region, path in discover_legacy(tmp_path, fcs=False).tables.items()} == {
        "reg001": ".csv",
        "reg002": ".csv",
    }


def test_codex_not_implemented_cells_table(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    with pytest.raises(NotImplementedError):
        codex(tmp_path, cells_table=tmp_path / "x.csv")


@pytest.mark.parametrize("imagej", [False, True])
def test_read_legacy_image_tiles_uncompressed_planes(tmp_path: Path, imagej: bool) -> None:
    # uncompressed single-strip planes are one zarr chunk each; they must be memory-mapped and tiled instead
    stack = np.random.default_rng(0).integers(0, 2**16, (2, 100, 130), dtype=np.uint16)
    files = [tmp_path / f"reg001_cyc001_ch{i + 1:03d}_M{i}.tif" for i in range(2)]
    for file, plane in zip(files, stack, strict=True):
        tifffile.imwrite(file, plane, imagej=imagej, rowsperstrip=plane.shape[0])
    hyperstack = tmp_path / "reg002_montage.tif"
    tifffile.imwrite(hyperstack, stack[None], imagej=True, metadata={"axes": "TCYX"})

    for source in (files, [hyperstack]):
        image, _ = read_legacy_image(source, None, {}, image_models_kwargs={"chunks": (1, 32, 64)})
        assert max(image.chunks[-2]) <= 32
        assert max(image.chunks[-1]) <= 64
        np.testing.assert_array_equal(image.compute(), stack)


def test_codex_single_tif_without_region(tmp_path: Path) -> None:
    path = tmp_path / "Kidney.tif"
    tifffile.imwrite(path, np.zeros((3, 32, 48), dtype=np.uint16), imagej=True, metadata={"axes": "CYX"})

    sdata = codex(path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["Kidney_image"]
    assert _level0(sdata, "Kidney_image").shape == (3, 32, 48)


def test_discover_legacy_regionless_table_ignores_sidecar_csv(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path / "exp", layout="per_marker", fcs=False, csv=False)
    (tmp_path / "antibody_list.csv").write_text("marker,clone\nCD3,UCHT1\n")
    cells = tmp_path / "exp" / "results" / "cells.csv"
    cells.parent.mkdir()
    cells.write_text("cell_id,x,y,cyc001_ch001:DAPI\n1,2.0,3.0,4.0\n")

    assert discover_legacy(tmp_path, fcs=True).tables == {"*": cells}

    (tmp_path / "clusters.csv").write_text("cellid,cluster\n1,2\n")
    (tmp_path / "xy.csv").write_text("x,y\n1.0,2.0\n")
    assert discover_legacy(tmp_path, fcs=True).tables == {"*": cells}

    (tmp_path / "other_cells.csv").write_text("cell_id:cell_id,x:x,y:y\n1,1.0,2.0\n")
    with pytest.raises(ValueError, match="cannot choose"):
        discover_legacy(tmp_path, fcs=True)


def test_codex_imread_kwargs_warns_when_unused(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    with pytest.warns(UserWarning, match="imread_kwargs"):
        codex(tmp_path, imread_kwargs={"nframes": 1}, image_models_kwargs=_FAST)


def test_codex_table_multi_region(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker", n_cells=50)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.tables) == ["table"]
    table = sdata.tables["table"]
    assert table.n_obs == 100
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert list(table.var_names) == dedupe_channel_names(data["markers"])
    assert sorted(sdata.shapes) == ["reg001_cells", "reg002_cells"]
    # the fixture returns the cells of the last region
    radius = sdata.shapes["reg002_cells"]["radius"].to_numpy()
    np.testing.assert_allclose(radius, np.sqrt(data["cells"]["size"] / np.pi), rtol=1e-6)
    reg002 = table[table.obs["region"] == "reg002_cells"]
    np.testing.assert_allclose(reg002.obsm["spatial"], data["cells"][["x", "y"]], rtol=1e-6)
    assert sdata.attrs["akoya"]["cell_tables"]["reg002"].endswith("reg002_compensated.fcs")


@pytest.mark.parametrize("fcs", [True, False])
def test_codex_old_flat_layout_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcs: bool) -> None:
    # the original reader read the image (and the CSV) relative to the working directory
    data = write_legacy_processor_tree(tmp_path / "data", layout="old_flat", fcs=fcs)
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path / "data", fcs=fcs, image_models_kwargs=_FAST)

    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data["images"]["reg001"][:1])
    table = sdata.tables["table"]
    assert table.n_obs == 50
    assert list(table.obs["region"].cat.categories) == ["reg001_cells"]
    np.testing.assert_allclose(table.X, data["cells"].filter(regex="^cyc").to_numpy(), rtol=1e-6)


def test_codex_fcs_false_falls_back_to_fcs_with_warning(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=True, csv=False)

    with pytest.warns(UserWarning, match="CSV"):
        sdata = codex(tmp_path, fcs=False, image_models_kwargs=_FAST)

    assert list(sdata.tables["table"].var_names) == dedupe_channel_names(data["markers"])


def test_codex_without_tables_reads_images_only(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=False)

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    assert not sdata.tables
    assert not sdata.shapes


def test_codex_table_float32_dense(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=True)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, fcs=False, image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    assert isinstance(table.X, np.ndarray)
    assert table.X.dtype == np.float32


def test_codex_regions_subsets_tables(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker")

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, regions=["reg002"], image_models_kwargs=_FAST)

    assert list(sdata.shapes) == ["reg002_cells"]
    assert list(sdata.tables["table"].obs["region"].cat.categories) == ["reg002_cells"]


def test_codex_single_fcs_file(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    fcs = tmp_path / "processed_2020-01-01" / "segm" / "segm-1" / "fcs" / "compensated" / "reg001_compensated.fcs"

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(fcs)

    assert not sdata.images
    assert list(sdata.shapes) == ["reg001_cells"]
    assert sdata.tables["table"].n_obs == 50


def test_codex_unreadable_regionless_table_is_skipped(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "reg001_montage.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    # a region-less table that is not assigned to a region: there are two images and no `region` column
    tifffile.imwrite(tmp_path / "reg002_montage.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    (tmp_path / "cells.csv").write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,3.0\n")
    (tmp_path / "clusters.csv").write_text("cellid,cluster\n1,2\n")

    with pytest.warns(UserWarning, match="Ignoring the cell table"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["reg001_image", "reg002_image"]
    assert not sdata.tables
    assert sdata.attrs["akoya"]["cell_tables"] == {}


def test_codex_regionless_table_belongs_to_single_stem_image(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "Kidney.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    (tmp_path / "cells.csv").write_text("cell_id,region,x,y,size,cyc001_ch001\n1,1,1.0,2.0,3.0,4.0\n")

    for regions in (None, ["Kidney"]):
        with pytest.warns(UserWarning, match="circles"):
            sdata = codex(tmp_path, regions=regions, image_models_kwargs=_FAST)
        assert list(sdata.tables["table"].obs["region"].cat.categories) == ["Kidney_cells"]
        assert list(sdata.shapes) == ["Kidney_cells"]
        assert sdata.attrs["akoya"]["cell_tables"] == {"*": str(tmp_path / "cells.csv")}


def test_codex_overlapping_tables_raise(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=True)
    (tmp_path / "all_cells.csv").write_text("cell_id,region,x,y,cyc001_ch001\n1,1,1.0,2.0,3.0\n")

    with pytest.raises(ValueError, match=r"Several cell tables contain the cells of region\(s\) \['reg001'\]"):
        codex(tmp_path, fcs=False, image_models_kwargs=_FAST)
