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
from spatialdata_io.readers.codex import discover_legacy
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
