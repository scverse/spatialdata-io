"""Tests for the PhenoCycler-Fusion reader on synthetic qptiff files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names

from spatialdata_io import phenocycler
from tests._akoya_data import write_qptiff

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def test_read_single_file(tmp_path: Path) -> None:
    path = tmp_path / "Tonsil_Scan1.qptiff"
    data = write_qptiff(path, biomarkers=["DAPI", "CD3", "CD3"])

    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["Tonsil_image"]
    assert get_channel_names(sdata.images["Tonsil_image"]) == ["DAPI", "CD3", "CD3 (2)"]
    np.testing.assert_array_equal(_level0(sdata, "Tonsil_image").values, data)
    assert sdata.attrs["akoya"]["regions"]["Tonsil"]["generation"] == "fusion2"
    assert sdata.attrs["spatialdata_io_reader"] == "phenocycler"


def test_read_directory_multiple_scans_and_subset(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI", "CD8"])

    sdata = phenocycler(str(tmp_path), image_models_kwargs=_FAST)
    assert sorted(sdata.images) == ["A_image", "B_image"]
    assert sorted(sdata.attrs["akoya"]["regions"]) == ["A", "B"]

    subset = phenocycler(tmp_path, regions=["B"], image_models_kwargs=_FAST)
    assert list(subset.images) == ["B_image"]
    assert list(subset.attrs["akoya"]["regions"]) == ["B"]

    with pytest.raises(ValueError, match="Z"):
        phenocycler(tmp_path, regions=["Z"])


def test_region_collision_keeps_full_stem(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "S_Scan2.qptiff", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["S_Scan1_image", "S_Scan2_image"]


def test_er_preferred(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint8")
    write_qptiff(tmp_path / "S_Scan1.er.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint16")

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]
    assert _level0(sdata, "S_image").dtype == np.uint16
    assert sdata.attrs["akoya"]["regions"]["S"]["path"].endswith("S_Scan1.er.qptiff")


def test_ignores_acquisition_files(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "images"
    acquisition = images / "phenocycler"
    acquisition.mkdir(parents=True)
    write_qptiff(images / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(acquisition / "S_Scan1_Cycle1.qptiff.raw", biomarkers=["DAPI"])
    write_qptiff(acquisition / "S.raw.qptiff", biomarkers=["DAPI"])
    write_qptiff(acquisition / "S_Scan1_Cycle1.qptiff", biomarkers=["DAPI"])
    write_qptiff(images / "S_Scan1.qptiff.intermediate", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]


def test_sidecars_hubmap_layout(tmp_path: Path) -> None:
    raw_images = tmp_path / "raw" / "images"
    processed = tmp_path / "lab_processed" / "images"
    raw_images.mkdir(parents=True)
    processed.mkdir(parents=True)
    write_qptiff(raw_images / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    wells = [
        {"wellName": "A1", "items": [{"markerName": "DAPI"}]},
        {"wellName": "A2", "items": [{"markerName": "DAPI"}, {"markerName": "CD3"}, {"markerName": "--"}]},
    ]
    xpd = {"formatType": "Protein", "wells": wells}
    (raw_images / "S_Scan1.xpd").write_text(json.dumps(xpd))
    (processed / "S_Scan1.qptiff.channels.csv").write_text("Channel,Marker\n1,DAPI\n2,CD3\n")

    # DAPI repeats per cycle and blanks do not count as channels, so the sidecars match the two pages
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    raw = sdata.attrs["akoya"]["regions"]["S"]["raw"]
    assert raw["xpd"] == xpd
    assert raw["channels_csv"] == [{"Channel": "1", "Marker": "DAPI"}, {"Channel": "2", "Marker": "CD3"}]


def test_sidecar_channel_mismatch_warns(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3", "CD8"])
    xpd = {"wells": [{"wellName": "A1", "items": [{"markerName": "DAPI"}, {"markerName": "CD4"}]}]}
    (tmp_path / "S_Scan1.xpd").write_text(json.dumps(xpd))

    with pytest.warns(UserWarning, match="3") as record:
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert any("2" in str(warning.message) for warning in record)
    assert get_channel_names(sdata.images["S_image"]) == ["DAPI", "CD3", "CD8"]


def test_channels_csv_mismatch_warns(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])
    (tmp_path / "S_Scan1.qptiff.channels.csv").write_text("Marker\nDAPI\nCD3\nCD8\n")

    with pytest.warns(UserWarning, match="channels.csv"):
        phenocycler(path, image_models_kwargs=_FAST)


def test_invalid_xpd_warns_and_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    (tmp_path / "S_Scan1.xpd").write_text("not json")

    with pytest.warns(UserWarning, match="not valid JSON"):
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert "xpd" not in sdata.attrs["akoya"]["regions"]["S"]["raw"]


def test_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="\\*.qptiff"):
        phenocycler(str(tmp_path))


def test_channel_names_override(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])

    sdata = phenocycler(path, channel_names=["nuclei", "T cells"], image_models_kwargs=_FAST)
    assert get_channel_names(sdata.images["S_image"]) == ["nuclei", "T cells"]
    channels = sdata.attrs["akoya"]["regions"]["S"]["channels"]
    assert [channel["name_deduped"] for channel in channels] == ["nuclei", "T cells"]

    with pytest.raises(ValueError, match="expected 2"):
        phenocycler(path, channel_names=["only one"])


def test_cells_table_and_labels_not_implemented(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])

    with pytest.raises(NotImplementedError):
        phenocycler(path, cells_table=tmp_path / "cells.csv")
    with pytest.raises(NotImplementedError):
        phenocycler(path, labels=tmp_path / "mask.tif")


def test_write_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3", "CD3"])
    sdata = phenocycler(path, image_models_kwargs=_FAST)

    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")

    assert get_channel_names(read.images["S_image"]) == get_channel_names(sdata.images["S_image"])
    assert read.attrs["akoya"] == sdata.attrs["akoya"]
    assert read.attrs["spatialdata_io_reader"] == "phenocycler"


def test_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("full page decode during read")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert isinstance(_level0(sdata, "S_image").data, da.Array)
