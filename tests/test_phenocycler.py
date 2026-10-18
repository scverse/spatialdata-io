"""Tests for the PhenoCycler-Fusion reader on synthetic qptiff files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation

from spatialdata_io import phenocycler
from spatialdata_io.__main__ import phenocycler_wrapper
from tests._akoya_data import forbid_pixel_reads, write_qptiff

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def _write_mask(path: Path, shape: tuple[int, int] = (256, 320)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint32)
    mask[10:20, 10:20], mask[30:40, 10:20] = 1, 2
    tifffile.imwrite(path, mask)
    return mask


def test_read_is_lazy_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "Tonsil_Scan1.qptiff"
    data = write_qptiff(path, biomarkers=["DAPI", "CD3", "CD3"])

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert isinstance(_level0(sdata, "Tonsil_image").data, da.Array)
    assert get_channel_names(sdata.images["Tonsil_image"]) == ["DAPI", "CD3", "CD3 (2)"]
    np.testing.assert_array_equal(_level0(sdata, "Tonsil_image").values, data)
    assert sdata.attrs["akoya"]["regions"]["Tonsil"]["generation"] == "fusion2"
    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")
    assert get_channel_names(read.images["Tonsil_image"]) == ["DAPI", "CD3", "CD3 (2)"]
    assert read.attrs["akoya"] == sdata.attrs["akoya"]
    assert read.attrs["spatialdata_io_reader"] == "phenocycler"


@pytest.mark.parametrize(
    ("name", "kwargs", "region", "channels"),
    [
        (
            "Slide_Scan1.unmixed.qptiff",
            {"biomarkers": [None, None], "names": ["PDL1 (Opal 520)", "DAPI"], "dtype": "float32"},
            "Slide_Scan1.unmixed",
            ["PDL1 (Opal 520)", "DAPI"],
        ),
        ("HandE_Scan1.qptiff", {"biomarkers": [None], "names": ["BF"], "rgb": True}, "HandE", ["r", "g", "b"]),
        # inForm component data; the brackets and comma of its name are not valid in an element name
        (
            "LuCa_[13860,52919]_1x1component_data.tif",
            {"biomarkers": [None], "names": ["A"], "dtype": "float32", "levels": 1},
            "LuCa__13860_52919__1x1component_data",
            ["A"],
        ),
    ],
    ids=["unmixed", "brightfield", "inform_component"],
)
def test_vectra_and_phenoimager_files(
    tmp_path: Path, name: str, kwargs: dict[str, Any], region: str, channels: list[str]
) -> None:
    data = write_qptiff(tmp_path / name, software="VectraPolaris 1.0", **kwargs)

    sdata = phenocycler(tmp_path / name, image_models_kwargs=_FAST)

    image = _level0(sdata, f"{region}_image")
    assert get_channel_names(sdata.images[f"{region}_image"]) == channels
    np.testing.assert_array_equal(image.values, np.moveaxis(data, -1, 0) if kwargs.get("rgb") else data)
    assert sdata.attrs["akoya"]["regions"][region]["generation"] == "qpi_other"


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ([], r"No \.qptiff files"),
        (["A [1].qptiff", "A_[1].qptiff"], r"same region\(s\) \['A__1_'\]"),
        (["plain.tif"], "not a PerkinElmer/Akoya QPTIFF"),
    ],
)
def test_bad_inputs_raise(tmp_path: Path, files: list[str], match: str) -> None:
    for name in files:
        if name.endswith(".tif"):
            tifffile.imwrite(tmp_path / name, np.zeros((8, 8), dtype=np.uint8))
        else:
            write_qptiff(tmp_path / name, biomarkers=["DAPI"])

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        phenocycler(tmp_path / files[0] if files and files[0].endswith(".tif") else tmp_path)


def test_regions_and_coordinate_systems(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"], stage_position_cm=(1.0, 2.0))
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI"], stage_position_cm=(1.01, 2.0))
    write_qptiff(tmp_path / "C_Scan1.qptiff", biomarkers=["DAPI"], slide_id="SLIDE2")
    write_qptiff(tmp_path / "C_Scan2.qptiff", biomarkers=["DAPI"], slide_id="SLIDE2")

    sdata = phenocycler(tmp_path, regions=["A", "B"], image_models_kwargs=_FAST)

    # scans of the same slide are placed at their stage positions
    assert sdata.coordinate_systems == ["global"]
    assert get_transformation(sdata["A_image"], get_all=True) == {"global": Translation([0, 0], axes=("x", "y"))}
    offset = get_transformation(sdata["B_image"], "global")
    assert isinstance(offset, Translation)
    np.testing.assert_allclose(offset.translation, [100 / 0.5069, 0], rtol=1e-3)
    # unrelated scans get their own coordinate systems; `_Scan<N>` is kept if it tells regions apart
    sdata = phenocycler(tmp_path, regions=["A", "C_Scan1", "C_Scan2"], image_models_kwargs=_FAST)
    assert sorted(sdata.coordinate_systems) == ["A", "C_Scan1", "C_Scan2"]
    assert get_transformation(sdata["C_Scan1_image"], get_all=True) == {"C_Scan1": Identity()}


def test_read_all_bit_depths(tmp_path: Path) -> None:
    for suffix, dtype in ((".qptiff", "uint8"), (".er.qptiff", "uint16")):
        write_qptiff(
            tmp_path / f"S{suffix}", biomarkers=["DAPI"], shape=(64, 80), dtype=dtype, stage_position_cm=(1, 2)
        )
    write_qptiff(tmp_path / "T.qptiff", biomarkers=["DAPI"], shape=(64, 80), stage_position_cm=(1.1, 2))
    mask = np.zeros((32, 40), dtype=np.uint32)
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    assert sorted(phenocycler(tmp_path, image_models_kwargs=_FAST).images) == ["S_image", "T_image"]
    sdata = phenocycler(
        tmp_path / "S.er.qptiff", labels=tmp_path / "mask.tif", read_all_bit_depths=True, image_models_kwargs=_FAST
    )

    assert sorted(sdata.images) == ["S_8bit_image", "S_image"]
    assert _level0(sdata, "S_image").dtype == np.uint16
    assert _level0(sdata, "S_8bit_image").dtype == np.uint8
    assert sdata.attrs["akoya"]["regions"]["S_8bit"]["path"].endswith("S.qptiff")
    image_transformations = get_transformation(sdata["S_image"], get_all=True)
    assert get_transformation(sdata["S_8bit_image"], get_all=True) == image_transformations
    # the mask is scaled onto the extended-range region only
    labels_transformation = get_transformation(sdata["S_labels"], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations == [Scale([2, 2], axes=("y", "x")), image_transformations["global"]]
    with pytest.raises(ValueError, match=r"Unknown region\(s\) \['S_8bit'\]"):
        phenocycler(tmp_path, labels={"S_8bit": tmp_path / "mask.tif"}, read_all_bit_depths=True)
    write_qptiff(tmp_path / "S_8bit.qptiff", biomarkers=["DAPI"])
    with pytest.raises(ValueError, match="S_8bit"):
        phenocycler(tmp_path, read_all_bit_depths=True)


def test_ignores_acquisition_files(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "images"
    acquisition = images / "phenocycler"
    acquisition.mkdir(parents=True)
    write_qptiff(images / "S_Scan1.QPTIFF", biomarkers=["DAPI", "CD3"])
    for name in ("phenocycler/S_Scan1_Cycle1.qptiff", "phenocycler/S.qptiff.raw", "S_Scan1.raw.qptiff"):
        write_qptiff(images / name, biomarkers=["DAPI"])

    assert list(phenocycler(tmp_path, image_models_kwargs=_FAST).images) == ["S_image"]


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
    (raw_images / "S_Scan1.xpd").write_text(json.dumps({"wells": wells}), encoding="utf-8-sig")
    (processed / "S_Scan1.qptiff.channels.csv").write_text("Channel,Marker\n1,DAPI\n2,CD3\n", encoding="utf-8-sig")

    # DAPI repeats per cycle and blanks do not count as channels, so the sidecars match the two pages
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    raw = sdata.attrs["akoya"]["regions"]["S"]["raw"]
    assert raw["xpd"] == {"wells": wells}
    assert raw["channels_csv"] == [{"Channel": "1", "Marker": "DAPI"}, {"Channel": "2", "Marker": "CD3"}]


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("S_Scan1.xpd", json.dumps({"wells": [{"items": [{"markerName": "CD4"}]}]}).encode(), "lists 1 distinct"),
        ("S_Scan1.qptiff.channels.csv", b"Marker\nDAPI\nCD3\nCD8\n", "channels.csv lists 3 distinct"),
        ("S_Scan1.xpd", b"not json", "not valid JSON"),
        ("S_Scan1.xpd", json.dumps({"wells": "x"}).encode(), "no readable markers"),
        ("S_Scan1.qptiff.channels.csv", "Marker\nDAPI\n".encode("utf-16"), "Ignoring"),
    ],
)
def test_bad_sidecars_warn(tmp_path: Path, name: str, content: bytes, match: str) -> None:
    data = write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    (tmp_path / name).write_bytes(content)

    with pytest.warns(UserWarning, match=match):
        sdata = phenocycler(tmp_path / "S_Scan1.qptiff", image_models_kwargs=_FAST)

    # the channel names always come from the qptiff
    assert get_channel_names(sdata.images["S_image"]) == ["DAPI", "CD3"]
    np.testing.assert_array_equal(_level0(sdata, "S_image").values, data)


def test_labels(tmp_path: Path) -> None:
    for name in ("Slide 1", "Slide 2"):
        write_qptiff(tmp_path / f"{name}.qptiff", biomarkers=["DAPI"])
    mask = _write_mask(tmp_path / "mask.tif")
    _write_mask(tmp_path / "small.tif", shape=(200, 300))

    # keys may be raw names; keys of unselected regions are ignored; other shapes warn
    with pytest.warns(UserWarning, match=r"\(200, 300\).*\(256, 320\).*not align"):
        sdata = phenocycler(
            tmp_path,
            regions=["Slide 1"],
            labels={"Slide 1": tmp_path / "small.tif", "Slide_2": tmp_path / "mask.tif"},
            image_models_kwargs=_FAST,
        )
    assert list(sdata.labels) == ["Slide_1_labels"]
    assert get_transformation(sdata["Slide_1_labels"], get_all=True) == {"global": Identity()}
    assert sdata.attrs["akoya"]["labels"] == {"Slide_1": str(tmp_path / "small.tif")}

    sdata = phenocycler(tmp_path / "Slide 1.qptiff", labels=tmp_path / "mask.tif", labels_models_kwargs=_FAST)
    np.testing.assert_array_equal(sdata.labels["Slide_1_labels"]["scale0"]["image"].values, mask)
    assert not sdata.tables


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ("mask.tif", "mapping"),
        ({"Z": "mask.tif"}, "'Z'"),
        ({"Slide 1": "mask.tif", "Slide_1": "mask.tif"}, "same region"),
    ],
)
def test_bad_labels_raise(tmp_path: Path, labels: str | dict[str, str], match: str) -> None:
    for name in ("Slide 1", "Slide 2"):
        write_qptiff(tmp_path / f"{name}.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")
    resolved = tmp_path / labels if isinstance(labels, str) else {k: tmp_path / v for k, v in labels.items()}

    with pytest.raises(ValueError, match=match):
        phenocycler(tmp_path, labels=resolved)


def test_cli(runner: CliRunner, tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_[1,2].qptiff", biomarkers=["DAPI", "HLA-A"])
    write_qptiff(tmp_path / "B.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mask(tmp_path / "mask.tif")
    args = ["--input", str(tmp_path), "--image-models-kwargs", json.dumps(_FAST)]

    # region ids and channel names may hold commas
    result = runner.invoke(
        phenocycler_wrapper,
        [
            *args,
            "-o",
            str(tmp_path / "a.zarr"),
            "--regions",
            "A_[1,2]",
            *("--channel-names", "DAPI", "--channel-names", "HLA-A,B,C"),
            *("--labels", json.dumps({"A_[1,2]": str(tmp_path / "mask.tif")})),
        ],
    )
    assert result.exit_code == 0, result.output
    sdata = read_zarr(tmp_path / "a.zarr")
    assert get_channel_names(sdata.images["A__1_2__image"]) == ["DAPI", "HLA-A,B,C"]
    assert list(sdata.labels) == ["A__1_2__labels"]

    for option, value, match in (
        ("--labels", '{"B": 1}', "labels"),
        ("--labels", "{not json", "labels"),
        ("--labels-models-kwargs", "null", "JSON object"),
    ):
        result = runner.invoke(phenocycler_wrapper, [*args, "-o", str(tmp_path / "b.zarr"), option, value])
        assert result.exit_code == 2
        assert match in result.output
