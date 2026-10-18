from __future__ import annotations

import json
import pickle
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
from spatialdata.transformations import Identity, Translation, get_transformation

from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    build_image_element,
    dedupe_channel_names,
    read_tiff,
    region_transformations,
    scans_to_attrs,
    select_regions,
)
from spatialdata_io.readers._utils._qptiff import AkoyaChannel, AkoyaScan

_NO_PYRAMID = {"scale_factors": None, "chunks": (1, 8, 8)}


def _scan(
    region: str = "reg1",
    names: tuple[str, ...] = ("DAPI", "CD3", "CD8"),
    stage: tuple[float, float] | None = None,
    pixel_size_um: float | None = 0.5,
    slide_id: str | None = "slide",
    rgb: bool = False,
) -> AkoyaScan:
    return AkoyaScan(
        path=Path(f"{region}.qptiff"),
        region=region,
        generation="fusion2",
        software="Fusion",
        slide_id=slide_id,
        pixel_size_um=pixel_size_um,
        nominal_pixel_size_um=0.5,
        stage_position_um=stage,
        channels=[AkoyaChannel(name=n, cycle=1, extra={"k": [1, 2]}) for n in names],
        rgb=rgb,
        raw={"page0_xml": "<x/>", "scan_profile": None},
    )


def test_dedupe_channel_names():
    assert dedupe_channel_names(["DAPI", "CD3", "DAPI", "DAPI"]) == ["DAPI", "CD3", "DAPI (2)", "DAPI (3)"]


def test_apply_channel_names_override_and_default():
    scan = _scan(names=("DAPI", "DAPI", "CD3"))
    assert apply_channel_names(scan, None) == ["DAPI", "DAPI (2)", "CD3"]
    assert apply_channel_names(scan, ["a", "b", "c"]) == ["a", "b", "c"]


def test_apply_channel_names_length_mismatch():
    with pytest.raises(ValueError, match="expected 3"):
        apply_channel_names(_scan(), ["a", "b"])


def test_region_transformations_single_is_global():
    assert region_transformations([_scan("a", stage=(5.0, 5.0))]) == {"a": {"global": Identity()}}


def test_region_transformations_shared_slide_translation():
    result = region_transformations([_scan("a", stage=(0.0, 0.0)), _scan("b", stage=(100.0, 50.0))])
    assert result == {
        "a": {"global": Translation([0.0, 0.0], axes=("x", "y"))},
        "b": {"global": Translation([200.0, 100.0], axes=("x", "y"))},
    }


def test_region_transformations_unrelated_get_own_cs():
    result = region_transformations(
        [_scan("A", stage=(0.0, 0.0), slide_id="s1"), _scan("B", stage=(100.0, 50.0), slide_id="s2")]
    )
    assert result == {"A": {"A": Identity()}, "B": {"B": Identity()}}


def test_region_transformations_mixed_pixel_sizes_own_cs():
    result = region_transformations([_scan("a", stage=(0.0, 0.0)), _scan("b", stage=(1.0, 1.0), pixel_size_um=0.25)])
    assert result == {"a": {"a": Identity()}, "b": {"b": Identity()}}
    # pixel sizes equal up to rounding share the sample space
    result = region_transformations([_scan("a", stage=(0.0, 0.0)), _scan("b", stage=(1.0, 1.0), pixel_size_um=0.5004)])
    assert set(result["b"]) == {"global"}


@pytest.mark.parametrize(
    "scans",
    [
        [_scan("a"), _scan("b")],
        [_scan("a", stage=(0.0, 0.0), pixel_size_um=None), _scan("b", stage=(1.0, 1.0))],
        [_scan("a", stage=(0.0, 0.0), slide_id=None), _scan("b", stage=(1.0, 1.0), slide_id=None)],
    ],
    ids=["no_stage_positions", "missing_pixel_size", "no_slide_id"],
)
def test_region_transformations_unplaceable_get_own_cs(scans: list[AkoyaScan]):
    assert region_transformations(scans) == {"a": {"a": Identity()}, "b": {"b": Identity()}}


def test_region_transformations_regions_without_image():
    # a region read without an image (e.g. only its cells) is unrelated to the scans
    shared = [_scan("a", stage=(0.0, 0.0)), _scan("b", stage=(100.0, 50.0))]
    result = region_transformations(shared, ["c", "a"])
    assert set(result["a"]) == set(result["b"]) == {"global"}
    assert result["c"] == {"c": Identity()}
    assert region_transformations([_scan("a")], ["c"]) == {"a": {"a": Identity()}, "c": {"c": Identity()}}
    assert region_transformations([], ["c"]) == {"c": {"global": Identity()}}


def test_scans_to_attrs_json_serialisable():
    scans = [_scan("a", stage=(1.0, 2.0)), _scan("b")]
    names = {"a": ["DAPI", "CD3", "CD8"], "b": ["x", "y", "z"]}
    attrs = scans_to_attrs(scans, names)
    json.dumps(attrs)
    entry = attrs["regions"]["a"]
    assert entry["stage_position_um"] == [1.0, 2.0]
    assert entry["path"] == "a.qptiff"
    assert [c["name_deduped"] for c in attrs["regions"]["b"]["channels"]] == ["x", "y", "z"]
    assert entry["channels"][0]["name"] == "DAPI"


def test_select_regions():
    assert select_regions(["a", "b"], None) == ["a", "b"]
    assert select_regions(["a", "b"], ["b"]) == ["b"]


def test_select_regions_deduplicates():
    assert select_regions(["a", "b"], ["b", "a", "b"]) == ["b", "a"]
    assert select_regions(["a", "b"], "b") == ["b"]


@pytest.mark.parametrize(
    ("regions", "error", "match"),
    [
        ({}, TypeError, "sequence of region ids"),
        ({"a": 1}, TypeError, "sequence of region ids"),
        ([], ValueError, "empty"),
    ],
)
def test_select_regions_invalid_raises(regions: object, error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        select_regions(["a", "b"], regions)  # type: ignore[arg-type]


def test_select_regions_unknown_raises():
    with pytest.raises(ValueError, match="'c'.*Available.*a.*b"):
        select_regions(["a", "b"], ["c"])


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_tiff_is_picklable(tmp_path: Path, compression: str | None) -> None:
    # compressed tiles are read through tifffile's zarr store, uncompressed data is memory-mapped
    path = tmp_path / "x.tif"
    data = np.arange(2 * 64 * 96, dtype=np.uint16).reshape(2, 64, 96)
    tifffile.imwrite(path, data, compression=compression, tile=(32, 32) if compression else None)

    image, _, _, read_by_tifffile = read_tiff(path, {}, (16, 16))

    assert read_by_tifffile
    restored = pickle.loads(pickle.dumps(image))
    np.testing.assert_array_equal(restored[0, :8, :8].compute(), data[0, :8, :8])
    np.testing.assert_array_equal(restored.compute(), data)


def test_build_image_rgb_and_multiplex():
    scan = _scan()
    img = da.from_array(np.random.default_rng(0).integers(0, 255, (3, 16, 16), dtype=np.uint8), chunks=(1, 8, 8))
    element = build_image_element(img, scan, ["DAPI", "CD3", "CD8"], {"global": Identity()}, _NO_PYRAMID)
    assert list(element.coords["c"].values) == ["DAPI", "CD3", "CD8"]
    assert isinstance(get_transformation(element, "global"), Identity)

    rgb_scan = _scan(rgb=True)
    rgb_img = da.from_array(np.zeros((16, 16, 3), dtype=np.uint8), chunks=(8, 8, 3))
    trans = Translation([4.0, 2.0], axes=("x", "y"))
    element = build_image_element(rgb_img, rgb_scan, ["r", "g", "b"], {"reg1": trans}, _NO_PYRAMID)
    assert list(element.coords["c"].values) == ["r", "g", "b"]
    assert element.shape == (3, 16, 16)
    assert get_transformation(element, get_all=True) == {"reg1": trans}


@pytest.mark.parametrize("imagej", [False, True])
@pytest.mark.parametrize(
    "data", [np.array([[[1, 8], [15, 0]]], dtype=np.uint16), np.array([[[1, 2], [300, 0]]], np.uint16)]
)
def test_read_tiff_big_endian(tmp_path: Path, imagej: bool, data: np.ndarray) -> None:
    # ImageJ/Fiji write big-endian tiffs by default; the memory-mapped path must use the file's byte order
    path = tmp_path / "x.tif"
    tifffile.imwrite(path, data, byteorder=">", imagej=imagej)

    image, _, _, _ = read_tiff(path, {}, (16, 16))

    assert image.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(image.compute(), data)
