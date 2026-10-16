from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from spatialdata.transformations import Identity, Translation, get_transformation

from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    build_image_element,
    dedupe_channel_names,
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


def test_region_transformations_identity_without_positions():
    result = region_transformations([_scan("a"), _scan("b")])
    assert all(isinstance(t, Identity) for t in result.values())
    assert set(result) == {"a", "b"}


def test_region_transformations_translation_same_slide():
    result = region_transformations([_scan("a", stage=(0.0, 0.0)), _scan("b", stage=(100.0, 50.0))])
    assert result["a"] == Translation([0.0, 0.0], axes=("x", "y"))
    assert result["b"] == Translation([200.0, 100.0], axes=("x", "y"))


def test_region_transformations_identity_different_slides():
    result = region_transformations(
        [_scan("a", stage=(0.0, 0.0), slide_id="s1"), _scan("b", stage=(100.0, 50.0), slide_id="s2")]
    )
    assert all(isinstance(t, Identity) for t in result.values())


def test_region_transformations_identity_missing_pixel_size():
    result = region_transformations([_scan("a", stage=(0.0, 0.0), pixel_size_um=None), _scan("b", stage=(1.0, 1.0))])
    assert all(isinstance(t, Identity) for t in result.values())


def test_region_transformations_single_scan_identity():
    assert isinstance(region_transformations([_scan("a", stage=(5.0, 5.0))])["a"], Identity)


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


def test_select_regions_unknown_raises():
    with pytest.raises(ValueError, match="'c'.*Available.*a.*b"):
        select_regions(["a", "b"], ["c"])


def test_build_image_rgb_and_multiplex():
    scan = _scan()
    img = da.from_array(np.random.default_rng(0).integers(0, 255, (3, 16, 16), dtype=np.uint8), chunks=(1, 8, 8))
    element = build_image_element(img, scan, ["DAPI", "CD3", "CD8"], Identity(), _NO_PYRAMID)
    assert list(element.coords["c"].values) == ["DAPI", "CD3", "CD8"]
    assert isinstance(get_transformation(element, "global"), Identity)

    rgb_scan = _scan(rgb=True)
    rgb_img = da.from_array(np.zeros((16, 16, 3), dtype=np.uint8), chunks=(8, 8, 3))
    trans = Translation([4.0, 2.0], axes=("x", "y"))
    element = build_image_element(rgb_img, rgb_scan, ["r", "g", "b"], trans, _NO_PYRAMID)
    assert list(element.coords["c"].values) == ["r", "g", "b"]
    assert element.shape == (3, 16, 16)
    assert get_transformation(element, "global") == trans
