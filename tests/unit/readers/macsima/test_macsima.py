import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
from ome_types import OME
from ome_types.model import (
    Image,
    MapAnnotation,
    Pixels,
    Pixels_DimensionOrder,
    PixelType,
    Plane,
    Plate,
    Reagent,
    Screen,
    StructuredAnnotations,
    Well,
)
from tifffile import imwrite

from spatialdata_io.readers.macsima import (
    ChannelMetadata,
    MultiChannelImage,
    _collect_map_annotation_values,
    _get_software_major_version,
    _get_software_version,
    _get_translations,
    _parse_ome_metadata,
    _parse_v0_ome_metadata,
    _parse_v1_ome_metadata,
    macsima,
)

RNG = da.random.default_rng(seed=0)


# Helper to create ChannelMetadata with some defaults
def make_ChannelMetadata(
    name: str,
    cycle: int,
    fluorophore: str = "",
    exposure: float = 0.0,
    imagetype: str = "StainCycle",
    well: str = "A01",
    roi: int = 0,
    translation_x: int = 0,
    translation_y: int = 0,
) -> ChannelMetadata:
    """Helper to construct ChannelMetadata with required defaults."""
    return ChannelMetadata(
        name=name,
        cycle=cycle,
        fluorophore=fluorophore,
        exposure=exposure,
        imagetype=imagetype,
        well=well,
        translation_x=translation_x,
        translation_y=translation_y,
        roi=roi,
    )


def test_exception_on_no_valid_files(tmp_path: Path) -> None:
    # Write a tiff file without metadata
    height = 10
    width = 10
    arr = np.zeros((1, height, width), dtype=np.uint16)
    path_no_metadata = Path(tmp_path) / "tiff_no_metadata.tiff"
    imwrite(path_no_metadata, arr, metadata=None, description=None, software=None, datetime=None)

    with pytest.raises(ValueError, match="No valid files were found"):
        macsima(tmp_path)


@pytest.mark.parametrize(
    "dimensions,expected",
    [
        (((10, 10), (10, 10)), False),
        (((10, 10), (15, 10)), True),
        (((10, 10), (10, 15)), True),
        (((15, 10), (10, 15)), True),
    ],
)
def test_check_differing_dimensions_works(dimensions: tuple[tuple[int, int], tuple[int, int]], expected: bool) -> None:
    imgs = []
    for img_dim in dimensions:
        arr = da.from_array(np.ones((1, img_dim[0], img_dim[1]), dtype=np.uint16))
        imgs.append(arr)

    ctx = (
        pytest.warns(UserWarning, match="Supplied images have different dimensions!")
        if expected
        else contextlib.nullcontext()
    )
    with ctx:
        assert MultiChannelImage._check_for_differing_xy_dimensions(imgs) == expected


def test_padding_on_differing_dimensions() -> None:
    # Simple test where all translations are 0
    # Here we expect to pad to the largest element.
    heights = [10, 10, 15, 20]
    widths = [10, 15, 10, 20]

    imgs = []
    for height, width in zip(heights, widths, strict=True):
        arr = da.from_array(np.ones((1, height, width), dtype=np.uint16))
        imgs.append(arr)

    channel_metadata = [make_ChannelMetadata(name="test", cycle=1)] * 4
    with pytest.warns(UserWarning, match="Padding images with 0s to same size of \\(20, 20\\)"):
        imgs_padded = MultiChannelImage._pad_images(imgs, channel_metadata)
    for img in imgs_padded:
        assert img.shape == (1, 20, 20)

    # More complex with non-zero translations
    # First test that padding does the minimal padding necessary.
    # To do this create images with very large, but identical translations. Since all of these should be normalized out we expect size 20x20 again.
    heights = [10, 10, 15, 20]
    widths = [10, 15, 10, 20]

    imgs = []
    for height, width in zip(heights, widths, strict=True):
        arr = da.from_array(np.ones((1, height, width), dtype=np.uint16))
        imgs.append(arr)
    channel_metadata = [make_ChannelMetadata(name="test", cycle=1, translation_x=100, translation_y=100)] * 4
    with pytest.warns(UserWarning, match="Padding images with 0s to same size of \\(20, 20\\)"):
        imgs_padded = MultiChannelImage._pad_images(imgs, channel_metadata)
    for img in imgs_padded:
        assert img.shape == (1, 20, 20)

    # Test with differing translations but same size.
    # As we translate the first image by 2 in x and 3 in y, we expect a 13x12 image
    heights = [10, 10]
    widths = [10, 10]

    imgs = []
    for height, width in zip(heights, widths, strict=True):
        arr = da.from_array(np.ones((1, height, width), dtype=np.uint16))
        imgs.append(arr)
    channel_metadata = [
        make_ChannelMetadata(name="test", cycle=1, translation_x=2, translation_y=3),
        make_ChannelMetadata(name="test", cycle=1, translation_x=0, translation_y=0),
    ]
    with pytest.warns(UserWarning, match="Padding images with 0s to same size of \\(13, 12\\)"):
        imgs_padded = MultiChannelImage._pad_images(imgs, channel_metadata)
    for img in imgs_padded:
        assert img.shape == (1, 13, 12)

    # Final test with differing image sizes, and translations that need to be normalized
    # For the total size, we need to check the sum of each image dimension + normalized translation
    # Here that would be image 2, with y = 15 + 5 - 3 = 17 (normalized to other image!) and x = 15 + 5 - 2 = 18

    heights = [10, 15]
    widths = [10, 15]

    imgs = []
    for height, width in zip(heights, widths, strict=True):
        arr = da.from_array(np.ones((1, height, width), dtype=np.uint16))
        imgs.append(arr)
    channel_metadata = [
        make_ChannelMetadata(name="test", cycle=1, translation_x=2, translation_y=3),
        make_ChannelMetadata(name="test", cycle=1, translation_x=5, translation_y=5),
    ]
    with pytest.warns(UserWarning, match="Padding images with 0s to same size of \\(17, 18\\)"):
        imgs_padded = MultiChannelImage._pad_images(imgs, channel_metadata)
    for img in imgs_padded:
        assert img.shape == (1, 17, 18)


def test_unsupported_parsing_styles() -> None:
    with pytest.raises(ValueError, match="Invalid option `not_a_parsing_style` for `MACSimaParsingStyle`."):
        macsima(Path(), parsing_style="not_a_parsing_style")


def test_mci_sort_by_channel() -> None:
    sizes = [100, 200, 300]
    c_names = ["test11", "test3", "test2"]
    cycles = [2, 0, 1]
    mci = MultiChannelImage(
        data=[RNG.random((size, size), chunks=(10, 10)) for size in sizes],
        metadata=[
            make_ChannelMetadata(name=c_name, cycle=cycle) for c_name, cycle in zip(c_names, cycles, strict=False)
        ],
    )
    assert mci.get_channel_names() == c_names
    assert [x.shape[0] for x in mci.data] == sizes
    mci.sort_by_channel()
    assert mci.get_channel_names() == ["test3", "test2", "test11"]
    assert [x.shape[0] for x in mci.data] == [200, 300, 100]


def test_mci_array_reference() -> None:
    arr1 = RNG.random((100, 100), chunks=(10, 10))
    arr2 = RNG.random((200, 200), chunks=(10, 10))
    mci = MultiChannelImage(
        data=[arr1, arr2],
        metadata=[
            make_ChannelMetadata(name="test1", cycle=0),
            make_ChannelMetadata(name="test2", cycle=1),
        ],
    )
    orig_arr1 = arr1.copy()

    # test we can subset by index and by name
    subset_mci = MultiChannelImage.subset_by_index(mci, [0])
    assert subset_mci.get_channel_names() == ["test1"]

    subset_mci_name = MultiChannelImage.subset_by_channel(mci, "test")
    assert subset_mci_name.get_channel_names() == ["test1", "test2"]

    # test that the subset is a view
    assert subset_mci.data[0] is arr1
    assert da.all(subset_mci.data[0] == orig_arr1)
    # test that a deepcopy is not a view
    deepcopy_mci: MultiChannelImage = deepcopy(mci)
    deepcopy_mci.data[0][0, 0] = deepcopy_mci.data[0][0, 0] + 1
    assert deepcopy_mci.data[0] is not arr1
    assert not da.all(deepcopy_mci.data[0] == orig_arr1)
    # test that the original mci is not changed
    assert da.all(mci.data[0] == orig_arr1)


def test_collect_map_annotation_values_with_no_duplicate_keys() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(value={"a": "1", "b": "2"}),
                MapAnnotation(value={"c": "3"}),
            ]
        )
    )

    result = _collect_map_annotation_values(ome)

    assert result == {"a": "1", "b": "2", "c": "3"}


def test_collect_map_annotations_values_with_duplicate_keys_identical_values() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(value={"a": "1", "b": "2"}),
                MapAnnotation(value={"b": "2", "c": "3"}),
            ]
        )
    )

    result = _collect_map_annotation_values(ome)
    # Key should only be returned once
    assert result == {"a": "1", "b": "2", "c": "3"}


def test_collect_map_annotations_values_with_duplicate_keys_different_values() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(value={"a": "1", "b": "2"}),
                MapAnnotation(value={"b": "99", "c": "3"}),
            ]
        )
    )
    result = _collect_map_annotation_values(ome)

    # The parser should return only the first found value.
    assert result == {"a": "1", "b": "2", "c": "3"}


def test_collect_map_annotation_values_handles_missing_sa_and_empty_list() -> None:
    # No structured_annotations at all
    ome1 = OME()
    assert _collect_map_annotation_values(ome1) == {}

    # structured_annotations present but with empty map_annotation list
    ome2 = OME(structured_annotations=StructuredAnnotations(map_annotations=[]))
    assert _collect_map_annotation_values(ome2) == {}


@pytest.mark.parametrize(
    "ma_values, expected",
    [
        ({"SoftwareVersion": " 1.2.3 "}, "1.2.3"),
        ({"Software version": " v0.9.0"}, "v0.9.0"),
    ],
)
def test_get_software_version_success(ma_values: dict[str, str], expected: str) -> None:
    assert _get_software_version(ma_values) == expected


@pytest.mark.parametrize(
    "ma_values",
    [
        ({}),
        ({"SoftwareVersion": ""}),
        ({"SoftwareVersion": "    "}),
        ({"Software version": ""}),
        ({"Software version": None}),
    ],
)
def test_get_software_version_failure(ma_values: dict[str, str | None]) -> None:
    with pytest.raises(ValueError, match="Could not extract Software Version"):
        _get_software_version(ma_values)


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.2.3", 1),
        ("  2.0.0 ", 2),
        ("v3.4.5", 3),
        ("V4.0.1", 4),
        ("10", 10),
    ],
)
def test_get_software_major_version_success(version: str, expected: int) -> None:
    assert _get_software_major_version(version) == expected


def test_get_software_major_version_failure() -> None:
    with pytest.raises(ValueError):
        _get_software_major_version("")


def test_parse_v0_ome_metadata_basic_extraction_and_conversions() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "Fluorochrome": "AF488",
                        "Exposure time": "123.4",
                        "Cycle": "5",
                        "ROI ID": "7",
                        "MICS cycle type": "AntigenCycle",
                    }
                )
            ]
        ),
        screens=[Screen(reagents=[Reagent(name="CD3__OKT3")])],
        plates=[Plate(wells=[Well(column=0, row=0, external_identifier="A01")])],
    )

    md = _parse_v0_ome_metadata(ome)

    assert md["name"] == "CD3"
    assert md["clone"] == "OKT3"
    assert md["fluorophore"] == "AF488"
    assert md["exposure"] == pytest.approx(123.4)
    assert md["cycle"] == 5
    assert md["roi"] == 7
    assert md["imagetype"] == "stain"  # harmonized!
    assert md["well"] == "A01"


def test_parse_v0_ome_metadata_handles_missing_or_invalid_numeric_fields() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "Exposure time": "not-a-number",
                        "Cycle": "NaN",
                        "ROI ID": "x",
                    }
                )
            ]
        ),
        screens=[Screen(reagents=[Reagent(name="MarkerOnly")])],
        plates=[Plate(wells=[Well(column=0, row=0, external_identifier=None)])],
    )

    md = _parse_v0_ome_metadata(ome)

    # name from reagent without "__"
    assert md["name"] == "MarkerOnly"
    assert md["clone"] is None
    assert md["exposure"] is None
    assert md["cycle"] is None
    assert md["roi"] is None
    # well remains None
    assert md["well"] is None


def test_parse_v0_ome_metadata_falls_back_to_MICScycleID_if_no_cycle_keyword() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "MICS cycle ID": "5",
                    }
                )
            ]
        ),
    )

    md = _parse_v0_ome_metadata(ome)
    assert md["cycle"] == 5


def test_parse_v0_ome_metadata_prefers_Cycle_over_MICScycleID_keyword() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[MapAnnotation(value={"MICS cycle ID": "5", "Cycle": "1"})]
        ),
    )

    md = _parse_v0_ome_metadata(ome)
    assert md["cycle"] == 1


def test_parse_v0_ome_metadata_bleach_cycle_appends_background() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "MICS cycle type": "BleachCycle",
                    }
                )
            ]
        ),
        screens=[Screen(reagents=[Reagent(name="CD4__RPA-T4")])],
    )

    md = _parse_v0_ome_metadata(ome)

    assert md["imagetype"] == "bleach"  # harmonized!
    assert md["name"] == "CD4_background"


def test_parse_v0_ome_metadata_handles_unknown_imagetypes() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "MICS cycle type": "NOT A VALID TYPE",
                    }
                )
            ]
        ),
        screens=[Screen(reagents=[Reagent(name="CD4__RPA-T4")])],
    )

    md = _parse_v0_ome_metadata(ome)

    # Unknown types should just be passed through
    assert md["imagetype"] == "NOT A VALID TYPE"


def test_parse_v1_ome_metadata_basic_extraction_and_conversions() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "Clone": "OKT3",
                        "Biomarker": "CD3",
                        "Fluorochrome": "AF488",
                        "ExposureTime": "45.6",
                        "Cycle": "3",
                        "RoiId": "10",
                        "ScanType": "S",
                    }
                )
            ]
        ),
        plates=[Plate(wells=[Well(column=0, row=0, external_identifier="B02")])],
    )

    md = _parse_v1_ome_metadata(ome)

    assert md["name"] == "CD3"
    assert md["clone"] == "OKT3"
    assert md["fluorophore"] == "AF488"
    assert md["exposure"] == pytest.approx(45.6)
    assert md["cycle"] == 3
    assert md["roi"] == 10
    assert md["imagetype"] == "stain"  # harmonized!
    assert md["well"] == "B02"


def test_parse_v1_ome_metadata_invalid_numerics_become_none() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "ExposureTime": "x",
                        "Cycle": "NaN",
                        "RoiId": "ABC",
                    }
                )
            ]
        ),
    )

    md = _parse_v1_ome_metadata(ome)

    assert md["exposure"] is None
    assert md["cycle"] is None
    assert md["roi"] is None


def test_parse_v1_ome_metadata_handles_unknown_imagetypes() -> None:
    ome = OME(
        structured_annotations=StructuredAnnotations(
            map_annotations=[
                MapAnnotation(
                    value={
                        "ScanType": "NOT A VALID TYPE",
                    }
                )
            ]
        ),
        screens=[Screen(reagents=[Reagent(name="CD4__RPA-T4")])],
    )

    md = _parse_v1_ome_metadata(ome)

    # Unknown types should just be passed through
    assert md["imagetype"] == "NOT A VALID TYPE"


def test_get_translations_returns_correct_values() -> None:
    ome = OME(
        images=[
            Image(
                pixels=Pixels(
                    dimension_order=Pixels_DimensionOrder("XYZCT"),
                    type=PixelType.UINT16,
                    size_x=1,
                    size_y=1,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    planes=[Plane(position_x=1, position_y=2, the_z=0, the_t=0, the_c=0)],
                )
            )
        ]
    )
    expected = {"translation_x": 1, "translation_y": 2}

    translations = _get_translations(ome)
    assert translations == expected


def test_get_translations_defaults_to_0_on_missing_data() -> None:
    ome = OME(
        images=[
            Image(
                pixels=Pixels(
                    dimension_order=Pixels_DimensionOrder("XYZCT"),
                    type=PixelType.UINT16,
                    size_x=1,
                    size_y=1,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    planes=[Plane(the_z=0, the_t=0, the_c=0)],
                )
            )
        ],
    )
    expected = {"translation_x": 0, "translation_y": 0}

    translations = _get_translations(ome)
    assert translations == expected


def make_ome(extra_ma: dict[str, Any] | None = None) -> OME:
    base = {}
    if extra_ma:
        base.update(extra_ma)
    return OME(
        images=[
            Image(
                pixels=Pixels(
                    dimension_order=Pixels_DimensionOrder("XYZCT"),
                    type=PixelType.UINT16,
                    size_x=1,
                    size_y=1,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    planes=[Plane(the_z=0, the_t=0, the_c=0)],
                )
            )
        ],
        structured_annotations=StructuredAnnotations(map_annotations=[MapAnnotation(value=base)]),
    )


def test_parse_ome_metadata_dispatches_to_v0() -> None:
    ome = make_ome(extra_ma={"SoftwareVersion": "0.9.0"})
    # enrich some so v0 parser has something to see
    ome.screens = [Screen(reagents=[Reagent(name="Marker0")])]

    md = _parse_ome_metadata(ome)

    # Assert that the v0 parser was used by checking a field
    # that can only come from v0 parsing logic (e.g. name from reagent)
    assert "name" in md
    assert md["name"] == "Marker0"


def test_parse_ome_metadata_dispatches_to_v1() -> None:
    ome = make_ome(extra_ma={"SoftwareVersion": "1.0.0", "Biomarker": "CD3"})
    md = _parse_ome_metadata(ome)

    assert md["name"] == "CD3"


def test_parse_ome_metadata_unknown_major_raises() -> None:
    ome = make_ome(extra_ma={"SoftwareVersion": "2.0.0"})

    with pytest.raises(ValueError, match="Unknown software version"):
        _parse_ome_metadata(ome)
