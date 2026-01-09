import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dask.array as da
import pandas as pd
import pytest
from click.testing import CliRunner
from ome_types import OME
from ome_types.model import (
    MapAnnotation,
    Plate,
    Reagent,
    Screen,
    StructuredAnnotations,
    Well,
)
from spatialdata import read_zarr
from spatialdata.models import get_channel_names

from spatialdata_io.__main__ import macsima_wrapper
from spatialdata_io.readers.macsima import (
    ChannelMetadata,
    MultiChannelImage,
    _collect_map_annotation_values,
    _get_software_major_version,
    _get_software_version,
    _parse_ome_metadata,
    _parse_v0_ome_metadata,
    _parse_v1_ome_metadata,
    macsima,
)
from tests._utils import skip_if_below_python_version

RNG = da.random.default_rng(seed=0)

if not (Path("./data/OMAP10_small").exists() or Path("./data/OMAP23_small").exists()):
    pytest.skip(
        "Requires the OMAP10 or OMAP23 datasets. "
        "The small OMAP10 dataset can be downloaded from TBD, for the full data see https://zenodo.org/records/7875938 "
        "The small OMAP23 dataset can be downloaded from TBD, for the full data set see https://zenodo.org/records/14008816",
        allow_module_level=True,
    )


# Helper to create ChannelMetadata with some defaults
def make_ChannelMetadata(
    name: str,
    cycle: int,
    fluorophore: str | None = None,
    exposure: float | None = None,
    imagetype: str | None = None,
    well: str | None = None,
    roi: int | None = None,
) -> ChannelMetadata:
    """Helper to construct ChannelMetadata with required defaults."""
    return ChannelMetadata(
        name=name,
        cycle=cycle,
        fluorophore=fluorophore or "",
        exposure=exposure if exposure is not None else 0.0,
        imagetype=imagetype or "StainCycle",
        well=well or "A01",
        roi=roi if roi is not None else 0,
    )


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("OMAP10_small", {"y": (0, 77), "x": (0, 94)}),
        ("OMAP23_small", {"y": (0, 77), "x": (0, 93)}),
    ],
)
def test_image_size(dataset: str, expected: dict[str, Any]) -> None:
    from spatialdata import get_extent

    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, transformations=False)  # Do not transform to make it easier to compare against pixel dimensions
    el = sdata[list(sdata.images.keys())[0]]
    cs = sdata.coordinate_systems[0]

    extent: dict[str, tuple[float, float]] = get_extent(el, coordinate_system=cs)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [("OMAP10_small", 4), ("OMAP23_small", 5)],
)
def test_total_channels(dataset: str, expected: int) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)
    el = sdata[list(sdata.images.keys())[0]]

    # get the number of channels
    channels: int = len(get_channel_names(el))
    assert channels == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("OMAP10_small", ["R1 DAPI", "R1 CD15", "R2 Bcl 2", "R2 CD1c"]),
        ("OMAP23_small", ["R1 DAPI", "R1 CD3", "R2 CD279", "R4 CD66b", "R15 DAPI_background"]),
    ],
)
def test_channel_names_with_cycle_in_name(dataset: str, expected: list[str]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, include_cycle_in_channel_name=True)
    el = sdata[list(sdata.images.keys())[0]]

    # get the channel names
    channels = get_channel_names(el)
    assert list(channels) == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("OMAP10_small", 2),
        ("OMAP23_small", 15),
    ],
)
def test_total_rounds(dataset: str, expected: list[int]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)
    table = sdata[list(sdata.tables)[0]]
    max_cycle = table.var["cycle"].max()
    assert max_cycle == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,skip_rounds,expected",
    [
        ("OMAP10_small", list(range(2, 4)), ["DAPI", "CD15"]),
        (
            "OMAP23_small",
            list(range(2, 16)),
            ["DAPI", "CD3"],
        ),
    ],
)
def test_skip_rounds(dataset: str, skip_rounds: list[int], expected: list[str]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, skip_rounds=skip_rounds)
    el = sdata[list(sdata.images.keys())[0]]

    # get the channel names
    channels = get_channel_names(el)
    assert list(channels) == expected, f"Expected {expected}, got {list(channels)}"


METADATA_COLUMN_ORDER = [
    "cycle",
    "imagetype",
    "well",
    "ROI",
    "fluorophore",
    "clone",
    "exposure",
]

EXPECTED_METADATA_OMAP10 = pd.DataFrame(
    {
        "name": ["DAPI", "CD15", "Bcl 2", "CD1c"],
        "cycle": [1, 1, 2, 2],
        "imagetype": ["stain", "stain", "stain", "stain"],
        "well": ["C-1", "C-1", "C-1", "C-1"],
        "ROI": [1, 1, 1, 1],
        "fluorophore": ["DAPI", "APC", "FITC", "PE"],
        "clone": [pd.NA, "VIMC6", "REA872", "REA694"],
        "exposure": [40.0, 2304.0, 96.0, 144.0],
    },
    index=["DAPI", "CD15", "Bcl 2", "CD1c"],
    columns=METADATA_COLUMN_ORDER,
)


EXPECTED_METADATA_OMAP23 = pd.DataFrame(
    {
        "name": ["DAPI", "CD3", "CD279", "CD66b", "DAPI_background"],
        "cycle": [1, 1, 2, 4, 15],
        "imagetype": ["stain", "stain", "stain", "stain", "bleach"],
        "well": ["D01", "D01", "D01", "D01", "D01"],
        "ROI": [1, 1, 1, 1, 1],
        "fluorophore": ["DAPI", "APC", "PE", "FITC", "DAPI"],
        "clone": [pd.NA, "REA1151", "REA1165", "REA306", pd.NA],
        "exposure": [51.0, 1212.52, 322.12, 856.68, 51.0],
    },
    index=["DAPI", "CD3", "CD279", "CD66b", "DAPI_background"],
    columns=METADATA_COLUMN_ORDER,
)


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected_df",
    [
        ("OMAP10_small", EXPECTED_METADATA_OMAP10),
        ("OMAP23_small", EXPECTED_METADATA_OMAP23),
    ],
)
def test_metadata_table(dataset: str, expected_df: pd.DataFrame) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)
    table = sdata[list(sdata.tables.keys())[0]]

    # Convert table.var to a DataFrame and align to expected columns
    actual = table.var[METADATA_COLUMN_ORDER]

    pd.testing.assert_frame_equal(actual, expected_df)


def test_parsing_style() -> None:
    with pytest.raises(ValueError):
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
        metadata=[make_ChannelMetadata(name="test1", cycle=0), make_ChannelMetadata(name="test2", cycle=1)],
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


@skip_if_below_python_version()
@pytest.mark.parametrize("dataset", ["OMAP10_small", "OMAP23_small"])
def test_cli_macsima(runner: CliRunner, dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            macsima_wrapper,
            [
                "--input",
                f,
                "--output",
                output_zarr,
                "--subset",
                500,
                "--c-subset",
                1,
                "--multiscale",
                False,
            ],
        )
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)


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
    import re

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


def make_ome_with_version(version_value: str, extra_ma: dict[str, Any] | None = None) -> OME:
    base = {"SoftwareVersion": version_value}
    if extra_ma:
        base.update(extra_ma)
    return OME(structured_annotations=StructuredAnnotations(map_annotations=[MapAnnotation(value=base)]))


def test_parse_ome_metadata_dispatches_to_v0() -> None:
    ome = make_ome_with_version("0.9.0")
    # enrich some so v0 parser has something to see
    ome.screens = [Screen(reagents=[Reagent(name="Marker0")])]

    md = _parse_ome_metadata(ome)

    # Assert that the v0 parser was used by checking a field
    # that can only come from v0 parsing logic (e.g. name from reagent)
    assert "name" in md
    assert md["name"] == "Marker0"


def test_parse_ome_metadata_dispatches_to_v1() -> None:
    ome = make_ome_with_version("1.0.0", extra_ma={"Biomarker": "CD3"})

    md = _parse_ome_metadata(ome)

    assert md["name"] == "CD3"


def test_parse_ome_metadata_unknown_major_raises() -> None:
    ome = make_ome_with_version("2.0.0")

    with pytest.raises(ValueError, match="Unknown software version"):
        _parse_ome_metadata(ome)
