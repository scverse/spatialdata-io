import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dask.array as da
import pytest
from click.testing import CliRunner
from spatialdata import read_zarr
from spatialdata.models import get_channel_names

from spatialdata_io.__main__ import macsima_wrapper
from spatialdata_io.readers.macsima import (
    ChannelMetadata,
    MultiChannelImage,
    macsima,
    parse_name_to_cycle,
)
from tests._utils import skip_if_below_python_version

RNG = da.random.default_rng(seed=0)

if not (Path("./data/Lung_adc_demo").exists() or Path("./data/MACSimaData_HCA").exists()):
    pytest.skip(
        "Requires the Lung_adc_demo or MACSimaData_HCA datasets, please check "
        "https://github.com/giovp/spatialdata-sandbox/macsima/Readme.md for instructions on how to get the data.",
        allow_module_level=True,
    )


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", {"y": (0, 15460), "x": (0, 13864)}),
        ("MACSimaData_HCA/HumanLiverH35", {"y": (0, 1154), "x": (0, 1396)}),
    ],
)
def test_image_size(dataset: str, expected: dict[str, Any]) -> None:
    from spatialdata import get_extent

    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f)
    el = sdata[list(sdata.images.keys())[0]]
    cs = sdata.coordinate_systems[0]

    extent: dict[str, tuple[float, float]] = get_extent(el, coordinate_system=cs)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [("Lung_adc_demo", 116), ("MACSimaData_HCA/HumanLiverH35", 102)],
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
        ("Lung_adc_demo", ["R0 DAPI", "R1 CD68", "R1 CD163"]),
        ("MACSimaData_HCA/HumanLiverH35", ["R0 DAPI", "R1 PE", "R1 DAPI"]),
    ],
)
def test_channel_names(dataset: str, expected: list[str]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, c_subset=3, include_cycle_in_channel_name=True)
    el = sdata[list(sdata.images.keys())[0]]

    # get the channel names
    channels = get_channel_names(el)
    assert list(channels) == expected


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", 68),
        ("MACSimaData_HCA/HumanLiverH35", 51),
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
        ("Lung_adc_demo", list(range(2, 68)), ["DAPI (1)", "CD68", "CD163", "DAPI (2)", "Control"]),
        (
            "MACSimaData_HCA/HumanLiverH35",
            list(range(2, 51)),
            ["DAPI (1)", "PE", "CD14", "Vimentin", "DAPI (2)", "WT1"],
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


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Lung_adc_demo", [0, 1, 1]),
        ("MACSimaData_HCA/HumanLiverH35", [0, 1, 1]),
    ],
)
def test_cycle_metadata(dataset: str, expected: list[str]) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = macsima(f, c_subset=3)
    table = sdata[list(sdata.tables.keys())[0]]

    # get the channel names
    cycles = table.var["cycle"]
    assert list(cycles) == expected


def test_parsing_style() -> None:
    with pytest.raises(ValueError):
        macsima(Path(), parsing_style="not_a_parsing_style")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("C-002_S-000_S_FITC_R-01_W-C-1_ROI-01_A-CD147_C-REA282.tif", 2),
        ("001_S_R-01_W-B-1_ROI-01_A-CD14REA599ROI1_C-REA599.ome.tif", 1),
    ],
)
def test_parsing_of_name_to_cycle(name: str, expected: int) -> None:
    result = parse_name_to_cycle(name)
    assert result == expected


def test_mci_sort_by_channel() -> None:
    sizes = [100, 200, 300]
    c_names = ["test11", "test3", "test2"]
    cycles = [2, 0, 1]
    mci = MultiChannelImage(
        data=[RNG.random((size, size), chunks=(10, 10)) for size in sizes],
        metadata=[ChannelMetadata(name=c_name, cycle=cycle) for c_name, cycle in zip(c_names, cycles, strict=False)],
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
        metadata=[ChannelMetadata(name="test1", cycle=0), ChannelMetadata(name="test2", cycle=1)],
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
@pytest.mark.parametrize("dataset", ["Lung_adc_demo", "MACSimaData_HCA/HumanLiverH35"])
def test_cli_macimsa(runner: CliRunner, dataset: str) -> None:
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
