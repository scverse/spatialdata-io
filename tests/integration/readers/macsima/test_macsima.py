import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from tifffile import imwrite

from spatialdata_io.__main__ import macsima_wrapper
from spatialdata_io.readers.macsima import macsima


def test_images_with_invalid_ome_metadata_are_excluded(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    # Write a tiff file without metadata
    # Use same dimensions as OMAP10_small, which we will use as a positive example
    height = 77
    width = 94
    arr = np.zeros((height, width, 1), dtype=np.uint16)
    path_no_metadata = Path(tmp_path) / "tiff_no_metadata.tiff"
    imwrite(path_no_metadata, arr, metadata=None, description=None, software=None, datetime=None)

    # Copy 1 image from OMAP10 small
    omap_10_image_path = (
        require_test_dataset("macsima_omap10") / "C-001_S-000_S_APC_R-01_W-C-1_ROI-01_A-CD15_C-VIMC6.tif"
    )
    shutil.copy(omap_10_image_path, Path(tmp_path))

    sdata = macsima(tmp_path)
    el = sdata[list(sdata.images.keys())[0]]
    channels = get_channel_names(el)
    assert channels == ["CD15"]


def test_multiple_subfolder_parsing_skips_emtpy_folders(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    parent_folder = tmp_path / "test_folder"
    shutil.copytree(require_test_dataset("macsima_omap23"), parent_folder / "OMAP23_small")
    os.makedirs(parent_folder / "empty_folder")

    with pytest.warns(UserWarning, match="No tif files found in .* skipping it"):
        sdata = macsima(parent_folder, parsing_style="processed_multiple_folders")
    assert len(sdata.images.keys()) == 1


@pytest.mark.parametrize(
    "dataset,expected",
    [
        pytest.param("macsima_omap10", {"y": (0, 77), "x": (0, 94)}, id="macsima_omap10"),
        pytest.param("macsima_omap23", {"y": (0, 77), "x": (0, 93)}, id="macsima_omap23"),
    ],
)
def test_image_size(dataset: str, expected: dict[str, Any], require_test_dataset: Callable[[str], Path]) -> None:
    from spatialdata import get_extent

    f = require_test_dataset(dataset)
    sdata = macsima(f, transformations=False)  # Do not transform to make it easier to compare against pixel dimensions
    el = sdata[list(sdata.images.keys())[0]]
    cs = sdata.coordinate_systems[0]

    extent: dict[str, tuple[float, float]] = get_extent(el, coordinate_system=cs)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == expected


@pytest.mark.parametrize(
    "dataset,expected",
    [
        pytest.param("macsima_omap10", 4, id="macsima_omap10"),
        pytest.param("macsima_omap23", 5, id="macsima_omap23"),
    ],
)
def test_total_channels(dataset: str, expected: int, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = macsima(f)
    el = sdata[list(sdata.images.keys())[0]]

    # get the number of channels
    channels: int = len(get_channel_names(el))
    assert channels == expected


@pytest.mark.parametrize(
    "dataset,expected",
    [
        pytest.param("macsima_omap10", ["R1 CD15", "R1 DAPI", "R2 Bcl 2", "R2 CD1c"], id="macsima_omap10"),
        pytest.param(
            "macsima_omap23",
            ["R1 CD3", "R1 DAPI", "R2 CD279", "R4 CD66b", "R15 DAPI_background"],
            id="macsima_omap23",
        ),
    ],
)
def test_channel_names_with_cycle_in_name(
    dataset: str, expected: list[str], require_test_dataset: Callable[[str], Path]
) -> None:
    f = require_test_dataset(dataset)
    sdata = macsima(f, include_cycle_in_channel_name=True)
    el = sdata[list(sdata.images.keys())[0]]

    # get the channel names
    channels = get_channel_names(el)
    assert list(channels) == expected


@pytest.mark.parametrize(
    "dataset,expected",
    [
        pytest.param("macsima_omap10", 2, id="macsima_omap10"),
        pytest.param("macsima_omap23", 15, id="macsima_omap23"),
    ],
)
def test_total_rounds(dataset: str, expected: list[int], require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = macsima(f)
    table = sdata[list(sdata.tables)[0]]
    max_cycle = table.var["cycle"].max()
    assert max_cycle == expected


@pytest.mark.parametrize(
    "dataset,skip_rounds,expected",
    [
        pytest.param("macsima_omap10", list(range(2, 4)), ["CD15", "DAPI"], id="macsima_omap10"),
        pytest.param(
            "macsima_omap23",
            list(range(2, 16)),
            ["CD3", "DAPI"],
            id="macsima_omap23",
        ),
    ],
)
def test_skip_rounds(
    dataset: str, skip_rounds: list[int], expected: list[str], require_test_dataset: Callable[[str], Path]
) -> None:
    f = require_test_dataset(dataset)
    sdata = macsima(f, skip_rounds=skip_rounds)
    el = sdata[list(sdata.images.keys())[0]]

    # get the channel names
    channels = get_channel_names(el)
    assert list(channels) == expected, f"Expected {expected}, got {list(channels)}"


def test_processed_single_folder_parsing_returns_a_single_image_stack(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    omap10_path = require_test_dataset("macsima_omap10")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_1")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_2")

    sdata = macsima(tmp_path, parsing_style="processed_single_folder")

    assert len(sdata.images) == 1
    # omap10_small has 4 channels, so we expect 8 here
    el = sdata[list(sdata.images.keys())[0]]
    assert len(get_channel_names(el)) == 8
    assert len(sdata.tables) == 1


def test_processed_single_folder_parsing_warns_when_specifying_filtered_folders(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    omap10_path = require_test_dataset("macsima_omap10")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_1")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_2")
    with pytest.warns(UserWarning, match="filtering only happens for processed_multi_folders"):
        macsima(tmp_path, parsing_style="processed_single_folder", filter_folder_names=["OMAP10_small_2"])


def test_processed_multiple_folders_returns_an_image_stack_per_subfolder(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    omap10_path = require_test_dataset("macsima_omap10")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_1")
    shutil.copytree(omap10_path, tmp_path / "OMAP10_small_2")

    sdata = macsima(tmp_path, parsing_style="processed_multiple_folders")

    assert len(sdata.images) == 2
    for el in sdata.images.keys():
        assert len(get_channel_names(sdata[el])) == 4
    assert len(sdata.tables) == 2


def test_processed_multiple_folders_skips_filtered_folder_names(
    tmp_path: Path, require_test_dataset: Callable[[str], Path]
) -> None:
    shutil.copytree(require_test_dataset("macsima_omap10"), tmp_path / "OMAP10_small")
    shutil.copytree(require_test_dataset("macsima_omap23"), tmp_path / "OMAP23_small")

    sdata = macsima(tmp_path, parsing_style="processed_multiple_folders", filter_folder_names=["OMAP10_small"])
    assert len(sdata.images) == 1
    assert list(sdata.images.keys()) == ["OMAP23_small_image"]
    assert len(sdata.tables) == 1
    assert list(sdata.tables.keys()) == ["OMAP23_small_table"]


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
        "name": ["CD15", "DAPI", "Bcl 2", "CD1c"],
        "cycle": [1, 1, 2, 2],
        "imagetype": ["stain", "stain", "stain", "stain"],
        "well": ["C-1", "C-1", "C-1", "C-1"],
        "ROI": [1, 1, 1, 1],
        "fluorophore": ["APC", "DAPI", "FITC", "PE"],
        "clone": ["VIMC6", pd.NA, "REA872", "REA694"],
        "exposure": [2304.0, 40.0, 96.0, 144.0],
    },
    index=["CD15", "DAPI", "Bcl 2", "CD1c"],
    columns=METADATA_COLUMN_ORDER,
)

EXPECTED_METADATA_OMAP23 = pd.DataFrame(
    {
        "name": ["CD3", "DAPI", "CD279", "CD66b", "DAPI_background"],
        "cycle": [1, 1, 2, 4, 15],
        "imagetype": ["stain", "stain", "stain", "stain", "bleach"],
        "well": ["D01", "D01", "D01", "D01", "D01"],
        "ROI": [1, 1, 1, 1, 1],
        "fluorophore": ["APC", "DAPI", "PE", "FITC", "DAPI"],
        "clone": ["REA1151", pd.NA, "REA1165", "REA306", pd.NA],
        "exposure": [1212.52, 51.0, 322.12, 856.68, 51.0],
    },
    index=["CD3", "DAPI", "CD279", "CD66b", "DAPI_background"],
    columns=METADATA_COLUMN_ORDER,
)


@pytest.mark.parametrize(
    "dataset,expected_df",
    [
        pytest.param("macsima_omap10", EXPECTED_METADATA_OMAP10, id="macsima_omap10"),
        pytest.param("macsima_omap23", EXPECTED_METADATA_OMAP23, id="macsima_omap23"),
    ],
)
def test_metadata_table(dataset: str, expected_df: pd.DataFrame, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = macsima(f)
    table = sdata[list(sdata.tables.keys())[0]]

    # Convert table.var to a DataFrame and align to expected columns
    actual = table.var[METADATA_COLUMN_ORDER]

    pd.testing.assert_frame_equal(actual, expected_df)


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("macsima_omap10", id="macsima_omap10"),
        pytest.param("macsima_omap23", id="macsima_omap23"),
    ],
)
def test_cli_macsima(runner: CliRunner, dataset: str, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
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
