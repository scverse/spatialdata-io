import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from click.testing import CliRunner
from spatialdata import read_zarr

from spatialdata_io.__main__ import xenium_wrapper
from spatialdata_io.readers.xenium import (
    cell_id_str_from_prefix_suffix_uint32,
    prefix_suffix_uint32_from_cell_id_str,
    xenium,
)
from tests._utils import skip_if_below_python_version


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    cell_id_prefix = np.array([1, 1437536272, 1437536273], dtype=np.uint32)
    dataset_suffix = np.array([1, 1, 2])

    cell_id_str = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    assert np.array_equal(cell_id_str, np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"]))


def test_prefix_suffix_uint32_from_cell_id_str() -> None:
    cell_id_str = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    cell_id_prefix, dataset_suffix = prefix_suffix_uint32_from_cell_id_str(cell_id_str)
    assert np.array_equal(cell_id_prefix, np.array([1, 1437536272, 1437536273], dtype=np.uint32))
    assert np.array_equal(dataset_suffix, np.array([1, 1, 2]))


def test_roundtrip_with_data_limits() -> None:
    # min and max values for uint32
    cell_id_prefix = np.array([0, 4294967295], dtype=np.uint32)
    dataset_suffix = np.array([1, 1])
    cell_id_str = np.array(["aaaaaaaa-1", "pppppppp-1"])
    f0 = cell_id_str_from_prefix_suffix_uint32
    f1 = prefix_suffix_uint32_from_cell_id_str
    assert np.array_equal(cell_id_prefix, f1(f0(cell_id_prefix, dataset_suffix))[0])
    assert np.array_equal(dataset_suffix, f1(f0(cell_id_prefix, dataset_suffix))[1])
    assert np.array_equal(cell_id_str, f0(*f1(cell_id_str)))


# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
# TODO: add tests for Xenium 3.0.0
@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("Xenium_V1_human_Breast_2fov_outs", "{'y': (0, 3529), 'x': (0, 5792), 'z': (10, 25)}"),
        ("Xenium_V1_human_Lung_2fov_outs", "{'y': (0, 3553), 'x': (0, 5793), 'z': (7, 32)}"),
    ],
)
def test_example_data(dataset: str, expected: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = xenium(f, cells_as_circles=False)
    from spatialdata import get_extent

    extent = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert str(extent) == expected


# TODO: add tests for Xenium 3.0.0
@skip_if_below_python_version()
@pytest.mark.parametrize("dataset", ["Xenium_V1_human_Breast_2fov_outs", "Xenium_V1_human_Lung_2fov_outs"])
def test_cli_xenium(runner: CliRunner, dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            xenium_wrapper,
            [
                "--input",
                f,
                "--output",
                output_zarr,
            ],
        )
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)
