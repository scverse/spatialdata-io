import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from click.testing import CliRunner
from spatialdata import match_table_to_element, read_zarr
from spatialdata.models import TableModel, get_table_keys

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
        (
            "Xenium_V1_human_Breast_2fov_outs",
            "{'y': (0, 3529), 'x': (0, 5792), 'z': (10, 25)}",
        ),
        (
            "Xenium_V1_human_Lung_2fov_outs",
            "{'y': (0, 3553), 'x': (0, 5793), 'z': (7, 32)}",
        ),
    ],
)
def test_example_data_data_extent(dataset: str, expected: str) -> None:
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
def test_example_data_index_integrity(dataset) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = xenium(f, cells_as_circles=False)

    # testing Xenium_V1_human_Breast_2fov_outs
    # fmt: off
    # test elements
    assert sdata["morphology_focus"]["scale0"]["image"].sel(c="DAPI", y=20.5, x=20.5).data.compute() == 94
    assert sdata["morphology_focus"]["scale0"]["image"].sel(c="AlphaSMA/Vimentin", y=3528.5, x=5775.5).data.compute() == 1
    assert sdata["cell_labels"]["scale0"]["image"].sel(y=73.5, x=33.5).data.compute() == 4088
    assert sdata["cell_labels"]["scale0"]["image"].sel(y=76.5, x=33.5).data.compute() == 4081
    assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=11.5, x=1687.5).data.compute() == 5030
    assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=3515.5, x=4618.5).data.compute() == 6392
    assert np.allclose(sdata['transcripts'].compute().loc[[0, 10000, 1113949]]['x'], [2.608911, 194.917831, 1227.499268])
    assert np.isclose(sdata['cell_boundaries'].loc['oipggjko-1'].geometry.centroid.x,736.4864931162789)
    assert np.isclose(sdata['nucleus_boundaries'].loc['oipggjko-1'].geometry.centroid.x,736.4931256878282)
    assert np.array_equal(sdata['table'].X.indices[:3], [1, 3, 34])
    # fmt: on

    # test table annotation
    region, region_key, instance_key = get_table_keys(sdata["table"])
    sdata["table"].obs.iloc[0]
    print(region, region_key, instance_key)
    assert region == "cell_labels"
    # to debug you can see that spatialdata.get_element_instances(sdata[region]) and sdata['table'].obs[instance_key] are different
    sdata["table"].obs.columns
    # TableModel.parse(sdata['table'], region=region, region_key=region_key, instance_key = 'labels_id', overwrite_metadata=True)
    match_table_to_element(sdata, element_name=region, table_name="table")

    pass

    # testing Xenium_V1_human_Lung_2fov_outs

    d = {"morphology_focus": {(0, 0, 0): None}}
    for element_name, indexes_values in d.items():
        print(f"Checking element '{element_name}'")
        for index, _value in indexes_values.items():
            print(f"  Checking index {index}")
            print(f"    Expected value: {sdata[element_name]}")
    sdata
    pass


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
