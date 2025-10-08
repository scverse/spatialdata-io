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
        (
            "Xenium_V1_Protein_Human_Kidney_tiny_outs",
            "{'y': (0, 6915), 'x': (0, 2963), 'z': (6, 22)}",
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
@pytest.mark.parametrize(
    "dataset",
    ["Xenium_V1_human_Breast_2fov_outs", "Xenium_V1_human_Lung_2fov_outs", "Xenium_V1_Protein_Human_Kidney_tiny_outs"],
)
def test_example_data_index_integrity(dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = xenium(f, cells_as_circles=False)

    if dataset == "Xenium_V1_human_Breast_2fov_outs":
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
        assert region == "cell_labels"
        matched_table = match_table_to_element(sdata, element_name=region, table_name="table")
        assert len(matched_table) == 7275
        assert matched_table.obs["cell_id"][:3].tolist() == [
            "aaaiikim-1",
            "aaaljapa-1",
            "aabhbgmg-1",
        ]
    elif dataset == "Xenium_V1_human_Lung_2fov_outs":
        # fmt: off
        # test elements
        assert sdata["morphology_focus"]["scale0"]["image"].sel(c="DAPI", y=0.5, x=2215.5).data.compute() == 1
        assert sdata["morphology_focus"]["scale0"]["image"].sel(c="DAPI", y=11.5, x=4437.5).data.compute() == 2007
        assert sdata["cell_labels"]["scale0"]["image"].sel(y=0.5, x=2940.5).data.compute() == 2605
        assert sdata["cell_labels"]["scale0"]["image"].sel(y=3.5, x=4801.5).data.compute() == 7618
        assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=8.5, x=4359.5).data.compute() == 7000
        assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=18.5, x=3015.5).data.compute() == 2764
        assert np.allclose(sdata['transcripts'].compute().loc[[0, 10000, 20000]]['x'], [174.258392, 12.210024, 214.759186])
        assert np.isclose(sdata['cell_boundaries'].loc['aaanbaof-1'].geometry.centroid.x, 43.96894317275074)
        assert np.isclose(sdata['nucleus_boundaries'].loc['aaanbaof-1'].geometry.centroid.x,43.31874577809517)
        assert np.array_equal(sdata['table'].X.indices[:3], [1, 8, 19])
        # fmt: on

        # test table annotation
        region, region_key, instance_key = get_table_keys(sdata["table"])
        assert region == "cell_labels"
        matched_table = match_table_to_element(sdata, element_name=region, table_name="table")
        assert len(matched_table) == 11898
        assert matched_table.obs["cell_id"][:3].tolist() == [
            "aaafiiei-1",
            "aaanbaof-1",
            "aabdiein-1",
        ]
    else:
        assert dataset == "Xenium_V1_Protein_Human_Kidney_tiny_outs"
        # fmt: off
        # test elements
        assert sdata["morphology_focus"]["scale0"]["image"].sel(c="VISTA", y=2876.5, x=32.5).data.compute() == 99
        assert sdata["morphology_focus"]["scale0"]["image"].sel(c="VISTA", y=4040.5, x=28.5).data.compute() == 103
        assert sdata["cell_labels"]["scale0"]["image"].sel(y=128.5, x=297.5).data.compute() == 358
        assert sdata["cell_labels"]["scale0"]["image"].sel(y=4059.5, x=637.5).data.compute() == 340
        assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=151.5, x=297.5).data.compute() == 368
        assert sdata["nucleus_labels"]["scale0"]["image"].sel(y=4039.5, x=93.5).data.compute() == 274
        assert np.allclose(sdata['transcripts'].compute().loc[[0, 10000, 20000]]['x'], [43.296875, 62.484375, 93.125])
        assert np.isclose(sdata['cell_boundaries'].loc['aadmbfof-1'].geometry.centroid.x, 64.54541104696033)
        assert np.isclose(sdata['nucleus_boundaries'].loc['aadmbfof-1'].geometry.centroid.x, 65.43305896114295)
        assert np.array_equal(sdata['table'].X.indices[:3], [3, 49, 53])
        # fmt: on

        # test table annotation
        region, region_key, instance_key = get_table_keys(sdata["table"])
        assert region == "cell_labels"
        matched_table = match_table_to_element(sdata, element_name=region, table_name="table")
        assert len(matched_table) == 358
        assert matched_table.obs["cell_id"][:3].tolist() == [
            "aadmbfof-1",
            "aageapbo-1",
            "aakefffb-1",
        ]


# TODO: add tests for Xenium 3.0.0
@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset",
    ["Xenium_V1_human_Breast_2fov_outs", "Xenium_V1_human_Lung_2fov_outs", "Xenium_V1_Protein_Human_Kidney_tiny_outs"],
)
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


@skip_if_below_python_version()
@pytest.mark.parametrize(
    (
        "dataset",
        "gex_only",
    ),
    [
        ("Xenium_V1_human_Lung_2fov_outs", False),
        ("Xenium_V1_human_Lung_2fov_outs", True),
        ("Xenium_V1_Human_Ovary_tiny_outs", False),
        ("Xenium_V1_Human_Ovary_tiny_outs", True),
        ("Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs", False),
        ("Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs", True),
        ("Xenium_V1_Protein_Human_Kidney_tiny_outs", False),
        ("Xenium_V1_Protein_Human_Kidney_tiny_outs", True),
    ],
)
def test_xenium_other_feature_types(dataset: str, gex_only: bool) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = xenium(f, cells_as_circles=False, gex_only=gex_only)
    if gex_only:
        assert set(sdata["table"].var["feature_types"]) == {"Gene Expression"}
    elif dataset == "Xenium_V1_human_Lung_2fov_outs":
        assert set(sdata["table"].var["feature_types"]) == {
            "Deprecated Codeword",
            "Gene Expression",
            "Negative Control Codeword",
            "Negative Control Probe",
            "Unassigned Codeword",
        }
    elif dataset in {"Xenium_V1_Human_Ovary_tiny_outs", "Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs"}:
        assert set(sdata["table"].var["feature_types"]) == {
            "Gene Expression",
            "Genomic Control",
            "Negative Control Codeword",
            "Negative Control Probe",
            "Unassigned Codeword",
        }
    elif dataset == "Xenium_V1_Protein_Human_Kidney_tiny_outs":
        assert set(sdata["table"].var["feature_types"]) == {
            "Gene Expression",
            "Genomic Control",
            "Negative Control Codeword",
            "Negative Control Probe",
            "Protein Expression",
            "Unassigned Codeword",
        }
        # Protein feature
        assert np.allclose(
            sdata["table"].X[0:3, sdata["table"].var_names.str.match("VISTA")].toarray().squeeze(), [0.7, 1.2, 0.0]
        )
        # RNA feature
        assert np.allclose(
            sdata["table"].X[[6, 7, 24], sdata["table"].var_names.str.match("ACTG2")].squeeze(), [1, 0, 2]
        )

    else:
        assert ValueError(f"Unexpected dataset {dataset}")
