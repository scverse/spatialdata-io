import math
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from click.testing import CliRunner
from spatialdata import match_table_to_element, read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io.__main__ import xenium_wrapper
from spatialdata_io.readers.xenium import xenium


# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
# TODO: add tests for Xenium 3.0.0
@pytest.mark.parametrize(
    "dataset,expected",
    [
        (
            "xenium_breast",
            "{'y': (0, 3529), 'x': (0, 5792), 'z': (10, 25)}",
        ),
        (
            "xenium_lung",
            "{'y': (0, 3553), 'x': (0, 5793), 'z': (7, 32)}",
        ),
        (
            "xenium_protein_kidney",
            "{'y': (0, 6915), 'x': (0, 2963), 'z': (6, 22)}",
        ),
    ],
    ids=["xenium_breast", "xenium_lung", "xenium_protein_kidney"],
)
def test_example_data_data_extent(dataset: str, expected: str, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = xenium(f, cells_as_circles=False)
    from spatialdata import get_extent

    extent = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert str(extent) == expected


# TODO: add tests for Xenium 3.0.0
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("xenium_breast", id="xenium_breast"),
        pytest.param("xenium_lung", id="xenium_lung"),
        pytest.param("xenium_protein_kidney", id="xenium_protein_kidney"),
    ],
)
def test_example_data_index_integrity(dataset: str, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = xenium(f, cells_as_circles=False)

    if dataset == "xenium_breast":
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
        assert sdata['cell_boundaries'].index.name == 'cell_id'
        index = sdata['nucleus_boundaries']['cell_id'].index[sdata['nucleus_boundaries']['cell_id'].eq('oipggjko-1')][0]
        assert np.isclose(sdata['nucleus_boundaries'].loc[index].geometry.centroid.x,736.4931256878282)
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
    elif dataset == "xenium_lung":
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
        assert sdata['cell_boundaries'].index.name == 'cell_id'
        index = sdata['nucleus_boundaries']['cell_id'].index[sdata['nucleus_boundaries']['cell_id'].eq('aaanbaof-1')][0]
        assert np.isclose(sdata['nucleus_boundaries'].loc[index].geometry.centroid.x,43.31874577809517)
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
        assert dataset == "xenium_protein_kidney"
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
        assert sdata['cell_boundaries'].index.name == 'cell_id'
        index = sdata['nucleus_boundaries']['cell_id'].index[sdata['nucleus_boundaries']['cell_id'].eq('aadmbfof-1')][0]
        assert np.isclose(sdata['nucleus_boundaries'].loc[index].geometry.centroid.x, 65.43305896114295)
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
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("xenium_breast", marks=pytest.mark.slow, id="xenium_breast"),
        pytest.param("xenium_lung", marks=pytest.mark.slow, id="xenium_lung"),
        pytest.param("xenium_protein_kidney", marks=pytest.mark.slow, id="xenium_protein_kidney"),
    ],
)
def test_cli_xenium(runner: CliRunner, dataset: str, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
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


@pytest.mark.parametrize(
    (
        "dataset",
        "gex_only",
    ),
    [
        pytest.param("xenium_lung", False, id="xenium_lung_all_features"),
        pytest.param("xenium_lung", True, id="xenium_lung_gex_only"),
        pytest.param("xenium_ovary", False, id="xenium_ovary_all_features"),
        pytest.param("xenium_ovary", True, id="xenium_ovary_gex_only"),
        pytest.param("xenium_multicell_ovary", False, id="xenium_multicell_ovary_all_features"),
        pytest.param("xenium_multicell_ovary", True, id="xenium_multicell_ovary_gex_only"),
        pytest.param("xenium_protein_kidney", False, id="xenium_protein_kidney_all_features"),
        pytest.param("xenium_protein_kidney", True, id="xenium_protein_kidney_gex_only"),
    ],
)
def test_xenium_other_feature_types(dataset: str, gex_only: bool, require_test_dataset: Callable[[str], Path]) -> None:
    f = require_test_dataset(dataset)
    sdata = xenium(f, cells_as_circles=False, gex_only=gex_only)
    if gex_only:
        assert set(sdata["table"].var["feature_types"]) == {"Gene Expression"}
    elif dataset == "xenium_lung":
        assert set(sdata["table"].var["feature_types"]) == {
            "Deprecated Codeword",
            "Gene Expression",
            "Negative Control Codeword",
            "Negative Control Probe",
            "Unassigned Codeword",
        }
    elif dataset in {"xenium_ovary", "xenium_multicell_ovary"}:
        assert set(sdata["table"].var["feature_types"]) == {
            "Gene Expression",
            "Genomic Control",
            "Negative Control Codeword",
            "Negative Control Probe",
            "Unassigned Codeword",
        }
    elif dataset == "xenium_protein_kidney":
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
