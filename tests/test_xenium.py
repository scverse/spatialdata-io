import math
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture
from spatialdata import match_table_to_element, read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io.__main__ import xenium_wrapper
from spatialdata_io.readers.xenium import (
    _cell_id_str_from_prefix_suffix_uint32_reference,
    _warn_if_scaled_protein,
    cell_id_str_from_prefix_suffix_uint32,
    prefix_suffix_uint32_from_cell_id_str,
    xenium,
)
from tests._utils import skip_if_below_python_version


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    cell_id_prefix = np.array([1, 1437536272, 1437536273], dtype=np.uint32)
    dataset_suffix = np.array([1, 1, 2])
    expected = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, expected)
    assert np.array_equal(reference, expected)


def test_cell_id_str_optimized_matches_reference() -> None:
    rng = np.random.default_rng(42)
    cell_id_prefix = rng.integers(0, 2**32, size=10_000, dtype=np.uint32)
    dataset_suffix = rng.integers(0, 10, size=10_000)

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, reference)


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
                str(f),
                "--output",
                str(output_zarr),
            ],
        )
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)


# A real pre-1.3.0 (XOA 1.0.2) bundle reduced to the CSV outputs a GEO deposit typically keeps:
# cells.csv.gz, cell/nucleus_boundaries.csv.gz, transcripts.csv.gz, cell_feature_matrix.h5,
# experiment.xenium -- no parquet and no cells.zarr.zip. Subset to 30 cells of the 10x Mouse Brain
# dataset (sddb 104cz).
XENIUM_CSV_ONLY = Path(__file__).parent / "fixtures" / "xenium-1.0.2-csv-tiny"


def test_xenium_csv_only_bundle() -> None:
    # defaults: cells_as_circles=False, cells_labels/nucleus_labels=True. The labels, which live only
    # in cells.zarr.zip, are reconstructed by rasterizing the boundary polygons read from the CSVs.
    sdata = xenium(
        XENIUM_CSV_ONLY,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
    )
    assert sdata["table"].n_obs == 30
    assert len(sdata["cell_boundaries"]) == 30
    assert len(sdata["nucleus_boundaries"]) == 30
    assert len(sdata["transcripts"]) > 0
    # labels reconstructed from the boundaries, keyed by the 30 integer cell_ids
    for name in ("cell_labels", "nucleus_labels"):
        assert _label_ids(sdata[name]) == set(range(1, 31))


def test_xenium_csv_only_circles() -> None:
    sdata = xenium(
        XENIUM_CSV_ONLY,
        cells_as_circles=True,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
    )
    assert len(sdata["cell_circles"]) == 30


def test_xenium_csv_only_uncompressed(tmp_path: Path) -> None:
    # some GEO deposits ship uncompressed .csv rather than .csv.gz
    import gzip
    import shutil

    for f in XENIUM_CSV_ONLY.iterdir():
        if f.suffixes[-2:] == [".csv", ".gz"]:
            with gzip.open(f, "rb") as src, open(tmp_path / f.stem, "wb") as dst:  # f.stem drops .gz
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(f, tmp_path / f.name)

    sdata = xenium(tmp_path, morphology_mip=False, morphology_focus=False, aligned_images=False)
    assert sdata["table"].n_obs == 30
    assert len(sdata["cell_boundaries"]) == 30


def test_xenium_csv_only_mtx_matrix(tmp_path: Path) -> None:
    # some GEO deposits ship the MatrixMarket cell_feature_matrix/ dir instead of the .h5
    import gzip
    import shutil

    import scanpy as sc
    from scipy.io import mmwrite

    for f in XENIUM_CSV_ONLY.iterdir():
        if f.name != "cell_feature_matrix.h5":
            shutil.copy(f, tmp_path / f.name)
    adata = sc.read_10x_h5(XENIUM_CSV_ONLY / "cell_feature_matrix.h5", gex_only=False)
    mdir = tmp_path / "cell_feature_matrix"
    mdir.mkdir()
    with gzip.open(mdir / "matrix.mtx.gz", "wb") as fh:
        mmwrite(fh, adata.X.T)  # type: ignore[arg-type]  # MatrixMarket is features x barcodes
    with gzip.open(mdir / "features.tsv.gz", "wt") as fh:
        for gid, name, ft in zip(adata.var["gene_ids"], adata.var_names, adata.var["feature_types"], strict=True):
            fh.write(f"{gid}\t{name}\t{ft}\n")
    with gzip.open(mdir / "barcodes.tsv.gz", "wt") as fh:
        fh.writelines(f"{b}\n" for b in adata.obs_names)

    sdata = xenium(tmp_path, morphology_mip=False, morphology_focus=False, aligned_images=False)
    assert sdata["table"].n_obs == 30
    assert "cell_labels" in sdata.labels


def test_warn_if_scaled_protein() -> None:
    # protein counts read outside HDF5 cannot be descaled (the factor lives only in the .h5), so warn
    from anndata import AnnData

    gex = AnnData(np.zeros((2, 1)), var=pd.DataFrame({"feature_types": ["Gene Expression"]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_scaled_protein(gex)  # no protein -> no warning

    prot = AnnData(np.zeros((2, 1)), var=pd.DataFrame({"feature_types": ["Protein Expression"]}))
    with pytest.warns(UserWarning, match="scaled units"):
        _warn_if_scaled_protein(prot)


# A v2/v3 (XOA 3.0.0) CSV-only bundle: hex cell_ids, and a label_id column in the boundary CSVs.
# Subset to 12 cells of the 10x Xenium Prime Mouse Brain dataset (sddb efli2b4vhc).
XENIUM_CSV_HEX = Path(__file__).parent / "fixtures" / "xenium-3.0.0-csv-hex-tiny"


def _label_ids(element: object) -> set[int]:
    import xarray as xr
    from xarray import DataTree

    if isinstance(element, DataTree):  # multiscale: take full-res scale0
        node = element["scale0"]
        element = node[str(next(iter(node.data_vars)))]
    assert isinstance(element, xr.DataArray)
    return set(np.unique(np.asarray(element.data)).tolist()) - {0}


def test_xenium_csv_only_hex_labels() -> None:
    # for hex cell_ids the integer raster label comes from the boundary CSVs' label_id column, and
    # multinucleate cells yield more nucleus labels than cells
    sdata = xenium(
        XENIUM_CSV_HEX, morphology_mip=False, morphology_focus=False, aligned_images=False, transcripts=False
    )
    assert sdata["table"].n_obs == 12
    assert isinstance(sdata["table"].obs["cell_id"].iloc[0], str)  # hex ids
    # the raster labels must be the true (non-contiguous) label_id, and must match the table column
    # the instance key resolves against -- not a 1..N rank (which would silently mis-join)
    cell_label_ids = _label_ids(sdata["cell_labels"])
    assert cell_label_ids == set(sdata["table"].obs["cell_labels"])
    assert max(cell_label_ids) > 12  # genuine label_ids have gaps; a 1..12 rank would fail this
    assert len(_label_ids(sdata["nucleus_labels"])) >= 12  # >= cells (multinucleate)


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


# ── CLI JSON kwargs tests (no real data needed) ───────────────────────────────


@pytest.mark.parametrize(
    "kwarg_name",
    ["--imread-kwargs", "--image-models-kwargs", "--labels-models-kwargs"],
)
def test_cli_xenium_invalid_json_rejected(runner: CliRunner, tmp_path: Path, kwarg_name: str) -> None:
    """Invalid JSON for any kwargs option must produce a non-zero exit and a clear error."""
    result = runner.invoke(
        xenium_wrapper,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            "not-valid-json{",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output


@pytest.mark.parametrize(
    ("kwarg_name", "kwarg_param"),
    [
        ("--imread-kwargs", "imread_kwargs"),
        ("--image-models-kwargs", "image_models_kwargs"),
        ("--labels-models-kwargs", "labels_models_kwargs"),
    ],
)
def test_cli_xenium_valid_json_forwarded(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture, kwarg_name: str, kwarg_param: str
) -> None:
    """Valid JSON kwargs must be parsed and forwarded to the xenium reader as a dict."""
    mock_xenium = mocker.patch("spatialdata_io.readers.xenium.xenium")
    mock_xenium.return_value = mocker.MagicMock()
    result = runner.invoke(
        xenium_wrapper,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            '{"chunks": 512}',
        ],
    )
    assert result.exit_code == 0, result.output
    call_kwargs = mock_xenium.call_args.kwargs
    assert call_kwargs[kwarg_param] == {"chunks": 512}
