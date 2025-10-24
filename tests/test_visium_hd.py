import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from click.testing import CliRunner
from spatialdata import get_extent, read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io.__main__ import visium_hd_wrapper
from spatialdata_io._constants._constants import VisiumHDKeys
from spatialdata_io.readers.visium_hd import (
    _decompose_projective_matrix,
    _projective_matrix_is_affine,
    visium_hd,
)
from tests._utils import skip_if_below_python_version

# --- UNIT TESTS FOR HELPER FUNCTIONS ---


def test_projective_matrix_is_affine() -> None:
    """Test the affine matrix check function."""
    # An affine matrix should have [0, 0, 1] as its last row
    affine_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0, 0, 1]])
    assert _projective_matrix_is_affine(affine_matrix)

    # A projective matrix is not affine if the last row is different
    projective_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0.01, 0.02, 1]])
    assert not _projective_matrix_is_affine(projective_matrix)


def test_decompose_projective_matrix() -> None:
    """Test the decomposition of a projective matrix into affine and shift components."""
    projective_matrix = np.array([[1, 2, 3], [4, 5, 6], [0.1, 0.2, 1]])
    affine, shift = _decompose_projective_matrix(projective_matrix)

    expected_affine = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])

    # The affine component should be correctly extracted
    assert np.allclose(affine, expected_affine)
    # Recomposing the affine and shift matrices should yield the original projective matrix
    assert np.allclose(affine @ shift, projective_matrix)


# --- END-TO-END TESTS ON EXAMPLE DATA ---

# TODO: Replace with the actual Visium HD test dataset folder name
# This dataset name is used to locate the test data in the './data/' directory.
# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml
# for instructions on how to download and place the data on disk.
DATASET_FOLDER = "Visium_HD_Mouse_Brain_Chunk"
DATASET_ID = "visium_hd_tiny"


@skip_if_below_python_version()
def test_visium_hd_data_extent() -> None:
    """Check the spatial extent of the loaded Visium HD data."""
    f = Path("./data") / DATASET_FOLDER
    if not f.is_dir():
        pytest.skip(f"Test data not found at '{f}'. Skipping extent test.")

    sdata = visium_hd(f, dataset_id=DATASET_ID)
    extent = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}

    # TODO: Replace with the actual expected extent of your test data
    expected_extent = "{'x': (1000, 7000), 'y': (2000, 8000)}"
    assert str(extent) == expected_extent


@skip_if_below_python_version()
@pytest.mark.parametrize(
    "params",
    [
        # Test case 1: Default binned data loading (squares)
        {
            "load_segmentations_only": False,
            "load_nucleus_segmentations": False,
            "bins_as_squares": True,
            "annotate_table_by_labels": False,
            "load_all_images": False,
        },
        # Test case 2: Binned data as circles
        {
            "load_segmentations_only": False,
            "load_nucleus_segmentations": False,
            "bins_as_squares": False,
            "annotate_table_by_labels": False,
            "load_all_images": False,
        },
        # Test case 3: Binned data with tables annotating labels instead of shapes
        {
            "load_segmentations_only": False,
            "load_nucleus_segmentations": False,
            "bins_as_squares": True,
            "annotate_table_by_labels": True,
            "load_all_images": False,
        },
        # Test case 4: Load binned data AND all segmentations (cell + nucleus)
        {
            "load_segmentations_only": False,
            "load_nucleus_segmentations": True,
            "bins_as_squares": True,
            "annotate_table_by_labels": False,
            "load_all_images": False,
        },
        # Test case 5: Load cell segmentations only
        {
            "load_segmentations_only": True,
            "load_nucleus_segmentations": False,
            "bins_as_squares": True,
            "annotate_table_by_labels": False,
            "load_all_images": False,
        },
        # Test case 6: Load all segmentations (cell + nucleus) only
        {
            "load_segmentations_only": True,
            "load_nucleus_segmentations": True,
            "bins_as_squares": True,
            "annotate_table_by_labels": False,
            "load_all_images": False,
        },
        # Test case 7: Load everything, including auxiliary images like CytAssist
        {
            "load_segmentations_only": False,
            "load_nucleus_segmentations": True,
            "bins_as_squares": True,
            "annotate_table_by_labels": False,
            "load_all_images": True,
        },
    ],
)
def test_visium_hd_data_integrity(params: dict[str, bool]) -> None:
    """Check the integrity of various components of the loaded SpatialData object."""
    f = Path("./data") / DATASET_FOLDER
    if not f.is_dir():
        pytest.skip(f"Test data not found at '{f}'. Skipping integrity test.")

    sdata = visium_hd(f, dataset_id=DATASET_ID, **params)

    # --- IMAGE CHECKS ---
    assert f"{DATASET_ID}_full_image" in sdata.images
    assert f"{DATASET_ID}_hires_image" in sdata.images
    assert f"{DATASET_ID}_lowres_image" in sdata.images
    if params.get("load_all_images", False):
        assert f"{DATASET_ID}_cytassist_image" in sdata.images

    # --- SEGMENTATION CHECKS (loaded in all modes if present) ---
    # TODO: Update placeholder values with actual data from your test dataset
    assert VisiumHDKeys.CELL_SEG_KEY_HD in sdata.tables
    assert f"{DATASET_ID}_{VisiumHDKeys.CELL_SEG_KEY_HD}" in sdata.shapes
    cell_table = sdata.tables[VisiumHDKeys.CELL_SEG_KEY_HD]
    assert cell_table.shape == (2485, 36738)  # Example shape (n_obs, n_vars)
    assert "cellid_000000001-1" in cell_table.obs_names  # Example cell ID

    if params["load_nucleus_segmentations"]:
        assert VisiumHDKeys.NUCLEUS_SEG_KEY_HD in sdata.tables
        assert f"{DATASET_ID}_{VisiumHDKeys.NUCLEUS_SEG_KEY_HD}" in sdata.shapes
        nuc_table = sdata.tables[VisiumHDKeys.NUCLEUS_SEG_KEY_HD]
        assert nuc_table.shape == (2485, 36738)  # Example shape
    else:
        assert VisiumHDKeys.NUCLEUS_SEG_KEY_HD not in sdata.tables

    # --- BINNED DATA CHECKS ---
    if params["load_segmentations_only"]:
        assert "square_002um" not in sdata.tables
    else:
        assert "square_008um" in sdata.tables
        table = sdata.tables["square_008um"]
        assert table.shape == (39000, 36738)  # Example shape
        assert "AAACCGGGTTTA-1" in table.obs_names  # Example barcode
        assert np.array_equal(table.X.indices[:3], [10, 20, 30])  # Example indices

        shape_name = f"{DATASET_ID}_square_008um"
        labels_name = f"{shape_name}_labels"
        if params["annotate_table_by_labels"]:
            assert labels_name in sdata.labels
            region, _, _ = get_table_keys(table)
            assert region == labels_name
        else:
            assert shape_name in sdata.shapes
            region, _, _ = get_table_keys(table)
            assert region == shape_name
            # Check for circles vs. squares
            if params["bins_as_squares"]:
                assert "radius" not in sdata.shapes[shape_name]
            else:
                assert "radius" in sdata.shapes[shape_name]


# --- CLI WRAPPER TEST ---


@skip_if_below_python_version()
def test_cli_visium_hd(runner: CliRunner) -> None:
    """Test the command-line interface for the Visium HD reader."""
    f = Path("./data") / DATASET_FOLDER
    if not f.is_dir():
        pytest.skip(f"Test data not found at '{f}'. Skipping CLI test.")

    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            visium_hd_wrapper,
            [
                "--path",
                str(f),
                "--output",
                str(output_zarr),
            ],
        )
        assert result.exit_code == 0, result.output
        # Verify the output can be read
        sdata = read_zarr(output_zarr)

        # A simple check to confirm data was loaded
        # The default dataset_id is inferred from the feature slice file name.
        # This assert may need adjustment based on your test data's file names.
        inferred_dataset_id = DATASET_FOLDER.replace("_outs", "")  # Example inference
        assert f"{inferred_dataset_id}_full_image" in sdata.images
