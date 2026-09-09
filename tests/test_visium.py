import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from shapely import Point
from spatialdata import SpatialData, get_extent, read_zarr
from spatialdata.models import ShapesModel, get_table_keys
from spatialdata.transformations import get_transformation

from spatialdata_io.__main__ import visium_wrapper
from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.readers.visium import visium

# --- END-TO-END TESTS ON EXAMPLE DATA ---
# This dataset name is used to locate the test data in the './data/' directory.
# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml
# for instructions on how to download and place the data on disk.
DATASET_FOLDER = "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer"
# the reader infers the same value from the name of the counts file
DATASET_ID = "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer"

if not (Path("./data") / DATASET_FOLDER).is_dir():
    pytest.skip(
        f"Requires the {DATASET_FOLDER} dataset (10x Genomics Space Ranger 2.1.0). The files and the "
        "layout they are expected in are listed in .github/workflows/prepare_test_data.yaml.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dataset_path() -> Path:
    return Path("./data") / DATASET_FOLDER


@pytest.fixture(scope="module")
def sdata(dataset_path: Path) -> SpatialData:
    return visium(dataset_path, dataset_id=DATASET_ID)


def test_visium_elements(sdata: SpatialData) -> None:
    """The reader builds the two downscaled images, the spots and the table."""
    assert list(sdata.images) == [f"{DATASET_ID}_hires_image", f"{DATASET_ID}_lowres_image"]
    assert list(sdata.shapes) == [DATASET_ID]
    assert list(sdata.tables) == ["table"]
    # no full resolution image is passed, so `<dataset_id>_full_image` is not created
    assert f"{DATASET_ID}_full_image" not in sdata.images
    assert sorted(sdata.coordinate_systems) == sorted(
        [DATASET_ID, f"{DATASET_ID}_downscaled_hires", f"{DATASET_ID}_downscaled_lowres"]
    )
    assert sdata.attrs["spatialdata_io_reader"] == "visium"


@pytest.mark.parametrize(
    "coordinate_system,expected",
    [
        (DATASET_ID, {"y": (0, 22630), "x": (-125, 23128)}),
        (f"{DATASET_ID}_downscaled_hires", {"y": (0, 1957), "x": (-11, 2000)}),
        (f"{DATASET_ID}_downscaled_lowres", {"y": (0, 587), "x": (-4, 600)}),
    ],
)
def test_visium_data_extent(sdata: SpatialData, coordinate_system: str, expected: dict[str, tuple[int, int]]) -> None:
    """Each coordinate system covers the image, plus the spots that fall outside of it."""
    extent = get_extent(sdata, exact=False, coordinate_system=coordinate_system)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == expected


def test_visium_images(sdata: SpatialData) -> None:
    """The two downscaled images are read as RGB, and each lives in its own coordinate system."""
    hires = sdata[f"{DATASET_ID}_hires_image"]
    lowres = sdata[f"{DATASET_ID}_lowres_image"]

    for image in (hires, lowres):
        assert image.dims == ("c", "y", "x")
        assert image.dtype == np.uint8
        assert image.coords["c"].values.tolist() == ["r", "g", "b"]

    assert hires.shape == (3, 1957, 2000)
    assert lowres.shape == (3, 587, 600)
    assert np.array_equal(hires.data[:, 1000, 1000].compute(), [8, 5, 92])
    assert np.array_equal(lowres.data[:, 300, 300].compute(), [13, 8, 63])

    # each image is unscaled in its own coordinate system, and scaled down in the one of the full resolution image
    assert sorted(get_transformation(hires, get_all=True)) == sorted([DATASET_ID, f"{DATASET_ID}_downscaled_hires"])
    assert sorted(get_transformation(lowres, get_all=True)) == sorted([DATASET_ID, f"{DATASET_ID}_downscaled_lowres"])


def test_visium_table(sdata: SpatialData) -> None:
    """The table holds the filtered counts, annotated by the spots."""
    table = sdata["table"]

    assert table.shape == (4169, 18085)
    assert table.obs_names[:3].tolist() == ["AACACTTGGCAAGGAA-1", "AACAGGATTCATAGTT-1", "AACAGGCCAACGATTA-1"]
    assert table.var_names[:3].tolist() == ["SAMD11", "NOC2L", "KLHL17"]
    assert table.obs_names.is_unique
    assert table.var_names.is_unique
    assert np.array_equal(table.X.indices[:3], [3, 6, 7])

    # the spot coordinates are moved to `obsm`, the remaining columns of `tissue_positions` are kept
    assert table.obs.columns.tolist() == ["in_tissue", "array_row", "array_col", "spot_id", "region"]
    assert table.obs["spot_id"].tolist() == list(range(len(table)))
    # the filtered matrix only contains the spots under the tissue
    assert table.obs["in_tissue"].eq(1).all()

    assert get_table_keys(table) == (DATASET_ID, "region", "spot_id")


def test_visium_circles_are_the_spot_coordinates(sdata: SpatialData, dataset_path: Path) -> None:
    """The circles are the spot coordinates of the table, in the order of the table."""
    circles = sdata[DATASET_ID]
    table = sdata["table"]

    # ground truth, read independently of the reader and reordered to follow the table
    positions = pd.read_csv(dataset_path / "spatial" / VisiumKeys.SPOTS_FILE_2, index_col=0)
    expected = positions.loc[table.obs_names, [VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].to_numpy()

    assert np.array_equal(circles.get_coordinates().to_numpy(), expected)
    assert np.array_equal(table.obsm["spatial"], expected)
    assert circles.index.tolist() == table.obs["spot_id"].tolist()
    assert all(isinstance(geometry, Point) for geometry in circles.geometry)

    scalefactors = json.loads((dataset_path / "spatial" / VisiumKeys.SCALEFACTORS_FILE).read_bytes())
    expected_radius = scalefactors["spot_diameter_fullres"] / 2.0
    assert np.array_equal(circles[ShapesModel.RADIUS_KEY], np.full(len(table), expected_radius))


def test_visium_dataset_id_is_inferred_from_the_counts_file(dataset_path: Path, sdata: SpatialData) -> None:
    """Without `dataset_id` the elements are named after the prefix of the counts file."""
    inferred = visium(dataset_path)

    assert list(inferred.images) == list(sdata.images)
    assert list(inferred.shapes) == list(sdata.shapes)
    assert sorted(inferred.coordinate_systems) == sorted(sdata.coordinate_systems)


# --- CLI WRAPPER TEST ---


def test_cli_visium(runner: CliRunner, dataset_path: Path) -> None:
    """The reader is reachable from the command line and the result can be written and read back."""
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            visium_wrapper,
            [
                "--input",
                str(dataset_path),
                "--output",
                str(output_zarr),
            ],
        )
        assert result.exit_code == 0, result.output

        sdata = read_zarr(output_zarr)
        assert list(sdata.shapes) == [DATASET_ID]
        assert sdata["table"].shape == (4169, 18085)
