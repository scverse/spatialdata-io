import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely import Point
from spatialdata.models import ShapesModel

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.readers.visium import visium

# This dataset name is used to locate the test data in the './data/' directory.
# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml
# for instructions on how to download and place the data on disk.
DATASET_FOLDER = "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer"
DATASET_ID = "visium_breast_cancer"

if not (Path("./data") / DATASET_FOLDER).is_dir():
    pytest.skip(
        f"Requires the {DATASET_FOLDER} dataset (10x Genomics Space Ranger 2.1.0). The files and the "
        "layout they are expected in are listed in .github/workflows/prepare_test_data.yaml.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dataset_path() -> Path:
    return Path("./data") / DATASET_FOLDER


def test_visium_circles_are_the_spot_coordinates(dataset_path: Path) -> None:
    """The circles are the spot coordinates of the table, in the order of the table.

    Regression test for the case in which the raw `tissue_positions` table was passed to
    `ShapesModel.parse()`, which raised
    `TypeError: ShapesModel.parse() does not support the type <class 'pandas.core.frame.DataFrame'>`.
    """
    sdata = visium(dataset_path, dataset_id=DATASET_ID)

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
