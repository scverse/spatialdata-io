import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely import Point
from spatialdata.models import ShapesModel

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.readers.visium import visium

SPOT_DIAMETER_FULLRES = 20.0

# the spots are given in an order which is neither the order of the counts file nor sorted, and one of them is not
# present in the counts file; this way the test detects the case in which the circles are built from the raw
# `tissue_positions` table instead of from the spot coordinates aligned to the table rows
SPOTS = {
    "barcode-2": (1, 0, 1, 210.0, 110.0),
    "barcode-0": (1, 0, 0, 200.0, 100.0),
    "spot-not-in-counts": (0, 9, 9, 999.0, 999.0),
    "barcode-1": (1, 1, 0, 220.0, 120.0),
}
BARCODES = ["barcode-0", "barcode-1", "barcode-2"]


@pytest.fixture
def visium_dataset(tmp_path: Path) -> Path:
    """Write a minimal Visium dataset (counts, tissue positions, scalefactors) to disk."""
    counts = pd.DataFrame(
        np.arange(len(BARCODES) * 2, dtype=np.float32).reshape(len(BARCODES), 2),
        index=BARCODES,
        columns=["gene-a", "gene-b"],
    )
    counts.to_csv(tmp_path / "counts.txt", sep="\t")

    spatial = tmp_path / "spatial"
    spatial.mkdir()
    positions = pd.DataFrame.from_dict(
        SPOTS,
        orient="index",
        columns=["in_tissue", "array_row", "array_col", VisiumKeys.SPOTS_Y, VisiumKeys.SPOTS_X],
    )
    positions.index.name = "barcode"
    positions.to_csv(spatial / VisiumKeys.SPOTS_FILE_2)

    scalefactors = {
        VisiumKeys.SCALEFACTORS_HIRES: 0.1,
        VisiumKeys.SCALEFACTORS_LOWRES: 0.01,
        "spot_diameter_fullres": SPOT_DIAMETER_FULLRES,
    }
    (spatial / VisiumKeys.SCALEFACTORS_FILE).write_text(json.dumps(scalefactors))
    return tmp_path


def test_visium_circles_match_spot_coordinates(visium_dataset: Path) -> None:
    """The circles are the spot coordinates of the table, in the order of the table.

    Regression test for the case in which the raw `tissue_positions` table was passed to `ShapesModel.parse()`.
    """
    sdata = visium(visium_dataset, dataset_id="test", counts_file="counts.txt")

    circles = sdata["test"]
    table = sdata["table"]

    # the table contains only the spots which are in the counts file, in the order of the counts file
    assert table.obs_names.tolist() == BARCODES

    expected = np.array([[SPOTS[barcode][4], SPOTS[barcode][3]] for barcode in BARCODES])
    assert np.array_equal(table.obsm["spatial"], expected)
    assert np.array_equal(circles.get_coordinates().to_numpy(), expected)

    assert circles.index.tolist() == table.obs["spot_id"].tolist()
    assert all(isinstance(geometry, Point) for geometry in circles.geometry)
    assert np.array_equal(circles[ShapesModel.RADIUS_KEY], np.full(len(BARCODES), SPOT_DIAMETER_FULLRES / 2.0))
