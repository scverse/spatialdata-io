import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from scipy.sparse import csc_matrix
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import get_transformation

from spatialdata_io import visium_hd
from spatialdata_io.__main__ import visium_hd_wrapper
from spatialdata_io._constants._constants import VisiumHDKeys


def _write_counts(path: Path, ids: list[int], counts: list[list[int]]) -> None:
    matrix = csc_matrix(np.asarray(counts, dtype=np.int32).T)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        for name in ("data", "indices", "indptr", "shape"):
            group.create_dataset(name, data=getattr(matrix, name))
        group.create_dataset("barcodes", data=np.asarray([f"cellid_{i:09d}-1" for i in ids], dtype="S"))
        features = group.create_group("features")
        for name, values in {
            "id": ["ENSG1", "ENSG2"],
            "name": ["G1", "G2"],
            "feature_type": ["Gene Expression", "Gene Expression"],
            "genome": ["GRCh38", "GRCh38"],
        }.items():
            features.create_dataset(name, data=np.asarray(values, dtype="S"))


@pytest.fixture
def segmented_output(tmp_path: Path) -> Path:
    """Small artificial files, read through the real HDF5/GeoJSON reader paths."""
    segmented = tmp_path / "segmented_outputs"
    spatial = segmented / "spatial"
    spatial.mkdir(parents=True)
    (spatial / "scalefactors_json.json").write_text(
        json.dumps({"tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1, "microns_per_pixel": 0.25})
    )
    with h5py.File(tmp_path / "feature_slice.h5", "w") as handle:
        handle.attrs["metadata_json"] = json.dumps({"hd_layout_json": json.dumps({"file_format": "1.0"})})
    _write_counts(segmented / "filtered_feature_cell_matrix.h5", [1], [[4, 7]])
    _write_counts(segmented / "raw_feature_cell_matrix.h5", [3, 1, 2, 4], [[5, 6], [4, 7], [99, 99], [0, 0]])
    # Reverse geometry order; cell 2 has no polygon, and cell 4 has zero counts.
    features = [
        {
            "type": "Feature",
            "properties": {"cell_id": i},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[i * 10, 0], [i * 10 + 2, 0], [i * 10 + 2, 2], [i * 10, 2], [i * 10, 0]]],
            },
        }
        for i in [4, 1, 3]
    ]
    (segmented / "cell_segmentations.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    return tmp_path


@pytest.mark.parametrize("filtered", [None, True, False])
def test_cell_count_selection(segmented_output: Path, filtered: bool | None) -> None:
    paths = [path for path in segmented_output.rglob("*") if path.is_file()]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    kwargs = {} if filtered is None else {"filtered_counts_file": filtered}
    with pytest.warns(UserWarning, match="No .*image"):
        sdata = visium_hd(segmented_output, dataset_id="sample", load_segmentations_only=True, **kwargs)
    table = sdata.tables[VisiumHDKeys.CELL_SEG_KEY_HD]
    shapes_key = f"sample_{VisiumHDKeys.CELL_SEG_KEY_HD}"
    shapes = sdata.shapes[shapes_key]
    ids = [3, 1, 4] if filtered is False else [1]
    expected = [[5, 6], [4, 7], [0, 0]] if filtered is False else [[4, 7]]
    assert table.obs_names.tolist() == [f"cellid_{i:09d}-1" for i in ids]
    assert shapes.index.tolist() == table.obs_names.tolist()
    assert table.var_names.tolist() == ["G1", "G2"]
    np.testing.assert_array_equal(table.X.toarray(), expected)
    np.testing.assert_array_equal(shapes.bounds.minx, np.asarray(ids) * 10)
    region, region_key, instance_key = get_table_keys(table)
    assert region == shapes_key or region == [shapes_key]
    assert table.obs[region_key].eq(shapes_key).all()
    assert table.obs[instance_key].tolist() == table.obs_names.tolist()
    np.testing.assert_array_equal(
        get_transformation(shapes, "sample_downscaled_hires").to_affine_matrix(("x", "y"), ("x", "y")),
        np.diag([0.5, 0.5, 1]),
    )
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_raw_cells_without_filtered_file(segmented_output: Path) -> None:
    (segmented_output / "segmented_outputs/filtered_feature_cell_matrix.h5").unlink()
    with pytest.warns(UserWarning, match="No .*image"):
        sdata = visium_hd(
            segmented_output, dataset_id="sample", load_segmentations_only=True, filtered_counts_file=False
        )
    assert sdata.tables[VisiumHDKeys.CELL_SEG_KEY_HD].n_obs == 3


def test_raw_cells_cli(segmented_output: Path, tmp_path: Path, runner: CliRunner) -> None:
    output = tmp_path / "converted.zarr"
    result = runner.invoke(
        visium_hd_wrapper,
        [
            "--input",
            str(segmented_output),
            "--output",
            str(output),
            "--dataset-id",
            "sample",
            "--load-segmentations-only",
            "True",
            "--filtered-counts-file",
            "False",
        ],
    )
    assert result.exit_code == 0, result.output
    table = read_zarr(output).tables[VisiumHDKeys.CELL_SEG_KEY_HD]
    assert table.obs_names.tolist() == ["cellid_000000003-1", "cellid_000000001-1", "cellid_000000004-1"]
    np.testing.assert_array_equal(table.X.toarray(), [[5, 6], [4, 7], [0, 0]])
