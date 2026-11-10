from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.io import write_elem
from scipy.sparse import csr_matrix
from spatialdata.models import get_table_keys
from tests.converters.test_atera_to_parquet import _write_atera_dataset
from xarray import DataTree

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io.readers import _atera_common as common
from spatialdata_io.readers._atera_common import _read_transcript_tile, _tile_row_ranges
from spatialdata_io.readers.atera import _get_table, read_table_for_cells, read_table_for_genes, read_var

# Atera `cell_id`s are large packed integers, unrelated to the `1..n_obs` values in the `cell_labels` raster.
CELL_IDS = [4_320_252_178, 4_975_876_403, 5_016_677_209, 5_217_424_078]


@pytest.fixture
def atera_path(tmp_path: Path) -> Path:
    n = len(CELL_IDS)
    obs = pd.DataFrame(
        {
            str(AteraKeys.CELL_ID): np.array(CELL_IDS, dtype=np.uint64),
            str(AteraKeys.CENTROID_X): np.arange(n, dtype=np.float32),
            str(AteraKeys.CENTROID_Y): np.arange(n, dtype=np.float32),
        },
        index=pd.Index([f"c{i}" for i in range(n)], dtype=object, name="barcode"),
    )
    var = pd.DataFrame(
        {
            str(AteraKeys.FEATURE_NAME): ["GENE_A", "GENE_B"],
            str(AteraKeys.VAR_FILTERED): [False, False],
            str(AteraKeys.VAR_HIGHLY_VARIABLE): [False, False],
            str(AteraKeys.VAR_FEATURE_TYPE): ["Gene Expression", "Gene Expression"],
            str(AteraKeys.VAR_GENOME): ["g", "g"],
        },
        index=pd.Index(["ID_A", "ID_B"], dtype=object, name=str(AteraKeys.FEATURE_ID)),
    )
    x = csr_matrix(np.arange(n * 2, dtype=np.float32).reshape(n, 2))
    for file in (AteraKeys.CELL_FEATURE_MATRIX_FILE, AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE):
        store = zarr.storage.ZipStore(tmp_path / file, mode="w")
        try:
            group = zarr.open_group(store, mode="w", zarr_format=2)
            write_elem(group, str(AteraKeys.X_GROUP), x)
            write_elem(group, str(AteraKeys.OBS_GROUP), obs)
            write_elem(group, str(AteraKeys.VAR_GROUP), var)
            # Real bundles also list the index (`barcode`/`feature_id`) in `column-order`, so it comes back from
            # `read_elem` as both the index and a column.
            for key, df in ((AteraKeys.OBS_GROUP, obs), (AteraKeys.VAR_GROUP, var)):
                elem = group[str(key)]
                elem.attrs["column-order"] = [str(df.index.name), *cast("list[str]", elem.attrs["column-order"])]
        finally:
            store.close()
    return tmp_path


def test_cell_labels_table_uses_label_id(atera_path: Path) -> None:
    table, _ = _get_table(atera_path, region="cell_labels")

    region, _, instance_key = get_table_keys(table)
    assert region == "cell_labels"
    assert instance_key == "label_id"
    # the `cell_labels` raster is numbered 1..n_obs by row position
    assert table.obs["label_id"].tolist() == [1, 2, 3, 4]
    assert table.obs["cell_id"].tolist() == CELL_IDS
    # genes detected per cell, read from the CSR `indptr` (cell 0 only has a zero in its first column)
    assert table.obs["n_genes"].tolist() == [1, 2, 2, 2]


def test_cell_boundaries_table_uses_cell_id(atera_path: Path) -> None:
    table, _ = _get_table(atera_path, region="cell_boundaries")

    assert get_table_keys(table) == ("cell_boundaries", "region", "cell_id")


def test_read_table_for_cells_keeps_original_label_ids(atera_path: Path) -> None:
    table = read_table_for_cells(atera_path, cell_ids=[CELL_IDS[2], CELL_IDS[0]])

    assert get_table_keys(table) == ("cell_labels", "region", "label_id")
    assert table.obs["cell_id"].tolist() == [CELL_IDS[2], CELL_IDS[0]]
    assert table.obs["label_id"].tolist() == [3, 1]


def test_read_table_for_genes_uses_label_id(atera_path: Path) -> None:
    table = read_table_for_genes(atera_path, "GENE_B")

    assert get_table_keys(table) == ("cell_labels", "region", "label_id")
    assert table.obs["label_id"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("table", ["_get_table", "read_table_for_cells", "read_table_for_genes"])
def test_table_obs_and_var_are_writable(atera_path: Path, table: str) -> None:
    # `read_elem` gives the index (`barcode`/`feature_id`) as `str` and the same-named column as `object`, which
    # `anndata` refuses to write because `pd.Series.equals` compares dtypes.
    if table == "_get_table":
        adata = _get_table(atera_path)[0]
    elif table == "read_table_for_cells":
        adata = read_table_for_cells(atera_path, cell_ids=CELL_IDS[:2])
    else:
        adata = read_table_for_genes(atera_path, "GENE_B")

    for name, df in (("obs", adata.obs), ("var", adata.var)):
        write_elem(zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=2), name, df)


def test_read_var_is_writable(atera_path: Path) -> None:
    write_elem(zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=2), "var", read_var(atera_path))


def test_tile_row_ranges() -> None:
    assert _tile_row_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert _tile_row_ranges(8, 4) == [(0, 4), (4, 8)]
    assert _tile_row_ranges(0, 4) == [(0, 0)]  # an empty tile keeps one (empty) partition


def test_chunked_tile_reads_match_whole_tile(tmp_path: Path) -> None:
    _write_atera_dataset(tmp_path, {"0,0": 7}, ["0,0"], ids={})
    feature_names = ["GENE_A", "GENE_B"]
    whole = _read_transcript_tile(tmp_path, "0,0", feature_names)
    # GENE_A owns rows 0-2 and GENE_B rows 3-6, so the 2-row chunks straddle the gene boundary.
    parts = [_read_transcript_tile(tmp_path, "0,0", feature_names, a, b) for a, b in _tile_row_ranges(7, 2)]

    assert pd.concat(parts, ignore_index=True).equals(whole)
    assert whole["feature_name"].tolist() == ["GENE_A"] * 3 + ["GENE_B"] * 4


def test_points_partitions_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_atera_dataset(tmp_path, {"0,0": 7, "0,1": 0}, ["0,0", "0,1"], ids={})
    monkeypatch.setattr(common, "_tile_row_ranges", lambda n: _tile_row_ranges(n, 3))
    points = common._get_points(tmp_path, pixel_size=1.0, feature_names=["GENE_A", "GENE_B"])

    assert points.npartitions == 4  # tile 0,0 -> 3 chunks (3+3+1), tile 0,1 -> 1 empty partition
    assert len(points) == 7


def test_morphology_images_use_embedded_pyramid(tmp_path: Path) -> None:
    import tifffile

    rng = np.random.default_rng(0)
    full = rng.integers(0, 255, (2, 64, 96), dtype=np.uint8)
    for idx, name in enumerate(["dapi", "vim"]):
        with tifffile.TiffWriter(tmp_path / f"ch{idx:04d}_{name}.ome.tif", ome=True) as tif:
            levels = [full[:, :: 2**i, :: 2**i] for i in range(3)]
            tif.write(
                levels[0],
                tile=(16, 16),
                subifds=len(levels) - 1,
                metadata={"axes": "CYX", "Channel": {"Name": ["dapi", "vim"]}},
            )
            for level in levels[1:]:
                tif.write(level, tile=(16, 16), subfiletype=1)

    image = cast(
        DataTree,
        common._get_morphology_images(tmp_path, image_models_kwargs={"chunks": (1, 32, 32), "scale_factors": [2, 2]}),
    )
    assert list(image.children) == ["scale0", "scale1", "scale2"]
    assert [image[k]["image"].shape for k in image.children] == [(2, 64, 96), (2, 32, 48), (2, 16, 24)]
    # The lowest level is read from the stored pyramid, not downsampled from `scale0`.
    np.testing.assert_array_equal(np.asarray(image["scale2"]["image"])[0], full[0, ::4, ::4])


def _write_he(path: Path, affine: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(
        path / "he_image.ome.tif", np.zeros((3, 40, 60), dtype=np.uint8), ome=True, metadata={"axes": "CYX"}
    )
    pd.DataFrame(affine).to_csv(path / "he_imagealignment.csv", header=False, index=False)


def test_aligned_he_image_is_placed_by_its_affine(tmp_path: Path) -> None:
    from spatialdata.transformations import get_transformation

    from spatialdata_io.readers.xenium import _add_aligned_images

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    images = _add_aligned_images(tmp_path)
    assert list(images) == ["he_image"]
    he = images["he_image"]
    assert list(he.coords["c"].values) == ["r", "g", "b"]
    m = get_transformation(he, "global").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_allclose(m, affine)


def test_keypoints_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from spatialdata_io.readers.atera import _check_keypoints

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    src = np.array([[5.0, 5.0], [30.0, 10.0], [50.0, 35.0]])
    dst = (np.c_[src, np.ones(3)] @ affine.T)[:, :2]
    kp = pd.DataFrame({"fixedX": dst[:, 0], "fixedY": dst[:, 1], "alignmentX": src[:, 0], "alignmentY": src[:, 1]})
    kp.to_csv(tmp_path / "keypoints.csv", index=False)
    assert _check_keypoints(tmp_path) == pytest.approx(0.0, abs=1e-9)
    assert "misplaced" not in caplog.text
    kp.assign(fixedX=kp["fixedX"] + 50).to_csv(tmp_path / "keypoints.csv", index=False)
    assert _check_keypoints(tmp_path) == pytest.approx(50.0)
    assert "misplaced" in caplog.text
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "keypoints.csv", index=False)  # unknown columns: skipped
    assert _check_keypoints(tmp_path) is None
