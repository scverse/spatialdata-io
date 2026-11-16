from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import shapely
import zarr
from anndata.io import write_elem
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import get_table_keys
from spatialdata.transformations import Affine
from tests.converters.test_atera_to_parquet import _write_atera_dataset
from xarray import DataTree

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io.readers import _atera_common as common
from spatialdata_io.readers._atera_common import _read_transcript_tile, _tile_row_ranges
from spatialdata_io.readers.atera import _get_table, read_table_for_cells, read_table_for_genes, read_var

# Atera `cell_id`s are large packed integers, unrelated to the `1..n_obs` values in the `cell_labels` raster.
CELL_IDS = [4_320_252_178, 4_975_876_403, 5_016_677_209, 5_217_424_078]
NEG_COUNTS = [0.0, 1.0, 0.0, 2.0]  # the negative control probe NEG_1, after the genes GENE_A and GENE_B


@pytest.fixture
def atera_path(tmp_path: Path) -> Path:
    _write_matrices(tmp_path, ["GENE_A", "GENE_B", "NEG_1"])
    _write_cells_transcripts_and_images(tmp_path, len(CELL_IDS))
    return tmp_path


def _write_matrices(path: Path, feature_names: list[str]) -> None:
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
            str(AteraKeys.FEATURE_NAME): feature_names,
            str(AteraKeys.VAR_FILTERED): [False] * 3,
            str(AteraKeys.VAR_HIGHLY_VARIABLE): [False] * 3,
            str(AteraKeys.VAR_FEATURE_TYPE): ["Gene Expression", "Gene Expression", "Negative Control Probe"],
            str(AteraKeys.VAR_GENOME): ["g"] * 3,
        },
        index=pd.Index(["ID_A", "ID_B", "ID_NEG"], dtype=object, name=str(AteraKeys.FEATURE_ID)),
    )
    x = csr_matrix(np.c_[np.arange(n * 2, dtype=np.float32).reshape(n, 2), NEG_COUNTS])
    for file in (AteraKeys.CELL_FEATURE_MATRIX_FILE, AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE):
        store = zarr.storage.ZipStore(path / file, mode="w")
        try:
            group = zarr.open_group(store, mode="w", zarr_format=2)
            write_elem(group, str(AteraKeys.X_GROUP), x if file == AteraKeys.CELL_FEATURE_MATRIX_FILE else x.tocsc())
            write_elem(group, str(AteraKeys.OBS_GROUP), obs)
            write_elem(group, str(AteraKeys.VAR_GROUP), var)
            # Real bundles also list the index (`barcode`/`feature_id`) in `column-order`, so it comes back from
            # `read_elem` as both the index and a column.
            for key, df in ((AteraKeys.OBS_GROUP, obs), (AteraKeys.VAR_GROUP, var)):
                elem = group[str(key)]
                elem.attrs["column-order"] = [str(df.index.name), *cast("list[str]", elem.attrs["column-order"])]
        finally:
            store.close()


PIXEL_SIZE = 0.5


def _write_cells_transcripts_and_images(path: Path, n: int) -> None:
    """The rest of a bundle: cell `i` is the 6 x 6 px square at ``10 i + 2``, with one 4-vertex polygon per mask."""
    import json

    import tifffile

    (path / AteraKeys.SPECS_FILE).write_text(json.dumps({"pixel_size": PIXEL_SIZE}))
    store = zarr.storage.ZipStore(path / AteraKeys.CELLS_FILE, mode="w")
    try:
        root = zarr.open_group(store, mode="w", zarr_format=2)
        mask = np.zeros((64, 96), dtype=np.uint32)
        for c in range(n):
            mask[10 * c + 2 : 10 * c + 8, 10 * c + 2 : 10 * c + 8] = c + 1
        bboxes = np.array([[10 * c + 2, 10 * c + 2, 10 * c + 8, 10 * c + 8] for c in range(n)], np.float32) * PIXEL_SIZE
        root.create_array("bboxes", data=bboxes)
        for i in ("0", "1"):
            root.create_array(f"masks/{i}", data=mask)
            square = bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]]
            root.create_array(f"polygon_sets/{i}/vertices", data=np.c_[square, np.zeros((n, 2), np.float32)])
            root.create_array(f"polygon_sets/{i}/num_vertices", data=np.full(n, 4, np.int32))
            root.create_array(f"polygon_sets/{i}/cell_index", data=np.arange(n, dtype=np.uint32))
            gridded = root.create_group(f"gridded_polygon_sets/{i}")
            gridded.attrs["grid_size"] = [100.0, 100.0]
            for level in range(5):
                gridded.create_array(f"{level}/0,0/num_vertices", data=np.full(n, 4, np.int32))
                gridded.create_array(f"{level}/0,0/cell_index", data=np.arange(n, dtype=np.uint32))
                corners = np.array([0, 0, 255, 0, 255, 255, 0, 255], np.uint8)
                gridded.create_array(f"{level}/0,0/relative_vertices", data=np.tile(corners, n))
    finally:
        store.close()

    # one tile; GENE_A owns rows 0-2 and GENE_B rows 3-5, the first transcript is unassigned
    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, mode="w")
    try:
        grid = zarr.open_group(store, mode="w", zarr_format=2).create_group(str(AteraKeys.GRID_GROUP))
        grid.attrs[str(AteraKeys.GRID_KEYS)] = ["0,0"]
        cell_id = np.zeros((6, 2), np.uint32)
        cell_id[0] = np.iinfo(np.uint32).max
        arrays = {
            "location": np.c_[np.arange(6), np.arange(6), np.zeros(6)].astype(np.float32),
            "gene_offset": np.array([[0, 3], [3, 6], [6, 6]], np.uint32),
            "cell_id": cell_id,
            "overlaps_nucleus": np.zeros((6, 1), np.uint8),
            "quality_score": np.full((6, 1), 30, np.float32),
            "codeword_identity": np.full((6, 1), 7, np.uint16),
        }
        for key, data in arrays.items():
            grid.create_array(f"0,0/{key}", data=data)
    finally:
        store.close()

    # each 2D file holds every channel, the "chNNNN" one carries the data; one 3D file per channel
    rng = np.random.default_rng(0)
    planes = rng.integers(0, 255, (2, 64, 96), dtype=np.uint8)
    (path / AteraKeys.MORPHOLOGY_2D_DIR).mkdir()
    (path / AteraKeys.MORPHOLOGY_3D_DIR).mkdir()
    for c, name in enumerate(["DAPI", "VIM"]):
        tifffile.imwrite(
            path / AteraKeys.MORPHOLOGY_2D_DIR / f"ch{c:04d}_{name.lower()}.ome.tif",
            planes,
            ome=True,
            tile=(32, 32),
            metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "VIM"]}},
        )
        tifffile.imwrite(
            path / AteraKeys.MORPHOLOGY_3D_DIR / f"ch{c:04d}_{name.lower()}_3d.ome.tif",
            np.stack([planes[c]] * 3),
            ome=True,
            tile=(32, 32),
            metadata={"axes": "ZYX", "Channel": {"Name": [name]}},
        )


def test_cell_labels_table_uses_label_id(atera_path: Path) -> None:
    table, _ = _get_table(atera_path, region="cell_labels")

    region, _, instance_key = get_table_keys(table)
    assert region == "cell_labels"
    assert instance_key == "label_id"
    # the `cell_labels` raster is numbered 1..n_obs by row position
    assert table.obs["label_id"].tolist() == [1, 2, 3, 4]
    assert table.obs["cell_id"].tolist() == CELL_IDS
    # genes detected per cell (cell 0 only has a zero in its first column); NEG_1 is not a gene
    assert table.obs["n_genes"].tolist() == [1, 2, 2, 2]


@pytest.mark.parametrize("table", ["_get_table", "read_table_for_cells"])
def test_x_holds_genes_and_obsm_the_controls(atera_path: Path, table: str) -> None:
    adata = _get_table(atera_path)[0] if table == "_get_table" else read_table_for_cells(atera_path, cell_ids=CELL_IDS)
    assert adata.var_names.tolist() == ["GENE_A", "GENE_B"]
    assert adata.obs["n_genes"].tolist() == [1, 2, 2, 2]
    controls = adata.obsm["control_counts"]
    assert isinstance(controls, pd.DataFrame)
    assert controls.columns.tolist() == ["NEG_1"]
    assert controls["NEG_1"].tolist() == NEG_COUNTS


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


@pytest.mark.parametrize("table", ["_get_table", "read_table_for_cells", "read_table_for_genes"])
def test_table_obs_index_does_not_shadow_the_barcode_column(atera_path: Path, table: str) -> None:
    # spatialdata's table joins call `obs.reset_index()` (spatialdata 0.8.0 _core/query/relational_query.py:259),
    # which fails with "cannot insert barcode, already exists" when the index carries the column's name
    if table == "_get_table":
        adata = _get_table(atera_path)[0]
    elif table == "read_table_for_cells":
        adata = read_table_for_cells(atera_path, cell_ids=CELL_IDS[:2])
    else:
        adata = read_table_for_genes(atera_path, "GENE_B")

    assert adata.obs.index.name is None
    assert adata.obs["barcode"].tolist() == adata.obs_names.tolist()
    cast(pd.DataFrame, adata.obs).reset_index()


def test_read_var_is_writable_and_indexed_like_the_table(atera_path: Path) -> None:
    var = read_var(atera_path)
    assert var.index.tolist() == ["GENE_A", "GENE_B", "NEG_1"]
    write_elem(zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=2), "var", var)


def test_tile_row_ranges() -> None:
    assert _tile_row_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert _tile_row_ranges(8, 4) == [(0, 4), (4, 8)]
    assert _tile_row_ranges(0, 4) == [(0, 0)]  # an empty tile keeps one (empty) partition


def test_chunked_tile_reads_match_whole_tile(tmp_path: Path) -> None:
    _write_atera_dataset(tmp_path, {"0,0": 7}, ["0,0"], ids={})
    feature_names = ["GENE_A", "GENE_B"]
    root = zarr.open_group(zarr.storage.ZipStore(tmp_path / AteraKeys.TRANSCRIPTS_FILE, read_only=True), mode="r")
    grid = root.get_group("grid")
    whole = _read_transcript_tile(grid, "0,0", feature_names)
    # GENE_A owns rows 0-2 and GENE_B rows 3-6, so the 2-row chunks straddle the gene boundary.
    parts = [_read_transcript_tile(grid, "0,0", feature_names, a, b) for a, b in _tile_row_ranges(7, 2)]

    assert pd.concat(parts, ignore_index=True).equals(whole)
    assert whole["feature_name"].tolist() == ["GENE_A"] * 3 + ["GENE_B"] * 4


def test_points_partitions_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_atera_dataset(tmp_path, {"0,0": 7, "0,1": 0}, ["0,0", "0,1"], ids={})
    monkeypatch.setattr(common, "_tile_row_ranges", lambda n: _tile_row_ranges(n, 3))
    points = common._get_points(tmp_path, pixel_size=1.0, feature_names=["GENE_A", "GENE_B"])

    assert points.npartitions == 4  # tile 0,0 -> 3 chunks (3+3+1), tile 0,1 -> 1 empty partition
    assert len(points) == 7


@pytest.mark.parametrize("scale_factors", [[2, 2], [4]])
def test_morphology_pyramid_is_built_from_level_0(atera_path: Path, scale_factors: list[int]) -> None:
    from spatialdata import bounding_box_query

    kwargs = {"chunks": (1, 32, 32), "scale_factors": scale_factors}
    image = cast(
        DataTree, common._get_morphology_images(atera_path / AteraKeys.MORPHOLOGY_2D_DIR, image_models_kwargs=kwargs)
    )
    factors = np.cumprod([1, *scale_factors])
    assert [image[f"scale{i}"]["image"].shape for i in range(len(factors))] == [(2, 64 // f, 96 // f) for f in factors]
    assert image["scale0"]["image"].coords["c"].values.tolist() == ["DAPI", "VIM"]
    # each scale has coordinates in scale0 px, so a query crops the same region at every scale
    crop = cast(
        DataTree,
        bounding_box_query(
            image, axes=("x", "y"), min_coordinate=[0, 0], max_coordinate=[32, 32], target_coordinate_system="global"
        ),
    )
    assert [crop[k]["image"].shape[1:] for k in crop.children] == [(32 // f, 32 // f) for f in factors]


def test_morphology_3d_names_channels_like_2d(atera_path: Path) -> None:
    image = common._get_morphology_images(atera_path / AteraKeys.MORPHOLOGY_3D_DIR, three_d=True)
    assert image.coords["c"].values.tolist() == ["DAPI", "VIM"]
    assert image.dims == ("c", "z", "y", "x")


def test_tiled_imread_reads_one_tile_per_task_and_pickles(atera_path: Path) -> None:
    import pickle

    import dask
    import tifffile

    file = atera_path / AteraKeys.MORPHOLOGY_2D_DIR / "ch0001_vim.ome.tif"
    image, dims = common._tiled_imread(file)
    assert dims == ("c", "y", "x")
    assert image.chunks == ((1, 1), (32, 32), (32, 32, 32))
    graph = pickle.loads(pickle.dumps(image))
    with dask.config.set(scheduler="processes", num_workers=2):
        np.testing.assert_array_equal(graph.compute(), tifffile.imread(file))


def _write_he(path: Path, affine: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(
        path / "S1_he_image.ome.tif", np.zeros((3, 40, 60), dtype=np.uint8), ome=True, metadata={"axes": "CYX"}
    )
    pd.DataFrame(affine).to_csv(path / "S1_he_imagealignment.csv", header=False, index=False)


@pytest.mark.parametrize("reader", ["atera", "xenium"])
def test_aligned_he_image_is_placed_by_its_affine(tmp_path: Path, reader: str) -> None:
    from spatialdata.transformations import get_transformation

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    images = _read_aligned(tmp_path, reader)
    assert list(images) == ["he_image"]
    he = images["he_image"]
    assert list(he.coords["c"].values) == ["r", "g", "b"]
    m = cast(Affine, get_transformation(he, "global")).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_allclose(m, affine)


def _write_keypoints(path: Path, affine: np.ndarray, name: str = "S1_keypoints.csv") -> pd.DataFrame:
    src = np.array([[5.0, 5.0], [30.0, 10.0], [50.0, 35.0]])
    dst = (np.c_[src, np.ones(3)] @ affine.T)[:, :2]
    kp = pd.DataFrame({"fixedX": dst[:, 0], "fixedY": dst[:, 1], "alignmentX": src[:, 0], "alignmentY": src[:, 1]})
    kp.to_csv(path / name, index=False)
    return kp


def test_keypoints_check(tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatialdata._logging import logger

    from spatialdata_io.readers.atera import _check_keypoints

    monkeypatch.setattr(logger, "propagate", True)

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    kp = _write_keypoints(tmp_path, affine)
    csv, kp_file = tmp_path / "S1_he_imagealignment.csv", tmp_path / "S1_keypoints.csv"
    assert _check_keypoints(csv, kp_file) == pytest.approx(0.0, abs=1e-9)
    assert "misplaced" not in caplog.text
    kp.assign(fixedX=kp["fixedX"] + 50).to_csv(kp_file, index=False)
    assert _check_keypoints(csv, kp_file) == pytest.approx(50.0)
    assert "misplaced" in caplog.text
    pd.DataFrame({"a": [1]}).to_csv(kp_file, index=False)
    with pytest.warns(UserWarning, match="expected columns"):
        assert _check_keypoints(csv, kp_file) is None


@pytest.mark.parametrize(
    "case", ["other_columns", "two_rows", "nan_rows", "collinear", "other_sample", "appledouble", "no_keypoints"]
)
def test_he_stays_unaligned_without_usable_keypoints(tmp_path: Path, case: str) -> None:
    import warnings

    from spatialdata.transformations import Identity, get_transformation

    from spatialdata_io.readers.xenium import _add_aligned_images

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    (tmp_path / "S1_he_imagealignment.csv").unlink()
    kp = _write_keypoints(tmp_path, affine)
    kp_file = tmp_path / "S1_keypoints.csv"
    if case == "other_columns":
        kp.set_axis(["a", "b", "c", "d"], axis=1).to_csv(kp_file, index=False)
    elif case == "two_rows":
        kp.iloc[:2].to_csv(kp_file, index=False)
    elif case == "nan_rows":
        kp.assign(fixedX=[1.0, np.nan, "x"]).to_csv(kp_file, index=False)
    elif case == "collinear":
        kp.assign(alignmentX=[1.0, 2.0, 3.0], alignmentY=[1.0, 2.0, 3.0]).to_csv(kp_file, index=False)
    elif case in ["other_sample", "appledouble"]:
        kp_file.rename(tmp_path / ("S2_keypoints.csv" if case == "other_sample" else "._S1_keypoints.csv"))
    else:
        kp_file.unlink()
    (tmp_path / "._S1_he_image.ome.tif").write_bytes(b"\x00" * 16)  # a macOS tar sidecar, not a TIFF

    expect_warning = case in ["other_columns", "two_rows", "nan_rows", "collinear"]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        images = _add_aligned_images(tmp_path)
    assert list(images) == ["he_image"]
    assert isinstance(get_transformation(images["he_image"], "global"), Identity)
    assert any("keypoint" in str(w.message) for w in record) == expect_warning


def test_affine_from_keypoints_reproduces_alignment(tmp_path: Path) -> None:
    from spatialdata_io.readers.xenium import _affine_from_keypoints

    # about the Atera v1 kidney alignment, with 0.2% anisotropy as in the preview scans
    affine = np.array([[0.0086, 1.2898, 914.1], [-1.2871, 0.0086, 32040.6], [0.0, 0.0, 1.0]])
    src = np.array([[16690.0, 14065.0], [22114.0, 25667.0], [13029.0, 25846.0], [16798.0, 35944.0]])
    dst = (np.c_[src, np.ones(len(src))] @ affine.T)[:, :2]
    kp = pd.DataFrame({"fixedX": dst[:, 0], "fixedY": dst[:, 1], "alignmentX": src[:, 0], "alignmentY": src[:, 1]})
    kp.to_csv(tmp_path / "S1_keypoints.csv", index=False)
    m = cast(Affine, _affine_from_keypoints(tmp_path / "S1_keypoints.csv")).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    np.testing.assert_allclose(m, affine, atol=1e-6)


def test_he_without_alignment_csv_is_placed_by_keypoints(tmp_path: Path) -> None:
    from spatialdata.transformations import get_transformation

    from spatialdata_io.readers.atera import _get_aligned_images

    affine = np.array([[0.0, 2.0, 10.0], [-2.0, 0.0, 300.0], [0.0, 0.0, 1.0]])
    _write_he(tmp_path, affine)
    (tmp_path / "S1_he_imagealignment.csv").unlink()
    src = np.array([[5.0, 5.0], [30.0, 10.0], [50.0, 35.0]])
    dst = (np.c_[src, np.ones(3)] @ affine.T)[:, :2]
    pd.DataFrame({"fixedX": dst[:, 0], "fixedY": dst[:, 1], "alignmentX": src[:, 0], "alignmentY": src[:, 1]}).to_csv(
        tmp_path / "S1_keypoints.csv", index=False
    )
    he = _get_aligned_images(tmp_path, {}, {})["he_image"]
    m = cast(Affine, get_transformation(he, "global")).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_allclose(m, affine, atol=1e-9)


def _read_aligned(path: Path, reader: str) -> dict:
    from spatialdata_io.readers.atera import _get_aligned_images
    from spatialdata_io.readers.xenium import _add_aligned_images

    return _get_aligned_images(path, {}, {}) if reader == "atera" else _add_aligned_images(path)


@pytest.mark.parametrize("reader", ["atera", "xenium"])
@pytest.mark.parametrize("planar", [False, True])
def test_aligned_he_image_planar_or_interleaved(tmp_path: Path, planar: bool, reader: str) -> None:
    import tifffile

    rgb = np.arange(3 * 40 * 60, dtype=np.uint32).reshape(3, 40, 60).astype(np.uint8)
    if planar:  # OME axes SYX, as in the Atera v1 Onboard 1.1.1 H&E
        tifffile.imwrite(tmp_path / "S1_he_image.ome.tif", rgb, ome=True, photometric="rgb", planarconfig="separate")
    else:  # YXS, as in the 1.0.4 kidney H&E
        tifffile.imwrite(tmp_path / "S1_he_image.ome.tif", np.moveaxis(rgb, 0, -1), ome=True, photometric="rgb")
    he = _read_aligned(tmp_path, reader)["he_image"]
    assert list(he.coords["c"].values) == ["r", "g", "b"]
    np.testing.assert_array_equal(np.asarray(he.transpose("c", "y", "x").data), rgb)


def test_atera_reads_writes_and_queries(atera_path: Path, tmp_path: Path) -> None:
    from spatialdata import bounding_box_query, read_zarr

    from spatialdata_io import atera

    sdata = atera(atera_path, transcripts=True)
    assert set(sdata.images) == {"morphology"}
    assert set(sdata.labels) == {"cell_labels", "nucleus_labels"}
    assert set(sdata.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    transcripts = sdata.points["transcripts"].compute()
    assert transcripts["cell_id"].tolist() == [-1, 0, 0, 0, 0, 0]  # the packed (uint32 max, uint32 max) is unassigned
    sdata.write(tmp_path / "out.zarr")
    back = read_zarr(tmp_path / "out.zarr")
    table, table_back = sdata.tables["table"], back.tables["table"]
    np.testing.assert_array_equal(cast(csr_matrix, table_back.X).toarray(), cast(Any, table.X).compute().toarray())
    pd.testing.assert_frame_equal(
        cast(pd.DataFrame, table_back.obsm["control_counts"]), cast(pd.DataFrame, table.obsm["control_counts"])
    )
    crop = cast(
        SpatialData,
        bounding_box_query(
            back, axes=("x", "y"), min_coordinate=[0, 0], max_coordinate=[15, 15], target_coordinate_system="global"
        ),
    )
    assert crop.tables["table"].obs["cell_id"].tolist() == CELL_IDS[:2]


def test_lazy_x_matches_the_store_and_pickles(atera_path: Path) -> None:
    import pickle

    x = cast(Any, _get_table(atera_path, row_chunk_size=3)[0].X)
    assert x.chunks[0] == (3, 1)
    expected = np.arange(len(CELL_IDS) * 2, dtype=np.float32).reshape(-1, 2)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(x)).compute().toarray(), expected)


def test_pixel_size_falls_back_to_the_ome_metadata_in_microns(tmp_path: Path) -> None:
    import tifffile

    (tmp_path / AteraKeys.MORPHOLOGY_2D_DIR).mkdir()
    tifffile.imwrite(
        tmp_path / AteraKeys.MORPHOLOGY_2D_DIR / "ch0000_dapi.ome.tif",
        np.zeros((1, 8, 8), np.uint8),
        ome=True,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 212.0,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 212.0,
            "PhysicalSizeYUnit": "nm",
        },
    )
    assert common._get_pixel_size(tmp_path) == pytest.approx(0.212)


@pytest.mark.parametrize("pipeline", ["BatchedCodecPipeline", "FusedCodecPipeline"])
def test_short_trailing_vlen_chunk_is_padded(tmp_path: Path, pipeline: str) -> None:
    import json
    import zipfile

    import zarr.core.codec_pipeline
    from numcodecs import VLenUTF8, get_codec

    if not hasattr(zarr.core.codec_pipeline, pipeline):
        pytest.skip(f"zarr {zarr.__version__} has no {pipeline}")
    root = tmp_path / "s.zarr"
    zarr.create_array(zarr.storage.LocalStore(root), shape=(5,), chunks=(3,), dtype=str, zarr_format=2)[:] = np.array(
        ["a", "bb", "ccc", "dd", "e"], dtype=object
    )
    meta = json.loads((root / ".zarray").read_text())
    compressor = get_codec(meta["compressor"]) if meta["compressor"] else None

    def zip_with_trailing_chunk(items: list[str]) -> Path:
        encoded = VLenUTF8().encode(np.array(items, dtype=object))
        (root / "1").write_bytes(compressor.encode(encoded) if compressor else encoded)
        with zipfile.ZipFile(tmp_path / "s.zip", "w") as zf:
            zf.writestr(".zgroup", json.dumps({"zarr_format": 2}))
            for f in root.iterdir():
                zf.write(f, f"s/{f.name}")
        return tmp_path / "s.zip"

    with zarr.config.set({"codec_pipeline.path": f"zarr.core.codec_pipeline.{pipeline}"}):
        # 10x's writer stores only the 2 items of the trailing chunk; zarr reads it at the nominal size of 3
        path = zip_with_trailing_chunk(["dd", "e"])
        with pytest.raises(ValueError, match="reshape"):
            zarr.open_group(zarr.storage.ZipStore(path, read_only=True), mode="r").get_array("s")[:]
        with common._open_zip(path) as group:
            assert np.asarray(group.get_array("s")[:]).tolist() == ["a", "bb", "ccc", "dd", "e"]
        path = zip_with_trailing_chunk(["dd"])  # one item short: corrupt, not padded
        with common._open_zip(path) as group, pytest.raises(ValueError, match="reshape"):
            group.get_array("s")[:]


def test_atera_reads_3d_morphology_with_default_kwargs(atera_path: Path) -> None:
    from spatialdata_io import atera

    image = atera(atera_path, morphology_3d_images=True).images["morphology_3d"]
    assert image["scale0"]["image"].dims == ("c", "z", "y", "x")
    assert image["scale0"]["image"].data.chunksize[:2] == (1, 1)
    assert [image[k]["image"].shape for k in image.children] == [(2, 3, 64 // f, 96 // f) for f in (1, 2, 4, 8, 16)]


def test_helpers_map_positions_through_the_bundle_not_the_table(atera_path: Path) -> None:
    from spatialdata_io import atera
    from spatialdata_io.readers.atera import read_cell_boundaries, read_transcripts_for_genes

    sdata = atera(atera_path)
    table = sdata.tables["table"]
    # an HVG-style gene subset and a reordered cell subset of the table
    sdata.tables["table"] = table[[2, 0], (table.var["feature_name"] == "GENE_B").to_numpy()].copy()
    points = read_transcripts_for_genes(sdata, atera_path, "GENE_B").points["transcripts_subset"].compute()
    assert points["x"].tolist() == [3.0, 4.0, 5.0]  # GENE_B owns rows 3-5 of the tile
    shapes = read_cell_boundaries(sdata, atera_path).shapes["cell_boundaries"]
    assert shapes.index.tolist() == CELL_IDS
    del sdata.tables["table"]  # nor do they need a table
    assert len(read_transcripts_for_genes(sdata, atera_path, ["GENE_A"]).points["transcripts_subset"]) == 3


@pytest.mark.parametrize("table", ["_get_table", "read_table_for_cells", "read_table_for_genes"])
def test_var_follows_the_xenium_layout(atera_path: Path, table: str) -> None:
    # The xenium() reader indexes var by gene symbol, with `gene_ids` (Ensembl) and `feature_types`; downstream
    # (gene search, QC) reads symbols from the index, ids from `gene_ids` and controls from `feature_types`.
    if table == "_get_table":
        adata = _get_table(atera_path)[0]
    elif table == "read_table_for_cells":
        adata = read_table_for_cells(atera_path, cell_ids=CELL_IDS[:2])
    else:
        adata = read_table_for_genes(atera_path, ["GENE_A", "GENE_B"])
    assert adata.var.index.tolist() == ["GENE_A", "GENE_B"]
    assert adata.var.index.name is None  # a name shared with the `feature_name` column breaks writes once suffixed
    assert adata.var["gene_ids"].tolist() == ["ID_A", "ID_B"]
    assert adata.var["feature_types"].tolist() == ["Gene Expression", "Gene Expression"]
    assert "feature_type" not in adata.var.columns


def test_duplicate_feature_names_are_made_unique_everywhere(atera_path: Path, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from spatialdata_io import atera
    from spatialdata_io.converters.atera_to_parquet import atera_to_parquet
    from spatialdata_io.readers.atera import read_transcripts_for_genes

    _write_matrices(atera_path, ["GENE_A", "GENE_A", "NEG_1"])
    sdata = atera(atera_path, transcripts=True)
    assert sdata.tables["table"].var_names.tolist() == ["GENE_A", "GENE_A-1"]
    transcripts = sdata.points["transcripts"].compute()
    assert transcripts["feature_name"].tolist() == ["GENE_A"] * 3 + ["GENE_A-1"] * 3
    sdata.write(tmp_path / "out.zarr")
    genes = read_table_for_genes(atera_path, ["GENE_A-1", "GENE_A"])
    np.testing.assert_array_equal(cast(csr_matrix, genes.X).toarray(), np.arange(8).reshape(4, 2)[:, ::-1])
    second = read_transcripts_for_genes(sdata, atera_path, "GENE_A-1").points["transcripts_subset"].compute()
    assert second["x"].tolist() == [3.0, 4.0, 5.0]
    table = pq.read_table(atera_to_parquet(atera_path, tmp_path / "t.parquet"))
    assert table["feature_name"].to_pylist() == ["GENE_A"] * 3 + ["GENE_A-1"] * 3


def test_empty_selections(atera_path: Path) -> None:
    from spatialdata_io import atera
    from spatialdata_io.readers.atera import read_cell_boundaries, read_transcripts_for_genes

    sdata = atera(atera_path)
    points = read_transcripts_for_genes(sdata, atera_path, "NEG_1").points["transcripts_subset"]  # no transcripts
    assert len(points) == 0
    assert points["feature_name"].dtype == "category"
    for table in [read_table_for_cells(atera_path, cell_ids=[]), read_table_for_cells(atera_path, bbox=(80, 0, 90, 5))]:
        assert table.shape == (0, 2)
        SpatialData(tables={"table": table})  # a valid element
    with pytest.raises(ValueError, match="No polygons"):
        read_cell_boundaries(sdata, atera_path, bbox=(80, 0, 90, 5))


def test_polygons_with_one_or_two_vertices_are_kept_empty(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from spatialdata._logging import logger

    monkeypatch.setattr(logger, "propagate", True)

    num_vertices = np.array([3, 2, 1, 0, 4])
    coords = np.arange(2 * num_vertices.sum(), dtype=np.float32).reshape(-1, 2)
    with caplog.at_level(logging.WARNING):
        shapes = common._polygons_to_shapes(coords, num_vertices, np.arange(5), None, 1.0, is_nucleus=False)
    assert shapes.geometry.is_empty.tolist() == [False, True, True, True, False]
    np.testing.assert_array_equal(shapely.get_coordinates(shapes.geometry.iloc[4])[:4], coords[6:10])
    assert "2 polygon(s) with 1 or 2 vertices" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "region"),
    [
        ({"cells_table": False, "transcripts": True}, None),
        ({"cells_labels": False, "cells_boundaries": False}, None),
        ({"cells_labels": False}, "cell_boundaries"),
        ({}, "cell_labels"),
    ],
)
def test_table_and_its_region_follow_the_flags(atera_path: Path, kwargs: dict, region: str | None) -> None:
    import warnings

    from spatialdata.models import TableModel

    from spatialdata_io import atera

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sdata = atera(atera_path, morphology_images=False, aligned_images=False, **kwargs)
    assert not [w for w in record if "which is not present in the SpatialData object" in str(w.message)]
    if kwargs.get("cells_table") is False:
        assert not sdata.tables
        assert len(sdata.points["transcripts"]) == 6
        assert sdata.shapes["cell_boundaries"].index.tolist() == CELL_IDS
    else:
        attrs = sdata.tables["table"].uns.get(TableModel.ATTRS_KEY, {})
        assert attrs.get("region") == region


def test_unique_feature_names_from_a_numpy_string_array() -> None:
    names = np.array(["G", "G", "NEG_1"], dtype=np.dtypes.StringDType())  # what read_elem gives on real bundles
    assert common._unique_feature_names(names).tolist() == ["G", "G-1", "NEG_1"]
