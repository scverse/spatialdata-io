# AnnData types `.X`/`.obs`/`.var` as broad unions; the assertions below intentionally rely on the concrete types.
# mypy: disable-error-code="union-attr,index,call-overload,arg-type,operator"
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.io import sparse_dataset
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from spatialdata_io.readers.atera import _get_table, read_table_for_cells, read_var
from spatialdata_io.writers.atera import write_atera_cell_feature_matrix

N_OBS, N_VAR = 25, 6
CELL_IDS = (np.arange(N_OBS, dtype=np.uint64) + 4_000_000_000) * 3
COLUMN_DTYPES = {"categorical", "categorical_i16", "categorical_i32", "integer", "float", "boolean", "string"}


def _make_table(x: Any = None) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(N_OBS, N_VAR)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_id": CELL_IDS,
            "centroid_column": rng.random(N_OBS),
            "centroid_row": rng.random(N_OBS),
            "cell_area": rng.random(N_OBS) * 100,
            "transcript_counts": counts.sum(1).astype(np.int64),
            "leiden_res_1.0": pd.Categorical(rng.integers(0, 3, N_OBS).astype(str)),
            # more than 127 categories: codes must not overflow int8
            "many": pd.Categorical([f"t{i}" for i in range(N_OBS)] * 1, categories=[f"t{i}" for i in range(200)]),
            "region": pd.Categorical(["cell_labels"] * N_OBS),
            "n_genes": (counts > 0).sum(1),
        },
        index=pd.Index([f"cell-{i}" for i in range(N_OBS)], name="barcode"),
    )
    var = pd.DataFrame(
        {
            "feature_name": [f"G{i}" for i in range(N_VAR)],
            "filtered": [False] * N_VAR,
            "highly_variable": [True, False] * (N_VAR // 2),
            "feature_type": ["Gene Expression"] * (N_VAR - 1) + ["Negative Control Probe"],
            "genome": ["hg38"] * N_VAR,
            "de_leiden_res_1.0_c0_logfc": rng.random(N_VAR).astype(np.float32),
        },
        index=pd.Index([f"ENS{i}" for i in range(N_VAR)], name="feature_id"),
    )
    table = ad.AnnData(X=csr_matrix(counts) if x is None else x(counts), obs=obs, var=var)
    table.obsm["X_umap"] = rng.random((N_OBS, 2)).astype(np.float32)
    table.obsm["spatial"] = obs[["centroid_column", "centroid_row"]].to_numpy()
    table.uns["leiden_res_1.0_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    table.uns["cell_feature_matrix_attrs"] = (
        '{"clusterings": [{"key_prefix": "leiden_res_1.0", "name": "Graph-based", "clustering_method": "Leiden",'
        ' "hyperparameters": "{}", "num_clusters": 3, "cluster_names": ["0", "1", "2"],'
        ' "cluster_prefixes": ["c0", "c1", "c2"], "diffexp": {"method": "Wilcoxon"}}]}'
    )
    return table


def _lazy(counts: np.ndarray) -> da.Array:
    blocks = [
        da.from_array(csr_matrix(counts[i : i + 7]), chunks=(7, N_VAR), asarray=False) for i in range(0, N_OBS, 7)
    ]
    return da.concatenate(blocks, axis=0)


def _open(path: Path) -> zarr.Group:
    return zarr.open_group(zarr.storage.ZipStore(path, read_only=True), mode="r")


def _validate_manifest(attrs: dict[str, Any]) -> None:
    """Hand-rolled check of the RFC's `anndata_zattrs/v0.1.0` JSON schema (no `jsonschema` dependency)."""
    assert set(attrs["version"]) == {"major", "minor"}
    props = attrs["matrix_properties"]
    assert set(props) == {"num_cells", "num_features", "major_axis", "feature_types"}
    assert props["major_axis"] in ("cell", "feature")
    for key in ("obs", "var", "obsm", "varm"):
        for col in attrs.get(key, []):
            assert col["dtype"] in COLUMN_DTYPES
            assert set(col) <= {"name", "dtype", "categories", "ordered", "length"}
            if col["dtype"].startswith("categorical"):
                assert len(col["categories"]) >= 1
            if key in ("obsm", "varm"):
                assert col["length"] >= 1
    for clustering in attrs.get("clusterings", []):
        required = ("key_prefix", "clustering_method", "hyperparameters", "num_clusters", "cluster_names")
        for req in (*required, "cluster_prefixes"):
            assert req in clustering
        assert clustering["clustering_method"] in ("Leiden", "Louvain", "k-means", "Hierarchical", "Other")


@pytest.mark.parametrize("x_kind", ["csr", "dask", "dense"])
@pytest.mark.parametrize("chunk_size", [7, 1_000_000])
def test_roundtrip(tmp_path: Path, x_kind: str, chunk_size: int) -> None:
    make = {"csr": None, "dask": _lazy, "dense": lambda c: c}[x_kind]
    table = _make_table(make)
    expected_x = (
        table.X.compute().toarray() if x_kind == "dask" else (table.X.toarray() if x_kind == "csr" else table.X)
    )
    path = write_atera_cell_feature_matrix(table, tmp_path, chunk_size=chunk_size, row_chunk_size=4)
    assert path == tmp_path / "cell_feature_matrix.zarr.zip"

    back, feature_names = _get_table(tmp_path, var_columns="all")
    assert np.array_equal(back.X.compute().toarray(), expected_x)
    assert feature_names == [f"G{i}" for i in range(N_VAR)]
    assert back.obs["cell_id"].tolist() == CELL_IDS.tolist()
    assert back.obs["label_id"].tolist() == list(range(1, N_OBS + 1))
    assert back.obs.index.tolist() == table.obs.index.tolist()
    assert back.obs["leiden_res_1.0"].astype(str).tolist() == table.obs["leiden_res_1.0"].astype(str).tolist()
    assert back.obs["many"].astype(str).tolist() == table.obs["many"].astype(str).tolist()
    assert np.allclose(back.obs["cell_area"], table.obs["cell_area"])
    assert np.allclose(back.obsm["spatial"], table.obsm["spatial"])
    assert np.allclose(back.obsm["X_umap"], table.obsm["X_umap"])
    assert back.var["feature_type"].astype(str).tolist() == table.var["feature_type"].tolist()
    assert "de_leiden_res_1.0_c0_logfc" in read_var(tmp_path).columns
    subset = read_table_for_cells(tmp_path, cell_ids=CELL_IDS[[3, 1]])
    assert subset.obs["cell_id"].tolist() == CELL_IDS[[3, 1]].tolist()


def test_writes_from_spatialdata_and_drops_reader_only_columns(tmp_path: Path) -> None:
    table = _make_table()
    sdata = SpatialData(tables={"table": table})
    table.obs["label_id"] = np.arange(1, N_OBS + 1, dtype=np.uint32)
    path = write_atera_cell_feature_matrix(sdata, tmp_path / "out.zarr.zip")
    root = _open(path)
    obs_columns = root["obs"].attrs["column-order"]
    assert not {"region", "n_genes", "label_id"} & set(obs_columns)
    assert "spatial" not in root["obsm"]
    # names and order are the table's own
    assert obs_columns == ["barcode", *[c for c in table.obs.columns if c not in ("region", "n_genes", "label_id")]]


def test_layout_dtypes_and_manifest(tmp_path: Path) -> None:
    path = write_atera_cell_feature_matrix(_make_table(), tmp_path)
    root = _open(path)
    attrs = dict(root.attrs)
    _validate_manifest(attrs)

    assert attrs["version"] == {"major": 0, "minor": 1}
    assert attrs["matrix_properties"] == {
        "num_cells": N_OBS,
        "num_features": N_VAR,
        "major_axis": "cell",
        "feature_types": ["Gene Expression", "Negative Control Probe"],
    }
    assert [c["name"] for c in attrs["obs"]] == root["obs"].attrs["column-order"]
    assert [c["name"] for c in attrs["var"]] == root["var"].attrs["column-order"]
    assert attrs["obsm"] == [{"name": "X_umap", "dtype": "float", "length": 2}]
    assert attrs["clusterings"][0]["key_prefix"] == "leiden_res_1.0"

    assert set(root.keys()) == {"X", "obs", "var", "obsm"}  # nothing beyond what the table has

    x = root["X"]
    assert x.attrs["encoding-type"] == "csr_matrix" and x.attrs["shape"] == [N_OBS, N_VAR]
    assert (x["data"].dtype, x["indices"].dtype, x["indptr"].dtype) == (np.int32, np.int64, np.int64)
    assert x["indptr"].shape == (N_OBS + 1,)

    obs = root["obs"]
    expected = {
        "cell_id": np.int64,
        "cell_area": np.float32,
        "transcript_counts": np.int64,
        "centroid_column": np.float32,
    }
    for name, dtype in expected.items():
        assert obs[name].dtype == dtype, name
    assert obs["barcode"].shape == (N_OBS,)
    assert obs.attrs["_index"] == "barcode"
    # categoricals are groups of codes/color/categories
    assert set(obs["leiden_res_1.0"].keys()) == {"codes", "color", "categories"}
    assert obs["leiden_res_1.0"]["color"].dtype == np.int32
    assert set(obs["many"].keys()) == {"codes", "categories"}  # no colors in `uns`, none invented
    assert obs["leiden_res_1.0"]["codes"].dtype == np.int8
    assert obs["many"]["codes"].dtype == np.int16
    manifest_dtypes = {c["name"]: c["dtype"] for c in attrs["obs"]}
    assert manifest_dtypes["leiden_res_1.0"] == "categorical" and manifest_dtypes["many"] == "categorical_i16"
    # colors are packed 0xRRGGBBAA, taken from `uns[f"{column}_colors"]`
    assert [int(c) & 0xFFFFFFFF for c in obs["leiden_res_1.0"]["color"][:]] == [0xFF0000FF, 0x00FF00FF, 0x0000FFFF]
    assert root["var"]["feature_type"]["codes"].shape == (N_VAR,)
    assert root["var"].attrs["_index"] == "feature_id"


def test_readable_with_plain_anndata(tmp_path: Path) -> None:
    table = _make_table()
    path = write_atera_cell_feature_matrix(table, tmp_path)
    root = _open(path)
    obs = ad.io.read_elem(root["obs"])
    assert obs.index.tolist() == table.obs.index.tolist()
    x = sparse_dataset(root["X"])[:]
    assert np.array_equal(x.toarray(), table.X.toarray())


def test_feature_major_is_csc(tmp_path: Path) -> None:
    table = _make_table()
    path = write_atera_cell_feature_matrix(table, tmp_path, major_axis="feature", chunk_size=7)
    root = _open(path)
    assert dict(root.attrs)["matrix_properties"]["major_axis"] == "feature"
    assert root["X"].attrs["encoding-type"] == "csc_matrix"
    assert root["X"]["indptr"].shape == (N_VAR + 1,)
    assert np.array_equal(sparse_dataset(root["X"])[:].toarray(), table.X.toarray())


def test_x_overflowing_int32_is_rejected(tmp_path: Path) -> None:
    table = _make_table()
    table.X = table.X.astype(np.float32) * 3e9
    with pytest.raises(ValueError, match="integer counts"):
        write_atera_cell_feature_matrix(table, tmp_path)
    assert not list(tmp_path.glob("*.zip"))
    assert not list(tmp_path.iterdir())  # no scratch files left behind


def test_missing_cell_id_is_rejected(tmp_path: Path) -> None:
    table = _make_table()
    del table.obs["cell_id"]
    with pytest.raises(ValueError, match="cell_id"):
        write_atera_cell_feature_matrix(table, tmp_path)


def test_x_layer_writes_raw_counts_when_x_is_normalised(tmp_path: Path) -> None:
    table = _make_table()
    counts = table.X.copy()
    table.layers["counts"] = counts
    table.X = table.X.astype(np.float32).multiply(0.37).tocsr()  # e.g. normalize_total + log1p in place
    with pytest.raises(ValueError, match="x_layer"):
        write_atera_cell_feature_matrix(table, tmp_path)
    with pytest.raises(KeyError, match="nope"):
        write_atera_cell_feature_matrix(table, tmp_path, x_layer="nope")

    write_atera_cell_feature_matrix(table, tmp_path, x_layer="counts")
    back, _ = _get_table(tmp_path, var_columns="all")
    assert np.array_equal(back.X.compute().toarray(), counts.toarray())


def test_colors_survive_a_roundtrip(tmp_path: Path) -> None:
    write_atera_cell_feature_matrix(_make_table(), tmp_path)
    back, _ = _get_table(tmp_path, var_columns="all")
    assert back.uns["leiden_res_1.0_colors"] == ["#ff0000ff", "#00ff00ff", "#0000ffff"]
    assert "many_colors" not in back.uns


def test_non_integer_x_is_rejected(tmp_path: Path) -> None:
    table = _make_table()
    table.X = table.X.astype(np.float32) * 0.5
    with pytest.raises(ValueError, match="integer counts"):
        write_atera_cell_feature_matrix(table, tmp_path)
