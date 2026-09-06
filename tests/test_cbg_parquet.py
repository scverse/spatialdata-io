"""Tests for the gene-major cell-by-gene writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
from anndata import AnnData

from spatialdata_io.experimental.cbg_parquet import write_cbg_row_groups
from spatialdata_io.experimental.feature_catalog import FeatureCatalog

GENES = ["GENEA", "GENEB", "GENEC"]
CELLS = ["cell-0", "cell-1", "cell-2", "cell-3"]

#: rows = cells, cols = genes. GENEC is deliberately all-zero.
DENSE = np.array(
    [
        [5.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [3.0, 7.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


@pytest.fixture
def table() -> AnnData:
    return AnnData(
        X=sp.csr_matrix(DENSE),
        obs=pd.DataFrame(index=CELLS),
        var=pd.DataFrame(index=GENES),
    )


@pytest.fixture
def catalog() -> FeatureCatalog:
    return FeatureCatalog.from_features_and_table([*GENES, "NegControlProbe_0001"], GENES)


@pytest.fixture
def written(tmp_path: Path, table: AnnData, catalog: FeatureCatalog) -> tuple[Path, dict]:
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(table, out, catalog=catalog)
    return out, manifest


def _row_group_for(directory: Path, manifest: dict, gene: str):
    rg = manifest["gene_to_row_group"][gene]
    file_index, local = divmod(rg, manifest["max_row_groups_per_file"])
    return pq.ParquetFile(directory / manifest["files"][file_index]).read_row_group(local)


# -- layout -------------------------------------------------------------------


def test_one_row_group_per_gene(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    total = sum(pq.ParquetFile(directory / f).metadata.num_row_groups for f in manifest["files"])
    assert total == len(GENES) == manifest["num_genes"]


def test_row_group_index_equals_feature_code(written: tuple[Path, dict], catalog: FeatureCatalog) -> None:
    """The invariant that lets one integer address both a transcript's gene and its CBG vector."""
    _, manifest = written
    for gene, rg in manifest["gene_to_row_group"].items():
        assert catalog.names.index(gene) == rg
    assert manifest["row_group_equals_feature_code"] is True


def test_controls_get_no_row_group(written: tuple[Path, dict]) -> None:
    _, manifest = written
    assert "NegControlProbe_0001" not in manifest["gene_to_row_group"]
    assert manifest["num_genes"] == len(GENES)


def test_schema_matches_celldega_reader(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    f = pq.ParquetFile(directory / manifest["files"][0])
    assert f.schema_arrow.names == ["cell_id", "expression", "gene"]
    meta = f.schema_arrow.metadata
    assert json.loads(meta[b"gene_to_row_group"]) == manifest["gene_to_row_group"]
    assert meta[b"storage_mode"] == b"row_groups_cbg_chunked"
    assert int(meta[b"num_genes"]) == len(GENES)


# -- values -------------------------------------------------------------------


def test_sparse_values_match_the_source_matrix(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    for col, gene in enumerate(GENES):
        rg = _row_group_for(directory, manifest, gene).to_pandas()
        expected = {i: DENSE[i, col] for i in range(len(CELLS)) if DENSE[i, col] != 0}
        assert dict(zip(rg["cell_id"], rg["expression"])) == expected
        assert set(rg["gene"]) <= {gene}


def test_zero_values_are_omitted(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    for gene in GENES:
        rg = _row_group_for(directory, manifest, gene)
        assert all(v != 0 for v in rg["expression"].to_pylist())


def test_all_zero_gene_is_an_empty_row_group(written: tuple[Path, dict]) -> None:
    """GENEC still occupies its slot so the feature_code invariant holds."""
    directory, manifest = written
    assert "GENEC" in manifest["gene_to_row_group"]
    assert _row_group_for(directory, manifest, "GENEC").num_rows == 0


def test_explicit_stored_zeros_are_dropped(tmp_path: Path, catalog: FeatureCatalog) -> None:
    """A sparse matrix can carry stored zeros; they are not expression."""
    m = sp.csr_matrix(DENSE)
    m[3, 0] = 0  # creates an explicit stored zero
    adata = AnnData(X=m, obs=pd.DataFrame(index=CELLS), var=pd.DataFrame(index=GENES))
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(adata, out, catalog=catalog)
    rg = _row_group_for(out, manifest, "GENEA").to_pandas()
    assert 3 not in set(rg["cell_id"])


def test_dense_matrix_is_supported(tmp_path: Path, catalog: FeatureCatalog) -> None:
    adata = AnnData(X=DENSE.copy(), obs=pd.DataFrame(index=CELLS), var=pd.DataFrame(index=GENES))
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(adata, out, catalog=catalog)
    rg = _row_group_for(out, manifest, "GENEB").to_pandas()
    assert dict(zip(rg["cell_id"], rg["expression"])) == {1: 2.0, 2: 7.0}


def test_layer_can_be_selected(tmp_path: Path, table: AnnData, catalog: FeatureCatalog) -> None:
    table.layers["scaled"] = sp.csr_matrix(DENSE * 10)
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(table, out, catalog=catalog, layer="scaled")
    rg = _row_group_for(out, manifest, "GENEA").to_pandas()
    assert dict(zip(rg["cell_id"], rg["expression"])) == {0: 50.0, 2: 30.0}


# -- cell codes ---------------------------------------------------------------


def test_cell_codes_default_to_table_order(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    rg = _row_group_for(directory, manifest, "GENEA").to_pandas()
    assert sorted(rg["cell_id"]) == [0, 2]


def test_explicit_cell_codes_are_honoured(tmp_path: Path, table: AnnData, catalog: FeatureCatalog) -> None:
    codes = {name: i for i, name in enumerate(reversed(CELLS))}
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(table, out, catalog=catalog, cell_codes=codes)
    rg = _row_group_for(out, manifest, "GENEA").to_pandas()
    assert sorted(rg["cell_id"]) == sorted([codes["cell-0"], codes["cell-2"]])


def test_missing_cell_code_is_reported(tmp_path: Path, table: AnnData, catalog: FeatureCatalog) -> None:
    with pytest.raises(ValueError, match="no cell_code"):
        write_cbg_row_groups(table, tmp_path / "cbg", catalog=catalog, cell_codes={"cell-0": 0})


def test_gene_missing_from_table_is_reported(tmp_path: Path, table: AnnData) -> None:
    bad = FeatureCatalog(names=("GENEA", "GHOST"), n_genes=2)
    with pytest.raises(ValueError, match="absent from table.var_names"):
        write_cbg_row_groups(table, tmp_path / "cbg", catalog=bad)


# -- chunking -----------------------------------------------------------------


def test_multi_file_chunking(tmp_path: Path, table: AnnData, catalog: FeatureCatalog) -> None:
    out = tmp_path / "cbg"
    manifest = write_cbg_row_groups(table, out, catalog=catalog, max_row_groups_per_file=2)
    assert manifest["files"] == ["chunk_0.parquet", "chunk_1.parquet"]
    assert [pq.ParquetFile(out / f).metadata.num_row_groups for f in manifest["files"]] == [2, 1]
    # the mapping must still resolve through the file split
    rg = _row_group_for(out, manifest, "GENEC")
    assert rg.num_rows == 0


def test_overwrite_guard(written: tuple[Path, dict], table: AnnData, catalog: FeatureCatalog) -> None:
    directory, _ = written
    with pytest.raises(FileExistsError):
        write_cbg_row_groups(table, directory, catalog=catalog)
