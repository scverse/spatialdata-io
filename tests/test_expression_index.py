"""Gene-major expression access without downloading the whole matrix."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from spatialdata_io.experimental.expression_index import (
    CSC_LAYER,
    GENE_STAT_COLUMNS,
    add_csc_layer,
    add_gene_statistics,
    csc_chunk_size,
    write_csc_layer,
)


def _table() -> ad.AnnData:
    # 4 cells x 3 genes
    #   gene 0: [1, 0, 0, 3]
    #   gene 1: [0, 2, 0, 4]
    #   gene 2: [5, 0, 0, 0]
    dense = np.array(
        [[1.0, 0.0, 5.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        dtype=np.float32,
    )
    return ad.AnnData(X=sp.csr_matrix(dense))


def test_statistics_count_the_zero_cells() -> None:
    table = _table()
    add_gene_statistics(table)

    # gene 0 is [1, 0, 0, 3]: mean 1, E[x^2] = 2.5, var 1.5, 2 of 4 cells non-zero
    assert table.var["mean"].iloc[0] == pytest.approx(1.0)
    assert table.var["std"].iloc[0] == pytest.approx(np.sqrt(1.5))
    assert table.var["max"].iloc[0] == pytest.approx(3.0)
    assert table.var["non_zero"].iloc[0] == pytest.approx(0.5)

    # a gene present in a single cell still reports that cell's value as the max
    assert table.var["max"].iloc[2] == pytest.approx(5.0)
    assert table.var["non_zero"].iloc[2] == pytest.approx(0.25)


def test_statistics_match_a_dense_computation() -> None:
    rng = np.random.default_rng(0)
    dense = (rng.random((200, 17)) < 0.3) * rng.random((200, 17)) * 10
    table = ad.AnnData(X=sp.csr_matrix(dense.astype(np.float32)))
    add_gene_statistics(table)

    assert np.allclose(table.var["mean"], dense.mean(axis=0), atol=1e-5)
    assert np.allclose(table.var["std"], dense.std(axis=0), atol=1e-5)
    assert np.allclose(table.var["max"], dense.max(axis=0), atol=1e-5)
    assert np.allclose(table.var["non_zero"], (dense > 0).mean(axis=0), atol=1e-9)


def test_every_documented_column_is_written() -> None:
    table = _table()
    written = add_gene_statistics(table)
    assert written == list(GENE_STAT_COLUMNS)
    for column in GENE_STAT_COLUMNS:
        assert column in table.var


def test_csc_layer_holds_the_same_values_gene_major() -> None:
    table = _table()
    assert add_csc_layer(table) == CSC_LAYER

    csc = table.layers[CSC_LAYER]
    assert sp.isspmatrix_csc(csc)
    assert (csc.toarray() == table.X.toarray()).all()

    # The point of the layer: one gene is a contiguous slice, so a client reads two
    # chunks instead of the entire matrix.
    start, end = csc.indptr[0], csc.indptr[1]
    assert sorted(csc.data[start:end]) == [1.0, 3.0]
    assert sorted(csc.indices[start:end]) == [0, 3]


def test_a_dense_matrix_is_left_alone() -> None:
    # Dense is already addressable by column, so a copy would double the storage for
    # nothing.
    table = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    assert add_csc_layer(table) is None
    assert CSC_LAYER not in table.layers


def test_no_expression_is_not_an_error() -> None:
    table = ad.AnnData(obs=None, var=None, shape=(0, 0))
    assert add_gene_statistics(table) == []
    assert add_csc_layer(table) is None


def test_the_layer_round_trips_through_zarr(tmp_path) -> None:
    # The whole approach depends on the transpose surviving a write, since a client reads
    # it back out of the store rather than from memory.
    table = _table()
    add_csc_layer(table)
    add_gene_statistics(table)

    path = tmp_path / "t.zarr"
    table.write_zarr(path)
    reloaded = ad.read_zarr(path)

    assert sp.isspmatrix_csc(reloaded.layers[CSC_LAYER])
    assert (reloaded.layers[CSC_LAYER].toarray() == table.X.toarray()).all()
    assert reloaded.var["mean"].iloc[0] == pytest.approx(1.0)


def test_chunks_are_sized_by_the_average_gene() -> None:
    # AnnData chunks for whole-matrix reads: 162,948 non-zeros for Xenium pancreas against
    # ~6,915 for one gene, so a gene costs 24x what it needs. These are sized by the gene.
    assert csc_chunk_size(nnz=2_607_168, n_genes=377) == pytest.approx(13_830, rel=0.01)
    # ...but never so small that the chunk count explodes on a sparse, gene-rich matrix.
    assert csc_chunk_size(nnz=1000, n_genes=5000) == 1024
    assert csc_chunk_size(nnz=0, n_genes=0) == 1


def test_written_layer_is_chunked_for_single_gene_reads(tmp_path) -> None:
    rng = np.random.default_rng(0)
    dense = (rng.random((500, 40)) < 0.2) * rng.random((500, 40))
    table = ad.AnnData(X=sp.csr_matrix(dense.astype(np.float32)))
    csc = table.X.tocsc()

    path = tmp_path / "t.zarr"
    table.write_zarr(path)
    info = write_csc_layer(path, csc)

    import zarr

    group = zarr.open_group(str(path / "layers" / info["layer"]), mode="r")
    assert group.attrs["encoding-type"] == "csc_matrix"
    assert group.attrs["shape"] == list(csc.shape)

    # A gene must not span many chunks, or reading one costs several requests.
    chunk = group["data"].chunks[0]
    per_gene = info["nnz"] / info["genes"]
    assert chunk >= per_gene

    # indptr is always read whole, so it stays a single chunk.
    assert group["indptr"].chunks[0] == group["indptr"].shape[0]


def test_written_layer_reads_back_as_the_same_matrix(tmp_path) -> None:
    table = _table()
    csc = table.X.tocsc()
    path = tmp_path / "t.zarr"
    table.write_zarr(path)
    write_csc_layer(path, csc)

    reloaded = ad.read_zarr(path)
    assert sp.isspmatrix_csc(reloaded.layers[CSC_LAYER])
    assert (reloaded.layers[CSC_LAYER].toarray() == table.X.toarray()).all()
