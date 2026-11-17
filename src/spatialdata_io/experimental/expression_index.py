"""Make gene-major expression reads cheap, without inventing a format.

``X`` is stored CSR (cell-major), which is right for the usual analysis access pattern of
"give me this cell's profile". A viewer wants column access: "give me this gene across all
cells". Fetching a column from CSR touches every chunk, so a client either downloads the
whole matrix or does nothing lazily at all -- 4.5 MB for Xenium pancreas, 54.8 MB for Prime
skin, and unbounded beyond that.

Two additions fix it, both plain AnnData that round-trips and that any tool can use:

``var`` statistics
    ``mean``, ``std``, ``max`` and ``non_zero`` per gene. Without these a client has to read
    the whole matrix just to populate a gene list, however the matrix is laid out.

a CSC layer
    A gene-major copy of ``X``. One gene becomes ``indptr[g]:indptr[g+1]`` -- a slice of one
    or two chunks, about 6,600 non-zeros for skin -- instead of the entire matrix.

The cost is a second copy of the non-zeros. Historical rebuilt skin measurements were
about 104 MB for this CSC layer versus 150.3 MB for the previous gene-major Parquet.
The layer's contents round-trip as AnnData, but custom chunk sizes need not survive an
ordinary rewrite. Fixed-length chunks target average gene density, not gene boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "add_gene_statistics",
    "add_csc_layer",
    "write_csc_layer",
    "GENE_STAT_COLUMNS",
    "CSC_LAYER",
]

#: ``var`` columns written by :func:`add_gene_statistics`.
GENE_STAT_COLUMNS = ("mean", "std", "max", "non_zero")

#: Layer holding the gene-major copy of ``X``.
CSC_LAYER = "X_csc"


def _column_stats(matrix: Any, n_genes: int) -> dict[str, NDArray[np.float64]]:
    """Per-gene mean, std, max and non-zero fraction, over all cells.

    Zeros count towards the denominator: they are measurements, not missing data, so ``std``
    is the population standard deviation across every cell and ``non_zero`` is the fraction
    of cells with a non-zero value. This matches what ``celldega.pre`` writes, and the
    client computes the same figures when these columns are absent.
    """
    import scipy.sparse as sp

    n_cells = matrix.shape[0]
    csc = matrix.tocsc() if sp.issparse(matrix) else sp.csc_matrix(matrix)

    total = np.asarray(csc.sum(axis=0)).ravel().astype(np.float64)
    squared = np.asarray(csc.multiply(csc).sum(axis=0)).ravel().astype(np.float64)
    counts = np.diff(csc.indptr).astype(np.float64)

    maxima = np.zeros(n_genes, dtype=np.float64)
    for gene in range(csc.shape[1]):
        start, end = csc.indptr[gene], csc.indptr[gene + 1]
        if end > start:
            maxima[gene] = float(csc.data[start:end].max())

    mean = total / n_cells
    # Var[x] = E[x^2] - E[x]^2, clipped because float error can make it slightly negative.
    variance = np.maximum(0.0, squared / n_cells - mean**2)

    return {
        "mean": mean,
        "std": np.sqrt(variance),
        "max": maxima,
        "non_zero": counts / n_cells,
    }


def add_gene_statistics(table: Any) -> list[str]:
    """Add per-gene summary statistics to ``table.var``, in place.

    Returns
    -------
    The column names written.
    """
    if table.X is None:
        return []

    stats = _column_stats(table.X, table.n_vars)
    for name in GENE_STAT_COLUMNS:
        table.var[name] = stats[name]
    return list(GENE_STAT_COLUMNS)


def add_csc_layer(table: Any, layer: str = CSC_LAYER) -> str | None:
    """Add a gene-major (CSC) copy of ``X`` as a layer, in place.

    Returns
    -------
    The layer name, or ``None`` when there is nothing to transpose.
    """
    import scipy.sparse as sp

    if table.X is None:
        return None
    if not sp.issparse(table.X):
        # A dense X is already randomly addressable by column; a CSC copy would only
        # double the storage for no gain.
        return None

    table.layers[layer] = table.X.tocsc()
    return layer


#: Target non-zeros per chunk when none is given: a few genes' worth, so one gene costs one
#: or two chunks instead of a slice of a huge one.
DEFAULT_GENES_PER_CHUNK = 2


def csc_chunk_size(nnz: int, n_genes: int, genes_per_chunk: int = DEFAULT_GENES_PER_CHUNK) -> int:
    """Chunk length that keeps a single gene to about one chunk.

    AnnData's default chunking is sized for whole-matrix reads -- 162,948 non-zeros per
    chunk for Xenium pancreas, against ~6,915 for one gene. Reading a gene then costs 24x
    what it needs. Sizing chunks by the average gene fixes that; the cost is more, smaller
    chunks, which compress slightly worse.
    """
    if n_genes <= 0 or nnz <= 0:
        return max(1, nnz)
    per_gene = max(1, nnz // n_genes)
    return max(1024, per_gene * genes_per_chunk)


def write_csc_layer(
    table_path: Any,
    csc: Any,
    *,
    layer: str = CSC_LAYER,
    chunk: int | None = None,
) -> dict[str, Any]:
    """Write a gene-major matrix into a written table's ``layers``, chunked for gene reads.

    Written directly rather than through AnnData because the chunking is the whole point,
    and AnnData sizes chunks for whole-matrix access.

    Parameters
    ----------
    table_path
        Path of the already-written table group inside the store.
    csc
        The gene-major matrix.
    layer
        Layer name under ``layers/``.
    chunk
        Non-zeros per chunk. Defaults to :func:`csc_chunk_size`.

    Returns
    -------
    A description of what was written, for the manifest.
    """
    from pathlib import Path

    import zarr

    n_genes = csc.shape[1]
    nnz = int(csc.nnz)
    chunk = chunk or csc_chunk_size(nnz, n_genes)

    root = zarr.open_group(str(Path(table_path) / "layers"), mode="a")
    if layer in root:
        del root[layer]
    group = root.create_group(layer)

    # AnnData's own encoding, so the layer round-trips as an ordinary csc_matrix.
    group.attrs["encoding-type"] = "csc_matrix"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = list(csc.shape)

    for name, values, dtype in (
        ("data", csc.data, np.float32),
        ("indices", csc.indices, np.int32),
        ("indptr", csc.indptr, np.int32),
    ):
        arr = np.asarray(values, dtype=dtype)
        # indptr is one value per gene and always read whole, so it stays a single chunk.
        chunks = (len(arr),) if name == "indptr" else (min(chunk, max(1, len(arr))),)
        group.create_array(name, shape=arr.shape, chunks=chunks, dtype=dtype)[:] = arr

    # AnnData and SpatialData read through consolidated metadata, so a group added after
    # the table was written is invisible to them until the index is refreshed. Browsers
    # are unaffected -- zarrita reads each node directly -- which makes this exactly the
    # kind of difference that shows up only on the Python side.
    zarr.consolidate_metadata(zarr.open_group(str(Path(table_path)), mode="a").store)

    return {"layer": layer, "chunk": chunk, "nnz": nnz, "genes": n_genes}
