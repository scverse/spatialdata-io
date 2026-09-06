"""Write a gene-major cell-by-gene matrix as one row group per gene.

This is what lets the overview render cheaply: selecting a gene fetches a single row
group holding that gene's non-zero cells, instead of touching any transcript data.

The schema matches Celldega's existing CBG reader exactly -- columns ``cell_id``
(the integer cell code, not a string barcode), ``expression``, ``gene``, plus
``gene_to_row_group`` / ``num_genes`` in the Parquet schema metadata.

One deliberate difference from Celldega's own writer: every gene in the catalog gets a
row group, in catalog order, even if it has no non-zero cells. That preserves the
invariant ``feature_code == cbg row group``, so a transcript's feature code and its
expression vector are addressed by the same integer. ``gene_to_row_group`` is still
written, so a client that looks the mapping up rather than assuming it works unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.regular_grid import DEFAULT_MAX_ROW_GROUPS_PER_FILE

__all__ = ["write_cbg_row_groups"]


def write_cbg_row_groups(
    table: Any,
    output_dir: str | Path,
    *,
    catalog: FeatureCatalog,
    cell_codes: dict[str, int] | None = None,
    layer: str | None = None,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write an AnnData table as gene-major CBG row groups.

    Parameters
    ----------
    table
        The annotating :class:`anndata.AnnData` table.
    output_dir
        Directory to write the chunk files into. Written atomically.
    catalog
        Feature catalog; its gene order defines the row-group order. Every gene in the
        catalog must be present in ``table.var_names``.
    cell_codes
        Mapping from ``obs`` name to integer cell code. Defaults to positional order,
        which is what :func:`write_shapes_regular_grid` uses by default.
    layer
        Optional ``table.layers`` key to read instead of ``table.X``.
    max_row_groups_per_file
        Genes per chunk file.
    compression
        Parquet compression codec.
    overwrite
        Replace ``output_dir`` if it exists.

    Returns
    -------
    The manifest fragment describing the written files.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")

    var_names = list(table.var_names)
    positions = {g: i for i, g in enumerate(var_names)}
    missing = [g for g in catalog.genes if g not in positions]
    if missing:
        raise ValueError(
            f"{len(missing)} catalog gene(s) are absent from table.var_names "
            f"(e.g. {missing[:3]}); the CBG would not cover every gene code."
        )

    if cell_codes is None:
        codes = np.arange(table.n_obs, dtype=np.uint32)
    else:
        unknown = [k for k in table.obs_names if k not in cell_codes]
        if unknown:
            raise ValueError(f"{len(unknown)} table cell(s) have no cell_code (e.g. {unknown[:3]})")
        codes = np.fromiter((cell_codes[k] for k in table.obs_names), dtype=np.uint32, count=table.n_obs)

    matrix = table.layers[layer] if layer is not None else table.X
    # CSC gives O(1) access to a single gene's column, which is the whole point here.
    csc = matrix.tocsc() if sp.issparse(matrix) else sp.csc_matrix(np.asarray(matrix))
    csc.sort_indices()

    schema = pa.schema(
        [
            pa.field("cell_id", pa.uint32()),
            pa.field("expression", pa.float32()),
            pa.field("gene", pa.string()),
        ]
    )

    gene_to_row_group = {gene: i for i, gene in enumerate(catalog.genes)}
    n_genes = len(catalog.genes)
    n_files = -(-n_genes // max_row_groups_per_file)
    width = len(str(n_files - 1)) if n_files > 1 else 1
    filenames = [f"chunk_{i:0{width}d}.parquet" for i in range(n_files)]

    schema = schema.with_metadata(
        {
            b"gene_to_row_group": json.dumps(gene_to_row_group).encode(),
            b"storage_mode": b"row_groups_cbg_chunked",
            b"num_genes": str(n_genes).encode(),
            b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
            b"profile": b"celldega_regular_grid_v1",
        }
    )

    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_values = 0
    try:
        writer: pq.ParquetWriter | None = None
        current_file = -1
        for row_group, gene in enumerate(catalog.genes):
            file_index = row_group // max_row_groups_per_file
            if file_index != current_file:
                if writer is not None:
                    writer.close()
                writer = pq.ParquetWriter(
                    staging / filenames[file_index], schema, compression=compression, write_statistics=False
                )
                current_file = file_index

            col = positions[gene]
            start, end = csc.indptr[col], csc.indptr[col + 1]
            cells = codes[csc.indices[start:end]]
            values = csc.data[start:end]
            # Explicit zeros can survive in a sparse matrix; the CBG stores non-zeros only.
            keep = values != 0
            cells, values = cells[keep], values[keep]
            n_values += len(values)

            assert writer is not None
            writer.write_table(
                pa.table(
                    {
                        "cell_id": pa.array(cells, type=pa.uint32()),
                        "expression": pa.array(values, type=pa.float32()),
                        "gene": pa.array([gene] * len(values), type=pa.string()),
                    },
                    schema=schema,
                )
            )
        if writer is not None:
            writer.close()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    return {
        "directory": output_dir.name,
        "files": filenames,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": n_genes,
        "num_genes": n_genes,
        "gene_to_row_group": gene_to_row_group,
        "n_values": n_values,
        "row_group_equals_feature_code": True,
    }
