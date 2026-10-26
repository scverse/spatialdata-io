from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.io import write_elem
from scipy.sparse import csr_matrix
from spatialdata.models import get_table_keys

from spatialdata_io._constants._constants import AteraKeys
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
