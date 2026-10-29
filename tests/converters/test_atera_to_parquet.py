from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import zarr
from anndata.io import write_elem
from numcodecs import Delta

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io.converters import atera_to_parquet as atera_module
from spatialdata_io.converters.atera_to_parquet import atera_to_parquet
from spatialdata_io.readers._atera_common import _tile_row_ranges

FEATURE_NAMES = ["GENE_A", "GENE_B"]


def _write_atera_dataset(
    path: Path, tile_sizes: dict[str, int], grid_keys: list[str], ids: dict[str, list[int]]
) -> None:
    """Write the minimal ``cell_feature_matrix.zarr.zip`` and ``transcripts.zarr.zip`` that `atera_to_parquet` reads.

    Each tile in ``tile_sizes`` gets that many transcripts, split between the two genes; a tile also gets an
    ``id`` array when it appears in ``ids``.
    """
    var = pd.DataFrame({str(AteraKeys.FEATURE_NAME): FEATURE_NAMES}, index=["ID_A", "ID_B"])
    store = zarr.storage.ZipStore(path / AteraKeys.CELL_FEATURE_MATRIX_FILE, mode="w")
    try:
        write_elem(zarr.open_group(store, mode="w", zarr_format=2), str(AteraKeys.VAR_GROUP), var)
    finally:
        store.close()

    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, mode="w")
    try:
        grid = zarr.open_group(store, mode="w", zarr_format=2).create_group(str(AteraKeys.GRID_GROUP))
        grid.attrs[str(AteraKeys.GRID_KEYS)] = grid_keys
        for tile_key, n in tile_sizes.items():
            tile = grid.create_group(tile_key)
            half = n // 2
            tile.create_array(str(AteraKeys.TRANSCRIPTS_LOCATION), data=np.zeros((n, 3), dtype=np.float32))
            tile.create_array(
                str(AteraKeys.TRANSCRIPTS_GENE_OFFSET), data=np.array([[0, half], [half, n]], dtype=np.uint32)
            )
            tile.create_array(str(AteraKeys.CELL_ID), data=np.zeros((n, 2), dtype=np.uint32))
            tile.create_array(str(AteraKeys.TRANSCRIPTS_OVERLAPS_NUCLEUS), data=np.zeros((n, 1), dtype=np.uint8))
            tile.create_array(str(AteraKeys.TRANSCRIPTS_QUALITY_SCORE), data=np.zeros((n, 1), dtype=np.float32))
            if tile_key in ids:
                tile.create_array(
                    str(AteraKeys.TRANSCRIPTS_ID),
                    data=np.array(ids[tile_key], dtype="<u8").reshape(-1, 1),
                    filters=[Delta(dtype="<u8", astype="<i8")],
                )
    finally:
        store.close()


def test_stored_transcript_ids(tmp_path: Path) -> None:
    ids = {"1,0": [40, 7, 1_000_000], "0,0": [], "0,1": [3, 99]}
    grid_keys = ["1,0", "0,0", "0,1"]
    _write_atera_dataset(tmp_path, {key: len(value) for key, value in ids.items()}, grid_keys, ids)

    table = pq.read_table(atera_to_parquet(tmp_path, tmp_path / "transcripts.parquet"))

    assert table["transcript_id"].to_pylist() == [i for key in grid_keys for i in ids[key]]


def test_generated_transcript_ids_follow_grid_keys(tmp_path: Path) -> None:
    tile_sizes = {"0,0": 2, "0,1": 0, "1,0": 3}
    grid_keys = ["1,0", "0,1", "0,0"]
    _write_atera_dataset(tmp_path, tile_sizes, grid_keys, ids={"1,0": [50, 51, 52], "0,1": []})

    table = pq.read_table(atera_to_parquet(tmp_path, tmp_path / "transcripts.parquet"))

    assert table["transcript_id"].to_pylist() == list(range(5))
    assert table["fov_name"].to_pylist() == ["1,0"] * 3 + ["0,0"] * 2


@pytest.mark.parametrize("with_stored_ids", [True, False])
def test_chunked_conversion_matches_unchunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_stored_ids: bool
) -> None:
    tile_sizes = {"0,0": 5, "0,1": 0, "1,0": 7}
    grid_keys = ["1,0", "0,1", "0,0"]
    ids = {"0,0": [9, 8, 7, 6, 5], "0,1": [], "1,0": list(range(100, 107))} if with_stored_ids else {}
    _write_atera_dataset(tmp_path, tile_sizes, grid_keys, ids)
    whole = pq.read_table(atera_to_parquet(tmp_path, tmp_path / "whole.parquet"))

    # 2-row chunks put chunk boundaries inside each gene's row range (the genes split a tile at n // 2).
    monkeypatch.setattr(atera_module, "_tile_row_ranges", lambda n: _tile_row_ranges(n, 2))
    chunked = pq.read_table(atera_to_parquet(tmp_path, tmp_path / "chunked.parquet"))

    assert chunked.equals(whole)
    assert chunked["feature_name"].to_pylist() == ["GENE_A"] * 3 + ["GENE_B"] * 4 + ["GENE_A"] * 2 + ["GENE_B"] * 3
