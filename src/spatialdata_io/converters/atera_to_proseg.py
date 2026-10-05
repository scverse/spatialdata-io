from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io.readers._atera_common import _read_transcript_tile
from spatialdata_io.readers.atera import read_var

__all__ = ["atera_to_proseg"]

# Sentinel string for unassigned transcripts; matches `proseg`'s own `--xenium` preset default for
# `--cell-id-unassigned` (see `set_xenium_presets` in `proseg`'s `main.rs`), so the output can be fed
# to `proseg` with no further column-name/sentinel configuration.
_CELL_ID_UNASSIGNED = "UNASSIGNED"

# Column names/dtypes match what `proseg`'s `--xenium` preset expects from a `transcripts.parquet`
# (see `read_xenium_transcripts_parquet` in `proseg`'s `src/spatialdata_input.rs`/`src/sampler/transcripts.rs`):
# string columns must all share one arrow string type (here, `Utf8`, matching `feature_name`'s type,
# since `proseg` infers the type to use for every other string column from it), `overlaps_nucleus` must
# be `UInt8`, `transcript_id` a `UInt64`, and the coordinate/quality columns `Float32`.
_PROSEG_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.uint64()),
        ("cell_id", pa.string()),
        ("overlaps_nucleus", pa.uint8()),
        ("feature_name", pa.string()),
        ("x_location", pa.float32()),
        ("y_location", pa.float32()),
        ("z_location", pa.float32()),
        ("qv", pa.float32()),
        ("fov_name", pa.string()),
    ]
)


def atera_to_proseg(path: str | Path, output: str | Path) -> Path:
    """Convert a *10x Genomics Atera* ``transcripts.zarr.zip`` into a `proseg`-compatible parquet file.

    `currently, proseg <https://github.com/dcjones/proseg>`_ only supports reading a generic/custom-column-name
    parquet file for the `Xenium <https://www.10xgenomics.com/products/xenium>`_ platform preset (any
    other platform/custom column names require a CSV, not a parquet, input). This writes a parquet
    file with the column names/dtypes `proseg` expects from Xenium's ``transcripts.parquet``
    (``transcript_id``, ``cell_id``, ``overlaps_nucleus``, ``feature_name``,
    ``x_location``/``y_location``/``z_location``, ``qv``, ``fov_name``), so the result can be run
    directly as ``proseg --xenium <output>``.

    Coordinates are written in the dataset's raw (micron) units, matching Xenium's own
    ``x_location``/``y_location``/``z_location`` convention (*not* the pixel units of the "global"
    coordinate system `atera()` assembles its `SpatialData` elements into).

    The input is read and written one spatial tile at a time (mirroring how `atera`'s own transcripts
    reader is built), so converting even a dataset with tens of millions of transcripts does not require
    holding the whole table in memory at once.

    Parameters
    ----------
    path
        Path to the Atera dataset (the same path passed to `atera`).
    output
        Path to write the parquet file to.

    Returns
    -------
    `output`, as a `Path`.
    """
    path = Path(path)
    output = Path(output)

    feature_names = read_var(path)[str(AteraKeys.FEATURE_NAME)].astype(str).tolist()

    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        tile_keys = sorted(group.get_group(str(AteraKeys.GRID_GROUP)).group_keys())
    finally:
        store.close()

    writer = pq.ParquetWriter(output, _PROSEG_SCHEMA)
    try:
        offset = 0
        for tile_key in tile_keys:
            tile_df = _read_transcript_tile(path, tile_key, feature_names)
            n = len(tile_df)
            if n == 0:
                continue

            cell_id = tile_df[str(AteraKeys.CELL_ID)].to_numpy()
            cell_id_str = np.where(cell_id == -1, _CELL_ID_UNASSIGNED, cell_id.astype(str))

            table = pa.table(
                {
                    "transcript_id": np.arange(offset, offset + n, dtype=np.uint64),
                    "cell_id": cell_id_str,
                    "overlaps_nucleus": tile_df["overlaps_nucleus"].to_numpy().astype(np.uint8),
                    "feature_name": tile_df[str(AteraKeys.FEATURE_NAME)].astype(str).to_numpy(),
                    "x_location": tile_df[str(AteraKeys.TRANSCRIPTS_X)].to_numpy().astype(np.float32),
                    "y_location": tile_df[str(AteraKeys.TRANSCRIPTS_Y)].to_numpy().astype(np.float32),
                    "z_location": tile_df[str(AteraKeys.TRANSCRIPTS_Z)].to_numpy().astype(np.float32),
                    "qv": tile_df["quality_score"].to_numpy().astype(np.float32),
                    "fov_name": np.full(n, tile_key),
                },
                schema=_PROSEG_SCHEMA,
            )
            writer.write_table(table)
            offset += n
    finally:
        writer.close()

    return output
