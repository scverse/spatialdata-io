"""Write a *10x Genomics Atera* ``cell_feature_matrix.zarr.zip``.

The layout reproduces the public ``Atera_v1_WTA_Kidney_non_diseased_FFPE_outs`` bundle (which follows the
"cell feature anndata" RFC): a Zarr v2 store serialised as an (uncompressed, since every chunk is already
Blosc-compressed) ZIP archive, with the root ``.zattrs`` manifest and the standard AnnData
``encoding-type``/``encoding-version``/``_index``/``column-order`` attrs, so that ``anndata.io.read_elem``
(and hence `spatialdata_io.atera`) reads it back unchanged.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from scipy import sparse
from spatialdata import SpatialData

from spatialdata_io._constants._constants import AteraKeys

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anndata import AnnData

__all__ = ["write_atera_cell_feature_matrix"]

_SCHEMA_VERSION = {"major": 0, "minor": 1}
_ANNDATA_ARRAY = {"encoding-type": "array", "encoding-version": "0.2.0"}
_ANNDATA_STRING = {"encoding-type": "string-array", "encoding-version": "0.2.0"}
_CHUNKS_PER_FLUSH = 4
_DEFAULT_MAX_WORKERS = 4
_INT32_MAX = np.iinfo(np.int32).max
_INT64_MAX = np.iinfo(np.int64).max

# Table columns that only exist on the in-memory `SpatialData` table (added by the reader), not in the bundle.
_TABLE_ONLY_OBS = {"region", "n_genes", str(AteraKeys.LABEL_ID)}


def write_atera_cell_feature_matrix(
    sdata: SpatialData | AnnData,
    path: str | Path,
    *,
    table_name: str = "table",
    x_layer: str | None = None,
    major_axis: Literal["cell", "feature"] = "cell",
    chunk_size: int = 1_000_000,
    row_chunk_size: int = 50_000,
    n_threads: int | None = None,
) -> Path:
    """Write a table out as ``cell_feature_matrix.zarr.zip``, in the layout of the public Atera bundle.

    The table's `obs`/`var` columns are written as they are (names, order, dtypes), except for the
    columns `atera` adds on read (``label_id``, ``n_genes``, ``region``) and ``obsm["spatial"]``.

    Parameters
    ----------
    sdata
        The `SpatialData` object holding the table (or the `AnnData` table itself), e.g. as returned by
        `spatialdata_io.atera`. ``X`` may be a lazy `dask` array of CSR blocks, a scipy sparse matrix or a
        dense array; the lazy case is streamed in row blocks so the full matrix is never held in memory.
    path
        Either the ``.zarr.zip`` file to write, or a directory (the file is then named
        ``cell_feature_matrix.zarr.zip``).
    table_name
        Which table of `sdata` to write.
    x_layer
        Write ``table.layers[x_layer]`` as ``X`` instead of ``table.X``. The bundle's ``X`` is raw integer
        counts, so use this (e.g. ``x_layer="counts"``) when ``table.X`` has been normalised in place.
    major_axis
        ``"cell"`` writes ``X`` as CSR (chunked by cell); ``"feature"`` writes the transposed layout (CSC),
        which needs the whole matrix in memory.
    chunk_size
        Number of elements per zarr chunk of the 1-D arrays (`X` ``data``/``indices``, numeric columns).
        String arrays are always a single chunk (zarr can't decode a ragged trailing variable-length chunk).
    row_chunk_size
        Rows per block when streaming a non-dask ``X``.
    n_threads
        How many row blocks of a lazy ``X`` to compute concurrently, and how many threads compress chunks.
        Defaults to ``min(cpu_count, 4)``: lazy blocks can be large, so more threads mostly adds memory pressure.

    Returns
    -------
    The path of the written ``.zarr.zip`` file.
    """
    if major_axis not in ("cell", "feature"):
        raise ValueError(f"`major_axis` must be 'cell' or 'feature', got {major_axis!r}.")
    table = sdata.tables[table_name] if isinstance(sdata, SpatialData) else sdata
    path = Path(path)
    if path.suffix != ".zip":
        path = path / str(AteraKeys.CELL_FEATURE_MATRIX_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    if x_layer is not None and x_layer not in table.layers:
        raise KeyError(f"`x_layer={x_layer!r}` is not in the table's layers ({list(table.layers)}).")
    x = table.X if x_layer is None else table.layers[x_layer]
    obs, obs_colors = _prepare_obs(table)
    var, var_colors = _prepare_var(table)
    n_obs, n_var = table.shape
    if tuple(x.shape) != (n_obs, n_var):  # pragma: no cover - AnnData enforces this
        raise ValueError("`X` shape does not match `obs`/`var`.")

    # lz4 compresses ~1.5x faster than the blosclz the 10x bundles use, at the same ratio. The codec is
    # recorded in each Blosc frame, so any zarr/numcodecs reader decodes it unchanged.
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    # The small groups (obs/var/obsm/varm/manifest) are built in an in-memory zarr store and copied into
    # the zip at the end. The big `X` arrays are compressed by us and streamed straight into the zip as
    # raw chunk entries, with their (final-shape) metadata coming from the in-memory store: zip entries
    # can't be replaced, so this avoids resizing arrays as the row blocks arrive.
    entries: dict[str, Any] = {}
    root = zarr.open_group(zarr.storage.MemoryStore(entries), mode="w", zarr_format=2)
    tmp_zip = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            obs_manifest = _write_dataframe(
                root, str(AteraKeys.OBS_GROUP), obs, "barcode", obs_colors, chunk_size, compressor
            )
            var_manifest = _write_dataframe(
                root, str(AteraKeys.VAR_GROUP), var, "feature_id", var_colors, chunk_size, compressor
            )
            obsm = _embeddings(table.obsm, n_obs, "obsm")
            obsm_manifest = _write_embeddings(root, str(AteraKeys.OBSM_GROUP), obsm, chunk_size, compressor)
            varm = _embeddings(table.varm, n_var, "varm")
            varm_manifest = _write_embeddings(root, str(AteraKeys.VARM_GROUP), varm, chunk_size, compressor)

            _write_x(root, zf, x, (n_obs, n_var), major_axis, chunk_size, row_chunk_size, n_threads, compressor)

            manifest = _build_manifest(
                table, obs_manifest, var_manifest, obsm_manifest, varm_manifest, var, major_axis
            )
            root.attrs.update(manifest)
            for key, buffer in entries.items():
                zf.writestr(key, buffer.to_bytes())
        os.replace(tmp_zip, path)
    finally:
        tmp_zip.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------------------------------
# obs / var preparation
# --------------------------------------------------------------------------------------------------


def _as_int(values: pd.Series, dtype: type[np.integer], limit: int, name: str) -> np.ndarray:
    arr = values.to_numpy()
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.array_equal(arr, np.rint(arr)):
            raise ValueError(f"column {name!r} must hold integers.")
    if arr.size and (arr.max() > limit or arr.min() < -limit - 1):
        raise ValueError(f"column {name!r} does not fit in {np.dtype(dtype).name}.")
    return arr.astype(dtype)


def _hex_to_int32(color: str) -> int:
    """Pack ``#rrggbb[aa]`` as the RFC's ``0xRRGGBBAA`` int32 (alpha defaults to opaque)."""
    h = color.lstrip("#")
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        raise ValueError(f"Invalid color {color!r}; expected '#rrggbb' or '#rrggbbaa'.")
    return int(np.uint32(int(h, 16)).view(np.int32))


def _colors_for(table: AnnData, column: str, k: int) -> np.ndarray | None:
    """Per-category colors from ``uns[f"{column}_colors"]`` (as `atera` fills it), or ``None`` if absent."""
    given = table.uns.get(f"{column}_colors")
    if given is None or len(given) != k:
        return None
    return np.array([_hex_to_int32(str(c)) for c in given], dtype=np.int32)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a dataframe's columns for writing: strings as `object`, floats as float32, categoricals as is."""
    columns: dict[str, Any] = {}
    for name in df.columns:
        col = df[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            columns[str(name)] = col
        elif pd.api.types.is_bool_dtype(col):
            columns[str(name)] = col.to_numpy(bool)
        elif pd.api.types.is_float_dtype(col):
            columns[str(name)] = col.to_numpy(np.float32)
        elif pd.api.types.is_numeric_dtype(col):
            columns[str(name)] = col.to_numpy()
        else:
            columns[str(name)] = col.astype(str).to_numpy(dtype=object)
    return pd.DataFrame(columns)


def _colors(table: AnnData, df: pd.DataFrame) -> dict[str, np.ndarray]:
    colors = {}
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            found = _colors_for(table, str(column), len(df[column].cat.categories))
            if found is not None:
                colors[str(column)] = found
    return colors


def _prepare_obs(table: AnnData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    obs = table.obs.reset_index(drop=True)
    if "barcode" not in obs.columns:
        obs.insert(0, "barcode", table.obs.index.astype(str))
    obs = obs.drop(columns=[c for c in obs.columns if c in _TABLE_ONLY_OBS])
    if str(AteraKeys.CELL_ID) not in obs.columns:
        raise ValueError(f"The table's obs must have a {AteraKeys.CELL_ID!r} column.")
    obs[str(AteraKeys.CELL_ID)] = _as_int(obs[str(AteraKeys.CELL_ID)], np.int64, _INT64_MAX, str(AteraKeys.CELL_ID))
    df = _clean_columns(obs)
    return df, _colors(table, df)


def _prepare_var(table: AnnData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    var = table.var.reset_index(drop=True)
    if "feature_id" not in var.columns:
        var.insert(0, "feature_id", table.var.index.astype(str))
    for required in (str(AteraKeys.FEATURE_NAME), str(AteraKeys.VAR_FEATURE_TYPE)):
        if required not in var.columns:
            raise ValueError(f"The table's var must have a {required!r} column.")
    df = _clean_columns(var)
    for name in (str(AteraKeys.VAR_FEATURE_TYPE), str(AteraKeys.VAR_GENOME)):
        if name in df.columns and not isinstance(df[name].dtype, pd.CategoricalDtype):
            df[name] = df[name].astype("category")
    return df, _colors(table, df)


# --------------------------------------------------------------------------------------------------
# zarr writing
# --------------------------------------------------------------------------------------------------


def _create(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    chunks: tuple[int, ...],
    compressor: Blosc,
    attrs: dict[str, str],
) -> None:
    arr = group.create_array(
        name, shape=values.shape, dtype=values.dtype, chunks=chunks, compressors=compressor, overwrite=True
    )
    if values.size:
        arr[...] = values
    arr.attrs.update(attrs)


def _write_1d(group: zarr.Group, name: str, values: np.ndarray, chunk_size: int, compressor: Blosc) -> None:
    if values.dtype.kind in "OUT":
        values = np.asarray(values, dtype=np.dtypes.StringDType())
        _create(group, name, values, (max(len(values), 1),), compressor, _ANNDATA_STRING)
    else:
        _create(group, name, values, (max(min(len(values), chunk_size), 1),), compressor, _ANNDATA_ARRAY)


def _codes_dtype(k: int) -> type[np.integer]:
    """Smallest signed integer type that holds the codes of `k` categories (the RFC's is int8)."""
    return np.int8 if k <= 127 else np.int16 if k <= 32767 else np.int32


def _descriptor(name: str, col: pd.Series) -> dict[str, Any]:
    if isinstance(col.dtype, pd.CategoricalDtype):
        # Like the public bundle, wider codes are flagged in the dtype (`categorical_i16`, ...): the
        # plain `categorical` of the RFC means int8 codes.
        bits = np.dtype(_codes_dtype(len(col.cat.categories))).itemsize * 8
        return {
            "name": name,
            "dtype": "categorical" if bits == 8 else f"categorical_i{bits}",
            "categories": [str(c) for c in col.cat.categories],
            "ordered": bool(col.cat.ordered),
        }
    if pd.api.types.is_bool_dtype(col):
        return {"name": name, "dtype": "boolean"}
    if pd.api.types.is_integer_dtype(col):
        return {"name": name, "dtype": "integer"}
    if pd.api.types.is_float_dtype(col):
        return {"name": name, "dtype": "float"}
    return {"name": name, "dtype": "string"}


def _write_dataframe(
    parent: zarr.Group,
    name: str,
    df: pd.DataFrame,
    index_name: str,
    colors: dict[str, np.ndarray],
    chunk_size: int,
    compressor: Blosc,
) -> list[dict[str, Any]]:
    group = parent.create_group(name, overwrite=True)
    manifest = []
    for column in df.columns:
        col = df[column]
        if isinstance(col.dtype, pd.CategoricalDtype):
            k = len(col.cat.categories)
            cat_group = group.create_group(column, overwrite=True)
            cat_group.attrs.update(
                {"encoding-type": "categorical", "encoding-version": "0.2.0", "ordered": bool(col.cat.ordered)}
            )
            codes_dtype = _codes_dtype(k)
            _write_1d(cat_group, "codes", col.cat.codes.to_numpy().astype(codes_dtype), chunk_size, compressor)
            if column in colors:
                _write_1d(cat_group, "color", colors[column], chunk_size, compressor)
            categories = col.cat.categories.astype(str).to_numpy(dtype=object)
            _write_1d(cat_group, "categories", categories, chunk_size, compressor)
        else:
            _write_1d(group, column, col.to_numpy(), chunk_size, compressor)
        manifest.append(_descriptor(column, col))
    group.attrs.update(
        {
            "_index": index_name,
            "column-order": list(df.columns),
            "encoding-type": "dataframe",
            "encoding-version": "0.2.0",
        }
    )
    return manifest


def _embeddings(mapping: Any, length: int, kind: str) -> dict[str, np.ndarray]:
    out = {}
    for key in mapping:
        if key == "spatial" and kind == "obsm":
            continue  # re-derived from the centroids by the reader
        value = mapping[key]
        arr = np.asarray(value.to_numpy() if isinstance(value, pd.DataFrame) else value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != length:
            raise ValueError(f"{kind}[{key!r}] has {arr.shape[0]} rows, expected {length}.")
        out[str(key)] = arr
    return out


def _write_embeddings(
    parent: zarr.Group, name: str, arrays: dict[str, np.ndarray], chunk_size: int, compressor: Blosc
) -> list[dict[str, Any]]:
    if not arrays:
        return []
    group = parent.create_group(name, overwrite=True)
    group.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    manifest = []
    for key, arr in arrays.items():
        rows = max(min(arr.shape[0], chunk_size // max(arr.shape[1], 1)), 1)
        _create(group, key, arr, (rows, arr.shape[1]), compressor, _ANNDATA_ARRAY)
        manifest.append({"name": key, "dtype": "float", "length": int(arr.shape[1])})
    return manifest


def _to_int32_counts(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind in "iu":
        if data.size and data.max() > _INT32_MAX:
            raise ValueError("X holds counts that do not fit in int32.")
        return data.astype(np.int32, copy=False)
    out = np.rint(data)
    if not np.array_equal(out, data) or (out.size and np.abs(out).max() > _INT32_MAX):
        raise ValueError(
            "X must hold integer counts that fit in int32. If it was normalised in place, pass `x_layer=` "
            "to write a raw-count layer (e.g. `x_layer='counts'`) instead."
        )
    return out.astype(np.int32)


def _n_workers(n_threads: int | None) -> int:
    # Each lazy row block can hold ~1 GB of non-zeros (50k-row blocks of a 24k-gene panel), so the default
    # is capped well below the CPU count: on a 15-core machine 4 beat 8/16 by ~30% by limiting peak memory.
    return n_threads or min(os.cpu_count() or 1, _DEFAULT_MAX_WORKERS)


def _csr_blocks(x: Any, n_obs: int, row_chunk_size: int, n_threads: int | None) -> Iterator[sparse.csr_matrix]:
    """Yield `x` as consecutive CSR row blocks, computing lazy blocks `n_threads` at a time."""
    if isinstance(x, da.Array):
        blocks = x.rechunk({1: -1}).to_delayed().ravel().tolist()
        batch = _n_workers(n_threads)
        batches = [blocks[i : i + batch] for i in range(0, len(blocks), batch)]

        def compute(delayed_blocks: list[Any]) -> tuple[Any, ...]:
            return dask.compute(*delayed_blocks, scheduler="threads", num_workers=batch)

        # Compute the next batch while the caller is encoding/writing the current one.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(compute, batches[0]) if batches else None
            for i in range(len(batches)):
                assert future is not None
                computed = future.result()
                future = pool.submit(compute, batches[i + 1]) if i + 1 < len(batches) else None
                for block in computed:
                    yield sparse.csr_matrix(block)
        return
    if sparse.issparse(x):
        x = x.tocsr()
    for start in range(0, n_obs, row_chunk_size):
        yield sparse.csr_matrix(x[start : start + row_chunk_size])


def _write_x(
    root: zarr.Group,
    zf: zipfile.ZipFile,
    x: Any,
    shape: tuple[int, int],
    major_axis: str,
    chunk_size: int,
    row_chunk_size: int,
    n_threads: int | None,
    compressor: Blosc,
) -> None:
    n_obs, n_var = shape
    group = root.create_group(str(AteraKeys.X_GROUP))

    if major_axis == "feature":
        blocks = list(_csr_blocks(x, n_obs, row_chunk_size, n_threads))
        full = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix(shape)
        csc = full.tocsc()
        csc.sort_indices()
        group.attrs.update({"encoding-type": "csc_matrix", "encoding-version": "0.1.0", "shape": [n_obs, n_var]})
        nnz_chunks = (max(min(csc.nnz, chunk_size), 1),)
        _create(group, "data", _to_int32_counts(csc.data), nnz_chunks, compressor, _ANNDATA_ARRAY)
        _create(group, "indices", csc.indices.astype(np.int64), nnz_chunks, compressor, _ANNDATA_ARRAY)
        indptr_chunks = (max(min(n_var + 1, chunk_size), 1),)
        _create(group, "indptr", csc.indptr.astype(np.int64), indptr_chunks, compressor, _ANNDATA_ARRAY)
        return

    group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": [n_obs, n_var]})
    workers = _n_workers(n_threads)
    streams = {
        "data": _ChunkStream(zf, "data", np.int32, chunk_size, compressor),
        "indices": _ChunkStream(zf, "indices", np.int64, chunk_size, compressor),
    }
    indptr = [np.zeros(1, dtype=np.int64)]
    nnz = 0
    rows = 0
    # Chunks are compressed on a thread pool (Blosc releases the GIL) and written to the zip in order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stream in streams.values():
            stream.pool = pool
            stream.max_inflight = 2 * workers
        for block in _csr_blocks(x, n_obs, row_chunk_size, n_threads):
            if block.shape[1] != n_var:
                raise ValueError(f"X block has {block.shape[1]} columns, expected {n_var}.")
            start, stop = int(block.indptr[0]), int(block.indptr[-1])
            indptr.append(block.indptr[1:].astype(np.int64) - start + nnz)
            nnz += stop - start
            rows += block.shape[0]
            streams["data"].append(_to_int32_counts(block.data[start:stop]))
            streams["indices"].append(block.indices[start:stop].astype(np.int64))
        for stream in streams.values():
            stream.finish()
    if rows != n_obs:
        raise ValueError(f"X yielded {rows} rows, expected {n_obs}.")

    # Metadata only: the chunk entries were already streamed into the zip.
    chunk = chunk_size if nnz >= chunk_size else max(nnz, 1)
    for name, dtype in (("data", np.int32), ("indices", np.int64)):
        arr = group.create_array(name, shape=(nnz,), dtype=dtype, chunks=(chunk,), compressors=compressor)
        arr.attrs.update(_ANNDATA_ARRAY)
    _create(
        group,
        "indptr",
        np.concatenate(indptr),
        (max(min(n_obs + 1, chunk_size), 1),),
        compressor,
        _ANNDATA_ARRAY,
    )


class _ChunkStream:
    """Compress a 1-D array that arrives in pieces into zarr v2 chunk entries (`X/<name>/<i>`) of a zip.

    Like zarr, the trailing partial chunk is padded with the fill value up to the full chunk size, unless
    the whole array fits in a single chunk (in which case the array's chunk shape is its length).
    """

    def __init__(self, zf: zipfile.ZipFile, name: str, dtype: Any, chunk_size: int, compressor: Blosc) -> None:
        self.zf, self.name, self.dtype = zf, name, np.dtype(dtype)
        self.chunk_size, self.compressor = chunk_size, compressor
        self.pool: ThreadPoolExecutor | None = None
        self.max_inflight = 2
        self._pending: list[np.ndarray] = []
        self._pending_len = 0
        self._inflight: deque[tuple[str, Future[bytes]]] = deque()
        self._n_chunks = 0

    def append(self, values: np.ndarray) -> None:
        self._pending.append(values)
        self._pending_len += len(values)
        if self._pending_len >= self.chunk_size * _CHUNKS_PER_FLUSH:
            self._flush(final=False)

    def finish(self) -> None:
        self._flush(final=True)
        while self._inflight:
            self._drain_one()

    def _flush(self, final: bool) -> None:
        if not self._pending_len:
            return
        buf = np.concatenate(self._pending) if len(self._pending) > 1 else self._pending[0]
        n_full = (len(buf) // self.chunk_size) * self.chunk_size
        for start in range(0, n_full, self.chunk_size):
            self._submit(buf[start : start + self.chunk_size])
        tail = buf[n_full:]
        if final and len(tail):
            if self._n_chunks:  # pad to a full chunk, as zarr does for an edge chunk
                padded = np.zeros(self.chunk_size, dtype=self.dtype)
                padded[: len(tail)] = tail
                tail = padded
            self._submit(tail)
            tail = tail[:0]
        self._pending = [tail] if len(tail) else []
        self._pending_len = len(tail)

    def _submit(self, chunk: np.ndarray) -> None:
        assert self.pool is not None
        key = f"{AteraKeys.X_GROUP}/{self.name}/{self._n_chunks}"
        self._n_chunks += 1
        self._inflight.append((key, self.pool.submit(self.compressor.encode, np.ascontiguousarray(chunk))))
        while len(self._inflight) > self.max_inflight:
            self._drain_one()

    def _drain_one(self) -> None:
        key, future = self._inflight.popleft()
        self.zf.writestr(key, future.result())


# --------------------------------------------------------------------------------------------------
# manifest + zip
# --------------------------------------------------------------------------------------------------


def _build_manifest(
    table: AnnData,
    obs: list[dict[str, Any]],
    var: list[dict[str, Any]],
    obsm: list[dict[str, Any]],
    varm: list[dict[str, Any]],
    var_df: pd.DataFrame,
    major_axis: str,
) -> dict[str, Any]:
    feature_types = list(dict.fromkeys(var_df[str(AteraKeys.VAR_FEATURE_TYPE)].astype(str)))
    manifest: dict[str, Any] = {
        "encoding-type": "anndata",
        "encoding-version": "0.1.0",
        str(AteraKeys.MANIFEST_VERSION): dict(_SCHEMA_VERSION),
        str(AteraKeys.MANIFEST_MATRIX_PROPERTIES): {
            "num_cells": int(table.n_obs),
            "num_features": int(table.n_vars),
            "major_axis": major_axis,
            "feature_types": feature_types,
        },
        str(AteraKeys.OBS_GROUP): obs,
        str(AteraKeys.VAR_GROUP): var,
    }
    if obsm:
        manifest[str(AteraKeys.OBSM_GROUP)] = obsm
    if varm:
        manifest[str(AteraKeys.VARM_GROUP)] = varm
    # Optional entries that can't be derived from the table; `atera` keeps them in `uns` as JSON.
    extras = table.uns.get(str(AteraKeys.UNS_MANIFEST))
    if extras:
        extras = json.loads(extras) if isinstance(extras, str) else dict(extras)
        for key in (
            AteraKeys.MANIFEST_CLUSTERINGS,
            AteraKeys.MANIFEST_CELL_ANNOTATIONS,
            AteraKeys.MANIFEST_DOWNSAMPLED_OBSM,
            AteraKeys.MANIFEST_DOWNSAMPLED_VARM,
        ):
            if extras.get(str(key)):
                manifest[str(key)] = extras[str(key)]
    return manifest
