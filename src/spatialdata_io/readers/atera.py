from __future__ import annotations

import logging
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from dask import delayed
from scipy.sparse import csc_matrix, csr_matrix
from spatialdata import SpatialData
from spatialdata.models import TableModel

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._atera_common import (
    DEFAULT_TABLE_ROW_CHUNK_SIZE,
    DEFAULT_VAR_COLUMNS,
    _cell_row_positions_in_bbox,
    _get_labels,
    _get_morphology_images,
    _get_pixel_size,
    _get_points,
    _get_polygons,
    _patched_ragged_vlen_chunk_decode,
    read_cell_boundaries,
    read_transcripts_for_genes,
)
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anndata import AnnData
    from xarray import DataArray, DataTree

__all__ = [
    "atera",
    "read_cell_boundaries",
    "read_table_for_cells",
    "read_table_for_genes",
    "read_transcripts_for_genes",
    "read_var",
]

# NOTE: this reads `cell_feature_matrix.zarr.zip`/`csc_cell_feature_matrix.zarr.zip` with
# `anndata`'s own zarr reading functions (`anndata.io.read_elem`/`anndata.io.sparse_dataset`). This
# only works on bundles whose zarr stores carry the standard AnnData `encoding-type`/
# `encoding-version`/`shape` attrs.
#
# Every non-table element (labels/shapes/points/morphology images) lives in `_atera_common.py`
# rather than duplicated here.


@inject_docs(xx=AteraKeys)
def atera(
    path: str | Path,
    *,
    cells_table: bool = True,
    cells_boundaries: bool = True,
    nucleus_boundaries: bool = True,
    cells_labels: bool = True,
    nucleus_labels: bool = True,
    transcripts: bool = False,
    morphology_images: bool = True,
    morphology_3d_images: bool = False,
    var_columns: list[str] | Literal["all"] = DEFAULT_VAR_COLUMNS,
    table_row_chunk_size: int = DEFAULT_TABLE_ROW_CHUNK_SIZE,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read a *10x Genomics Atera* dataset into a SpatialData object.

    This function reads the following files:

        - ``{xx.SPECS_FILE!r}``: File containing specifications, including the pixel size. If absent, the
          pixel size is instead read from the ``PhysicalSizeX`` OME-XML metadata of the first
          ``{xx.MORPHOLOGY_2D_DIR!r}`` OME-TIFF.
        - ``{xx.CELL_FEATURE_MATRIX_FILE!r}``: Zipped zarr store with the cell-by-gene matrix and cell metadata.
        - ``{xx.CELLS_FILE!r}``: Zipped zarr store with cell/nucleus labels and boundary polygons.
        - ``{xx.TRANSCRIPTS_FILE!r}``: Zipped zarr store with per-transcript locations (optional, large).
        - ``{xx.MORPHOLOGY_2D_DIR!r}``: Directory of single-channel morphology OME-TIFF images.
        - ``{xx.MORPHOLOGY_3D_DIR!r}``: Directory of single-channel 3D morphology OME-TIFF images (optional, huge).

    Unlike most other 10x Genomics formats, all the tabular/raster/point data (aside from the morphology
    images) is stored in zipped `zarr <https://zarr.dev/>`_ stores. ``{xx.CELL_FEATURE_MATRIX_FILE!r}`` and
    ``{xx.CSC_CELL_FEATURE_MATRIX_FILE!r}`` use the standard AnnData zarr encoding and are read with
    `anndata`'s own zarr IO (``anndata.io.read_elem``/``anndata.io.sparse_dataset``).

    Parameters
    ----------
    path
        Path to the dataset.
    cells_table
        Whether to read the cell annotations in the `AnnData` table.
    cells_boundaries
        Whether to read cell boundaries (polygons).
    nucleus_boundaries
        Whether to read nucleus boundaries (polygons).
    cells_labels
        Whether to read cell labels (raster).
    nucleus_labels
        Whether to read nucleus labels (raster).
    transcripts
        Whether to read transcripts (points). This is opt-in and defaults to `False` because the transcripts
        table can have tens of millions of rows; when read, it is loaded lazily with `dask`.
    morphology_images
        Whether to read the 2D morphology images.
    morphology_3d_images
        Whether to read the 3D morphology images. This is opt-in and defaults to `False` because these images
        can be very large.
    var_columns
        Which columns of `var` to keep, out of the full table (which includes ~135 per-cluster
        differential-expression columns). Defaults to a small subset (`{xx.FEATURE_NAME!r}`,
        `{xx.VAR_FILTERED!r}`, `{xx.VAR_HIGHLY_VARIABLE!r}`, `{xx.FEATURE_ID!r}`,
        `{xx.VAR_FEATURE_TYPE!r}`, `{xx.VAR_GENOME!r}`) to avoid holding the full table in memory;
        pass `"all"` to keep every column, or use `read_var` to load the full table separately.
    table_row_chunk_size
        Number of `obs` rows read per chunk when lazily loading the table's `X` (see `_get_table`).
        Larger values mean fewer, larger `dask` tasks (faster, but more peak memory per chunk);
        smaller values mean more, smaller ones.
    imread_kwargs
        Keyword arguments passed to `tifffile.TiffFile` when reading the morphology images (see
        `_tiled_imread`); the images are read tile-by-tile off `tifffile`'s own `zarr` store rather
        than via `dask_image.imread.imread`, so this only accepts `TiffFile` constructor kwargs.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    labels_models_kwargs
        Keyword arguments to pass to the labels models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )

    pixel_size = _get_pixel_size(path)

    needs_cells_zarr = cells_boundaries or nucleus_boundaries or cells_labels or nucleus_labels
    if not cells_table and (needs_cells_zarr or transcripts):
        logging.info("Reading the table is required for the requested elements; setting cells_table=True.")
        cells_table = True

    # Prefer `cell_labels` as the table's annotation target (matching the `xenium` reader's
    # convention), but fall back to `cell_boundaries` when labels aren't being loaded, so the
    # table doesn't end up annotating an element that isn't actually present in `sdata`.
    default_region = "cell_labels" if cells_labels else "cell_boundaries" if cells_boundaries else "cell_labels"

    table: AnnData | None = None
    feature_names: list[str] | None = None
    if cells_table:
        table, feature_names = _get_table(
            path, var_columns=var_columns, row_chunk_size=table_row_chunk_size, region=default_region
        )

    shapes: dict[str, Any] = {}
    labels: dict[str, DataArray | DataTree] = {}
    points: dict[str, Any] = {}
    images: dict[str, DataArray | DataTree] = {}

    if needs_cells_zarr:
        cells_store = zarr.storage.ZipStore(path / AteraKeys.CELLS_FILE, read_only=True)
        cells_group = zarr.open_group(cells_store, mode="r")

        cell_ids = table.obs[str(AteraKeys.CELL_ID)].to_numpy() if table is not None else None

        if nucleus_labels:
            labels["nucleus_labels"] = _get_labels(cells_group, mask_index=0, labels_models_kwargs=labels_models_kwargs)
        if cells_labels:
            labels["cell_labels"] = _get_labels(cells_group, mask_index=1, labels_models_kwargs=labels_models_kwargs)
        if nucleus_boundaries:
            shapes["nucleus_boundaries"] = _get_polygons(
                cells_group, mask_index=0, cell_ids=cell_ids, pixel_size=pixel_size, is_nucleus=True
            )
        if cells_boundaries:
            shapes["cell_boundaries"] = _get_polygons(
                cells_group, mask_index=1, cell_ids=cell_ids, pixel_size=pixel_size, is_nucleus=False
            )

    if transcripts:
        if feature_names is None:
            raise ValueError("Reading transcripts requires `cells_table=True` (to map gene indices to names).")
        points["transcripts"] = _get_points(path, pixel_size, feature_names)

    if morphology_images:
        images["morphology"] = _get_morphology_images(
            path / AteraKeys.MORPHOLOGY_2D_DIR, imread_kwargs, image_models_kwargs
        )
    if morphology_3d_images:
        images["morphology_3d"] = _get_morphology_images(
            path / AteraKeys.MORPHOLOGY_3D_DIR, imread_kwargs, image_models_kwargs, three_d=True
        )

    tables = {"table": table} if table is not None else {}
    sdata = SpatialData(images=images, labels=labels, points=points, tables=tables, shapes=shapes)
    sdata = _set_reader_metadata(sdata, "atera")
    sdata.attrs[str(AteraKeys.PIXEL_SIZE)] = pixel_size
    return sdata


def _read_x_chunk(zarr_path: Path, zarr_key: AteraKeys, start: int, stop: int) -> csr_matrix:
    """Read one contiguous row-range of `X` via `anndata.io.sparse_dataset`'s own row slicing.

    Opens its own zip store (rather than sharing one across chunks) so this function can safely be
    called from independent `dask` tasks.
    """
    from anndata.io import sparse_dataset

    store = zarr.storage.ZipStore(zarr_path / zarr_key, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        return sparse_dataset(group[str(AteraKeys.X_GROUP)])[start:stop]
    finally:
        store.close()


def _lazy_x(zarr_path: Path, zarr_key: AteraKeys, n_obs: int, n_var: int, dtype: np.dtype, row_chunk_size: int) -> da.Array:
    """Build a `dask`-backed `n_obs x n_var` CSR array, delegating each chunk's read to `sparse_dataset`."""
    meta = csr_matrix((0, n_var), dtype=dtype)
    blocks = []
    for start in range(0, n_obs, row_chunk_size):
        stop = min(start + row_chunk_size, n_obs)
        block = delayed(_read_x_chunk)(zarr_path, AteraKeys.CELL_FEATURE_MATRIX_FILE, start, stop)
        blocks.append(da.from_delayed(block, shape=(stop - start, n_var), dtype=dtype, meta=meta))
    return da.concatenate(blocks, axis=0)


def _get_table(
    path: Path,
    var_columns: list[str] | Literal["all"] = DEFAULT_VAR_COLUMNS,
    row_chunk_size: int = DEFAULT_TABLE_ROW_CHUNK_SIZE,
    region: str = "cell_labels",
) -> tuple[AnnData, list[str]]:
    """Read ``cell_feature_matrix.zarr.zip`` into an AnnData table using native `anndata` zarr IO.

    `obs`/`var`/`obsm` are read with `anndata.io.read_elem` (which also means `obs`/`var` come back
    indexed by whichever column the store's ``_index`` attr names -- e.g. `barcode`/`feature_id`).
    `X` is built as a lazily `dask`-backed array in row chunks (see `_lazy_x`), each chunk read via
    `anndata.io.sparse_dataset` rather than manual `indptr` arithmetic.

    `region` should be the name of a `SpatialElement` that will actually be present in the
    `SpatialData` object the table is assembled into (see the ``default_region`` logic in
    `atera`), so the table doesn't end up annotating an element that was never loaded.
    """
    from anndata import AnnData
    from anndata.io import read_elem, sparse_dataset

    store = zarr.storage.ZipStore(path / AteraKeys.CELL_FEATURE_MATRIX_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")

        with _patched_ragged_vlen_chunk_decode():
            var_df = read_elem(group[str(AteraKeys.VAR_GROUP)])
            feature_names = var_df[str(AteraKeys.FEATURE_NAME)].astype(str).tolist()
            if var_columns != "all":
                var_df = var_df[list(var_columns)]

            obs_df = read_elem(group[str(AteraKeys.OBS_GROUP)])

            obsm = read_elem(group[str(AteraKeys.OBSM_GROUP)]) if str(AteraKeys.OBSM_GROUP) in group else {}

        x_ds = sparse_dataset(group[str(AteraKeys.X_GROUP)])
        n_obs, n_var = x_ds.shape
        dtype = x_ds.dtype
    finally:
        store.close()

    x = _lazy_x(path, AteraKeys.CELL_FEATURE_MATRIX_FILE, n_obs, n_var, dtype, row_chunk_size)

    adata = AnnData(X=x, obs=obs_df, var=var_df, obsm=obsm)
    adata.obsm["spatial"] = adata.obs[[str(AteraKeys.CENTROID_X), str(AteraKeys.CENTROID_Y)]].to_numpy()
    adata.obs["region"] = pd.Categorical([region] * n_obs)

    table = TableModel.parse(
        adata,
        region=region,
        region_key="region",
        instance_key=str(AteraKeys.CELL_ID),
    )
    return table, feature_names


def read_var(path: str | Path) -> pd.DataFrame:
    """Read the complete `var` table from ``cell_feature_matrix.zarr.zip`` via `anndata.io.read_elem`.

    Includes the ~135 per-cluster differential-expression columns that `atera` drops by default
    (see `var_columns` on `atera`) to avoid holding that wider table in memory.

    Use this to build your own `var`/`AnnData` with whichever columns you need, e.g.
    ``sdata.tables["table"].var = read_var(path)[["feature_name", "de_leiden_res_1.0_c0_score"]]``.

    Parameters
    ----------
    path
        Path to the dataset (the same path originally passed to `atera`).

    Returns
    -------
    The complete `var` `DataFrame`, indexed by `feature_id`.
    """
    from anndata.io import read_elem

    path = Path(path)
    store = zarr.storage.ZipStore(path / AteraKeys.CELL_FEATURE_MATRIX_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        with _patched_ragged_vlen_chunk_decode():
            var_df = read_elem(group[str(AteraKeys.VAR_GROUP)])
    finally:
        store.close()
    return var_df


def read_table_for_cells(
    path: str | Path,
    cell_ids: Sequence[int] | np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    var_columns: list[str] | Literal["all"] = DEFAULT_VAR_COLUMNS,
) -> AnnData:
    """Read the cell-by-gene table for only a subset of cells, without loading the full `X`.

    Unlike ``atera(path, cells_table=True)``, which eagerly builds the whole `X`, this reads only
    the requested rows directly off `anndata.io.sparse_dataset`'s own fancy-indexing support
    (``x_ds[row_positions]``), which already handles arbitrary/unsorted row selections internally.

    Exactly one of `cell_ids`/`bbox` must be given.

    Parameters
    ----------
    path
        Path to the dataset (the same path originally passed to `atera`).
    cell_ids
        Specific cell ids to read (matching `obs["cell_id"]`); the returned table preserves this
        order.
    bbox
        ``(xmin, ymin, xmax, ymax)``, in the same "global" coordinate system as the rest of an
        `atera()`-read `SpatialData` (see `read_cell_boundaries`). Selects every cell whose
        bounding box overlaps this region.
    var_columns
        Same as on `atera`; see there.

    Returns
    -------
    An `AnnData` `TableModel` with only the requested cells.
    """
    if (cell_ids is None) == (bbox is None):
        raise ValueError("Exactly one of `cell_ids`/`bbox` must be given.")
    from anndata import AnnData
    from anndata.io import read_elem, sparse_dataset

    path = Path(path)
    store = zarr.storage.ZipStore(path / AteraKeys.CELL_FEATURE_MATRIX_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        with _patched_ragged_vlen_chunk_decode():
            var_df = read_elem(group[str(AteraKeys.VAR_GROUP)])
            if var_columns != "all":
                var_df = var_df[list(var_columns)]
            obs_df = read_elem(group[str(AteraKeys.OBS_GROUP)])

        if cell_ids is not None:
            index = pd.Index(obs_df[str(AteraKeys.CELL_ID)].to_numpy())
            row_positions = index.get_indexer(np.asarray(cell_ids))
            missing = np.asarray(cell_ids)[row_positions == -1]
            if missing.size:
                raise ValueError(f"cell_id(s) not found in {AteraKeys.CELL_FEATURE_MATRIX_FILE!s}: {missing.tolist()}")
        else:
            row_positions = _cell_row_positions_in_bbox(path, bbox)

        x_ds = sparse_dataset(group[str(AteraKeys.X_GROUP)])
        x = x_ds[row_positions]
        obs_subset = obs_df.iloc[row_positions]
    finally:
        store.close()

    n_obs = len(obs_subset)
    adata = AnnData(X=x, obs=obs_subset, var=var_df)
    adata.obsm["spatial"] = adata.obs[[str(AteraKeys.CENTROID_X), str(AteraKeys.CENTROID_Y)]].to_numpy()
    adata.obs["region"] = pd.Categorical(["cell_labels"] * n_obs)

    return TableModel.parse(
        adata,
        region="cell_labels",
        region_key="region",
        instance_key=str(AteraKeys.CELL_ID),
    )


def read_table_for_genes(path: str | Path, genes: str | Sequence[str]) -> AnnData:
    """Read the cell-by-gene table for only a subset of genes, across all cells, without loading full `X`.

    Column-wise mirror of `read_table_for_cells`: reads from the bundle's redundant, column-major
    ``csc_cell_feature_matrix.zarr.zip`` (a transpose of the same matrix, otherwise unused by
    `atera`) and pulls the requested columns directly off `anndata.io.sparse_dataset`'s own
    fancy-indexing support (``x_ds[:, col_positions]``).

    Parameters
    ----------
    path
        Path to the dataset (the same path originally passed to `atera`).
    genes
        Gene name(s) to read (matching `var["feature_name"]`); the returned table preserves this
        order.

    Returns
    -------
    An `AnnData` with every cell but only the requested genes, sharing `obs` row order with the
    rest of an `atera()`-read `SpatialData` (so its `X` columns can be dropped directly onto
    `sdata.tables["table"].obs`/`sdata.shapes["cell_boundaries"]` for plotting).
    """
    from anndata import AnnData
    from anndata.io import read_elem, sparse_dataset

    genes = [genes] if isinstance(genes, str) else list(genes)
    path = Path(path)
    store = zarr.storage.ZipStore(path / AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")

        with _patched_ragged_vlen_chunk_decode():
            var_df = read_elem(group[str(AteraKeys.VAR_GROUP)])[
                [str(AteraKeys.FEATURE_ID), str(AteraKeys.FEATURE_NAME)]
            ]
            feature_names = var_df[str(AteraKeys.FEATURE_NAME)].astype(str).tolist()
            index = pd.Index(feature_names)
            col_positions = index.get_indexer(np.asarray(genes))
            missing = np.asarray(genes)[col_positions == -1]
            if missing.size:
                raise ValueError(
                    f"gene(s) not found in {AteraKeys.VAR_GROUP!s}.{AteraKeys.FEATURE_NAME!s}: {missing.tolist()}"
                )

            var_subset = var_df.iloc[col_positions]
            var_subset.index = pd.Index(np.asarray(genes), name=str(AteraKeys.FEATURE_NAME))

            obs_df = read_elem(group[str(AteraKeys.OBS_GROUP)])
            n_obs = len(obs_df)

        x_ds = sparse_dataset(group[str(AteraKeys.X_GROUP)])
        x: csc_matrix = x_ds[:, col_positions]
    finally:
        store.close()

    adata = AnnData(X=x, obs=obs_df, var=var_subset)
    adata.obs["region"] = pd.Categorical(["cell_labels"] * n_obs)

    return TableModel.parse(
        adata,
        region="cell_labels",
        region_key="region",
        instance_key=str(AteraKeys.CELL_ID),
    )
