from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csc_matrix, csr_matrix
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation, Identity

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._atera_common import (
    DEFAULT_TABLE_ROW_CHUNK_SIZE,
    DEFAULT_VAR_COLUMNS,
    _cell_row_positions_in_bbox,
    _get_array,
    _get_group,
    _get_morphology_images,
    _get_pixel_size,
    _get_points,
    _get_polygons,
    _open_zip,
    _read_cell_ids,
    _read_feature_names,
    _tiled_imread,
    _unique_feature_names,
    read_cell_boundaries,
    read_transcripts_for_genes,
)
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata
from spatialdata_io.readers.xenium import _affine_from_keypoints, _get_labels

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import dask.array as da
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
    aligned_images: bool = True,
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
        Whether to read the cell annotations in the `AnnData` table. As in `xenium`, `X` and `var` hold the Gene
        Expression features. The counts of the other features (negative control probes and codewords, unassigned
        codewords) go to a second table, ``table_controls``, with the same cells, `obs` index and annotation,
        a sparse `X` and those features in `var`.
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
    aligned_images
        Whether to read the post-run images aligned to the morphology images (``<sample>_he_image.ome.tif`` /
        ``<sample>_if_image.ome.tif`` in the bundle root), placed with their ``*imagealignment.csv`` affine. A
        warning is logged if that affine does not map ``<sample>_keypoints.csv`` onto itself. An H&E without an
        alignment CSV is placed by an affine fit to its keypoints, or left unaligned (with a warning) if they are
        unusable.
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
        Keyword arguments passed to `tifffile.TiffFile` when reading the morphology and aligned images (see
        `_tiled_imread`). The images are read one tile per `dask` task rather than with `dask_image.imread.imread`,
        so this only accepts `TiffFile` constructor kwargs.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    labels_models_kwargs
        Keyword arguments to pass to the labels models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"The Atera bundle directory {str(path)!r} does not exist (resolved to {str(path.resolve())!r}, "
            f"relative paths are resolved against the current working directory {str(Path.cwd())!r})."
        )
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )

    pixel_size = _get_pixel_size(path)

    # The table annotates `cell_labels` (instance ids `obs["label_id"]`, as in `xenium`), else `cell_boundaries`
    # (`obs["cell_id"]`), else nothing: the nucleus elements are numbered per nucleus, not per cell.
    region = "cell_labels" if cells_labels else "cell_boundaries" if cells_boundaries else None
    tables: dict[str, AnnData] = {}
    if cells_table:
        tables["table"], tables["table_controls"], _ = _get_table(path, var_columns, table_row_chunk_size, region)

    shapes: dict[str, Any] = {}
    labels: dict[str, DataArray | DataTree] = {}
    points: dict[str, Any] = {}
    images: dict[str, DataArray | DataTree] = {}

    if cells_boundaries or nucleus_boundaries or cells_labels or nucleus_labels:
        cells_group = zarr.open_group(zarr.storage.ZipStore(path / AteraKeys.CELLS_FILE, read_only=True), mode="r")
        cell_ids = _read_cell_ids(path)

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
        points["transcripts"] = _get_points(path, pixel_size, _read_feature_names(path))

    if morphology_images:
        images["morphology"] = _get_morphology_images(
            path / AteraKeys.MORPHOLOGY_2D_DIR, imread_kwargs, image_models_kwargs
        )
    if morphology_3d_images:
        # (c, z, y, x): the 3-tuple default chunks gain one z plane per chunk, and z is not downsampled
        kwargs_3d = dict(image_models_kwargs)
        if isinstance(chunks := kwargs_3d.get("chunks"), tuple | list) and len(chunks) == 3:
            kwargs_3d["chunks"] = (chunks[0], 1, *chunks[1:])
        kwargs_3d["scale_factors"] = [
            f if isinstance(f, dict) else {"z": 1, "y": f, "x": f} for f in kwargs_3d.get("scale_factors") or []
        ]
        images["morphology_3d"] = _get_morphology_images(
            path / AteraKeys.MORPHOLOGY_3D_DIR, imread_kwargs, kwargs_3d, three_d=True
        )

    if aligned_images:
        images |= _get_aligned_images(path, imread_kwargs, image_models_kwargs)

    sdata = SpatialData(images=images, labels=labels, points=points, tables=tables, shapes=shapes)
    sdata = _set_reader_metadata(sdata, "atera")
    sdata.attrs[str(AteraKeys.PIXEL_SIZE)] = pixel_size
    return sdata


KEYPOINTS_MAX_RESIDUAL_PX = 10.0


def _get_aligned_images(
    path: Path, imread_kwargs: Mapping[str, Any], image_models_kwargs: Mapping[str, Any]
) -> dict[str, DataArray | DataTree]:
    """The post-run H&E and IF images, read like the morphology images, in the global (morphology px) system.

    Each is placed by its alignment CSV or, for an H&E without one, by an affine fit to its keypoints (as in `xenium`).
    """
    images = {}
    for file in sorted(path.glob("*.ome.tif")):
        name = next((n for n in ["he_image", "if_image"] if file.name.endswith(f"{n}.ome.tif")), None)
        if name is None or file.name.startswith("._"):  # macOS sidecars
            continue
        alignment_file = file.with_name(file.name.replace(".ome.tif", "alignment.csv"))
        kp_file = file.with_name(file.name.removesuffix("he_image.ome.tif") + "keypoints.csv")
        image, dims = _tiled_imread(file, imread_kwargs)
        transformation: BaseTransformation = Identity()
        if alignment_file.is_file():
            affine = pd.read_csv(alignment_file, header=None).to_numpy(dtype=float)
            transformation = Affine(affine, input_axes=("x", "y"), output_axes=("x", "y"))
            if name == "he_image":
                _check_keypoints(alignment_file, kp_file)
        elif name == "he_image":
            transformation = _affine_from_keypoints(kp_file) or transformation
        if "c" not in dims:
            image, dims = image[None], ("c", *dims)
        n_channels = image.shape[dims.index("c")]
        images[name] = Image2DModel.parse(
            image,
            dims=dims,
            c_coords=["r", "g", "b", "a"][:n_channels] if name == "he_image" and n_channels in [3, 4] else None,
            transformations={"global": transformation},
            **image_models_kwargs,
        )
    return images


def _check_keypoints(alignment_file: Path, kp_file: Path) -> float | None:
    """Warn when the alignment affine does not map the keypoints onto each other; return the median residual (px)."""
    if not kp_file.is_file():
        return None
    columns = ["fixedX", "fixedY", "alignmentX", "alignmentY"]
    kp = pd.read_csv(kp_file)
    if not set(columns) <= set(kp.columns):
        warnings.warn(f"{kp_file.name}: expected columns {columns}, found {list(kp.columns)}.", stacklevel=2)
        return None
    xy = kp[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        warnings.warn(f"{kp_file.name}: {len(xy)} finite keypoints, at least 3 are needed.", stacklevel=2)
        return None
    affine = pd.read_csv(alignment_file, header=None).to_numpy(dtype=float)
    src, dst = xy[:, 2:], xy[:, :2]
    median = float(np.median(np.linalg.norm((np.c_[src, np.ones(len(src))] @ affine.T)[:, :2] - dst, axis=1)))
    if median > KEYPOINTS_MAX_RESIDUAL_PX:
        logger.warning(
            f"{alignment_file.name} maps {kp_file.name} with a median residual of {median:.1f} px "
            f"(> {KEYPOINTS_MAX_RESIDUAL_PX}); the aligned image may be misplaced."
        )
    return median


def _align_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Make a column that shares the index's name have the index's dtype, and leave the index unnamed.

    The bundles' `obs`/`var` list the index (`barcode`/`feature_id`) as a column too. Under pandas 3,
    `read_elem` returns that index as `str` but the column as `object`, and `anndata` refuses to write
    a name that is both index and column unless the two compare equal, which `pd.Series.equals` only does
    for matching dtypes.
    """
    name = df.index.name
    if name is not None and name in df.columns:
        df[name] = df[name].astype(df.index.dtype)
        # the column keeps the values; an index of the same name breaks `obs.reset_index()`, which spatialdata's
        # table joins call (spatialdata 0.8.0 _core/query/relational_query.py:259): "cannot insert barcode"
        df.index.name = None
    return df


def _read_dataframe(group: zarr.Group, key: AteraKeys) -> pd.DataFrame:
    """Read ``group[key]`` with `anndata.io.read_elem` as a `DataFrame` that `anndata` can write back."""
    from anndata.io import read_elem

    return _align_index_column(cast(pd.DataFrame, read_elem(group[str(key)])))


def _read_category_colors(
    group: zarr.Group, key: AteraKeys, columns: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """Colors of the categorical columns of ``group[key]`` that have a ``color`` child, as ``#rrggbbaa``.

    ``anndata.io.read_elem`` ignores the ``color`` array (packed int32 ``0xRRGGBBAA``), so they are kept in
    `uns` (``{column}_colors``, as `scanpy` does) for `write_atera_cell_feature_matrix` to write back.
    """
    colors = {}
    for name, elem in cast(zarr.Group, group[str(key)]).members():
        if isinstance(elem, zarr.Group) and "color" in elem and (columns is None or name in columns):
            colors[f"{name}_colors"] = [
                f"#{int(c) & 0xFFFFFFFF:08x}" for c in np.asarray(cast(zarr.Array, elem["color"])[:])
            ]
    return colors


def _read_manifest_extras(group: zarr.Group) -> str | None:
    """The root manifest entries that can't be derived from the table, JSON-encoded for `adata.uns`.

    `uns` holds a string rather than nested lists/dicts so the table stays writable by `anndata`;
    `write_atera_cell_feature_matrix` writes these entries back out.
    """
    keys = (
        AteraKeys.MANIFEST_CLUSTERINGS,
        AteraKeys.MANIFEST_CELL_ANNOTATIONS,
        AteraKeys.MANIFEST_DOWNSAMPLED_OBSM,
        AteraKeys.MANIFEST_DOWNSAMPLED_VARM,
    )
    attrs = dict(group.attrs)
    extras = {str(k): attrs[str(k)] for k in keys if attrs.get(str(k))}
    return json.dumps(extras) if extras else None


def _add_label_id(obs_df: pd.DataFrame) -> None:
    """Add ``obs["label_id"]``: the value each cell takes in the ``cell_labels`` raster.

    ``cells.zarr.zip`` ``masks/1`` is labelled ``1..n_obs`` by row position, which is *not* the (large,
    packed) ``obs["cell_id"]``. Must be called on the full `obs` before any row subsetting.
    """
    obs_df[str(AteraKeys.LABEL_ID)] = np.arange(1, len(obs_df) + 1, dtype=np.uint32)


# store `var` columns that the `xenium` layout renames
_XENIUM_VAR_NAMES = {str(AteraKeys.VAR_FEATURE_TYPE): "feature_types"}


def _xenium_var(var: pd.DataFrame) -> pd.DataFrame:
    """Lay out the store's `var` as the `xenium` reader does.

    Gene symbols are the index (made unique with a suffix), Ensembl ids go to ``gene_ids`` and the feature type to
    ``feature_types`` (the bundles index by ``feature_id`` and name the type ``feature_type``). The type values are
    already the Xenium ones (``Gene Expression``, ``Negative Control Probe``, ...).
    """
    out = var.rename(columns=_XENIUM_VAR_NAMES)
    out["gene_ids"] = var.index.astype(str).to_numpy()
    out.index = _unique_feature_names(var[str(AteraKeys.FEATURE_NAME)])
    return out


def _is_gene(var: pd.DataFrame) -> np.ndarray:
    """Which features are ``Gene Expression``, the only ones `xenium` keeps in `X` by default (``gex_only=True``)."""
    if "feature_types" not in var.columns:
        return np.ones(len(var), dtype=bool)
    return (var["feature_types"] == "Gene Expression").to_numpy()


def _read_obs_var(group: zarr.Group) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`obs` with ``label_id`` and the full `var` in the `xenium` layout, in store order."""
    obs = _read_dataframe(group, AteraKeys.OBS_GROUP)
    _add_label_id(obs)
    return obs, _xenium_var(_read_dataframe(group, AteraKeys.VAR_GROUP))


def _assemble_tables(
    x: Any,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    controls: csr_matrix | csc_matrix,
    var_columns: list[str] | Literal["all"],
    region: str | None,
    obsm: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[AnnData, AnnData]:
    """The Gene Expression table and the ``table_controls`` table of all other features, for the same cells.

    The other features are the negative control probes and codewords and the unassigned codewords, so `X` and
    ``n_genes`` of the first table count genes only. The second has a sparse `X` (CSR, as stored), the same `obs`
    index and instance id columns, and annotates the same element. `var_columns` name the store's columns.
    """
    from anndata import AnnData

    is_gene = _is_gene(var)
    columns = list(
        var.columns if var_columns == "all" else [_XENIUM_VAR_NAMES.get(c, c) for c in var_columns] + ["gene_ids"]
    )
    obsm = {**obsm, "spatial": obs[[str(AteraKeys.CENTROID_X), str(AteraKeys.CENTROID_Y)]].to_numpy()}
    table = _parse_table(AnnData(X=x, obs=obs, var=var.loc[is_gene, columns], obsm=obsm), region)
    ids = obs[[str(AteraKeys.CELL_ID), str(AteraKeys.LABEL_ID)]].copy()
    controls_table = AnnData(X=csr_matrix(controls), obs=ids, var=var.loc[~is_gene, columns])
    return table, _parse_table(controls_table, region)


def _parse_table(adata: AnnData, region: str | None) -> AnnData:
    """Annotate `region` (if any) with `adata`, using whichever `obs` column holds that element's instance ids.

    ``cell_labels`` raster values are ``label_id``; ``cell_boundaries`` shapes are indexed by ``cell_id``.
    """
    if region is None or adata.n_obs == 0:  # spatialdata rejects an empty annotating table ("Regions ... do not match")
        return TableModel.parse(adata)
    instance_key = AteraKeys.LABEL_ID if region == "cell_labels" else AteraKeys.CELL_ID
    adata.obs["region"] = pd.Categorical([region] * adata.n_obs)
    return TableModel.parse(adata, region=region, region_key="region", instance_key=str(instance_key))


def _get_table(
    path: Path,
    var_columns: list[str] | Literal["all"] = DEFAULT_VAR_COLUMNS,
    row_chunk_size: int = DEFAULT_TABLE_ROW_CHUNK_SIZE,
    region: str | None = "cell_labels",
) -> tuple[AnnData, AnnData, list[str]]:
    """Read ``cell_feature_matrix.zarr.zip`` into the gene table and the ``table_controls`` table using native `anndata` zarr IO.

    `obs`/`var`/`obsm` are read with `anndata.io.read_elem` (which also means `obs`/`var` come back
    indexed by whichever column the store's ``_index`` attr names -- e.g. `barcode`/`feature_id`).
    `X` is a lazy `dask` array of `row_chunk_size`-row CSR chunks (`anndata.experimental.read_elem_lazy`).

    `region` is the element the table annotates; it must be read too (see `atera`). None annotates nothing.
    """
    from anndata.experimental import read_elem_lazy
    from anndata.io import read_elem, sparse_dataset

    with _open_zip(path / AteraKeys.CELL_FEATURE_MATRIX_FILE) as group:
        obs, var = _read_obs_var(group)
        colors = _read_category_colors(group, AteraKeys.OBS_GROUP)
        # keyed by the column's name in `var` (`feature_types_colors`), as scanpy looks them up
        var_colors = {
            f"{_XENIUM_VAR_NAMES.get(k.removesuffix('_colors'), k.removesuffix('_colors'))}_colors": v
            for k, v in _read_category_colors(
                group, AteraKeys.VAR_GROUP, None if var_columns == "all" else list(var_columns)
            ).items()
        }
        obsm = (
            cast(dict[str, Any], read_elem(group[str(AteraKeys.OBSM_GROUP)]))
            if str(AteraKeys.OBSM_GROUP) in group
            else {}
        )
        manifest_extras = _read_manifest_extras(group)
        # the graph keeps the store, which reopens on first use (also after pickling)
        x_group = _get_group(group, str(AteraKeys.X_GROUP))
        x = cast("da.Array", read_elem_lazy(x_group, chunks=(row_chunk_size, -1)))
        stored_per_row = np.diff(_get_array(x_group, "indptr")[:])
    is_gene = _is_gene(var)
    controls: csr_matrix | csc_matrix
    if (path / AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE).is_file():  # column reads are cheap in the CSC copy
        with _open_zip(path / AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE) as group:
            controls = cast(csc_matrix, sparse_dataset(group[str(AteraKeys.X_GROUP)])[:, np.flatnonzero(~is_gene)])
    else:
        controls = cast(csr_matrix, x[:, ~is_gene].compute())
    # genes detected per cell: the stored entries of each CSR row (its `indptr` step) minus those of the controls
    obs["n_genes"] = stored_per_row - controls.getnnz(axis=1)
    table, controls_table = _assemble_tables(x[:, is_gene], obs, var, controls, var_columns, region, obsm)
    table.uns.update({**colors, **var_colors})
    controls_table.uns.update(var_colors)
    if manifest_extras is not None:
        table.uns[str(AteraKeys.UNS_MANIFEST)] = manifest_extras
    return table, controls_table, var.index.tolist()


def read_var(path: str | Path) -> pd.DataFrame:
    """Read the complete `var` table from ``cell_feature_matrix.zarr.zip`` via `anndata.io.read_elem`.

    Includes the ~135 per-cluster differential-expression columns that `atera` drops by default
    (see `var_columns` on `atera`) to avoid holding that wider table in memory.

    It has the same index as the table's `var` (gene symbols), plus the non-gene features, so columns can be added
    by name, e.g. ``table.var["score"] = read_var(path)["de_leiden_res_1.0_c0_score"]``.

    Parameters
    ----------
    path
        Path to the dataset (the same path originally passed to `atera`).

    Returns
    -------
    The complete `var` `DataFrame` of every feature, in the `xenium` layout (see `atera`).
    """
    with _open_zip(Path(path) / AteraKeys.CELL_FEATURE_MATRIX_FILE) as group:
        return _xenium_var(_read_dataframe(group, AteraKeys.VAR_GROUP))


def read_table_for_cells(
    path: str | Path,
    cell_ids: Sequence[int] | np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    var_columns: list[str] | Literal["all"] = DEFAULT_VAR_COLUMNS,
) -> AnnData:
    """Read the cell-by-gene table for only a subset of cells, without loading the full `X`.

    Unlike ``atera(path, cells_table=True)``, whose `X` is lazy over every cell, this reads only
    the requested rows, in memory, directly off `anndata.io.sparse_dataset`'s own fancy-indexing support
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
    An `AnnData` `TableModel` with only the requested cells and the Gene Expression features.
    """
    if (cell_ids is None) == (bbox is None):
        raise ValueError("Exactly one of `cell_ids`/`bbox` must be given.")
    from anndata.io import sparse_dataset

    path = Path(path)
    with _open_zip(path / AteraKeys.CELL_FEATURE_MATRIX_FILE) as group:
        obs, var = _read_obs_var(group)
        if cell_ids is not None:
            row_positions = pd.Index(obs[str(AteraKeys.CELL_ID)].to_numpy()).get_indexer(pd.Index(np.asarray(cell_ids)))
            missing = np.asarray(cell_ids)[row_positions == -1]
            if missing.size:
                raise ValueError(f"cell_id(s) not found in {AteraKeys.CELL_FEATURE_MATRIX_FILE!s}: {missing.tolist()}")
        else:
            assert bbox is not None
            row_positions = _cell_row_positions_in_bbox(path, bbox)
        x = cast(csr_matrix, sparse_dataset(group[str(AteraKeys.X_GROUP)])[row_positions])
    obs = obs.iloc[row_positions].copy()
    is_gene = _is_gene(var)
    obs["n_genes"] = x[:, is_gene].getnnz(axis=1)
    return _assemble_tables(x[:, is_gene], obs, var, x[:, ~is_gene], var_columns, "cell_labels")[0]


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
    from anndata.io import sparse_dataset

    genes = [genes] if isinstance(genes, str) else list(genes)
    path = Path(path)
    with _open_zip(path / AteraKeys.CSC_CELL_FEATURE_MATRIX_FILE) as group:
        obs, var = _read_obs_var(group)
        col_positions = var.index.get_indexer(pd.Index(genes))
        missing = np.asarray(genes)[col_positions == -1]
        if missing.size:
            raise ValueError(
                f"gene(s) not found in {AteraKeys.VAR_GROUP!s}.{AteraKeys.FEATURE_NAME!s}: {missing.tolist()}"
            )
        x = cast(csc_matrix, sparse_dataset(group[str(AteraKeys.X_GROUP)])[:, col_positions])
    obsm: dict[str, Any] = {"spatial": obs[[str(AteraKeys.CENTROID_X), str(AteraKeys.CENTROID_Y)]].to_numpy()}
    return _parse_table(AnnData(X=x, obs=obs, var=var.iloc[col_positions], obsm=obsm), "cell_labels")
