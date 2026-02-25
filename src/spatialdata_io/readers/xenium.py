from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import packaging.version
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tifffile
import zarr
from dask.dataframe import read_parquet
from dask_image.imread import imread
from geopandas import GeoDataFrame
from shapely import GeometryType, Polygon, from_ragged_array
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations.transformations import Affine, Identity, Scale
from xarray import DataArray, DataTree

from spatialdata_io._constants._constants import XeniumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io._utils import deprecation_alias, zarr_open
from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow as pa
    from anndata import AnnData
    from spatialdata._types import ArrayLike

__all__ = ["xenium", "xenium_aligned_image", "xenium_explorer_selection"]


@deprecation_alias(
    cells_as_shapes="cells_as_circles",
    cell_boundaries="cells_boundaries",
    cell_labels="cells_labels",
)
@inject_docs(xx=XeniumKeys)
def xenium(
    path: str | Path,
    *,
    cells_boundaries: bool = True,
    nucleus_boundaries: bool = True,
    cells_as_circles: bool = False,
    cells_labels: bool = True,
    nucleus_labels: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    aligned_images: bool = True,
    cells_table: bool = True,
    n_jobs: int | None = None,
    gex_only: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read a *10x Genomics Xenium* dataset into a SpatialData object.

    This function reads the following files:

        - ``{xx.XENIUM_SPECS!r}``: File containing specifications.
        - ``{xx.NUCLEUS_BOUNDARIES_FILE!r}``: Polygons of nucleus boundaries.
        - ``{xx.CELL_BOUNDARIES_FILE!r}``: Polygons of cell boundaries.
        - ``{xx.TRANSCRIPTS_FILE!r}``: File containing transcripts.
        - ``{xx.CELL_FEATURE_MATRIX_FILE!r}``: File containing cell feature matrix.
        - ``{xx.CELL_METADATA_FILE!r}``: File containing cell metadata.
        - ``{xx.MORPHOLOGY_MIP_FILE!r}``: File containing morphology mip.
        - ``{xx.MORPHOLOGY_FOCUS_FILE!r}``: File containing morphology focus.

    .. seealso::

        - `10x Genomics Xenium file format <https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-at-a-glance>`_.
        - `Release notes for the Xenium format <https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/release-notes/release-notes-for-xoa>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    cells_boundaries
        Whether to read cell boundaries (polygons).
    nucleus_boundaries
        Whether to read nucleus boundaries (polygons).
    cells_as_circles
        Whether to read cells also as circles (the center and the radius of each circle is computed from the
        corresponding labels cell).
    cells_labels
        Whether to read cell labels (raster). The polygonal version of the cell labels are simplified
        for visualization purposes, and using the raster version is recommended for analysis.
    nucleus_labels
        Whether to read nucleus labels (raster). The polygonal version of the nucleus labels are simplified
        for visualization purposes, and using the raster version is recommended for analysis.
    transcripts
        Whether to read transcripts.
    morphology_mip
        Whether to read the morphology mip image (available in versions < 2.0.0).
    morphology_focus
        Whether to read the morphology focus image.
    aligned_images
        Whether to also parse, when available, additional H&E or IF aligned images. For more control over the aligned
        images being read, in particular, to specify the axes of the aligned images, please set this parameter to
        `False` and use the `xenium_aligned_image` function directly.
    cells_table
        Whether to read the cell annotations in the `AnnData` table.
    n_jobs
        .. deprecated::
            ``n_jobs`` is not used anymore and will be removed in a future release. The reading time of shapes is now
            greatly improved and does not require parallelization.
    gex_only
        Whether to load only the "Gene Expression" feature type.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    labels_models_kwargs
        Keyword arguments to pass to the labels models.

    Returns
    -------
    :class:`spatialdata.SpatialData`

    Notes
    -----
    Old versions. Until spatialdata-io v0.6.0: `cells_as_circles` was `True` by default; the table was associated to the
    circles when `cells_as_circles` was `True`, and the table was associated to the polygons when `cells_as_circles`
    was `False`; the radii of the circles were computed form the nuclei instead of the cells.

    Performance. You can improve visualization performance (at the cost of accuracy) by setting `cells_as_circles` to `True`.

    Examples
    --------
    This code shows how to change the annotation target of the table from the cell circles to the cell labels.
    >>> from spatialdata_io import xenium
    >>> sdata = xenium("path/to/raw/data", cells_as_circles=True)
    >>> sdata["table"].obs["region"] = "cell_labels"
    >>> sdata.set_table_annotates_spatialelement(
    ...     table_name="table", region="cell_labels", region_key="region", instance_key="cell_labels"
    ... )
    >>> sdata.write("path/to/data.zarr")
    """
    if n_jobs is not None:
        warnings.warn(
            "The `n_jobs` parameter is deprecated and will be removed in a future release. "
            "The reading time of shapes is now greatly improved and does not require parallelization.",
            DeprecationWarning,
            stacklevel=2,
        )
    image_models_kwargs, labels_models_kwargs = _initialize_raster_models_kwargs(
        image_models_kwargs, labels_models_kwargs
    )
    path = Path(path)
    with open(path / XeniumKeys.XENIUM_SPECS) as f:
        specs = json.load(f)
    # to trigger the warning if the version cannot be parsed
    version = _parse_version_of_xenium_analyzer(specs, hide_warning=False)

    specs["region"] = "cell_circles" if cells_as_circles else "cell_labels"

    # the table is required in some cases
    if not cells_table:
        if cells_as_circles:
            logging.info(
                'When "cells_as_circles" is set to `True` reading the table is required; setting `cell_annotations` to '
                "`True`."
            )
            cells_table = True
        if cells_boundaries or nucleus_boundaries:
            logging.info(
                'When "cell_boundaries" or "nucleus_boundaries" is set to `True` reading the table is required; '
                "setting `cell_annotations` to `True`."
            )
            cells_table = True

    if cells_table:
        return_values = _get_tables_and_circles(path, cells_as_circles, specs, gex_only)
        if cells_as_circles:
            table, circles = return_values
        else:
            table = return_values
    else:
        table = None

    # open cells.zarr.zip once and reuse across all functions that need it
    cells_zarr: zarr.Group | None = None
    need_cells_zarr = (
        nucleus_boundaries
        or nucleus_labels
        or cells_boundaries
        or cells_labels
        or (version is not None and version >= packaging.version.parse("2.0.0") and table is not None)
    )
    if need_cells_zarr:
        cells_zarr_store = zarr.storage.ZipStore(path / XeniumKeys.CELLS_ZARR, read_only=True)
        cells_zarr = zarr.open(cells_zarr_store, mode="r")

    # pre-compute cell_id strings from the zarr once, to avoid redundant conversion.
    cells_zarr_cell_id_str: np.ndarray | None = None
    if cells_zarr is not None and version is not None and version >= packaging.version.parse("1.3.0"):
        cell_id_raw = cells_zarr["cell_id"][...]
        cell_id_prefix, dataset_suffix = cell_id_raw[:, 0], cell_id_raw[:, 1]
        cells_zarr_cell_id_str = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)

    if version is not None and version >= packaging.version.parse("2.0.0") and table is not None:
        assert cells_zarr is not None
        cell_summary_table = _get_cells_metadata_table_from_zarr(cells_zarr, cells_zarr_cell_id_str)
        try:
            _assert_arrays_equal_sampled(
                cell_summary_table[XeniumKeys.CELL_ID].values, table.obs[XeniumKeys.CELL_ID].values
            )
        except AssertionError:
            warnings.warn(
                'The "cell_id" column in the cells metadata table does not match the "cell_id" column in the annotation'
                " table. This could be due to trying to read a new version that is not supported yet. Please "
                "report this issue.",
                UserWarning,
                stacklevel=2,
            )
        table.obs[XeniumKeys.Z_LEVEL] = cell_summary_table[XeniumKeys.Z_LEVEL].values
        table.obs[XeniumKeys.NUCLEUS_COUNT] = cell_summary_table[XeniumKeys.NUCLEUS_COUNT].values

    polygons = {}
    labels = {}
    tables = {}
    points = {}
    images = {}

    # Build the label_index <-> cell_id mappings from the zarr once, reuse for both labels
    # and boundaries. For v2.0+ this is deterministic from the zarr polygon_sets
    # (label_id = cell_index + 1). For older versions, use seg_mask_value (cells only).
    # For nuclei in v2.0+, this correctly handles multinucleate cells: each nucleus gets its
    # own label_index, avoiding the bug of merging multiple nuclei into a single polygon.
    # Older versions do not support multinucleate cells, so cell_id-based grouping is correct.
    nucleus_indices_mapping: pd.DataFrame | None = None
    cell_indices_mapping: pd.DataFrame | None = None
    if cells_zarr_cell_id_str is not None and cells_zarr is not None and "polygon_sets" in cells_zarr:
        if nucleus_boundaries or nucleus_labels:
            nucleus_indices_mapping = _get_indices_mapping_from_zarr(cells_zarr, cells_zarr_cell_id_str, mask_index=0)
        if cells_boundaries or cells_labels:
            cell_indices_mapping = _get_indices_mapping_from_zarr(cells_zarr, cells_zarr_cell_id_str, mask_index=1)
    elif cells_zarr_cell_id_str is not None:
        if cells_boundaries or cells_labels:
            cell_indices_mapping = _get_indices_mapping_legacy(cells_zarr, cells_zarr_cell_id_str, specs=specs)

    if nucleus_labels:
        labels["nucleus_labels"] = _get_labels(cells_zarr, mask_index=0, labels_models_kwargs=labels_models_kwargs)
    if cells_labels:
        labels["cell_labels"] = _get_labels(cells_zarr, mask_index=1, labels_models_kwargs=labels_models_kwargs)
        if cell_indices_mapping is not None and table is not None:
            try:
                _assert_arrays_equal_sampled(
                    cell_indices_mapping["cell_id"].values, table.obs[str(XeniumKeys.CELL_ID)].values
                )
            except AssertionError:
                warnings.warn(
                    "The cell_id column in the cell_labels_table does not match the cell_id column derived from the "
                    "cell labels data. This could be due to trying to read a new version that is not supported yet. "
                    "Please report this issue.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                table.obs["cell_labels"] = cell_indices_mapping["label_index"].values
                if not cells_as_circles:
                    table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] = "cell_labels"

    if nucleus_boundaries:
        polygons["nucleus_boundaries"] = _get_polygons(
            path,
            XeniumKeys.NUCLEUS_BOUNDARIES_FILE,
            specs,
            indices_mapping=nucleus_indices_mapping,
            is_nucleus=True,
        )

    if cells_boundaries:
        polygons["cell_boundaries"] = _get_polygons(
            path,
            XeniumKeys.CELL_BOUNDARIES_FILE,
            specs,
            indices_mapping=cell_indices_mapping,
        )

    if transcripts:
        points["transcripts"] = _get_points(path, specs)

    if version is None or version < packaging.version.parse("2.0.0"):
        if morphology_mip:
            images["morphology_mip"] = _get_images(
                path,
                XeniumKeys.MORPHOLOGY_MIP_FILE,
                imread_kwargs,
                image_models_kwargs,
            )
        if morphology_focus:
            images["morphology_focus"] = _get_images(
                path,
                XeniumKeys.MORPHOLOGY_FOCUS_FILE,
                imread_kwargs,
                image_models_kwargs,
            )
    else:
        if morphology_focus:
            morphology_focus_dir = path / XeniumKeys.MORPHOLOGY_FOCUS_DIR
            files = {f for f in os.listdir(morphology_focus_dir) if f.endswith(".ome.tif") and not f.startswith("._")}

            if XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_IMAGE.value.format(0) in files:
                # v2 or v3
                first_tiff_path = morphology_focus_dir / XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_IMAGE.value.format(0)
                if len(files) not in [1, 4]:
                    raise ValueError(
                        "Expected 1 (no segmentation kit) or 4 (segmentation kit) files in the morphology focus directory, "
                        f"found {len(files)}: {files}"
                    )
                if files != {XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_IMAGE.value.format(i) for i in range(len(files))}:
                    raise ValueError(
                        "Expected files in the morphology focus directory to be named as "
                        f"{XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_IMAGE.value.format(0)} to "
                        f"{XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_IMAGE.value.format(len(files) - 1)}, found {files}"
                    )
                if len(files) == 1:
                    channel_names = {
                        0: XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_0.value,
                    }
                else:
                    channel_names = {
                        0: XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_0.value,
                        1: XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_1.value,
                        2: XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_2.value,
                        3: XeniumKeys.MORPHOLOGY_FOCUS_CHANNEL_3.value,
                    }
            else:
                # slow import
                from ome_types import from_xml

                # v4
                if XeniumKeys.MORPHOLOGY_FOCUS_V4_DAPI_FILENAME.value not in files:
                    raise ValueError(
                        "Expected files in the morphology focus directory to be named as "
                        f"chNNNN_<name>.ome.tif starting with {XeniumKeys.MORPHOLOGY_FOCUS_V4_DAPI_FILENAME.value}"
                    )
                first_tiff_path = morphology_focus_dir / XeniumKeys.MORPHOLOGY_FOCUS_V4_DAPI_FILENAME.value
                ome = from_xml(tifffile.tiffcomment(first_tiff_path), validate=False)

                # Get channel names from the OME XML
                ome_channels = ome.images[0].pixels.channels
                channels = []
                for ome_ch in ome_channels:
                    if ome_ch.name is None:
                        raise ValueError(f"Found a channel without a name in {first_tiff_path}")
                    # Parse the channel index from the channel id
                    match = re.match(r"Channel:(\d+)", ome_ch.id)
                    invalid_format_msg: str = (
                        "Expected OME channel ID to be of the form 'Channel:<index>'. "
                        + f"Found: {ome_ch.id} in file {first_tiff_path}"
                    )
                    if match is None:
                        raise ValueError(invalid_format_msg)
                    try:
                        channel_idx = int(match.group(1))
                    except ValueError as e:
                        raise ValueError(invalid_format_msg) from e
                    channels.append((channel_idx, ome_ch.name))
                channel_names = dict(sorted(channels))

            # this reads the scale 0 for all the 1 or 4 channels (the other files are parsed automatically)
            # dask.image.imread will call tifffile.imread which will give a warning saying that reading multi-file
            # pyramids is not supported; since we are reading the full scale image and reconstructing the pyramid, we
            # can ignore this

            class IgnoreSpecificMessage(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    # Ignore specific log message
                    if "OME series cannot read multi-file pyramids" in record.getMessage():
                        return False
                    return True

            tf_logger = tifffile.logger()
            tf_logger.addFilter(IgnoreSpecificMessage())
            image_models_kwargs = dict(image_models_kwargs)
            assert "c_coords" not in image_models_kwargs, (
                "The channel names for the morphology focus images are handled internally"
            )
            image_models_kwargs["c_coords"] = list(channel_names.values())
            images["morphology_focus"] = _get_images(
                morphology_focus_dir,
                first_tiff_path.name,
                imread_kwargs,
                image_models_kwargs,
            )
            del image_models_kwargs["c_coords"]
            tf_logger.removeFilter(IgnoreSpecificMessage())

    if table is not None:
        tables["table"] = table

    elements_dict = {
        "images": images,
        "labels": labels,
        "points": points,
        "tables": tables,
        "shapes": polygons,
    }
    if cells_as_circles:
        elements_dict["shapes"][specs["region"]] = circles
    sdata = SpatialData(**elements_dict)

    # find and add additional aligned images
    if aligned_images:
        extra_images = _add_aligned_images(path, imread_kwargs, image_models_kwargs)
        for key, value in extra_images.items():
            sdata.images[key] = value

    return _set_reader_metadata(sdata, "xenium")


def _assert_arrays_equal_sampled(a: ArrayLike, b: ArrayLike, n: int = 100) -> None:
    """Assert two arrays are equal by checking a random sample of entries."""
    assert len(a) == len(b), f"Array lengths differ: {len(a)} != {len(b)}"
    idx = np.random.default_rng(0).choice(len(a), size=min(n, len(a)), replace=False)
    np.testing.assert_array_equal(np.asarray(a[idx]), np.asarray(b[idx]))


def _decode_cell_id_column(cell_id_column: pd.Series) -> pd.Series:
    if isinstance(cell_id_column.iloc[0], bytes):
        return cell_id_column.str.decode("utf-8")
    if not isinstance(cell_id_column.iloc[0], str):
        cell_id_column.index = cell_id_column.index.astype(str)
    return cell_id_column


def _get_polygons(
    path: Path,
    file: str,
    specs: dict[str, Any],
    indices_mapping: pd.DataFrame | None = None,
    is_nucleus: bool = False,
) -> GeoDataFrame:
    """Parse boundary polygons from a parquet file.

    Parameters
    ----------
    indices_mapping
        When provided (from ``_get_indices_mapping_from_zarr`` or ``_get_indices_mapping_legacy``),
        contains ``cell_id`` and ``label_index`` columns. The parquet ``label_id`` column is used
        for fast integer-based change detection (to locate all the vertices of each polygon).
        When None, falls back to cell_id-based grouping from the parquet (Xenium < 2.0).
    is_nucleus
        When True (nucleus boundaries), use ``label_index`` as the GeoDataFrame index and store
        ``cell_id`` as a column. This gives each nucleus a distinct integer id matching the raster
        labels, correctly handling multinucleate cells.
        When False (cell boundaries), use ``cell_id`` as the GeoDataFrame index.
    """
    columns_to_read = [str(XeniumKeys.BOUNDARIES_VERTEX_X), str(XeniumKeys.BOUNDARIES_VERTEX_Y)]
    if indices_mapping is not None:
        columns_to_read.append("label_id")
    else:
        columns_to_read.append(str(XeniumKeys.CELL_ID))
    table = pq.read_table(path / file, columns=columns_to_read)

    x = table.column(str(XeniumKeys.BOUNDARIES_VERTEX_X)).to_numpy()
    y = table.column(str(XeniumKeys.BOUNDARIES_VERTEX_Y)).to_numpy()
    coords = np.column_stack([x, y])

    n = len(x)

    if indices_mapping is not None:
        id_col = table.column("label_id")
        id_arr = id_col.to_numpy()
        change_mask = id_arr[1:] != id_arr[:-1]
    else:
        id_col = table.column(str(XeniumKeys.CELL_ID))
        change_mask = pc.not_equal(id_col.slice(0, n - 1), id_col.slice(1)).to_numpy(zero_copy_only=False)
    group_starts = np.where(np.concatenate([[True], change_mask]))[0]
    n_unique_ids = pc.count_distinct(id_col).as_py()
    if len(group_starts) != n_unique_ids:
        raise ValueError(
            f"In {file}, rows belonging to the same polygon must be contiguous. "
            f"Expected {n_unique_ids} group starts, but found {len(group_starts)}. "
            f"This indicates non-consecutive polygon rows."
        )

    group_ends = np.concatenate([group_starts[1:], [n]])

    # offsets for ragged array:
    # offsets[0] (ring_offsets): describing to which rings the vertex positions belong to
    # offsets[1] (geom_offsets): describing to which polygons the rings belong to
    ring_offsets = np.concatenate([[0], group_ends])  # vertex positions
    geom_offsets = np.arange(len(group_starts) + 1)  # [0, 1, 2, ..., n_polygons]

    geoms = from_ragged_array(GeometryType.POLYGON, coords, offsets=(ring_offsets, geom_offsets))

    if indices_mapping is not None:
        assert len(indices_mapping) == len(group_starts), (
            f"Expected {len(group_starts)} polygons, but indices_mapping has {len(indices_mapping)} entries."
        )
        if is_nucleus:
            # Use label_index (int) as GeoDataFrame index, cell_id as column.
            geo_df = GeoDataFrame(
                {"geometry": geoms, str(XeniumKeys.CELL_ID): indices_mapping["cell_id"].values},
                index=indices_mapping["label_index"].values,
            )
        else:
            # Use cell_id (str) as GeoDataFrame index.
            geo_df = GeoDataFrame({"geometry": geoms}, index=indices_mapping["cell_id"].values)
    else:
        # Fall back to extracting unique cell IDs from parquet (slow for large_string columns).
        unique_ids = id_col.filter(np.concatenate([[True], change_mask])).to_pylist()
        index = _decode_cell_id_column(pd.Series(unique_ids))
        geo_df = GeoDataFrame({"geometry": geoms}, index=index.values)

    scale = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    return ShapesModel.parse(geo_df, transformations={"global": scale})


def _get_labels(
    cells_zarr: zarr.Group,
    mask_index: int,
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> DataArray:
    """Read the labels raster from cells.zarr.zip masks/{mask_index}."""
    if mask_index not in [0, 1]:
        raise ValueError(f"mask_index must be 0 or 1, found {mask_index}.")
    masks = da.from_array(cells_zarr["masks"][f"{mask_index}"])
    return Labels2DModel.parse(masks, dims=("y", "x"), transformations={"global": Identity()}, **labels_models_kwargs)


def _get_indices_mapping_from_zarr(
    cells_zarr: zarr.Group,
    cells_zarr_cell_id_str: np.ndarray,
    mask_index: int,
) -> pd.DataFrame:
    """Build the label_index <-> cell_id mapping from the zarr polygon_sets.

    From the 10x Genomics docs: "the label ID is equal to the cell index + 1",
    where cell_index is polygon_sets/{mask_index}/cell_index. This is deterministic
    and avoids reading the slow parquet boundary files.

    For cells (mask_index=1): cell_index is 0..N-1 (1:1 with cells), so
    label_index = arange(1, N+1).
    For nuclei (mask_index=0): cell_index maps each nucleus to its parent cell,
    so label_index = arange(1, M+1) and cell_id = cell_id_str[cell_index[i]].
    """
    cell_index = cells_zarr[f"polygon_sets/{mask_index}/cell_index"][...]
    label_index = np.arange(1, len(cell_index) + 1, dtype=np.int64)
    cell_id = cells_zarr_cell_id_str[cell_index]
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "label_index": label_index,
        }
    )


def _get_indices_mapping_legacy(
    cells_zarr: zarr.Group,
    cell_id_str: ArrayLike,
    specs: dict[str, Any],
) -> pd.DataFrame | None:
    """Build the label_index <-> cell_id mapping for versions < 2.0.0.

    Uses seg_mask_value from the zarr (available in v1.3.0+).
    """
    version = _parse_version_of_xenium_analyzer(specs)
    if version is None or version < packaging.version.parse("1.3.0"):
        return None
    label_index = cells_zarr["seg_mask_value"][...]
    return pd.DataFrame(
        {
            "cell_id": cell_id_str,
            "label_index": label_index.astype(np.int64),
        }
    )


@inject_docs(xx=XeniumKeys)
def _get_cells_metadata_table_from_zarr(
    cells_zarr: zarr.Group,
    cell_id_str: ArrayLike,
) -> AnnData:
    """Read cells metadata from ``{xx.CELLS_ZARR}``.

    Read the cells summary table, which contains the z_level information for versions < 2.0.0, and also the
    nucleus_count for versions >= 2.0.0.
    """
    x = cells_zarr["cell_summary"][...]
    column_names = cells_zarr["cell_summary"].attrs["column_names"]
    df = pd.DataFrame(x, columns=column_names)

    df[XeniumKeys.CELL_ID] = cell_id_str
    return df


def _get_points(path: Path, specs: dict[str, Any]) -> pa.Table:
    table = read_parquet(path / XeniumKeys.TRANSCRIPTS_FILE)

    # check if we need to decode bytes
    sample = table[XeniumKeys.FEATURE_NAME].head(1)
    needs_decode = isinstance(sample.iloc[0], bytes)

    # get unique categories (fast)
    categories = table[XeniumKeys.FEATURE_NAME].drop_duplicates().compute()
    if needs_decode:
        categories = categories.str.decode("utf-8")
    cat_dtype = pd.CategoricalDtype(categories=categories)

    # decode column if needed, then convert to categorical
    if needs_decode:
        table[XeniumKeys.FEATURE_NAME] = table[XeniumKeys.FEATURE_NAME].map_partitions(
            lambda s: s.str.decode("utf-8").astype(cat_dtype), meta=pd.Series(dtype=cat_dtype)
        )
    else:
        table[XeniumKeys.FEATURE_NAME] = table[XeniumKeys.FEATURE_NAME].astype(cat_dtype)

    transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    points = PointsModel.parse(
        table,
        coordinates={
            "x": XeniumKeys.TRANSCRIPTS_X,
            "y": XeniumKeys.TRANSCRIPTS_Y,
            "z": XeniumKeys.TRANSCRIPTS_Z,
        },
        feature_key=XeniumKeys.FEATURE_NAME,
        instance_key=XeniumKeys.CELL_ID,
        transformations={"global": transform},
        sort=True,
    )
    return points


def _get_tables_and_circles(
    path: Path, cells_as_circles: bool, specs: dict[str, Any], gex_only: bool
) -> AnnData | tuple[AnnData, AnnData]:
    adata = _read_10x_h5(path / XeniumKeys.CELL_FEATURE_MATRIX_FILE, gex_only=gex_only)
    metadata = pd.read_parquet(path / XeniumKeys.CELL_METADATA_FILE)
    _assert_arrays_equal_sampled(metadata.cell_id.astype(str), adata.obs_names.values)
    circ = metadata[[XeniumKeys.CELL_X, XeniumKeys.CELL_Y]].to_numpy()
    adata.obsm["spatial"] = circ
    metadata.drop([XeniumKeys.CELL_X, XeniumKeys.CELL_Y], axis=1, inplace=True)
    # avoids anndata's ImplicitModificationWarning
    metadata.index = adata.obs_names
    adata.obs = metadata
    adata.obs["region"] = specs["region"]
    adata.obs["region"] = adata.obs["region"].astype("category")
    adata.obs[XeniumKeys.CELL_ID] = _decode_cell_id_column(adata.obs[XeniumKeys.CELL_ID])
    table = TableModel.parse(
        adata,
        region=specs["region"],
        region_key="region",
        instance_key=str(XeniumKeys.CELL_ID),
    )
    if cells_as_circles:
        transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
        radii = np.sqrt(adata.obs[XeniumKeys.CELL_AREA].to_numpy() / np.pi)
        circles = ShapesModel.parse(
            circ,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs[XeniumKeys.CELL_ID].copy(),
        )
        return table, circles
    return table


def _get_images(
    path: Path,
    file: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> DataArray | DataTree:
    image = imread(path / file, **imread_kwargs)
    if "c_coords" in image_models_kwargs and "dummy" in image_models_kwargs["c_coords"]:
        # Napari currently interprets 4 channel images as RGB; a series of PRs to fix this is almost ready but they will
        # not be merged soon.
        # Here, since the new data from the xenium analyzer version 2.0.0 gives 4-channel images that are not RGBA,
        # let's add a dummy channel as a temporary workaround.
        image = da.concatenate([image, da.zeros_like(image[0:1])], axis=0)
    return Image2DModel.parse(
        image,
        transformations={"global": Identity()},
        dims=("c", "y", "x"),
        rgb=None,
        **image_models_kwargs,
    )


def _add_aligned_images(
    path: Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> dict[str, DataTree]:
    """Discover and parse aligned images."""
    images = {}
    ome_tif_files = list(path.glob("*.ome.tif"))
    csv_files = list(path.glob("*.csv"))
    for file in ome_tif_files:
        element_name = None
        for suffix in [
            XeniumKeys.ALIGNED_HE_IMAGE_SUFFIX,
            XeniumKeys.ALIGNED_IF_IMAGE_SUFFIX,
        ]:
            if file.name.endswith(suffix):
                element_name = suffix.replace(XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_REMOVE, "")
                break
        if element_name is not None:
            # check if an alignment file exists
            expected_filename = file.name.replace(
                XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_REMOVE,
                XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_ADD,
            )
            alignment_files = [f for f in csv_files if f.name == expected_filename]
            assert len(alignment_files) <= 1, f"Found more than one alignment file for {file.name}."
            alignment_file = alignment_files[0] if alignment_files else None

            # parse the image
            image = xenium_aligned_image(file, alignment_file, imread_kwargs, image_models_kwargs)
            images[element_name] = image
    return images


def xenium_aligned_image(
    image_path: str | Path,
    alignment_file: str | Path | None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    dims: tuple[str, ...] | None = None,
    rgba: bool = False,
    c_coords: list[str] | None = None,
) -> DataTree:
    """Read an image aligned to a Xenium dataset, with an optional alignment file.

    Parameters
    ----------
    image_path
        Path to the image.
    alignment_file
        Path to the alignment file, if not passed it is assumed that the image is aligned.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    dims
        Dimensions of the image (tuple of axes names); valid strings are "c", "x" and "y". If not passed, the function
        will try to infer the dimensions from the image shape. Please use this argument when the default behavior fails.
        Example: for an image with shape (1, y, 1, x, 3), use dims=("anystring", "y", "dummy", "x", "c"). Values that
        are not "c", "x" or "y" are considered dummy dimensions and will be squeezed (the data must have len 1 for
        those axes).
    rgba
        Interprets the `c` channel as RGBA, by setting the channel names to `r`, `g`, `b` (`a`). When `c_coords` is not
        `None`, this argument is ignored.
    c_coords
        Channel names for the image. By default, the function will try to infer the channel names from the image
        shape and name (by detecting if the name suggests that the image is a H&E image). Example: for an RGB image with
        shape (3, y, x), use c_coords=["r", "g", "b"].

    Returns
    -------
    The single-scale or multi-scale aligned image element.
    """
    image_path = Path(image_path)
    assert image_path.exists(), f"File {image_path} does not exist."
    image = imread(image_path, **imread_kwargs)

    # Depending on the version of pipeline that was used, some images have shape (1, y, x, 3) and others (3, y, x) or
    # (4, y, x).
    # since y and x are always different from 1, let's differentiate from the two cases here, independently of the
    # pipeline version.
    # Note that a more robust approach is to look at the xml metadata in the ome.tif; we should use this in a future PR.
    # In fact, it could be that the len(image.shape) == 4 has actually dimes (1, x, y, c) and not (1, y, x, c). This is
    # not a problem because the transformation is constructed to be consistent, but if is the case, the data orientation
    # would be transposed compared to the original image, not ideal.
    if dims is None:
        if len(image.shape) == 4:
            assert image.shape[0] == 1
            assert image.shape[-1] == 3
            image = image.squeeze(0)
            dims = ("y", "x", "c")
        else:
            assert len(image.shape) == 3
            assert image.shape[0] in [3, 4]
            dims = ("c", "y", "x")
    else:
        logging.info(f"Image has shape {image.shape}, parsing with dims={dims}.")
        image = DataArray(image, dims=dims)
        # squeeze spurious dimensions away
        to_squeeze = [dim for dim in dims if dim not in ["c", "x", "y"]]
        dims = tuple(dim for dim in dims if dim in ["c", "x", "y"])
        for dim in to_squeeze:
            image = image.squeeze(dim)

    if alignment_file is None:
        transformation = Identity()
    else:
        alignment_file = Path(alignment_file)
        assert alignment_file.exists(), f"File {alignment_file} does not exist."
        alignment = pd.read_csv(alignment_file, header=None).values
        transformation = Affine(alignment, input_axes=("x", "y"), output_axes=("x", "y"))

    if c_coords is None and (rgba or image_path.name.endswith(XeniumKeys.ALIGNED_HE_IMAGE_SUFFIX)):
        c_index = dims.index("c")
        n_channels = image.shape[c_index]
        if n_channels == 3:
            c_coords = ["r", "g", "b"]
        elif n_channels == 4:
            c_coords = ["r", "g", "b", "a"]

    return Image2DModel.parse(
        image,
        dims=dims,
        transformations={"global": transformation},
        c_coords=c_coords,
        **image_models_kwargs,
    )


def _selection_to_polygon(df: pd.DataFrame, pixel_size: float) -> Polygon:
    xy_keys = [XeniumKeys.EXPLORER_SELECTION_X, XeniumKeys.EXPLORER_SELECTION_Y]
    return Polygon(df[xy_keys].values / pixel_size)


def xenium_explorer_selection(
    path: str | Path, pixel_size: float = 0.2125, return_list: bool = False
) -> Polygon | list[Polygon]:
    """Read the coordinates of a selection `.csv` file exported from the `Xenium Explorer  <https://www.10xgenomics.com/support/software/xenium-explorer/latest>`_.

    This file can be generated by the "Freehand Selection" or the "Rectangular Selection".
    The output `Polygon` can be used for a polygon query on the pixel coordinate
    system (by default, this is the `"global"` coordinate system for Xenium data).
    If `spatialdata_xenium_explorer  <https://github.com/quentinblampey/spatialdata_xenium_explorer>`_ was used,
    the `pixel_size` argument must be set to the one used during conversion with `spatialdata_xenium_explorer`.

    In case multiple polygons were selected on the Explorer and exported into a single file, it will return a list of polygons.

    Parameters
    ----------
    path
        Path to the `.csv` file containing the selection coordinates
    pixel_size
        Size of a pixel in microns. By default, the Xenium pixel size is used.
    return_list
        If `True`, returns a list of Polygon even if only one polygon was selected

    Returns
    -------
    :class:`shapely.geometry.polygon.Polygon`
    """
    df = pd.read_csv(path, skiprows=2)

    if XeniumKeys.EXPLORER_SELECTION_KEY not in df:
        polygon = _selection_to_polygon(df, pixel_size)
        return [polygon] if return_list else polygon

    return [_selection_to_polygon(sub_df, pixel_size) for _, sub_df in df.groupby(XeniumKeys.EXPLORER_SELECTION_KEY)]


def _parse_version_of_xenium_analyzer(
    specs: dict[str, Any],
    hide_warning: bool = True,
) -> packaging.version.Version | None:
    # After using xeniumranger (e.g. 3.0.1.1) to resegment data from previous versions (e.g. xenium-1.6.0.7), a new dict is added to
    # `specs`, named 'xenium_ranger', which contains the key 'version' and whose value specifies the version of xeniumranger used to
    # resegment the data (e.g. 'xenium-3.0.1.1').
    # When parsing the outs/ folder from the resegmented data, this version (rather than the original 'analysis_sw_version') is used
    # whenever a code branch is dependent on the data version
    if specs.get(XeniumKeys.XENIUM_RANGER):
        string = specs[XeniumKeys.XENIUM_RANGER]["version"]
    else:
        string = specs[XeniumKeys.ANALYSIS_SW_VERSION]

    pattern = r"^(?:x|X)enium-(\d+\.\d+\.\d+(\.\d+-\d+)?)"

    result = re.search(pattern, string)
    # Example
    # Input: xenium-2.0.0.6-35-ga7e17149a
    # Output: 2.0.0.6-35

    warning_message = (
        f"Could not parse the version of the Xenium Analyzer from the string: {string}. This may happen for "
        "experimental version of the data. Please report in GitHub https://github.com/scverse/spatialdata-io/issues.\n"
        "The reader will continue assuming the latest version of the Xenium Analyzer."
    )

    if result is None:
        if not hide_warning:
            warnings.warn(warning_message, stacklevel=2)
        return None

    group = result.groups()[0]
    try:
        return packaging.version.parse(group)
    except packaging.version.InvalidVersion:
        if not hide_warning:
            warnings.warn(warning_message, stacklevel=2)
        return None


def _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix: ArrayLike, dataset_suffix: ArrayLike) -> ArrayLike:
    """Reference implementation of cell_id_str_from_prefix_suffix_uint32.

    Readable but slow for large arrays due to Python-level string operations.
    Kept as ground truth for testing the optimized version.

    See https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cellID
    """
    # convert to hex, remove the 0x prefix
    cell_id_prefix_hex = [hex(x)[2:] for x in cell_id_prefix]

    # shift the hex values: '0'->'a', ..., '9'->'j', 'a'->'k', ..., 'f'->'p'
    hex_shift = {str(i): chr(ord("a") + i) for i in range(10)} | {
        chr(ord("a") + i): chr(ord("a") + 10 + i) for i in range(6)
    }
    cell_id_prefix_hex_shifted = ["".join([hex_shift[c] for c in x]) for x in cell_id_prefix_hex]

    # merge the prefix and the suffix
    cell_id_str = [
        str(x[0]).rjust(8, "a") + f"-{x[1]}" for x in zip(cell_id_prefix_hex_shifted, dataset_suffix, strict=False)
    ]

    return np.array(cell_id_str)


def cell_id_str_from_prefix_suffix_uint32(cell_id_prefix: ArrayLike, dataset_suffix: ArrayLike) -> ArrayLike:
    """Convert cell ID prefix/suffix uint32 pairs to the Xenium string representation.

    Each uint32 prefix is converted to 8 hex nibbles, each mapped to a character
    (0->'a', 1->'b', ..., 15->'p'), then joined with "-{suffix}".

    See https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cellID
    """
    cell_id_prefix = np.asarray(cell_id_prefix, dtype=np.uint32)
    dataset_suffix = np.asarray(dataset_suffix)

    # Extract 8 hex nibbles (4 bits each) from each uint32, most significant first.
    # Each nibble maps to a character: 0->'a', 1->'b', ..., 9->'j', 10->'k', ..., 15->'p'.
    # Leading zero nibbles become 'a', equivalent to rjust(8, 'a') padding.
    shifts = np.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=np.uint32)
    nibbles = (cell_id_prefix[:, np.newaxis] >> shifts) & 0xF
    char_codes = (nibbles + ord("a")).astype(np.uint8)

    # View the (n, 8) uint8 array as n byte-strings of length 8
    prefix_strs = char_codes.view("S8").ravel().astype("U8")

    suffix_strs = np.char.add("-", dataset_suffix.astype("U"))
    return np.char.add(prefix_strs, suffix_strs)


def prefix_suffix_uint32_from_cell_id_str(
    cell_id_str: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    # parse the string into the prefix and suffix
    cell_id_prefix_str, dataset_suffix = zip(*[x.split("-") for x in cell_id_str], strict=False)
    dataset_suffix_int = [int(x) for x in dataset_suffix]

    # reverse the shifted hex conversion
    hex_unshift = {chr(ord("a") + i): str(i) for i in range(10)} | {
        chr(ord("a") + 10 + i): chr(ord("a") + i) for i in range(6)
    }
    cell_id_prefix_hex = ["".join([hex_unshift[c] for c in x]) for x in cell_id_prefix_str]

    # Convert hex (no need to add the 0x prefix)
    cell_id_prefix = [int(x, 16) for x in cell_id_prefix_hex]

    return np.array(cell_id_prefix, dtype=np.uint32), np.array(dataset_suffix_int)
