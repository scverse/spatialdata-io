from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
import xmltodict
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations.transformations import Identity, Scale

from spatialdata_io._constants._constants import SeqfishKeys as SK
from spatialdata_io._docs import inject_docs

__all__ = ["seqfish"]

LARGE_IMAGE_THRESHOLD = 100_000_000


@inject_docs(vx=SK, megapixels_value=str(int(LARGE_IMAGE_THRESHOLD / 1e6)))
def seqfish(
    path: str | Path,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    load_shapes: bool = True,
    cells_as_circles: bool = False,
    rois: list[str] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    raster_models_scale_factors: list[int] | None = None,
) -> SpatialData:
    """
    Read *seqfish* formatted dataset.

    This function reads the following files:

        - ```<roi_prefix>_{vx.COUNTS_FILE!r}{vx.CSV_FILE!r}```: Counts and metadata file.
        - ```<roi_prefix>_{vx.CELL_COORDINATES!r}{vx.CSV_FILE!r}```: Cell coordinates file.
        - ```<roi_prefix>_{vx.DAPI!r}{vx.TIFF_FILE!r}```: High resolution tiff image.
        - ```<roi_prefix>_{vx.SEGMENTATION!r}{vx.TIFF_FILE!r}```: Cell mask file.
        - ```<roi_prefix>_{vx.TRANSCRIPT_COORDINATES!r}{vx.CSV_FILE!r}```: Transcript coordinates file.

    .. seealso::

        - `seqfish output <https://spatialgenomics.com/data/>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    load_images
        Whether to load the images.
    load_labels
        Whether to load cell segmentation.
    load_points
        Whether to load the transcript locations.
    load_shapes
        Whether to load cells as shape.
    cells_as_circles
        Whether to read cells also as circles instead of labels.
    rois
        Which ROIs (specified as strings, without trailing "_") to load (the ROI strings are used as prefixes for the
        filenames). If `None`, all ROIs are loaded.
    imread_kwargs
        Keyword arguments to pass to :func:`dask_image.imread.imread`.
    raster_models_scale_factors
        Scale factors to downscale high-resolution images and labels. The scale factors will be automatically set to
        obtain a multi-scale image for all the images and labels that are larger than {megapixels_value} megapixels.

    Returns
    -------
    :class:`spatialdata.SpatialData`

    Examples
    --------
    This code shows how to change the annotation target of the table from the cell labels to the cell boundaries.
    Please check that the string Roi1 is used in the naming of your dataset, otherwise adjust the code below.
    >>> from spatialdata_io import seqfish
    >>> sdata = seqfish("path/to/raw/data")
    >>> sdata["table_Roi1"].obs["region"] = "Roi1_Boundaries"
    >>> sdata.set_table_annotates_spatialelement(
    ...     table_name="table_Roi1", region="Roi1_Boundaries", region_key="region", instance_key="instance_id"
    ... )
    >>> sdata.write("path/to/data.zarr")
    """
    path = Path(path)
    count_file_pattern = re.compile(rf"(.*?)_{re.escape(SK.CELL_COORDINATES)}{re.escape(SK.CSV_FILE)}$")
    count_files = [f for f in os.listdir(path) if count_file_pattern.match(f)]
    if not count_files:
        raise ValueError(
            f"No files matching the pattern {count_file_pattern} were found. Cannot infer the naming scheme."
        )

    rois_str_set = set()
    for count_file in count_files:
        found = count_file_pattern.match(count_file)
        if found is None:
            raise ValueError(f"File {count_file} does not match the expected pattern.")
        rois_str_set.add(found.group(1))
    logger.info(f"Found ROIs: {rois_str_set}")
    rois_str = list(rois_str_set)

    if isinstance(rois, list):
        for roi in rois:
            if str(roi) not in rois_str_set:
                raise ValueError(f"ROI{roi} not found.")
        rois_str = rois
    elif rois is not None:
        raise ValueError("Invalid type for 'roi'. Must be list[str] or None.")

    def get_cell_file(roi: str) -> str:
        return f"{roi}_{SK.CELL_COORDINATES}{SK.CSV_FILE}"

    def get_count_file(roi: str) -> str:
        return f"{roi}_{SK.COUNTS_FILE}{SK.CSV_FILE}"

    def get_dapi_file(roi: str) -> str:
        return f"{roi}_{SK.DAPI}{SK.TIFF_FILE}"

    def get_cell_segmentation_labels_file(roi: str) -> str:
        return f"{roi}_{SK.SEGMENTATION}{SK.TIFF_FILE}"

    def get_cell_segmentation_shapes_file(roi: str) -> str:
        return f"{roi}_{SK.BOUNDARIES}{SK.GEOJSON_FILE}"

    def get_transcript_file(roi: str) -> str:
        return f"{roi}_{SK.TRANSCRIPT_COORDINATES}{SK.CSV_FILE}"

    # parse table information
    tables: dict[str, ad.AnnData] = {}
    for roi_str in rois_str:
        # parse cell gene expression data
        count_matrix = get_count_file(roi_str)
        df = pd.read_csv(path / count_matrix, delimiter=",")
        instance_id = df.iloc[:, 0].astype(str)
        expression = df.drop(columns=["Unnamed: 0"])
        expression.set_index(instance_id, inplace=True)
        adata = ad.AnnData(expression)

        # parse cell spatial information
        cell_file = get_cell_file(roi_str)
        cell_info = pd.read_csv(path / cell_file, delimiter=",")
        cell_info["label"] = cell_info["label"].astype("str")
        # below, the obsm are assigned by position, not by index. Here we check that we can do it
        assert cell_info["label"].to_numpy().tolist() == adata.obs.index.to_numpy().tolist()
        cell_info.set_index("label", inplace=True)
        adata.obs[SK.AREA] = cell_info[SK.AREA]
        adata.obsm[SK.SPATIAL_KEY] = cell_info[[SK.CELL_X, SK.CELL_Y]].to_numpy()

        # map tables to cell labels (defined later)
        region = os.path.splitext(get_cell_segmentation_labels_file(roi_str))[0]
        adata.obs[SK.REGION_KEY] = region
        adata.obs[SK.REGION_KEY] = adata.obs[SK.REGION_KEY].astype("category")
        adata.obs[SK.INSTANCE_KEY_TABLE] = instance_id.to_numpy().astype(np.uint16)
        adata.obs = adata.obs.reset_index(drop=True)
        tables[f"table_{roi_str}"] = TableModel.parse(
            adata,
            region=region,
            region_key=SK.REGION_KEY.value,
            instance_key=SK.INSTANCE_KEY_TABLE.value,
        )

    # parse scale factors to scale images and labels
    scaled = {}
    for roi_str in rois_str:
        scaled[roi_str] = Scale(
            np.array(_get_scale_factors_scale0(path / get_dapi_file(roi_str))),
            axes=("y", "x"),
        )

    def _get_scale_factors(raster_path: Path, raster_models_scale_factors: list[int] | None) -> list[int] | None:
        n_pixels = _get_n_pixels(raster_path)
        if n_pixels > LARGE_IMAGE_THRESHOLD and raster_models_scale_factors is None:
            return [2, 2, 2]
        else:
            return raster_models_scale_factors

    if load_images:
        images = {}
        for x in rois_str:
            image_path = path / get_dapi_file(x)
            scale_factors = _get_scale_factors(image_path, raster_models_scale_factors)

            images[f"{os.path.splitext(get_dapi_file(x))[0]}"] = Image2DModel.parse(
                imread(image_path, **imread_kwargs),
                dims=("c", "y", "x"),
                scale_factors=scale_factors,
                transformations={x: scaled[x]},
            )
    else:
        images = {}

    if load_labels:
        labels = {}
        for x in rois_str:
            labels_path = path / get_cell_segmentation_labels_file(x)
            scale_factors = _get_scale_factors(labels_path, raster_models_scale_factors)

            labels[f"{os.path.splitext(get_cell_segmentation_labels_file(x))[0]}"] = Labels2DModel.parse(
                imread(labels_path, **imread_kwargs).squeeze(),
                dims=("y", "x"),
                scale_factors=scale_factors,
                transformations={x: scaled[x]},
            )
    else:
        labels = {}

    points = {}
    if load_points:
        for x in rois_str:

            # prepare data
            name = f"{os.path.splitext(get_transcript_file(x))[0]}"
            p = pd.read_csv(path / get_transcript_file(x), delimiter=",")
            instance_key_points = SK.INSTANCE_KEY_POINTS.value if SK.INSTANCE_KEY_POINTS.value in p.columns else None

            coordinates = {"x": SK.TRANSCRIPTS_X, "y": SK.TRANSCRIPTS_Y, "z": SK.TRANSCRIPTS_Z}
            if SK.TRANSCRIPTS_Z not in p.columns:
                coordinates.pop("z")
                warnings.warn(
                    f"Column {SK.TRANSCRIPTS_Z} not found in {get_transcript_file(x)}.", UserWarning, stacklevel=2
                )

            # call parser
            points[name] = PointsModel.parse(
                p,
                coordinates=coordinates,
                feature_key=SK.FEATURE_KEY.value,
                instance_key=instance_key_points,
                transformations={x: Identity()},
            )

    shapes = {}
    if cells_as_circles:
        for x, adata in zip(rois_str, tables.values()):
            shapes[f"{os.path.splitext(get_cell_file(x))[0]}"] = ShapesModel.parse(
                adata.obsm[SK.SPATIAL_KEY],
                geometry=0,
                radius=np.sqrt(adata.obs[SK.AREA].to_numpy() / np.pi),
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
                transformations={x: Identity()},
            )
    if load_shapes:
        for x, adata in zip(rois_str, tables.values()):
            # this assumes that the index matches the instance key of the table. A more robust approach could be
            # implemented, as described here https://github.com/scverse/spatialdata-io/issues/249
            shapes[f"{os.path.splitext(get_cell_segmentation_shapes_file(x))[0]}"] = ShapesModel.parse(
                path / get_cell_segmentation_shapes_file(x),
                transformations={x: scaled[x]},
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
            )

    sdata = SpatialData(images=images, labels=labels, points=points, tables=tables, shapes=shapes)

    return sdata


def _is_ome_tiff_multiscale(ome_tiff_file: Path) -> bool:
    """
    Check if the OME-TIFF file is multi-scale.

    Parameters
    ----------
    ome_tiff_file
        Path to the OME-TIFF file.

    Returns
    -------
    Whether the OME-TIFF file is multi-scale.
    """
    # for some image files we couldn't find the multiscale information in the omexml metadata, and this method proves to
    # be more robust
    try:
        zarr_tiff_store = tifffile.imread(ome_tiff_file, is_ome=True, level=1, aszarr=True)
        zarr_tiff_store.close()
    except IndexError:
        return False
    return True


def _get_n_pixels(ome_tiff_file: Path) -> int:
    with tifffile.TiffFile(ome_tiff_file, is_ome=True) as tif:
        page = tif.pages[0]
        shape = page.shape
        n_pixels = np.array(shape).prod()
        assert isinstance(n_pixels, int)
        return n_pixels


def _get_scale_factors_scale0(DAPI_path: Path) -> list[float]:
    with tifffile.TiffFile(DAPI_path, is_ome=True) as tif:
        ome_metadata = xmltodict.parse(tif.ome_metadata)
        scalefactor_x = ome_metadata["OME"]["Image"]["Pixels"]["@PhysicalSizeX"]
        scalefactor_y = ome_metadata["OME"]["Image"]["Pixels"]["@PhysicalSizeY"]
        return [float(scalefactor_x), float(scalefactor_y)]
