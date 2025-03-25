from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
from dask_image.imread import imread
from spatialdata import SpatialData
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

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["seqfish"]


@inject_docs(vx=SK)
def seqfish(
    path: str | Path,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    load_shapes: bool = True,
    cells_as_circles: bool = False,
    rois: list[int] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    raster_models_scale_factors: list[float] | None = None,
) -> SpatialData:
    """Read *seqfish* formatted dataset.

    This function reads the following files:

        - ```{vx.ROI!r}{vx.COUNTS_FILE!r}{vx.CSV_FILE!r}```: Counts and metadata file.
        - ```{vx.ROI!r}{vx.CELL_COORDINATES!r}{vx.CSV_FILE!r}```: Cell coordinates file.
        - ```{vx.ROI!r}{vx.DAPI!r}{vx.TIFF_FILE!r}```: High resolution tiff image.
        - ```{vx.ROI!r}{vx.SEGMENTATION!r}{vx.TIFF_FILE!r}```: Cell mask file.
        - ```{vx.ROI!r}{vx.TRANSCRIPT_COORDINATES!r}{vx.CSV_FILE!r}```: Transcript coordinates file.

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
        Which ROIs (specified as integers) to load. Only necessary if multiple ROIs present.
    imread_kwargs
        Keyword arguments to pass to :func:`dask_image.imread.imread`.

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
    count_file_pattern = re.compile(rf"(.*?){re.escape(SK.CELL_COORDINATES)}{re.escape(SK.CSV_FILE)}$")
    count_files = [f for f in os.listdir(path) if count_file_pattern.match(f)]
    if not count_files:
        raise ValueError(
            f"No files matching the pattern {count_file_pattern} were found. Cannot infer the naming scheme."
        )

    roi_pattern = re.compile(f"^{SK.ROI}(\\d+)")
    found_rois = {m.group(1) for i in os.listdir(path) if (m := roi_pattern.match(i))}
    if rois is None:
        rois_str = [f"{SK.ROI}{roi}" for roi in found_rois]
    elif isinstance(rois, list):
        for roi in rois:
            if str(roi) not in found_rois:
                raise ValueError(f"ROI{roi} not found.")
        rois_str = [f"{SK.ROI}{roi}" for roi in rois]
    else:
        raise ValueError("Invalid type for 'roi'. Must be list[int] or None.")

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
            np.array(_get_scale_factors(path / get_dapi_file(roi_str), SK.SCALEFEFACTOR_X, SK.SCALEFEFACTOR_Y)),
            axes=("y", "x"),
        )

    if load_images:
        images = {
            f"{os.path.splitext(get_dapi_file(x))[0]}": Image2DModel.parse(
                imread(path / get_dapi_file(x), **imread_kwargs),
                dims=("c", "y", "x"),
                scale_factors=raster_models_scale_factors,
                transformations={"global": scaled[x]},
            )
            for x in rois_str
        }
    else:
        images = {}

    if load_labels:
        labels = {
            f"{os.path.splitext(get_cell_segmentation_labels_file(x))[0]}": Labels2DModel.parse(
                imread(path / get_cell_segmentation_labels_file(x), **imread_kwargs).squeeze(),
                dims=("y", "x"),
                scale_factors=raster_models_scale_factors,
                transformations={"global": scaled[x]},
            )
            for x in rois_str
        }
    else:
        labels = {}

    points = {}
    if load_points:
        for x in rois_str:
            # prepare data
            name = f"{os.path.splitext(get_transcript_file(x))[0]}"
            p = pd.read_csv(path / get_transcript_file(x), delimiter=",")
            instance_key_points = SK.INSTANCE_KEY_POINTS.value if SK.INSTANCE_KEY_POINTS.value in p.columns else None

            # call parser
            points[name] = PointsModel.parse(
                p,
                coordinates={"x": SK.TRANSCRIPTS_X, "y": SK.TRANSCRIPTS_Y},
                feature_key=SK.FEATURE_KEY.value,
                instance_key=instance_key_points,
                transformations={"global": Identity()},
            )

    shapes = {}
    if cells_as_circles:
        for x, adata in zip(rois_str, tables.values(), strict=False):
            shapes[f"{os.path.splitext(get_cell_file(x))[0]}"] = ShapesModel.parse(
                adata.obsm[SK.SPATIAL_KEY],
                geometry=0,
                radius=np.sqrt(adata.obs[SK.AREA].to_numpy() / np.pi),
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
                transformations={"global": Identity()},
            )
    if load_shapes:
        for x in rois_str:
            # this assumes that the index matches the instance key of the table. A more robust approach could be
            # implemented, as described here https://github.com/scverse/spatialdata-io/issues/249
            shapes[f"{os.path.splitext(get_cell_segmentation_shapes_file(x))[0]}"] = ShapesModel.parse(
                path / get_cell_segmentation_shapes_file(x),
                transformations={"global": scaled[x]},
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
            )

    sdata = SpatialData(images=images, labels=labels, points=points, tables=tables, shapes=shapes)

    return sdata


def _get_scale_factors(DAPI_path: Path, scalefactor_x_key: str, scalefactor_y_key: str) -> list[float]:
    with tifffile.TiffFile(DAPI_path) as tif:
        ome_metadata = tif.ome_metadata
        root = ET.fromstring(ome_metadata)
        for element in root.iter():
            if scalefactor_x_key in element.attrib.keys():
                scalefactor_x = element.attrib[scalefactor_x_key]
                scalefactor_y = element.attrib[scalefactor_y_key]
    return [float(scalefactor_x), float(scalefactor_y)]
