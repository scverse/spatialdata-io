from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

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

__all__ = ["seqfish"]


@inject_docs(vx=SK)
def seqfish(
    path: str | Path,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    load_shapes: bool = True,
    load_additional_shapes: Literal["segmentation", "boundaries", "all"] | str | None = None,
    ROIs: list[int] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *seqfish* formatted dataset.

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
    load_additional_shapes
        Whether to load additional shapes such as segmentation or boundaries for both cells and nuclei.
    ROIs
        Which ROIs (specified as integers) to load. Only necessary if multiple ROIs present.
        If "all" is specified, reads all remaining .tiff and .geojson files in the directory.
    imread_kwargs
        Keyword arguments to pass to :func:`dask_image.imread.imread`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    count_file_pattern = re.compile(rf"(.*?){re.escape(SK.CELL_COORDINATES)}.*{re.escape(SK.CSV_FILE)}$")
    count_files = [i for i in os.listdir(path) if count_file_pattern.match(i)]
    roi_pattern = re.compile(f"^{SK.ROI}(\\d+)")
    n_rois = {m.group(1) for i in os.listdir(path) if (m := roi_pattern.match(i))}
    if not count_files:
        raise ValueError(
            f"No files matching the pattern {count_file_pattern} were found. Cannot infer the naming scheme."
        )
    matched = count_file_pattern.match(count_files[0])
    if matched is None:
        raise ValueError(f"File {count_files[0]} does not match the pattern {count_file_pattern}")

    rois_str: list[str] = []
    if ROIs is None:
        for roi in n_rois:
            rois_str.append(f"{SK.ROI}{roi}")
    elif isinstance(ROIs, list):
        for roi in ROIs:
            if str(roi) not in n_rois:
                raise ValueError(f"ROI{roi} not found.")
            rois_str.append(f"{SK.ROI}{roi}")
    else:
        raise ValueError("Invalid type for 'roi'. Must be list[int] or None.")

    def get_cell_file(roi: str) -> str:
        return f"{roi}_{SK.CELL_COORDINATES}{SK.CSV_FILE}"

    def get_count_file(roi: str) -> str:
        return f"{roi}_{SK.COUNTS_FILE}{SK.CSV_FILE}"

    def get_dapi_file(roi: str) -> str:
        return f"{roi}_{SK.DAPI}{SK.TIFF_FILE}"

    def get_cell_mask_file(roi: str) -> str:
        return f"{roi}_{SK.SEGMENTATION}{SK.TIFF_FILE}"

    def get_transcript_file(roi: str) -> str:
        return f"{roi}_{SK.TRANSCRIPT_COORDINATES}{SK.CSV_FILE}"

    scaled = {}
    adatas: dict[str, ad.AnnData] = {}
    for roi_str in rois_str:
        assert isinstance(roi_str, str) or roi_str is None
        cell_file = get_cell_file(roi_str)
        count_matrix = get_count_file(roi_str)
        adata = ad.read_csv(path / count_matrix, delimiter=",")
        cell_info = pd.read_csv(path / cell_file, delimiter=",")
        adata.obsm[SK.SPATIAL_KEY] = cell_info[[SK.CELL_X, SK.CELL_Y]].to_numpy()
        adata.obs[SK.AREA] = np.reshape(cell_info[SK.AREA].to_numpy(), (-1, 1))
        region = f"cells_{roi_str}"
        adata.obs[SK.REGION_KEY] = region
        adata.obs[SK.INSTANCE_KEY_TABLE] = adata.obs.index.astype(int)
        adatas[roi_str] = adata
        scaled[roi_str] = Scale(
            np.array(_get_scale_factors(path / get_dapi_file(roi_str), SK.SCALEFEFACTOR_X, SK.SCALEFEFACTOR_Y)),
            axes=("y", "x"),
        )

    scale_factors = [2, 2, 2, 2]

    if load_images:
        images = {
            f"{os.path.splitext(get_dapi_file(x))[0]}": Image2DModel.parse(
                imread(path / get_dapi_file(x), **imread_kwargs),
                dims=("c", "y", "x"),
                scale_factors=scale_factors,
                transformations={"global": scaled[x]},
            )
            for x in rois_str
        }
    else:
        images = {}

    if load_labels:
        labels = {
            f"{os.path.splitext(get_cell_mask_file(x))[0]}": Labels2DModel.parse(
                imread(path / get_cell_mask_file(x), **imread_kwargs).squeeze(),
                dims=("y", "x"),
                scale_factors=scale_factors,
                transformations={"global": Identity()},
            )
            for x in rois_str
        }
    else:
        labels = {}

    if load_points:
        points = {
            f"{os.path.splitext(get_transcript_file(x))[0]}": PointsModel.parse(
                pd.read_csv(path / get_transcript_file(x), delimiter=","),
                coordinates={"x": SK.TRANSCRIPTS_X, "y": SK.TRANSCRIPTS_Y},
                feature_key=SK.FEATURE_KEY.value,
                # instance_key=SK.INSTANCE_KEY_POINTS.value, # TODO: should be optional but parameter might get deprecated anyway
                transformations={"global": Identity()},
            )
            for x in rois_str
        }
    else:
        points = {}

    tables = {}
    for name, adata in adatas.items():
        adata.obs[SK.REGION_KEY] = adata.obs[SK.REGION_KEY].astype("category")
        adata.obs = adata.obs.reset_index(drop=True)
        tables[name] = TableModel.parse(
            adata,
            region=f"cells_{name}",
            region_key=SK.REGION_KEY.value,
            instance_key=SK.INSTANCE_KEY_TABLE.value,
        )

    if load_shapes:
        shapes = {
            f"{os.path.splitext(get_cell_file(x))[0]}": ShapesModel.parse(
                adata.obsm[SK.SPATIAL_KEY],
                geometry=0,
                radius=np.sqrt(adata.obs[SK.AREA].to_numpy() / np.pi),
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
                transformations={"global": Identity()},
            )
            for x, adata in zip(rois_str, adatas.values())
        }
    else:
        shapes = {}

    if load_additional_shapes is not None:
        shape_file_names = []
        for filename in os.listdir(path):
            if filename.endswith((SK.TIFF_FILE, SK.GEOJSON_FILE)):
                if load_additional_shapes == "all":
                    if not any(key in filename for key in images.keys()) and not any(
                        key in filename for key in labels.keys()
                    ):
                        shape_file_names.append(filename)
                elif load_additional_shapes == "segmentation":
                    if SK.SEGMENTATION in filename and not any(key in filename for key in labels.keys()):
                        shape_file_names.append(filename)
                elif load_additional_shapes == "boundaries":
                    if SK.BOUNDARIES in filename:
                        shape_file_names.append(filename)
                elif isinstance(load_additional_shapes, str):
                    if load_additional_shapes in filename:
                        shape_file_names.append(filename)
                else:
                    raise ValueError(f"No file found with identifier {load_additional_shapes}")

        for x in range(len(shape_file_names)):
            shapes[f"{os.path.splitext(shape_file_names[x])[0]}"] = ShapesModel.parse(
                path / shape_file_names[x],
                index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
                transformations={"global": Identity()},
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
