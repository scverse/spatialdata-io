from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Union  # noqa: F401
from typing import Any, Optional

import pandas as pd
from anndata import AnnData
from dask_image.imread import imread
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatialdata import Image2DModel, Scale, ShapesModel, SpatialData, TableModel
from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._logging import logger
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import _read_counts

__all__ = ["visium"]


@inject_docs(vx=VisiumKeys)
def visium(
    path: str | Path,
    dataset_id: Optional[str] = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> AnnData:
    """
    Read *10x Genomics* Visium formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.COUNTS_FILE!r}```: Counts and metadata file.
        - ``{vx.IMAGE_HIRES_FILE!r}``: High resolution image.
        - ``{vx.IMAGE_LOWRES_FILE!r}``: Low resolution image.
        - ``{vx.IMAGE_TIF_FILE!r}``: High resolution tif image.
        - ``{vx.SCALEFACTORS_FILE!r}``: Scalefactors file.
        - ``{vx.SPOTS_FILE!r}``: Spots positions file.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    # get library_id
    patt = re.compile(f".*{VisiumKeys.COUNTS_FILE}")
    library_id = [i for i in os.listdir(path) if patt.match(i)][0].replace(f"_{VisiumKeys.COUNTS_FILE}", "")
    if dataset_id is not None:
        if dataset_id != library_id:
            logger.warning(
                f"`dataset_id: {dataset_id}` does not match `library_id: {library_id}`. `dataset_id: {dataset_id}` will be used to build SpatialData."
            )
    else:
        dataset_id = library_id

    adata, library_id = _read_counts(
        path, count_file=f"{library_id}_{VisiumKeys.COUNTS_FILE}", library_id=library_id, **kwargs
    )

    coords = pd.read_csv(
        path / VisiumKeys.SPOTS_FILE,
        header=1,
        index_col=0,
    )
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    coords = adata.obs[[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].values
    adata.obs.drop(columns=[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y], inplace=True)

    scalefactors = json.loads((path / VisiumKeys.SCALEFACTORS_FILE).read_bytes())
    input_cs = CoordinateSystem("xy", axes=["x", "y"])
    output_hires_cs = CoordinateSystem("hires", axes=["x", "y"])
    output_lowres_cs = CoordinateSystem("lowres", axes=["x", "y"])
    transform_lowres = Scale(
        [scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]],
        input_cs,
        output_lowres_cs,
    )
    transform_hires = Scale(
        [scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]],
        input_cs,
        output_hires_cs,
    )

    shapes = {}
    circles = ShapesModel.parse(
        coords,
        shape_type="circle",
        shape_size=scalefactors["spot_diameter_fullres"],
        index=adata.obs_names,
    )
    circles.uns["transform"] = [transform_lowres, transform_hires, circles.uns["transform"]]
    shapes[dataset_id] = circles
    table = TableModel.parse(adata, region="/polygons/cell_boundaries", instance_key="cell_id")

    images = {}
    input_cs = CoordinateSystem("cxy", axes=["c", "x", "y"])
    output_hires_cs = CoordinateSystem("hires", axes=["c", "x", "y"])
    output_lowres_cs = CoordinateSystem("lowres", axes=["c", "x", "y"])
    transform_lowres = Scale(
        [1.0, scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]],
        input_cs,
        output_lowres_cs,
    )
    transform_hires = Scale(
        [1.0, scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]],
        input_cs,
        output_hires_cs,
    )

    full_image = imread(path / VisiumKeys.IMAGE_TIF_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    full_image = DataArray(full_image, dims=["c", "y", "x"], name=dataset_id)
    full_image.attrs = {"transform": None}
    image_hires = imread(path / VisiumKeys.IMAGE_HIRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_hires = DataArray(image_hires, dims=["c", "y", "x"], name=dataset_id)
    image_hires.attrs = {"transform": transform_hires}
    image_lowres = imread(path / VisiumKeys.IMAGE_LOWRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_lowres = DataArray(image_lowres, dims=["c", "y", "x"], name=dataset_id)
    image_lowres.attrs = {"transform": transform_lowres}

    multiscale = MultiscaleSpatialImage.from_dict(
        d={
            "scale0": full_image,
            "scale1": image_hires,
            "scale2": image_lowres,
        }
    )
    Image2DModel().validate(multiscale)
    images = {dataset_id: multiscale}

    return SpatialData(table=table, shapes=shapes, images=images)
