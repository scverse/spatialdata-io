from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import pandas as pd
from dask_image.imread import imread
from spatialdata import Image2DModel, ShapesModel, SpatialData, TableModel
from spatialdata._core.transformations import Identity, Scale
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
) -> SpatialData:
    """
    Read *10x Genomics* Visium formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.COUNTS_FILE!r}```: Counts and metadata file.
        - ``{vx.IMAGE_HIRES_FILE!r}``: High resolution image.
        - ``{vx.IMAGE_LOWRES_FILE!r}``: Low resolution image.
        - ``<dataset_id>_`{vx.IMAGE_TIF_SUFFIX!r}```: High resolution tif image.
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
    first_file = [i for i in os.listdir(path) if patt.match(i)][0]
    if f"_{VisiumKeys.COUNTS_FILE}" in first_file:
        library_id = first_file.replace(f"_{VisiumKeys.COUNTS_FILE}", "")
    else:
        raise ValueError(
            f"Cannot determine the library_id. Expecting a file with format <library_id>_{VisiumKeys.COUNTS_FILE}. Has "
            f"the files been renamed?"
        )
    if dataset_id is not None:
        if dataset_id != library_id:
            logger.warning(
                f"`dataset_id: {dataset_id}` does not match `library_id: {library_id}`. `dataset_id: {dataset_id}` "
                f"will be used to build SpatialData."
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
    shapes = {}
    circles = ShapesModel.parse(
        coords,
        shape_type="Circle",
        shape_size=scalefactors["spot_diameter_fullres"],
        index=adata.obs_names,
        transform=Identity(),
    )
    shapes[dataset_id] = circles
    table = TableModel.parse(adata)

    transform_original = Identity()
    transform_lowres = Scale(
        [scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]], axes=("y", "x")
    )
    transform_hires = Scale(
        [scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]], axes=("y", "x")
    )

    full_image = (
        imread(path / f"{dataset_id}{VisiumKeys.IMAGE_TIF_SUFFIX}", **imread_kwargs).squeeze().transpose(2, 0, 1)
    )
    full_image = DataArray(full_image, dims=("c", "y", "x"), name=dataset_id)
    full_image.attrs = {"transform": transform_original}

    image_hires = imread(path / VisiumKeys.IMAGE_HIRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_hires = DataArray(image_hires, dims=("c", "y", "x"), name=dataset_id)
    image_hires.attrs = {"transform": transform_hires}

    image_lowres = imread(path / VisiumKeys.IMAGE_LOWRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_lowres = DataArray(image_lowres, dims=("c", "y", "x"), name=dataset_id)
    image_lowres.attrs = {"transform": transform_lowres}

    full_image_parsed = Image2DModel.parse(full_image, multiscale_factors=[2, 2, 2, 2], **image_models_kwargs)
    image_hires_parsed = Image2DModel.parse(image_hires)
    image_lowres_parsed = Image2DModel.parse(image_lowres)

    images = {
        dataset_id + "_full_image": full_image_parsed,
        dataset_id + "_hires_image": image_hires_parsed,
        dataset_id + "_lowres_image": image_lowres_parsed,
    }

    return SpatialData(table=table, shapes=shapes, images=images)
