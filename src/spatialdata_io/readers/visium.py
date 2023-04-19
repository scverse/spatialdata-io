from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale
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
        - ``<dataset_id>_`{vx.IMAGE_TIF_ALTERNATIVE_SUFFIX!r}```: High resolution tif image, old naming convention.
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
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

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
        header=0,
        index_col=0,
    )
    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    coords = adata.obs[[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].values
    adata.obsm["spatial"] = coords
    adata.obs.drop(columns=[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y], inplace=True)
    adata.obs["spot_id"] = np.arange(len(adata))
    adata.var_names_make_unique()

    scalefactors = json.loads((path / VisiumKeys.SCALEFACTORS_FILE).read_bytes())

    transform_original = Identity()
    transform_lowres = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]]),
        axes=("y", "x"),
    )
    transform_hires = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]]),
        axes=("y", "x"),
    )
    shapes = {}
    circles = ShapesModel.parse(
        coords,
        geometry=0,
        radius=scalefactors["spot_diameter_fullres"] / 2.0,
        index=adata.obs["spot_id"].copy(),
        transformations={
            "global": transform_original,
            "downscaled_hires": transform_hires,
            "downscaled_lowres": transform_lowres,
        },
    )
    shapes[dataset_id] = circles
    adata.obs["region"] = dataset_id
    table = TableModel.parse(adata, region=dataset_id, region_key="region", instance_key="spot_id")

    if (path / f"{dataset_id}{VisiumKeys.IMAGE_TIF_SUFFIX}").exists():
        tif_path = path / f"{dataset_id}{VisiumKeys.IMAGE_TIF_SUFFIX}"
    elif (path / f"{dataset_id}{VisiumKeys.IMAGE_TIF_ALTERNATIVE_SUFFIX}").exists():
        tif_path = path / f"{dataset_id}{VisiumKeys.IMAGE_TIF_ALTERNATIVE_SUFFIX}"
    else:
        raise FileNotFoundError(
            f"Cannot find {VisiumKeys.IMAGE_TIF_SUFFIX} or {VisiumKeys.IMAGE_TIF_ALTERNATIVE_SUFFIX}."
        )

    full_image = imread(tif_path, **imread_kwargs).squeeze().transpose(2, 0, 1)
    full_image = DataArray(full_image, dims=("c", "y", "x"), name=dataset_id)

    image_hires = imread(path / VisiumKeys.IMAGE_HIRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_hires = DataArray(image_hires, dims=("c", "y", "x"), name=dataset_id)

    image_lowres = imread(path / VisiumKeys.IMAGE_LOWRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
    image_lowres = DataArray(image_lowres, dims=("c", "y", "x"), name=dataset_id)

    full_image_parsed = Image2DModel.parse(
        # drop alpha channel
        full_image.sel({"c": slice(0, 3)}),
        scale_factors=[2, 2, 2, 2],
        transformations={"global": transform_original},
        **image_models_kwargs,
    )

    image_hires_parsed = Image2DModel.parse(image_hires, transformations={"downscaled_hires": Identity()})
    image_lowres_parsed = Image2DModel.parse(image_lowres, transformations={"downscaled_lowres": Identity()})

    images = {
        dataset_id + "_lowres_image": image_lowres_parsed,
        dataset_id + "_hires_image": image_hires_parsed,
        dataset_id + "_full_image": full_image_parsed,
    }

    return SpatialData(images=images, shapes=shapes, table=table)
