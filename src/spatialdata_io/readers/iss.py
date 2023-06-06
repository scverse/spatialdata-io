from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import anndata as ad
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, TableModel, Labels2DModel
from spatialdata.transformations.transformations import Identity
from xarray import DataArray

from spatialdata_io._constants._constants import ISSKeys
from spatialdata_io._docs import inject_docs

__all__ = ["iss"]


@inject_docs(vx=ISSKeys)
def iss(
    path: str | Path,
    raw_relative_path: str | Path,
    label_relative_path: str | Path,
    h5ad_relative_path: str | Path,
    dataset_id: Optional[str] = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    label_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *Sanger ISS* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>``: Counts and metadata file.
        - ``{vx.RAW_IMAGE!r}``: Raw raster image.
        - ``{vx.LABEL_IMAGE!r}``: Label image.

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
    label_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Label2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    adata = ad.read(path / h5ad_relative_path)
    adata.obs["cell_id"] = np.arange(len(adata))
    adata.var_names_make_unique()
    adata.obs["region"] = dataset_id
    table = TableModel.parse(adata, region=dataset_id, region_key="region", instance_key="cell_id")

    transform_original = Identity()

    label_image = imread(path / label_relative_path, **imread_kwargs).squeeze()
    label_image = DataArray(label_image, dims=("y", "x"), name=dataset_id + "_label_image")

    label_image_parsed = Labels2DModel.parse(
        label_image,
        scale_factors=[2, 4, 8, 16],
        transformations={"global": transform_original},
        **label_models_kwargs,
    )

    raw_image = imread(path / raw_relative_path, **imread_kwargs).squeeze()[[0]]
    raw_image = DataArray(raw_image, dims=("c", "y", "x"), name=dataset_id)

    raw_image_parsed = Image2DModel.parse(
        raw_image,
        # scale_factors=list(2**np.arange(8)),
        scale_factors=[2, 4, 8, 16],
        transformations={"global": transform_original},
        **image_models_kwargs,
    )

    return SpatialData(
        images={dataset_id + "_raw_image": raw_image_parsed},
        labels={dataset_id + "_label_image": label_image_parsed},
        table=table
    )
