from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations.transformations import Identity
from xarray import DataArray

from spatialdata_io._docs import inject_docs

__all__ = ["iss"]


@inject_docs()
def iss(
    path: str | Path,
    raw_relative_path: str | Path,
    label_relative_path: str | Path,
    h5ad_relative_path: str | Path,
    dataset_id: str = "region",
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    label_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *Sanger ISS* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>``: Counts and metadata file.
        - ``<raw_relative_path>``: Raw raster image.
        - ``<label_relative_path>``: Label image.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing the data.
    raw_relative_path : str or Path
        Relative path to the raw raster image file.
    label_relative_path : str or Path
        Relative path to the label image file.
    h5ad_relative_path : str or Path
        Relative path to the counts and metadata file.
    dataset_id : str
        Dataset identifier.
    imread_kwargs : Mapping[str, Any], optional
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs : Mapping[str, Any], optional
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    label_models_kwargs : Mapping[str, Any], optional
        Keyword arguments passed to :class:`spatialdata.models.Label2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
        The spatial data object containing the ISS data.
    """
    path = Path(path)

    adata = ad.read(path / h5ad_relative_path)
    adata.obs["cell_id"] = np.arange(len(adata))
    adata.var_names_make_unique()
    adata.obs["region"] = dataset_id
    table = TableModel.parse(adata, region=dataset_id, region_key="region", instance_key="cell_id")

    transform_original = Identity()

    label_image = imread(path / label_relative_path, **imread_kwargs).squeeze()
    label_image = DataArray(label_image, dims=("y", "x"), name=f"{dataset_id}_label_image")

    label_image_parsed = Labels2DModel.parse(
        label_image,
        scale_factors=[2, 4, 8, 16],
        transformations={"global": transform_original},
        **label_models_kwargs,
    )

    raw_image = imread(path / raw_relative_path, **imread_kwargs)
    raw_image = DataArray(raw_image, dims=("c", "y", "x"), name=dataset_id)

    raw_image_parsed = Image2DModel.parse(
        raw_image,
        scale_factors=[2, 4, 8, 16],
        transformations={"global": transform_original},
        **image_models_kwargs,
    )

    return SpatialData(
        images={f"{dataset_id}_raw_image": raw_image_parsed},
        labels={f"{dataset_id}_label_image": label_image_parsed},
        table=table,
    )
