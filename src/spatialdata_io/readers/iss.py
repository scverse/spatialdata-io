from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
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
    labels_relative_path: str | Path,
    h5ad_relative_path: str | Path,
    instance_key: str | None = None,
    dataset_id: str = "region",
    multiscale_image: bool = True,
    multiscale_labels: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *Sanger ISS* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>``: Counts and metadata file.
        - ``<raw_relative_path>``: Raw raster image.
        - ``<labels_relative_path>``: Label image.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    raw_relative_path
        Relative path to the raw raster image file.
    labels_relative_path
        Relative path to the label image file.
    h5ad_relative_path
        Relative path to the counts and metadata file.
    instance_key
        Which column of the `AnnData` table (in the `obs` `DataFrame`) contains the instance identifiers
        (e.g. a `'cell_id'` column); if not specified, such information is assumed to be contained in the index of the
        `AnnData` object.
    dataset_id
        Dataset identifier.
    multiscale_image
        Whether to process the raw image into a multiscale image.
    multiscale_labels
        Whether to process the label image into a multiscale image.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    The spatial data object containing the ISS data.
    """
    REGION = f"{dataset_id}_labels_image"
    REGION_KEY = "region"
    path = Path(path)

    adata = ad.read(path / h5ad_relative_path)
    if instance_key is None:
        instance_key = "instance_id"
        adata.obs[instance_key] = adata.obs.index.astype(int)
    adata.var_names_make_unique()
    adata.obs[REGION_KEY] = REGION
    table = TableModel.parse(adata, region=REGION, region_key=REGION_KEY, instance_key=instance_key)

    transform_original = Identity()

    labels_image = imread(path / labels_relative_path, **imread_kwargs).squeeze()
    labels_image = DataArray(labels_image[:, :], dims=("y", "x"))

    labels_image_parsed = Labels2DModel.parse(
        labels_image,
        scale_factors=[2, 2, 2, 2] if multiscale_labels else None,
        transformations={"global": transform_original},
        **labels_models_kwargs,
    )

    raw_image = imread(path / raw_relative_path, **imread_kwargs)
    raw_image = DataArray(raw_image, dims=("c", "y", "x"))

    raw_image_parsed = Image2DModel.parse(
        raw_image,
        scale_factors=[2, 2, 2, 2] if multiscale_image else None,
        transformations={"global": transform_original},
        **image_models_kwargs,
    )

    return SpatialData(
        images={f"{dataset_id}_raw_image": raw_image_parsed},
        labels={REGION: labels_image_parsed},
        table=table,
    )
