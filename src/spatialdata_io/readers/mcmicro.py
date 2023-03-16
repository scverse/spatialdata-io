from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from dask_image.imread import imread
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel

from spatialdata_io._constants._constants import McmicroKeys

__all__ = ["mcmicro"]


def mcmicro(
    path: str | Path,
    dataset_id: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read a *Mcmicro* output into a SpatialData object.

    .. seealso::

        - `Mcmicro pipeline  <https://mcmicro.org/>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    dataset_id
        Dataset identifier.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    samples = os.listdir(path / McmicroKeys.IMAGES_DIR)
    if len(samples) > 1:
        raise ValueError("Only one sample per dataset is supported.")
    if (dataset_id + McmicroKeys.IMAGE_SUFFIX) not in samples:
        raise ValueError("Dataset id is not consistent with sample name.")

    images = {}
    images[f"{dataset_id}_image"] = _get_images(
        path,
        dataset_id,
        imread_kwargs,
        image_models_kwargs,
    )
    labels = {}
    labels[f"{dataset_id}_cells"] = _get_labels(
        path,
        dataset_id,
        "cell",
        imread_kwargs,
        image_models_kwargs,
    )
    labels[f"{dataset_id}_nuclei"] = _get_labels(
        path,
        dataset_id,
        "nuclei",
        imread_kwargs,
        image_models_kwargs,
    )

    table = _get_table(path, dataset_id)

    return SpatialData(images=images, labels=labels, table=table)


def _get_images(
    path: Path,
    sample: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(path / McmicroKeys.IMAGES_DIR / f"{sample}{McmicroKeys.IMAGE_SUFFIX}", **imread_kwargs)
    return Image2DModel.parse(image, **image_models_kwargs)


def _get_labels(
    path: Path,
    sample: str,
    labels_kind: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(
        path
        / McmicroKeys.LABELS_DIR
        / f"{McmicroKeys.LABELS_PREFIX}{sample}"
        / f"{labels_kind}{McmicroKeys.IMAGE_SUFFIX}",
        **imread_kwargs,
    ).squeeze()
    return Labels2DModel.parse(image, **image_models_kwargs)


def _get_table(
    path: Path,
    sample: str,
) -> AnnData:
    table = pd.read_csv(path / McmicroKeys.QUANTIFICATION_DIR / f"{sample}{McmicroKeys.CELL_FEATURES_SUFFIX}")
    markers = pd.read_csv(path / McmicroKeys.MARKERS_FILE)
    markers.index = markers.marker_name
    var = markers.marker_name.tolist()
    coords = [McmicroKeys.COORDS_X.value, McmicroKeys.COORDS_Y.value]
    adata = AnnData(
        table[var].to_numpy(),
        obs=table.drop(columns=var + coords),
        var=markers,
        obsm={"spatial": table[coords].to_numpy()},
        dtype=np.float_,
    )
    adata.obs["region"] = f"{sample}_cells"

    return TableModel.parse(
        adata, region=f"{sample}_cells", region_key="region", instance_key=McmicroKeys.INSTANCE_KEY.value
    )
