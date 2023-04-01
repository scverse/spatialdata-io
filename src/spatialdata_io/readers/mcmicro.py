from __future__ import annotations

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
    tma
        Whether output is from a tissue microarray analysis
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    if (path / McmicroKeys.IMAGES_DIR_TMA).exists():
        tma = True
    else:
        tma = False

    if not tma:
        image_dir = path / McmicroKeys.IMAGES_DIR_WSI
        if not image_dir.exists():
            raise ValueError(f"{path} does not contain {McmicroKeys.IMAGES_DIR_WSI} directory")

        samples = list(image_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX))
        if len(samples) > 1:
            raise ValueError("Only one sample per dataset is supported.")
    else:
        image_dir = path / McmicroKeys.IMAGES_DIR_TMA
        samples = list(image_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX))

    images = {}
    for sample in samples:
        image_id = sample.with_name(sample.stem).with_suffix("").stem
        if tma:
            image_id = f"core_{image_id}"
        images[f"{image_id}_image"] = _get_images(
            sample,
            imread_kwargs,
            image_models_kwargs,
        )

    labels = {}
    labels[f"{image_id}_cells"] = _get_labels(
        path,
        image_id,
        "cell",
        imread_kwargs,
        image_models_kwargs,
    )
    labels[f"{image_id}_nuclei"] = _get_labels(
        path,
        image_id,
        "nuclei",
        imread_kwargs,
        image_models_kwargs,
    )

    table = _get_table(path, image_id)

    return SpatialData(images=images, labels=labels, table=table)


def _get_images(
    path: Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(path, **imread_kwargs)
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
