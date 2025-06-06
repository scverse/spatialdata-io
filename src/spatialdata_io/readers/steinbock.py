from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations.transformations import Identity

from spatialdata_io._constants._constants import SteinbockKeys

if TYPE_CHECKING:
    from collections.abc import Mapping

    from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
    from spatial_image import SpatialImage

__all__ = ["steinbock"]


def steinbock(
    path: str | Path,
    labels_kind: Literal["deepcell", "ilastik"] = "deepcell",
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read a *Steinbock* output into a SpatialData object.

    .. seealso::

        - `Steinbock pipeline  <https://bodenmillergroup.github.io/steinbock/latest/>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    labels_kind
        Kind of labels to use. Either ``deepcell`` or ``ilastik``.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    labels_kind = SteinbockKeys(f"masks_{labels_kind}")  # type: ignore[assignment]

    samples = [i.replace(SteinbockKeys.IMAGE_SUFFIX, "") for i in os.listdir(path / SteinbockKeys.IMAGES_DIR)]
    samples_labels = [i.replace(SteinbockKeys.LABEL_SUFFIX, "") for i in os.listdir(path / labels_kind)]
    images = {}
    labels = {}
    if len(set(samples).difference(set(samples_labels))):
        logger.warning(
            f"Samples {set(samples).difference(set(samples_labels))} have images but no labels. They will be ignored."
        )
    for sample in samples:
        images[f"{sample}_image"] = _get_images(
            path,
            sample,
            imread_kwargs,
            image_models_kwargs,
        )
        labels[f"{sample}_labels"] = _get_labels(
            path,
            sample,
            labels_kind,
            imread_kwargs,
            image_models_kwargs,
        )

    adata = ad.read(path / SteinbockKeys.CELLS_FILE)
    idx = adata.obs.index.str.split(" ").map(lambda x: int(x[1]))
    regions = adata.obs.image.str.replace(".tiff", "", regex=False)
    regions = regions.apply(lambda x: f"{x}_labels")
    adata.obs["cell_id"] = idx
    adata.obs["region"] = regions
    adata.obsm["spatial"] = adata.obs[["centroid-0", "centroid-1"]].to_numpy()

    # duplicate of adata.obs['image']
    del adata.obs["Image"]

    # for backwards compatibility with the demo dataset from spatialdata-io
    if "Final Concentration / Dilution" in adata.var.columns:
        # / is an invalid character
        adata.var["Final Concentration"] = adata.var["Final Concentration / Dilution"]
        del adata.var["Final Concentration / Dilution"]

    # replace all spaces with underscores
    adata.var.columns = adata.var.columns.str.replace(" ", "_")

    if len({f"{s}_labels" for s in samples}.difference(set(regions.unique()))):
        raise ValueError("Samples in table and images are inconsistent, please check.")
    table = TableModel.parse(adata, region=regions.unique().tolist(), region_key="region", instance_key="cell_id")

    return SpatialData(images=images, labels=labels, table=table)


def _get_images(
    path: Path,
    sample: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialImage | MultiscaleSpatialImage:
    image = imread(path / SteinbockKeys.IMAGES_DIR / f"{sample}{SteinbockKeys.IMAGE_SUFFIX}", **imread_kwargs)
    return Image2DModel.parse(data=image, transformations={sample: Identity()}, rgb=None, **image_models_kwargs)


def _get_labels(
    path: Path,
    sample: str,
    labels_kind: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialImage | MultiscaleSpatialImage:
    image = imread(path / labels_kind / f"{sample}{SteinbockKeys.LABEL_SUFFIX}", **imread_kwargs).squeeze()
    return Labels2DModel.parse(data=image, transformations={sample: Identity()}, **image_models_kwargs)
