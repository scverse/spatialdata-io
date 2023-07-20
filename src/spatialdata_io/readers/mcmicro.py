from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from dask_image.imread import imread
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations import Identity
from yaml.loader import SafeLoader

from spatialdata_io._constants._constants import McmicroKeys

__all__ = ["mcmicro"]


def mcmicro(
    path: str | Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    label_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
    label_models_kwargs
        Keyword arguments to pass to the label models

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    params = _load_params(path)
    tma: bool = params["workflow"]["tma"]
    transformations = {"global": Identity()}

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
            transformations,
            imread_kwargs,
            image_models_kwargs,
        )

    samples_labels = list((path / McmicroKeys.LABELS_DIR).glob("*/*" + McmicroKeys.IMAGE_SUFFIX))

    labels = {}
    for label_path in samples_labels:
        if not tma:
            # TODO: when support python >= 3.9 chance to str.removesuffix(McmicroKeys.IMAGE_SUFFIX)
            segmentation_stem = label_path.with_name(label_path.stem).with_suffix("").stem
            labels[f"{image_id}_{segmentation_stem}"] = _get_labels(
                label_path,
                transformations,
                imread_kwargs,
                label_models_kwargs,
            )
        else:
            segmentation_stem = label_path.with_name(label_path.stem).with_suffix("").stem
            core_id_search = re.search(r"\d+$", label_path.parent.stem)
            core_id = int(core_id_search.group()) if core_id_search else None
            labels[f"core_{core_id}_{segmentation_stem}"] = _get_labels(
                label_path,
                imread_kwargs,
                label_models_kwargs,
            )

    table = _get_table(path, tma)

    return SpatialData(images=images, labels=labels, table=table)


def _load_params(path: Path) -> Any:
    params_path = path / McmicroKeys.PARAMS_FILE
    with open(params_path) as fp:
        params = yaml.load(fp, SafeLoader)
    return params


def _get_images(
    path: Path,
    transformations: Mapping[str, Identity],
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(path, **imread_kwargs)
    return Image2DModel.parse(image, transformations=transformations, **image_models_kwargs)


def _get_labels(
    path: Path,
    transformations: Mapping[str, Identity],
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    label_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(
        path,
        **imread_kwargs,
    ).squeeze()
    return Labels2DModel.parse(image, transformations=transformations, **label_models_kwargs)


def _get_table(
    path: Path,
    tma: bool,
) -> AnnData:
    markers = pd.read_csv(path / McmicroKeys.MARKERS_FILE)
    markers.index = markers.marker_name
    var = markers.marker_name.tolist()
    coords = [McmicroKeys.COORDS_X.value, McmicroKeys.COORDS_Y.value]

    table_paths = list((path / McmicroKeys.QUANTIFICATION_DIR).glob("*.csv"))
    regions = []
    adatas = None
    for table_path in table_paths:
        if not tma:
            adata, region = _create_anndata(table_path, markers, var, coords, tma)

            return TableModel.parse(
                adata, region=region, region_key="region", instance_key=McmicroKeys.INSTANCE_KEY.value
            )
        else:
            adata, region = _create_anndata(table_path, markers, var, coords, tma)
            regions.append(region)

            if not adatas:
                adatas = adata
            else:
                adatas = adatas.concatenate(adata, index_unique=None)

    return TableModel.parse(adatas, region=regions, region_key="region", instance_key=McmicroKeys.INSTANCE_KEY.value)


def _create_anndata(
    csv_path: Path,
    markers: pd.DataFrame,
    var: list[str],  # mypy has a bug with ellips, should be list[str, ...]
    coords: list[str],
    tma: bool,
) -> tuple[AnnData, str]:
    label_basename = csv_path.stem.split("_")[-1]
    pattern = r"^(.*?)--"
    sample_id_search = re.search(pattern, csv_path.stem)
    sample_id = sample_id_search.groups()[0] if sample_id_search else None
    if not sample_id:
        raise ValueError(
            f"Csv filename should be in form <SAMPLE_ID>--<SEGMENTATION>_<LABEL_NAME>, got {csv_path.stem} "
        )
    table = pd.read_csv(csv_path)

    if not tma:
        region_value = sample_id + "_" + label_basename
    else:
        # Ensure unique CellIDs when concatenating anndata objects. Ensures unique values for INSTANCE_KEY
        region_value = "core_" + sample_id + "_" + label_basename
        table[McmicroKeys.INSTANCE_KEY] = "core_" + sample_id + "_" + table[McmicroKeys.INSTANCE_KEY].astype(str)
    table.index = table[McmicroKeys.INSTANCE_KEY]
    adata = AnnData(
        table[var].to_numpy(),
        obs=table.drop(columns=var + coords),
        var=markers,
        obsm={"spatial": table[coords].to_numpy()},
        dtype=np.float_,
    )
    adata.obs["region"] = pd.Categorical([region_value] * len(adata))
    return adata, region_value
