from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from dask_image.imread import imread
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations import Identity, Translation, set_transformation
from yaml.loader import SafeLoader

from spatialdata_io._constants._constants import McmicroKeys

__all__ = ["mcmicro"]


def _get_transformation(
    tma: int | None = None,
    tma_centroids: pd.DataFrame | None = None,
    raster_data: SpatialImage | MultiscaleSpatialImage | None = None,
) -> dict[str, Identity]:
    if tma is None:
        assert tma_centroids is None
        return {"global": Identity()}
    else:
        assert tma_centroids is not None
        assert raster_data is not None
        xy_centroids = tma_centroids[["x", "y"]].loc[tma].to_numpy()
        x_offset = np.median(raster_data["x"])
        y_offset = np.median(raster_data["y"])
        xy = xy_centroids - np.array([x_offset, y_offset])
        return {"global": Translation(xy, axes=("x", "y"))}


def mcmicro(
    path: str | Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read a *Mcmicro* output into a SpatialData object.

    .. seealso::

        - `Mcmicro pipeline  <https://mcmicro.org/>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    labels_models_kwargs
        Keyword arguments to pass to the labels models

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    params = _load_params(path)
    tma: bool = params["workflow"]["tma"]
    tma_centroids: pd.DataFrame | None = None

    markers = pd.read_csv(path / McmicroKeys.MARKERS_FILE)
    markers.index = markers.marker_name
    assert markers.channel_number.is_monotonic_increasing
    marker_names = markers.marker_name.tolist()

    images = {}
    if tma:
        centroids_file = path / McmicroKeys.COREOGRAPH_CENTROIDS
        tma_centroids = pd.read_csv(centroids_file, header=None, names=["y", "x"], index_col=False, sep=" ")
        tma_centroids.index = tma_centroids.index + 1

    image_dir = path / McmicroKeys.IMAGES_DIR_WSI
    if not image_dir.exists():
        raise ValueError(f"{path} does not contain {McmicroKeys.IMAGES_DIR_WSI} directory")

    samples = list(image_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX))
    if len(samples) > 1:
        raise ValueError("Only one sample per dataset is supported.")

    # if tma is true, the image in `samples` is the global image with all the tma cores; let's use it and then
    # reassign `samples` to each individual core
    if tma:
        assert len(samples) == 1
        data = imread(samples[0], **imread_kwargs)
        # , scale_factors=[2, 2]
        data = Image2DModel.parse(data, transformations=_get_transformation(), rgb=None, **image_models_kwargs)
        images["tma_map"] = data

        image_dir = path / McmicroKeys.IMAGES_DIR_TMA
        samples = list(image_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX))
        image_dir_masks = image_dir / "masks"
        samples_masks = list(image_dir_masks.glob("*"))

    for sample in samples:
        core_id: str = sample.with_name(sample.stem).with_suffix("").stem
        if tma:
            image_id = f"core_{core_id}"
        else:
            image_id = core_id

        data = imread(sample, **imread_kwargs)
        data = Image2DModel.parse(data, c_coords=marker_names, rgb=None, **image_models_kwargs)
        transformations = _get_transformation(
            tma=int(core_id) if tma else None, tma_centroids=tma_centroids, raster_data=data
        )
        set_transformation(data, transformation=transformations, set_all=True)
        images[f"{image_id}_image"] = data

    # in exemplar-001 the raw images are aligned with the illumination images, not with the registration image
    raw_dir = path / McmicroKeys.RAW_DIR
    if raw_dir.exists():
        raw_images = list(raw_dir.glob("*"))
        for raw_image in raw_images:
            raw_name = raw_image.with_name(raw_image.stem).with_suffix("").stem

            data = imread(raw_image, **imread_kwargs)
            images[raw_name] = Image2DModel.parse(
                data, transformations={raw_name: Identity()}, rgb=None, **image_models_kwargs
            )

    illumination_dir = path / McmicroKeys.ILLUMINATION_DIR
    if illumination_dir.exists():
        illumination_images = list(illumination_dir.glob("*"))
        for illumination_image in illumination_images:
            illumination_name = illumination_image.with_name(illumination_image.stem).with_suffix("").stem
            raw_name = illumination_name.removesuffix(McmicroKeys.ILLUMINATION_SUFFIX_DFP)
            raw_name = raw_name.removesuffix(McmicroKeys.ILLUMINATION_SUFFIX_FFP)

            data = imread(illumination_image, **imread_kwargs)
            images[illumination_name] = Image2DModel.parse(
                data, transformations={raw_name: Identity()}, rgb=None, **image_models_kwargs
            )

    samples_labels = list((path / McmicroKeys.LABELS_DIR).glob("*/*" + McmicroKeys.IMAGE_SUFFIX))

    labels = {}
    for labels_path in samples_labels:
        if not tma:
            # TODO: when support python >= 3.9 chance to str.removesuffix(McmicroKeys.IMAGE_SUFFIX)
            segmentation_stem = labels_path.with_name(labels_path.stem).with_suffix("").stem

            data = imread(labels_path, **imread_kwargs).squeeze()
            data = Labels2DModel.parse(data, transformations=_get_transformation(), **labels_models_kwargs)
            labels[f"{image_id}_{segmentation_stem}"] = data
        else:
            segmentation_stem = labels_path.with_name(labels_path.stem).with_suffix("").stem
            core_id_search = re.search(r"\d+$", labels_path.parent.stem)
            if core_id_search is None:
                raise ValueError(f"Cannot infer core_id from {labels_path.parent}")
            else:
                core_id = core_id_search.group()
                assert core_id is not None

            data = imread(labels_path, **imread_kwargs).squeeze()
            data = Labels2DModel.parse(data, **labels_models_kwargs)
            transformations = _get_transformation(tma=int(core_id), tma_centroids=tma_centroids, raster_data=data)
            set_transformation(data, transformation=transformations, set_all=True)
            labels[f"core_{core_id}_{segmentation_stem}"] = data

    if tma:
        for mask_path in samples_masks:
            mask_stem = mask_path.stem
            core_id = mask_stem.split("_")[0]

            data = imread(mask_path, **imread_kwargs).squeeze()
            data = Labels2DModel.parse(data, **labels_models_kwargs)
            transformations = _get_transformation(tma=int(core_id), tma_centroids=tma_centroids, raster_data=data)
            set_transformation(data, transformation=transformations, set_all=True)
            labels[f"core_{McmicroKeys.IMAGES_DIR_TMA}_{mask_stem}"] = data

    tables_dict = _get_tables(path, markers, tma)

    return SpatialData(images=images, labels=labels, tables=tables_dict)


def _load_params(path: Path) -> Any:
    params_path = path / McmicroKeys.PARAMS_FILE
    with open(params_path) as fp:
        params = yaml.load(fp, SafeLoader)
    return params


def _get_tables(
    path: Path,
    marker_df: pd.DataFrame,
    tma: bool,
) -> dict[str, AnnData]:
    var = marker_df.marker_name.tolist()
    coords = [McmicroKeys.COORDS_X.value, McmicroKeys.COORDS_Y.value]

    table_paths = list((path / McmicroKeys.QUANTIFICATION_DIR).glob("*.csv"))
    regions = []
    adatas = None
    tables_dict = {}
    for table_path in table_paths:
        if not tma:
            table_name = table_path.stem
            adata, region = _create_anndata(csv_path=table_path, markers=marker_df, var=var, coords=coords, tma=tma)
            table = TableModel.parse(
                adata, region=region, region_key="region", instance_key=McmicroKeys.INSTANCE_KEY.value
            )
            tables_dict[table_name] = table
        else:
            adata, region = _create_anndata(csv_path=table_path, markers=marker_df, var=var, coords=coords, tma=tma)
            regions.append(region)
            # TODO: check validity of output
            if not adatas:
                adatas = adata
            else:
                adatas = ad.concat([adatas, adata], index_unique="_")
            table = TableModel.parse(
                adatas, region=regions, region_key="region", instance_key=McmicroKeys.INSTANCE_KEY.value
            )
            tables_dict["segmentation_table"] = table

    return tables_dict


def _create_anndata(
    csv_path: Path,
    markers: pd.DataFrame,
    var: list[str],  # mypy has a bug with ellips, should be list[str, ...]
    coords: list[str],
    tma: bool,
) -> tuple[AnnData, str]:
    labels_basename = csv_path.stem.split("_")[-1]
    pattern = r"^(.*?)--"
    sample_id_search = re.search(pattern, csv_path.stem)
    sample_id = sample_id_search.groups()[0] if sample_id_search else None
    if not sample_id:
        raise ValueError(
            f"Csv filename should be in form <SAMPLE_ID>--<SEGMENTATION>_<labels_NAME>, got {csv_path.stem} "
        )
    table = pd.read_csv(csv_path)

    if not tma:
        region_value = sample_id + "_" + labels_basename
    else:
        region_value = "core_" + sample_id + "_" + labels_basename
        table[McmicroKeys.INSTANCE_KEY] = table[McmicroKeys.INSTANCE_KEY]
    adata = AnnData(
        table[var].to_numpy(),
        obs=table.drop(columns=var + coords),
        var=markers,
        obsm={"spatial": table[coords].to_numpy()},
        dtype=float,
    )
    adata.obs["region"] = pd.Categorical([region_value] * len(adata))
    return adata, region_value
