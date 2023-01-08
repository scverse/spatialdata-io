from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from anndata import AnnData
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from skimage.transform import estimate_transform
from spatialdata import SpatialData
from spatialdata._core.core_utils import xy_cs
from spatialdata._core.models import Image2DModel, Labels2DModel, ShapesModel
from spatialdata._core.transformations import Affine
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from typin import Any, Mapping

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

__all__ = ["cosmx"]


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: str,
    shape_size: float | int = 1,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> AnnData:
    """
    Read *Cosmx Nanostring* data.

    This function reads the following files:

        - ``<dataset_id>_`{cx.COUNTS_SUFFIX!r}```: Counts matrix.
        - ``<dataset_id>_`{cx.METADATA_SUFFIX!r}```: Metadata file.
        - ``<dataset_id>_`{cx.FOV_SUFFIX!r}```: Field of view file.
        - ``{cx.IMAGES_DIR!r}``: Directory containing the images.
        - ``{cx.LABELS_DIR!r}``: Directory containing the labels.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    dataset_id
        Name of the dataset.
    shape_size
        Size of the shape to be used for the centroids of the labels.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    # check for file existence
    counts_file = path / f"{dataset_id}_{CosmxKeys.COUNTS_SUFFIX}"
    if not counts_file.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}.")
    meta_file = path / f"{dataset_id}_{CosmxKeys.METADATA_SUFFIX}"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}.")
    fov_file = path / f"{dataset_id}_{CosmxKeys.FOV_SUFFIX}"
    if not fov_file.exists():
        raise FileNotFoundError(f"Found field of view file: {fov_file}.")
    images_dir = path / CosmxKeys.IMAGES_DIR
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}.")
    labels_dir = path / CosmxKeys.LABELS_DIR
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}.")

    counts = pd.read_csv(path / counts_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    counts.index = counts.index.astype(str).str.cat(counts.pop(CosmxKeys.REGION_KEY).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    obs[CosmxKeys.REGION_KEY] = pd.Categorical(obs[CosmxKeys.REGION_KEY].astype(str))
    obs[CosmxKeys.INSTANCE_KEY] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[CosmxKeys.REGION_KEY].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
    )
    adata.var_names = counts.columns

    fovs_counts = set(adata.obs.fov.astype(str).unique())

    # prepare to read images and labels
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")
    pat = re.compile(r".*_F(\d+)")

    # read images
    images = {}
    for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            images[fov] = Image2DModel.parse(
                imread(path / CosmxKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze(), name=fov, **image_models_kwargs
            )

    # read labels
    labels = {}
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            labels[fov] = Labels2DModel.parse(
                imread(path / CosmxKeys.LABELS_DIR / fname, **imread_kwargs).squeeze(), name=fov, **image_models_kwargs
            )

    fovs_images = set(images.keys()).intersection(set(labels.keys()))
    fovs_diff = fovs_images.difference(fovs_counts)
    if len(fovs_diff):
        logger.warning(
            f"Found images and labels for {len(fovs_images)} FOVs, but only {len(fovs_counts)} FOVs in the counts file.\n"
            + f"The following FOVs are missing: {fovs_diff} \n"
            + "`SpatialData` returns intersection of FOVs for counts and images/labels.",
        )

    circles = {}
    for fov in fovs_images:
        idx = adata.obs.fov.astype(str) == fov
        loc = adata[idx, :].obs[[CosmxKeys.X_LOCAL, CosmxKeys.Y_LOCAL]].values
        glob = adata[idx, :].obs[[CosmxKeys.X_GLOBAL, CosmxKeys.Y_GLOBAL]].values
        loc_to_glob_transform = _estimate_transform(loc, glob)
        circ = ShapesModel.parse(loc, shape_type="circle", shape_size=shape_size)
        implicit_transform = circ.uns["transform"]
        circ.uns["transform"] = [implicit_transform, loc_to_glob_transform]
        circles[fov] = circ

    adata.obs.drop(columns=[CosmxKeys.X_LOCAL, CosmxKeys.Y_LOCAL, CosmxKeys.X_GLOBAL, CosmxKeys.Y_GLOBAL], inplace=True)

    # TODO: what to do with fov file?
    # if fov_file is not None:
    #     fov_positions = pd.read_csv(path / fov_file, header=0, index_col=CosmxKeys.REGION_KEY)
    #     for fov, row in fov_positions.iterrows():
    #         try:
    #             adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
    #         except KeyError:
    #             logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
    #             continue

    return SpatialData(images=images, labels=labels, shapes=circles, table=adata)


def _estimate_transform(src: ArrayLike, tgt: ArrayLike) -> Affine:
    out = estimate_transform(ttype="affine", src=src, dst=tgt)
    out_cs = deepcopy(xy_cs)
    out_cs.name = "xy_global"
    return Affine(out.params, input_coordinate_system=xy_cs, output_coordinate_system=out_cs)
