from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata._core.models import Image2DModel, Labels2DModel, TableModel
from spatialdata._logging import logger

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

# from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis  # , CoordinateSystem
# from spatialdata._core.transformations import Affine
# from spatialdata._core.core_utils import xy_cs
# from skimage.transform import estimate_transform
__all__ = ["cosmx"]

# x_axis = NgffAxis(name="x", type="space", unit="discrete")
# y_axis = NgffAxis(name="y", type="space", unit="discrete")
# c_axis = NgffAxis(name="c", type="channel", unit="index")


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: Optional[str] = None,
    # shape_size: float | int = 1,
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

    # tries to infer dataset_id from the name of the counts file
    if dataset_id is None:
        counts_files = [f for f in os.listdir(path) if str(f).endswith(CosmxKeys.COUNTS_SUFFIX)]
        if len(counts_files) == 1:
            found = re.match(rf"(.*)_{CosmxKeys.COUNTS_SUFFIX}", counts_files[0])
            if found:
                dataset_id = found.group(1)
    if dataset_id is None:
        raise ValueError("Could not infer `dataset_id` from the name of the counts file. Please specify it manually.")

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

    table = TableModel.parse(
        adata,
        region=adata.obs.fov.astype(str).tolist(),
        region_key=CosmxKeys.REGION_KEY.value,
        instance_key=CosmxKeys.INSTANCE_KEY.value,
    )

    fovs_counts = set(table.obs.fov.astype(str).unique())

    # TODO(giovp): uncomment once transform is ready
    # input_cs = CoordinateSystem("cxy", axes=[c_axis, y_axis, x_axis])
    # input_cs_labels = CoordinateSystem("cxy", axes=[y_axis, x_axis])
    # output_cs = CoordinateSystem("global", axes=[c_axis, y_axis, x_axis])
    # output_cs_labels = CoordinateSystem("global", axes=[y_axis, x_axis])

    # affine_transforms_images = {}
    # affine_transforms_labels = {}

    # for fov in fovs_counts:
    #     idx = table.obs.fov.astype(str) == fov
    #     loc = table[idx, :].obs[[CosmxKeys.X_LOCAL, CosmxKeys.Y_LOCAL]].values
    #     glob = table[idx, :].obs[[CosmxKeys.X_GLOBAL, CosmxKeys.Y_GLOBAL]].values
    #     out = estimate_transform(ttype="affine", src=loc, dst=glob)
    #     affine_transforms_images[fov] = Affine(
    #         out.params, input_coordinate_system=input_cs, output_coordinate_system=output_cs
    #     )
    #     affine_transforms_labels[fov] = Affine(
    #         out.params, input_coordinate_system=input_cs_labels, output_coordinate_system=output_cs_labels
    #     )

    table.obsm["global"] = table.obs[[CosmxKeys.X_GLOBAL, CosmxKeys.Y_GLOBAL]].to_numpy()
    table.obsm["spatial"] = table.obs[[CosmxKeys.X_LOCAL, CosmxKeys.Y_LOCAL]].to_numpy()
    table.obs.drop(columns=[CosmxKeys.X_LOCAL, CosmxKeys.Y_LOCAL, CosmxKeys.X_GLOBAL, CosmxKeys.Y_GLOBAL], inplace=True)

    # prepare to read images and labels
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")
    pat = re.compile(r".*_F(\d+)")

    # check if fovs are correct for images and labels
    fovs_images = []
    for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fovs_images.append(str(int(pat.findall(fname)[0])))

    fovs_labels = []
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fovs_labels.append(str(int(pat.findall(fname)[0])))

    fovs_images_and_labels = set(fovs_images).intersection(set(fovs_labels))
    fovs_diff = fovs_images_and_labels.difference(set(fovs_counts))
    if len(fovs_diff):
        raise logger.warning(
            f"Found images and labels for {len(fovs_images)} FOVs, but only {len(fovs_counts)} FOVs in the counts file.\n"
            + f"The following FOVs are missing: {fovs_diff} \n"
            + "... will use only fovs in Table."
        )

    # read images
    images = {}
    for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                images[fov] = Image2DModel.parse(
                    imread(path / CosmxKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze(),
                    name=fov,
                    # transform=affine_transforms_images[fov],
                    **image_models_kwargs,
                )
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")

    # read labels
    labels = {}
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                labels[fov] = Labels2DModel.parse(
                    imread(path / CosmxKeys.LABELS_DIR / fname, **imread_kwargs).squeeze(),
                    name=fov,
                    # transform=affine_transforms_labels[fov],
                    **image_models_kwargs,
                )
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping labels {fname}.")

    # TODO: what to do with fov file?
    # if fov_file is not None:
    #     fov_positions = pd.read_csv(path / fov_file, header=0, index_col=CosmxKeys.REGION_KEY)
    #     for fov, row in fov_positions.iterrows():
    #         try:
    #             adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
    #         except KeyError:
    #             logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
    #             continue

    return SpatialData(images=images, labels=labels, table=table)
