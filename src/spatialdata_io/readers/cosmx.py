from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask_image.imread import imread
from scipy.sparse import csr_matrix

# from spatialdata._core.core_utils import xy_cs
from skimage.transform import estimate_transform
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel

# from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis  # , CoordinateSystem
from spatialdata.transformations.transformations import Affine, Identity

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

__all__ = ["cosmx"]

# x_axis = NgffAxis(name="x", type="space", unit="discrete")
# y_axis = NgffAxis(name="y", type="space", unit="discrete")
# c_axis = NgffAxis(name="c", type="channel", unit="index")


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: Optional[str] = None,
    transcripts: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
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
    transcripts
        Whether to also read in transcripts information.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

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
    if transcripts:
        transcripts_file = path / f"{dataset_id}_{CosmxKeys.TRANSCRIPTS_SUFFIX}"
        if not transcripts_file.exists():
            raise FileNotFoundError(f"Transcripts file not found: {transcripts_file}.")
    else:
        transcripts_file = None
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
    counts.index = counts.index.astype(str).str.cat(counts.pop(CosmxKeys.FOV).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    obs[CosmxKeys.FOV] = pd.Categorical(obs[CosmxKeys.FOV].astype(str))
    obs[CosmxKeys.REGION_KEY] = pd.Categorical(obs[CosmxKeys.FOV].astype(str).apply(lambda s: s + "_labels"))
    obs[CosmxKeys.INSTANCE_KEY] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[CosmxKeys.FOV].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
    )
    adata.var_names = counts.columns

    table = TableModel.parse(
        adata,
        region=list(set(adata.obs[CosmxKeys.REGION_KEY].astype(str).tolist())),
        region_key=CosmxKeys.REGION_KEY.value,
        instance_key=CosmxKeys.INSTANCE_KEY.value,
    )

    fovs_counts = list(map(str, adata.obs.fov.astype(int).unique()))

    affine_transforms_to_global = {}

    for fov in fovs_counts:
        idx = table.obs.fov.astype(str) == fov
        loc = table[idx, :].obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].values
        glob = table[idx, :].obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].values
        out = estimate_transform(ttype="affine", src=loc, dst=glob)
        affine_transforms_to_global[fov] = Affine(
            # out.params, input_coordinate_system=input_cs, output_coordinate_system=output_cs
            out.params,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

    table.obsm["global"] = table.obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].to_numpy()
    table.obsm["spatial"] = table.obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].to_numpy()
    table.obs.drop(
        columns=[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL, CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL],
        inplace=True,
    )

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
                aff = affine_transforms_to_global[fov]
                im = imread(path / CosmxKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze()
                flipped_im = da.flip(im, axis=0)
                parsed_im = Image2DModel.parse(
                    flipped_im,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_image": aff,
                    },
                    dims=("y", "x", "c"),
                    **image_models_kwargs,
                )
                images[f"{fov}_image"] = parsed_im
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")

    # read labels
    labels = {}
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                la = imread(path / CosmxKeys.LABELS_DIR / fname, **imread_kwargs).squeeze()
                flipped_la = da.flip(la, axis=0)
                parsed_la = Labels2DModel.parse(
                    flipped_la,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_labels": aff,
                    },
                    dims=("y", "x"),
                    **image_models_kwargs,
                )
                labels[f"{fov}_labels"] = parsed_la
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping labels {fname}.")

    points: dict[str, DaskDataFrame] = {}
    if transcripts:
        # assert transcripts_file is not None
        # from pyarrow.csv import read_csv
        #
        # ptable = read_csv(path / transcripts_file)  # , header=0)
        # for fov in fovs_counts:
        #     aff = affine_transforms_to_global[fov]
        #     sub_table = ptable.filter(pa.compute.equal(ptable.column(CosmxKeys.FOV), int(fov))).to_pandas()
        #     sub_table[CosmxKeys.INSTANCE_KEY] = sub_table[CosmxKeys.INSTANCE_KEY].astype("category")
        #     # we rename z because we want to treat the data as 2d
        #     sub_table.rename(columns={"z": "z_raw"}, inplace=True)
        #     points[fov] = PointsModel.parse(
        #         sub_table,
        #         coordinates={"x": CosmxKeys.X_LOCAL_TRANSCRIPT, "y": CosmxKeys.Y_LOCAL_TRANSCRIPT},
        #         feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
        #         instance_key=CosmxKeys.INSTANCE_KEY,
        #         transformations={
        #             fov: Identity(),
        #             "global": aff,
        #             "global_only_labels": aff,
        #         },
        #     )
        # let's convert the .csv to .parquet and let's read it with pyarrow.parquet for faster subsetting
        import tempfile

        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            print("converting .csv to .parquet to improve the speed of the slicing operations... ", end="")
            assert transcripts_file is not None
            transcripts_data = pd.read_csv(path / transcripts_file, header=0)
            transcripts_data.to_parquet(Path(tmpdir) / "transcripts.parquet")
            print("done")

            ptable = pq.read_table(Path(tmpdir) / "transcripts.parquet")
            for fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                sub_table = ptable.filter(pa.compute.equal(ptable.column(CosmxKeys.FOV), int(fov))).to_pandas()
                sub_table[CosmxKeys.INSTANCE_KEY] = sub_table[CosmxKeys.INSTANCE_KEY].astype("category")
                # we rename z because we want to treat the data as 2d
                sub_table.rename(columns={"z": "z_raw"}, inplace=True)
                points[f"{fov}_points"] = PointsModel.parse(
                    sub_table,
                    coordinates={"x": CosmxKeys.X_LOCAL_TRANSCRIPT, "y": CosmxKeys.Y_LOCAL_TRANSCRIPT},
                    feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                    instance_key=CosmxKeys.INSTANCE_KEY,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_labels": aff,
                    },
                )

    # TODO: what to do with fov file?
    # if fov_file is not None:
    #     fov_positions = pd.read_csv(path / fov_file, header=0, index_col=CosmxKeys.FOV)
    #     for fov, row in fov_positions.iterrows():
    #         try:
    #             adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
    #         except KeyError:
    #             logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
    #             continue

    return SpatialData(images=images, labels=labels, points=points, table=table)
