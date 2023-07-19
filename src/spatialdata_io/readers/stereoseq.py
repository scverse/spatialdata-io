from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel

from spatialdata_io._constants._constants import StereoseqKeys

__all__ = ["stereoseq"]


def stereoseq(
    path: str | Path,
    dataset_id: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *stereoseq* formatted dataset.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier. If not given will be determined automatically
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    imread_kwargs = dict(imread_kwargs)
    image_models_kwargs = dict(image_models_kwargs)

    image_patterns = [
        re.compile(r".*" + re.escape(StereoseqKeys.MASK_TIF)),
        re.compile(r".*" + re.escape(StereoseqKeys.REGIST_TIF)),
        re.compile(r".*" + re.escape(StereoseqKeys.TISSUE_TIF)),
        re.compile(r".*" + re.escape(StereoseqKeys.FOV_TIF)),
    ]

    square_bin = dataset_id + StereoseqKeys.GEF_FILE
    gef_patterns = [
        re.compile(r".*" + re.escape(StereoseqKeys.RAW_GEF)),
        re.compile(r".*" + re.escape(StereoseqKeys.CELLBIN_GEF)),
        re.compile(r".*" + re.escape(StereoseqKeys.TISSUECUT_GEF)),
        re.compile(r".*" + re.escape(square_bin)),
    ]

    image_filenames = [
        i for i in os.listdir(path / StereoseqKeys.REGISTER) if any(pattern.match(i) for pattern in image_patterns)
    ]
    cell_mask_file = [x for x in image_filenames if (f"{StereoseqKeys.MASK_TIF}" in x)]
    image_filenames.remove(cell_mask_file[0])
    print(image_filenames)

    cellbin_gef_filename = [
        i for i in os.listdir(path / StereoseqKeys.REGISTER) if any(pattern.match(i) for pattern in gef_patterns)
    ]
    squarebin_gef_filename = [
        i for i in os.listdir(path / StereoseqKeys.TISSUECUT) if any(pattern.match(i) for pattern in gef_patterns)
    ]

    table_pattern = re.compile(r".*" + re.escape(StereoseqKeys.CELL_CLUSTER_H5AD))
    table_filename = [i for i in os.listdir(path / StereoseqKeys.CELLCLUSTER) if table_pattern.match(i)]

    # create table using .h5ad and cellbin.gef
    adata = ad.read_h5ad(path / StereoseqKeys.CELLCLUSTER / table_filename[0])
    path_cellbin = path / StereoseqKeys.REGISTER / cellbin_gef_filename[0]
    cellbin_gef = h5py.File(str(path_cellbin), "r")

    # add cell info to obs
    obs = pd.DataFrame(
        cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_DATASET][:],
        columns=[
            StereoseqKeys.CELL_ID,
            StereoseqKeys.COORD_X,
            StereoseqKeys.COORD_Y,
            StereoseqKeys.OFFSET,
            StereoseqKeys.GENECOUNT,
            StereoseqKeys.EXP_COUNT,
            StereoseqKeys.DNBCOUNT,
            StereoseqKeys.CELL_AREA,
            StereoseqKeys.CELL_TYPE_ID,
            StereoseqKeys.CLUSTER_ID,
        ],
    )

    obsm_spatial = obs[[StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y]].to_numpy()
    obs = obs.drop([StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y], axis=1)
    obs[StereoseqKeys.CELL_EXON] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_EXON][:]

    # add gene info to var
    var = pd.DataFrame(
        cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.FEATURE_KEY][:],
        columns=[
            StereoseqKeys.GENE_NAME,
            StereoseqKeys.OFFSET,
            StereoseqKeys.CELL_COUNT,
            StereoseqKeys.EXP_COUNT,
            StereoseqKeys.MAX_MID_COUNT,
        ],
    )
    var[StereoseqKeys.GENE_NAME] = var[StereoseqKeys.GENE_NAME].str.decode("utf-8")
    var[StereoseqKeys.GENE_EXON] = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.GENE_EXON][:]

    # add cell border to obsm
    cell_coordinates = [x for i in range(1, 33) for x in ("x_" + str(i), "y_" + str(i))]
    n_cells, n_coords, xy = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_BORDER][:].shape
    arr = cellbin_gef[StereoseqKeys.CELL_BIN][StereoseqKeys.CELL_BORDER][:]
    new_arr = arr.reshape(n_cells, n_coords * xy)
    obsm = pd.DataFrame(new_arr, columns=cell_coordinates)

    obs.index = adata.obs.index
    adata.obs = pd.merge(adata.obs, obs, left_index=True, right_index=True)
    var.index = adata.var.index
    adata.var = pd.merge(adata.var, var, left_index=True, right_index=True)
    obsm.index = adata.obs.index
    adata.obsm[StereoseqKeys.CELL_BORDER] = obsm
    adata.obsm[StereoseqKeys.SPATIAL_KEY] = obsm_spatial

    # add region and instance_id to obs for the TableModel
    adata.obs[StereoseqKeys.REGION_KEY] = StereoseqKeys.REGION
    adata.obs[StereoseqKeys.REGION_KEY] = adata.obs[StereoseqKeys.REGION_KEY].astype("category")
    adata.obs[StereoseqKeys.INSTANCE_KEY] = adata.obs.index

    # create points model using SquareBin.gef
    path_squarebin = path / StereoseqKeys.TISSUECUT / squarebin_gef_filename[0]
    squarebin_gef = h5py.File(str(path_squarebin), "r")

    df_by_bin = {}
    for i in squarebin_gef[StereoseqKeys.GENE_EXP].keys():
        # get gene info
        arr = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.FEATURE_KEY][:]
        df_gene = pd.DataFrame(
            arr, columns=[StereoseqKeys.FEATURE_KEY, StereoseqKeys.OFFSET, StereoseqKeys.GENE_COUNT]
        )
        df_gene[StereoseqKeys.FEATURE_KEY] = df_gene[StereoseqKeys.FEATURE_KEY].str.decode("utf-8")
        df_gene = df_gene.rename(columns={"count": "counts"})  # #138 df_gene.count will throw error if not renamed

        # create df for points model
        arr = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.EXPRESSION][:]
        df_points = pd.DataFrame(
            arr, columns=[StereoseqKeys.COORD_X, StereoseqKeys.COORD_Y, StereoseqKeys.GENE_COUNT]
        )
        df_points = df_points.astype(np.float32)
        df_points[StereoseqKeys.EXON] = squarebin_gef[StereoseqKeys.GENE_EXP][i][StereoseqKeys.EXON][:]
        df_points[StereoseqKeys.FEATURE_KEY] = [
            name for name, cell_count in zip(df_gene.gene, df_gene.counts) for _ in range(cell_count)
        ]  # unroll gene names by count such that there exists a mapping between coordinate counts and gene names
        df_by_bin[i] = df_points

        # add more gene info to var
        df_gene = df_gene.rename(columns={"counts": "count"})
        df_gene = df_gene.set_index(StereoseqKeys.FEATURE_KEY)
        df_gene.index.name = None
        df_gene = df_gene.add_suffix("_" + str(i))
        adata.var = pd.concat([adata.var, df_gene], axis=1)

    images = {
        f"{name}": Image2DModel.parse(
            imread(path / StereoseqKeys.REGISTER / name, **imread_kwargs), dims=("c", "y", "x"), **image_models_kwargs
        )
        for name in image_filenames
    }

    labels = {
        f"{cell_mask_file[0]}": Labels2DModel.parse(
            imread(path / StereoseqKeys.REGISTER / cell_mask_file[0], **imread_kwargs).squeeze(), dims=("y", "x")
        )
    }

    points = {
        f"transcripts_{bin}": PointsModel.parse(
            df,
            coordinates={"x": StereoseqKeys.COORD_X, "y": StereoseqKeys.COORD_Y},
            feature_key=StereoseqKeys.FEATURE_KEY,
        )
        for bin, df in df_by_bin.items()
    }
    table = TableModel.parse(
        adata,
        region=StereoseqKeys.REGION,
        region_key=StereoseqKeys.REGION_KEY,
        instance_key=StereoseqKeys.INSTANCE_KEY,
    )
    # shapes = {
    #     StereoseqKeys.REGION_KEY: ShapesModel.parse(
    #         adata.obsm[StereoseqKeys.SPATIAL_KEY], geometry=3, index=adata.obs[StereoseqKeys.INSTANCE_KEY]
    #     )
    # }
    sdata = SpatialData(images=images, labels=labels, table=table, points=points)

    return sdata
