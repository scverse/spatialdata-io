from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from spatialdata_io._constants._constants import CurioKeys
from spatialdata_io._docs import inject_docs

__all__ = ["curio"]


@inject_docs(vx=CurioKeys)
def curio(
    path: str | Path,
) -> SpatialData:
    """
    Read *Curio* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.ANNDATA_FILE!r}```: Counts and metadata file.
        - ``<dataset_id>_`{vx.CLUSTER_ASSIGNMENT!r}```: Cluster assignment file.
        - ``<dataset_id>_`{vx.METRICS_FILE!r}```: Metrics file.
        - ``<dataset_id>_`{vx.VAR_FEATURES_CLUSTERS!r}```: Variable features clusters file.
        - ``<dataset_id>_`{vx.VAR_FEATURES_MORANSI!r}```: Variable features Moran's I file.

    <dataset_id> is automatically inferred from the path.

    Parameters
    ----------
    path
        Path to the directory containing the data.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    path_files = (
        CurioKeys.ANNDATA_FILE,
        CurioKeys.CLUSTER_ASSIGNMENT,
        CurioKeys.METRICS_FILE,
        CurioKeys.VAR_FEATURES_CLUSTERS,
        CurioKeys.VAR_FEATURES_MORANSI,
    )

    file_names = {}
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)):
            for path_file in path_files:
                if path_file in i:
                    file_names[path_file] = i
    assert len(file_names) == len(path_files), f"Missing files: {set(path_files) - set(file_names.keys())}"

    adata = ad.read_h5ad(path / file_names[CurioKeys.ANNDATA_FILE])
    cluster_assign = pd.read_csv(path / file_names[CurioKeys.CLUSTER_ASSIGNMENT], sep="\t", header=None)
    metrics = pd.read_csv(path / file_names[CurioKeys.METRICS_FILE], sep=r"\,", header=0)
    var_features_clusters = pd.read_csv(path / file_names[CurioKeys.VAR_FEATURES_CLUSTERS], sep="\t", header=0)
    var_features_moransi = pd.read_csv(path / file_names[CurioKeys.VAR_FEATURES_MORANSI], sep="\t", header=0)

    # adding cluster information in adata.obs
    assert np.array_equal(cluster_assign[0].to_numpy(), adata.obs.index.to_numpy())
    adata.obs = adata.obs.assign(cluster=cluster_assign[1].values)
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    # adding metrics information in adata.uns
    categories = metrics[CurioKeys.CATEGORY].unique()
    for cat in categories:
        df = metrics.loc[metrics[CurioKeys.CATEGORY] == cat]
        adata.uns[cat] = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    adata.uns[CurioKeys.TOP_CLUSTER_DEFINING_FEATURES] = var_features_clusters

    # adding Moran's I information in adata.var, for the variable for which it is available
    assert set(adata.var_names).issuperset(var_features_moransi.index)
    adata.var.join(var_features_moransi, how="outer")

    adata.obs[CurioKeys.REGION_KEY] = CurioKeys.REGION
    adata.obs[CurioKeys.REGION_KEY] = adata.obs[CurioKeys.REGION_KEY].astype("category")
    adata.obs[CurioKeys.INSTANCE_KEY] = adata.obs.index

    table = TableModel.parse(
        adata,
        region=CurioKeys.REGION.value,
        region_key=CurioKeys.REGION_KEY.value,
        instance_key=CurioKeys.INSTANCE_KEY.value,
    )

    # adding geometry information in a shapes element (we redundantly leave it in obsm['spatial'])
    assert np.array_equal(adata.obsm["spatial"], adata.obsm["X_spatial"])
    xy = adata.obsm["spatial"]
    del adata.obsm["X_spatial"]

    shapes = ShapesModel.parse(xy, geometry=0, radius=10, index=adata.obs[CurioKeys.INSTANCE_KEY])

    return SpatialData(table=table, shapes={CurioKeys.REGION.value: shapes})
