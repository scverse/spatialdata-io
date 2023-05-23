from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import TableModel

from spatialdata_io._constants._constants import CurioKeys
from spatialdata_io._docs import inject_docs

__all__ = ["curio"]


@inject_docs(vx=CurioKeys)
def curio(
    path: str | Path,
    dataset_id: Optional[str] = None,
) -> SpatialData:
    """
    Read *Curio* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.ANNDATA_FILE!r}```: Counts and metadata file.
        - ``<dataset_id>_`{vx.CLUSTER_ASSIGNMENT!r}```: Cluster assignment file.
        - ``<dataset_id>_`{vx.METRICS_FILE!r}```: Metrics file.
        - ``<dataset_id>_`{vx.VAR_FEATURES_CLUSTERS!r}```: Variable features clusters file.
        - ``<dataset_id>_`{vx.VAR_FEATURES_MORANSI!r}```: Variable features Moran's I file.

    .. seealso::

        - `CODEX output <https://help.codex.bio/codex/processor/technical-notes/expected-output>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    path_files = [
        CurioKeys.ANNDATA_FILE,
        CurioKeys.CLUSTER_ASSIGNMENT,
        CurioKeys.METRICS_FILE,
        CurioKeys.VAR_FEATURES_CLUSTERS,
        CurioKeys.VAR_FEATURES_MORANSI,
    ]

    if dataset_id is not None:
        file_names = [f"{dataset_id}_{file_name}" for file_name in path_files]
    else:
        file_names = []
        for file_name in path_files:
            file_names.extend(str(path.glob(file_name)))

    adata = ad.read_h5ad(path / file_names[0])
    cluster_assign = pd.read_csv(path / file_names[1], sep="\t", header=None)
    metrics = pd.read_csv(path / file_names[2], sep=r"\,", header=0)
    var_features_clusters = pd.read_csv(path / file_names[3], sep="\t", header=0)
    var_features_moransi = pd.read_csv(path / file_names[4], sep="\t", header=0)

    adata.obs = adata.obs.assign(cluster=cluster_assign[1].values)
    categories = metrics[CurioKeys.CATEGORY].unique()
    for cat in categories:
        df = metrics.loc[metrics[CurioKeys.CATEGORY] == cat]
        adata.uns[cat] = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    adata.uns[CurioKeys.TOP_CLUSTER_DEFINING_FEATURES] = var_features_clusters
    adata.var.join(var_features_moransi, how="outer")

    table = TableModel.parse(adata)

    return SpatialData(table=table)
