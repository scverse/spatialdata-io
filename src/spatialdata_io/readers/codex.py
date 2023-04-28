from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import imageio.v3 as iio
import pandas as pd
import readfcs
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, TableModel

from spatialdata_io._constants._constants import CodexKeys
from spatialdata_io._docs import inject_docs

__all__ = ["codex"]


@inject_docs(vx=CodexKeys)
def codex(
    path: str | Path,
    fcs: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> SpatialData:
    """
    Read *CODEX* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.FCS_FILE!r}```: Counts and metadata file.
        - ``<dataset_id>_`{vx.IMAGE_TIF!r}```: High resolution tif image.

    .. seealso::

        - `CODEX output <https://help.codex.bio/codex/processor/technical-notes/expected-output>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    fcs
        Whether a .fcs file is provided. If False, a .csv file is expected.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    patt = re.compile(f"{CodexKeys.FCS_FILE}") if fcs else re.compile(f"{CodexKeys.FCS_FILE_CSV}")
    path_files = [i for i in os.listdir(path) if patt.match(i)]
    if path_files and f"{CodexKeys.FCS_FILE}" or f"{CodexKeys.FCS_FILE_CSV}" in path_files[0]:
        fcs = (
            readfcs.ReadFCS(path_files[0]).data
            if f"{CodexKeys.FCS_FILE}" in path_files[0]
            else pd.read_csv(path_files[0], header=0, index_col=None)
        )
    else:
        raise ValueError("Cannot determine data set. Expecting a file with format .fcs or .csv")

    adata = _df_to_anndata(fcs)

    region = adata.obs[f"{CodexKeys.REGION_KEY}"].unique()[0].tolist()
    table = TableModel.parse(
        adata, region=region, region_key=f"{CodexKeys.REGION_KEY}", instance_key=f"{CodexKeys.INSTANCE_KEY}"
    )

    im_patt = re.compile(f"{CodexKeys.IMAGE_TIF}")
    path_files = [i for i in os.listdir(path) if im_patt.match(i)]
    if path_files and f"{CodexKeys.IMAGE_TIF}" in path_files[0]:
        image = iio.imread(path_files[0])
        images = {
            "images": Image2DModel.parse(
                image,
                scale_factors=[2, 2, 2, 2],
            )
        }
        sdata = SpatialData(image=images, table=table)
    else:
        logger.warning(f"Cannot find {CodexKeys.IMAGE_TIF} file. Will build spatialdata with table only.")
        sdata = SpatialData(table=table)

    return sdata


def _df_to_anndata(df: pd.DataFrame) -> ad.AnnData:
    """
    Convert a DataFrame to an anndata object.
    """
    adata = ad.AnnData(df.filter(regex="cyc.*"))
    adata.obs = df[df.columns.drop(list(df.filter(regex="cyc.*")))]
    adata.obs.set_index("cell_id", inplace=True, drop=False)
    adata.obsm["spatial"] = df[["x", "y"]].values
    adata.var_names_make_unique()
    return adata
