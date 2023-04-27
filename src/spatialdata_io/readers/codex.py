from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import anndata as ad
import readfcs
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import TableModel

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
    # get library_id
    patt = re.compile(f".*{CodexKeys.FCS_FILE}")
    first_file = [i for i in os.listdir(path) if patt.match(i)][0]
    if f"_{CodexKeys.FCS_FILE}" in first_file:
        # read the .fcs table
        fcs = readfcs.ReadFCS()
        counts = fcs.filter(regex="cyc.*")
        # counts = fcs.data.iloc[:, 9:]
        adata = ad.AnnData(counts)
        adata.obs = fcs[fcs.columns.drop(list(fcs.filter(regex='cyc.*')))]
        adata.obs.set_index('cell_id', inplace=True, drop=False)
        adata.obsm["spatial"] = fcs[['x', 'y']].values
        adata.var_names_make_unique()
    else:
        raise ValueError(
            f"Cannot determine the library_id. Expecting a file with format <library_id>_{CodexKeys.FCS_FILE}. Has "
            f"the files been renamed?"
        )

    table = TableModel.parse(adata, region=dataset_id, region_key="region:region", instance_key="cell_id:cell_id")

    if (path / f"{dataset_id}{CodexKeys.IMAGE_TIF_SUFFIX}").exists():
        path / f"{dataset_id}{CodexKeys.IMAGE_TIF_SUFFIX}"
    else:
        raise FileNotFoundError(
            f"Cannot find {CodexKeys.IMAGE_TIF_SUFFIX} or {CodexKeys.IMAGE_TIF_ALTERNATIVE_SUFFIX}."
        )

    image = iio.imread(path / ".tif")
    segmentation = iio.imread(path / "segmentation.tif")

    images = {"images": SpatialData.Image2DModel.parse(image)}
    labels = {"labels": SpatialData.Labels2DModel.parse(segmentation, dims=("y", "x"))}
    return SpatialData(images=images, labels=labels, table=table)
