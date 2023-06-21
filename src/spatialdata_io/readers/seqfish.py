from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)

from spatialdata_io._constants._constants import SeqfishKeys
from spatialdata_io._docs import inject_docs

__all__ = ["seqfish"]


@inject_docs(vx=SeqfishKeys)
def seqfish(
    path: str | Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *seqfish* formatted dataset.

    This function reads the following files:

        - ```{vx.COUNTS_FILE!r}{vx.SECTION!r}{vx.CSV_FILE!r}```: Counts and metadata file.
        - ```{vx.CELL_COORDINATES!r}{vx.SECTION!r}{vx.CSV_FILE!r}```: Cell coordinates file.
        - ```{vx.DAPI!r}{vx.SECTION!r}{vx.OME_TIFF_FILE!r}```: High resolution tiff image.
        - ```{vx.CELL_MASK_FILE!r}{vx.SECTION!r}{vx.TIFF_FILE!r}```: Cell mask file.
        - ```{vx.TRANSCRIPT_COORDINATES!r}{vx.SECTION!r}{vx.CSV_FILE!r}```: Transcript coordinates file.

    .. seealso::

        - `seqfish output <https://spatialgenomics.com/data/>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    csv_pattern = re.compile(r".*" + re.escape(SeqfishKeys.CSV_FILE))
    tiff_pattern = re.compile(r".*" + re.escape(SeqfishKeys.TIFF_FILE))
    ome_tiff_pattern = re.compile(r".*" + re.escape(SeqfishKeys.OME_TIFF_FILE))

    table_files = [i for i in os.listdir(path) if csv_pattern.match(i) or tiff_pattern.match(i) or ome_tiff_pattern.match(i)]
    count_matrices = [x for x in table_files if (SeqfishKeys.COUNTS_FILE in x)]

    if not table_files or not count_matrices:
        raise ValueError("No files required to build table were found.")

    adatas = {}

    for _sections, count_matrix in enumerate(count_matrices):
        section = int(re.findall(r"\d+", count_matrix)[0])
        cell_file = [x for x in table_files if (f"{SeqfishKeys.CELL_COORDINATES}{SeqfishKeys.SECTION}{section}" in x)][
            0
        ]
        adata = ad.read_csv(path / count_matrix, delimiter=",")
        cell_info = pd.read_csv(path / cell_file, delimiter=",")
        adata.obsm[SeqfishKeys.SPATIAL_KEY] = cell_info[[SeqfishKeys.CELL_X, SeqfishKeys.CELL_Y]].to_numpy()
        adata.obs[SeqfishKeys.AREA] = np.reshape(cell_info[SeqfishKeys.AREA].to_numpy(), (-1, 1))
        adata.obs[SeqfishKeys.SECTION_KEY] = section
        adatas[section] = adata

    dapi_file = [x for x in table_files if (f"{SeqfishKeys.DAPI}" in x)]
    cell_mask_file = [x for x in table_files if (f"{SeqfishKeys.CELL_MASK_FILE}" in x)]
    transcript_file = [x for x in table_files if (f"{SeqfishKeys.TRANSCRIPT_COORDINATES}" in x)]

    images = {
        f"label_{x+1}": Image2DModel.parse(imread(path / dapi_file[x - 1], **imread_kwargs), dims=("c", "y", "x"))
        for x in range(1, _sections + 1)
    }
    labels = {
        f"image_{x+1}": Labels2DModel.parse(
            imread(path / cell_mask_file[x - 1], **imread_kwargs).squeeze(), dims=("y", "x"))
        for x in range(1, _sections + 1)
    }
    points = {
        f"transcripts_{x+1}": PointsModel.parse(
            pd.read_csv(path / transcript_file[x - 1], delimiter=","),
            coordinates={"x": SeqfishKeys.TRANSCRIPTS_X, "y": SeqfishKeys.TRANSCRIPTS_Y},
            feature_key=SeqfishKeys.FEATURE_KEY,
            instance_key=SeqfishKeys.INSTANCE_KEY_POINTS,
        )
        for x in range(1, _sections + 1)
    }

    adata = ad.concat(adatas)
    adata.obs[SeqfishKeys.REGION_KEY] = SeqfishKeys.REGION
    adata.obs[SeqfishKeys.REGION_KEY] = adata.obs[SeqfishKeys.REGION_KEY].astype("category")
    adata.obs[SeqfishKeys.INSTANCE_KEY_TABLE] = adata.obs.index

    table = TableModel.parse(
        adata,
        region=SeqfishKeys.REGION.value,
        region_key=SeqfishKeys.REGION_KEY.value,
        instance_key=SeqfishKeys.INSTANCE_KEY_TABLE.value,
    )

    shapes = {
        SeqfishKeys.REGION.value: ShapesModel.parse(
            adata.obsm[SeqfishKeys.SPATIAL_KEY], geometry=0, radius=10, index=adata.obs[SeqfishKeys.INSTANCE_KEY_TABLE]
        )
    }
    sdata = SpatialData(images=images, labels=labels, points=points, table=table, shapes=shapes)

    return sdata
