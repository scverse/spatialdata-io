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
from spatialdata.transformations import Identity

from spatialdata_io._constants._constants import SeqfishKeys as SK
from spatialdata_io._docs import inject_docs

__all__ = ["seqfish"]


@inject_docs(vx=SK)
def seqfish(
    path: str | Path,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    sections: list[int] | None = None,
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
    load_images
        Whether to load the images.
    load_labels
        Whether to load the labels.
    load_points
        Whether to load the points.
    sections
        Which sections (specified as integers) to load. By default, all sections are loaded.
    imread_kwargs
        Keyword arguments to pass to :func:`dask_image.imread.imread`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    count_file_pattern = re.compile(rf"(.*?)_{SK.CELL_COORDINATES}_{SK.SECTION}[0-9]+" + re.escape(SK.CSV_FILE))
    count_files = [i for i in os.listdir(path) if count_file_pattern.match(i)]
    if not count_files:
        # no file matching tbe pattern found
        raise ValueError(
            f"No files matching the pattern {count_file_pattern} were found. Cannot infer the naming scheme."
        )
    matched = count_file_pattern.match(count_files[0])
    if matched is None:
        raise ValueError(f"File {count_files[0]} does not match the pattern {count_file_pattern}")
    prefix = matched.group(1)

    n = len(count_files)
    all_sections = list(range(1, n + 1))
    if sections is None:
        sections = all_sections
    else:
        for section in sections:
            if section not in all_sections:
                raise ValueError(f"Section {section} not found in the data.")
    sections_str = [f"{SK.SECTION}{x}" for x in sections]

    def get_cell_file(section: str) -> str:
        return f"{prefix}_{SK.CELL_COORDINATES}_{section}{SK.CSV_FILE}"

    def get_count_file(section: str) -> str:
        return f"{prefix}_{SK.COUNTS_FILE}_{section}{SK.CSV_FILE}"

    def get_dapi_file(section: str) -> str:
        return f"{prefix}_{SK.DAPI}_{section}{SK.OME_TIFF_FILE}"

    def get_cell_mask_file(section: str) -> str:
        return f"{prefix}_{SK.CELL_MASK_FILE}_{section}{SK.TIFF_FILE}"

    def get_transcript_file(section: str) -> str:
        return f"{prefix}_{SK.TRANSCRIPT_COORDINATES}_{section}{SK.CSV_FILE}"

    adatas: dict[str, ad.AnnData] = {}
    for section in sections_str:  # type: ignore[assignment]
        assert isinstance(section, str)
        cell_file = get_cell_file(section)
        count_matrix = get_count_file(section)
        adata = ad.read_csv(path / count_matrix, delimiter=",")
        cell_info = pd.read_csv(path / cell_file, delimiter=",")
        adata.obsm[SK.SPATIAL_KEY] = cell_info[[SK.CELL_X, SK.CELL_Y]].to_numpy()
        adata.obs[SK.AREA] = np.reshape(cell_info[SK.AREA].to_numpy(), (-1, 1))
        region = f"cells_{section}"
        adata.obs[SK.REGION_KEY] = region
        adata.obs[SK.INSTANCE_KEY_TABLE] = adata.obs.index.astype(int)
        adatas[section] = adata

    scale_factors = [2, 2, 2, 2]

    if load_images:
        images = {
            f"image_{x}": Image2DModel.parse(
                imread(path / get_dapi_file(x), **imread_kwargs),
                dims=("c", "y", "x"),
                scale_factors=scale_factors,
                transformations={x: Identity()},
            )
            for x in sections_str
        }
    else:
        images = {}

    if load_labels:
        labels = {
            f"labels_{x}": Labels2DModel.parse(
                imread(path / get_cell_mask_file(x), **imread_kwargs).squeeze(),
                dims=("y", "x"),
                scale_factors=scale_factors,
                transformations={x: Identity()},
            )
            for x in sections_str
        }
    else:
        labels = {}

    if load_points:
        points = {
            f"transcripts_{x}": PointsModel.parse(
                pd.read_csv(path / get_transcript_file(x), delimiter=","),
                coordinates={"x": SK.TRANSCRIPTS_X, "y": SK.TRANSCRIPTS_Y},
                feature_key=SK.FEATURE_KEY.value,
                instance_key=SK.INSTANCE_KEY_POINTS.value,
                transformations={x: Identity()},
            )
            for x in sections_str
        }
    else:
        points = {}

    adata = ad.concat(adatas.values())
    adata.obs[SK.REGION_KEY] = adata.obs[SK.REGION_KEY].astype("category")
    adata.obs = adata.obs.reset_index(drop=True)
    table = TableModel.parse(
        adata,
        region=[f"cells_{x}" for x in sections_str],
        region_key=SK.REGION_KEY.value,
        instance_key=SK.INSTANCE_KEY_TABLE.value,
    )

    shapes = {
        f"cells_{x}": ShapesModel.parse(
            adata.obsm[SK.SPATIAL_KEY],
            geometry=0,
            radius=np.sqrt(adata.obs[SK.AREA].to_numpy() / np.pi),
            index=adata.obs[SK.INSTANCE_KEY_TABLE].copy(),
            transformations={x: Identity()},
        )
        for x, adata in adatas.items()
    }

    sdata = SpatialData(images=images, labels=labels, points=points, table=table, shapes=shapes)

    return sdata
