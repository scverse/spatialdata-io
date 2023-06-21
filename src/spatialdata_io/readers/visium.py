from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import _read_counts

__all__ = ["visium"]


@inject_docs(vx=VisiumKeys)
def visium(
    path: str | Path,
    dataset_id: Optional[str] = None,
    counts_file: str = VisiumKeys.COUNTS_FILE,
    fullres_image_file: Optional[str | Path] = None,
    tissue_positions_file: Optional[str | Path] = None,
    scalefactors_file: Optional[str | Path] = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> SpatialData:
    """
    Read *10x Genomics* Visium formatted dataset.

    This function reads the following files:

        - ``{vx.COUNTS_FILE!r}``: Counts and metadata file.
        - ``{vx.IMAGE_HIRES_FILE!r}``: High resolution image.
        - ``{vx.IMAGE_LOWRES_FILE!r}``: Low resolution image.
        - ``{vx.SCALEFACTORS_FILE!r}``: Scalefactors file.
        - ``{vx.SPOTS_FILE_1!r}`` (SpaceRanger 1) or ``{vx.SPOTS_FILE_2!r}`` (SpaceRanger 2):
            Spots positions file.
        - ``fullres_image_file``: large microscopy image used as input for space ranger.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier. If not given will be determined automatically
        from the ``{vx.COUNTS_FILE!r}`` file.
    counts_file
        Name of the counts file. Use only if counts is not in `h5` format.
    fullres_image_file
        Path to the full-resolution image.
    tissue_positions_file
        Path to the tissue positions file.
    scalefactors_file
        Path to the scalefactors file.
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
    # get library_id
    try:
        patt = re.compile(f".*{VisiumKeys.COUNTS_FILE}")
        first_file = [i for i in os.listdir(path) if patt.match(i)][0]

        if f"_{VisiumKeys.COUNTS_FILE}" in first_file:
            library_id = first_file.replace(f"_{VisiumKeys.COUNTS_FILE}", "")
        else:
            raise ValueError(
                f"Cannot determine the library_id. Expecting a file with format <library_id>_{VisiumKeys.COUNTS_FILE}. Has "
                f"the files been renamed?"
            )
        counts_file = f"{library_id}_{VisiumKeys.COUNTS_FILE}"
    except IndexError as e:
        logger.error(
            f"{e}. \nError is due to the fact that the library id could not be found, this is the case when the `counts_file` is `.mtx`.",
        )
        if dataset_id is None:
            raise ValueError("Cannot determine the `library_id`. Please provide `dataset_id`.")
        library_id = dataset_id
        if counts_file is None:
            raise ValueError("Cannot determine the library_id. Please provide `counts_file`.")

    if dataset_id is not None:
        if dataset_id != library_id:
            logger.warning(
                f"`dataset_id: {dataset_id}` does not match `library_id: {library_id}`. `dataset_id: {dataset_id}` "
                f"will be used to build SpatialData."
            )
    else:
        dataset_id = library_id

    adata, dataset_id = _read_counts(path, counts_file=counts_file, library_id=dataset_id, **kwargs)

    if (path / "spatial" / VisiumKeys.SPOTS_FILE_1).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_1) in str(tissue_positions_file)
    ):
        read_coords = partial(pd.read_csv, header=None, index_col=0)
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_1
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    elif (path / "spatial" / VisiumKeys.SPOTS_FILE_2).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_2) in str(tissue_positions_file)
    ):
        read_coords = partial(pd.read_csv, header=1, index_col=0)
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_2
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    else:
        raise ValueError(f"Cannot find `tissue_positions` file in `{path}`.")
    coords = read_coords(tissue_positions_file)

    coords.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    coords = adata.obs[[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].values
    adata.obsm["spatial"] = coords
    adata.obs.drop(columns=[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y], inplace=True)
    adata.obs["spot_id"] = np.arange(len(adata))
    adata.var_names_make_unique()
    if (path / "spatial" / VisiumKeys.SCALEFACTORS_FILE).exists() or (
        scalefactors_file is not None and (path / scalefactors_file).exists()
    ):
        scalefactors_file = (
            path / "spatial" / VisiumKeys.SCALEFACTORS_FILE if scalefactors_file is None else path / scalefactors_file
        )
        scalefactors = json.loads(scalefactors_file.read_bytes())

    transform_original = Identity()
    transform_lowres = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]]),
        axes=("y", "x"),
    )
    transform_hires = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]]),
        axes=("y", "x"),
    )
    shapes = {}
    circles = ShapesModel.parse(
        coords,
        geometry=0,
        radius=scalefactors["spot_diameter_fullres"] / 2.0,
        index=adata.obs["spot_id"].copy(),
        transformations={
            "global": transform_original,
            "downscaled_hires": transform_hires,
            "downscaled_lowres": transform_lowres,
        },
    )
    shapes[dataset_id] = circles
    adata.obs["region"] = dataset_id
    table = TableModel.parse(adata, region=dataset_id, region_key="region", instance_key="spot_id")

    images = {}
    if fullres_image_file is not None:
        fullres_image_file = path / Path(fullres_image_file)
        if fullres_image_file.exists():
            if "MAX_IMAGE_PIXELS" in imread_kwargs:
                from PIL import Image as ImagePIL

                ImagePIL.MAX_IMAGE_PIXELS = imread_kwargs.pop("MAX_IMAGE_PIXELS")
            full_image = imread(fullres_image_file, **imread_kwargs).squeeze().transpose(2, 0, 1)
            full_image = DataArray(full_image, dims=("c", "y", "x"), name=dataset_id)
            images[dataset_id + "_full_image"] = Image2DModel.parse(
                full_image,
                scale_factors=[2, 2, 2, 2],
                transformations={"global": transform_original},
                **image_models_kwargs,
            )
        else:
            logger.warning(f"File {fullres_image_file} does not exist, skipping...")

    if (path / VisiumKeys.IMAGE_HIRES_FILE).exists():
        image_hires = imread(path / VisiumKeys.IMAGE_HIRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
        image_hires = DataArray(image_hires, dims=("c", "y", "x"), name=dataset_id)
        images[dataset_id + "_hires_image"] = Image2DModel.parse(
            image_hires, transformations={"downscaled_hires": Identity()}
        )
    if (path / VisiumKeys.IMAGE_LOWRES_FILE).exists():
        image_lowres = imread(path / VisiumKeys.IMAGE_LOWRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
        image_lowres = DataArray(image_lowres, dims=("c", "y", "x"), name=dataset_id)
        images[dataset_id + "_lowres_image"] = Image2DModel.parse(
            image_lowres, transformations={"downscaled_lowres": Identity()}
        )

    return SpatialData(images=images, shapes=shapes, table=table)
