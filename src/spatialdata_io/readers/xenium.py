from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anndata import AnnData
from dask.dataframe import read_parquet
from dask_image.imread import imread
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pyarrow import Table
from shapely import Polygon
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale

from spatialdata_io._constants._constants import XeniumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

__all__ = ["xenium"]


@inject_docs(xx=XeniumKeys)
def xenium(
    path: str | Path,
    n_jobs: int = 1,
    cells_as_shapes: bool = False,
    nucleus_boundaries: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read a *10X Genomics Xenium* dataset into a SpatialData object.

    This function reads the following files:

        - ``{xx.XENIUM_SPECS!r}``: File containing specifications.
        - ``{xx.NUCLEUS_BOUNDARIES_FILE!r}``: Polygons of nucleus boundaries.
        - ``{xx.CELL_BOUNDARIES_FILE!r}``: Polygons of cell boundaries.
        - ``{xx.TRANSCRIPTS_FILE!r}``: File containing transcripts.
        - ``{xx.CELL_FEATURE_MATRIX_FILE!r}``: File containing cell feature matrix.
        - ``{xx.CELL_METADATA_FILE!r}``: File containing cell metadata.
        - ``{xx.MORPHOLOGY_MIP_FILE!r}``: File containing morphology mip.
        - ``{xx.MORPHOLOGY_FOCUS_FILE!r}``: File containing morphology focus.

    .. seealso::

        - `10X Genomics Xenium file format  <https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    n_jobs
        Number of jobs to use for parallel processing.
    cells_as_shapes
        Whether to read cells also as shapes. Useful for visualization.
    nucleus_boundaries
        Whether to read nucleus boundaries.
    transcripts
        Whether to read transcripts.
    morphology_mip
        Whether to read morphology mip.
    morphology_focus
        Whether to read morphology focus.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    path = Path(path)
    with open(path / XeniumKeys.XENIUM_SPECS) as f:
        specs = json.load(f)

    specs["region"] = "cell_circles" if cells_as_shapes else "cell_boundaries"
    return_values = _get_tables_and_circles(path, cells_as_shapes, specs)
    if cells_as_shapes:
        table, circles = return_values
    else:
        table = return_values
    polygons = {}

    if nucleus_boundaries:
        polygons["nucleus_boundaries"] = _get_polygons(
            path,
            XeniumKeys.NUCLEUS_BOUNDARIES_FILE,
            specs,
            n_jobs,
        )

    polygons["cell_boundaries"] = _get_polygons(
        path, XeniumKeys.CELL_BOUNDARIES_FILE, specs, n_jobs, idx=table.obs[str(XeniumKeys.CELL_ID)].copy()
    )

    points = {}
    if transcripts:
        points["transcripts"] = _get_points(path, specs)

    images = {}
    if morphology_mip:
        images["morphology_mip"] = _get_images(
            path,
            XeniumKeys.MORPHOLOGY_MIP_FILE,
            specs,
            imread_kwargs,
            image_models_kwargs,
        )
    if morphology_focus:
        images["morphology_focus"] = _get_images(
            path,
            XeniumKeys.MORPHOLOGY_MIP_FILE,
            specs,
            imread_kwargs,
            image_models_kwargs,
        )
    if cells_as_shapes:
        return SpatialData(images=images, shapes=polygons | {specs["region"]: circles}, points=points, table=table)
    return SpatialData(images=images, shapes=polygons, points=points, table=table)


def _get_polygons(
    path: Path, file: str, specs: dict[str, Any], n_jobs: int, idx: Optional[ArrayLike] = None
) -> GeoDataFrame:
    def _poly(arr: ArrayLike) -> Polygon:
        return Polygon(arr[:-1])

    # seems to be faster than pd.read_parquet
    df = pq.read_table(path / file).to_pandas()

    out = Parallel(n_jobs=n_jobs)(
        delayed(_poly)(i.to_numpy())
        for _, i in df.groupby(XeniumKeys.CELL_ID)[[XeniumKeys.BOUNDARIES_VERTEX_X, XeniumKeys.BOUNDARIES_VERTEX_Y]]
    )
    geo_df = GeoDataFrame({"geometry": out})
    if idx is not None:
        geo_df.index = idx
    scale = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    return ShapesModel.parse(geo_df, transformations={"global": scale})


def _get_points(path: Path, specs: dict[str, Any]) -> Table:
    table = read_parquet(path / XeniumKeys.TRANSCRIPTS_FILE)
    table["feature_name"] = table["feature_name"].apply(lambda x: x.decode("utf-8"), meta=("feature_name", "object"))

    transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    points = PointsModel.parse(
        table,
        coordinates={"x": XeniumKeys.TRANSCRIPTS_X, "y": XeniumKeys.TRANSCRIPTS_Y, "z": XeniumKeys.TRANSCRIPTS_Y},
        feature_key=XeniumKeys.FEATURE_NAME,
        instance_key=XeniumKeys.CELL_ID,
        transformations={"global": transform},
    )
    return points


def _get_tables_and_circles(
    path: Path, cells_as_shapes: bool, specs: dict[str, Any]
) -> AnnData | tuple[AnnData, AnnData]:
    adata = _read_10x_h5(path / XeniumKeys.CELL_FEATURE_MATRIX_FILE)
    metadata = pd.read_parquet(path / XeniumKeys.CELL_METADATA_FILE)
    np.testing.assert_array_equal(metadata.cell_id.astype(str).values, adata.obs_names.values)
    circ = metadata[[XeniumKeys.CELL_X, XeniumKeys.CELL_Y]].to_numpy()
    adata.obsm["spatial"] = circ
    metadata.drop([XeniumKeys.CELL_X, XeniumKeys.CELL_Y], axis=1, inplace=True)
    adata.obs = metadata
    adata.obs["region"] = specs["region"]
    table = TableModel.parse(adata, region=specs["region"], region_key="region", instance_key=str(XeniumKeys.CELL_ID))
    if cells_as_shapes:
        transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
        radii = np.sqrt(adata.obs[XeniumKeys.CELL_NUCLEUS_AREA].to_numpy() / np.pi)
        circles = ShapesModel.parse(
            circ,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs[XeniumKeys.CELL_ID].copy(),
        )
        return table, circles
    return table


def _get_images(
    path: Path,
    file: str,
    specs: dict[str, Any],
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialImage | MultiscaleSpatialImage:
    image = imread(path / file, **imread_kwargs)
    return Image2DModel.parse(
        image, transformations={"global": Identity()}, dims=("c", "y", "x"), **image_models_kwargs
    )
