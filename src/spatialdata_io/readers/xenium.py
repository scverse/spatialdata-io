from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anndata import AnnData
from dask_image.imread import imread
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pyarrow import Table
from shapely import Polygon
from spatial_image import SpatialImage
from spatialdata import (
    Image2DModel,
    PointsModel,
    PolygonsModel,
    NgffScale,
    ShapesModel,
    SpatialData,
    TableModel,
)
from spatialdata._types import ArrayLike

from spatialdata_io._constants._constants import XeniumKeys
from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

# format specification https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html#polygon_vertices

__all__ = ["xenium"]


def _get_polygons(path: Path, file: str, specs: dict[str, Any], n_jobs: int) -> GeoDataFrame:
    def _poly(arr: ArrayLike) -> Polygon:
        return Polygon(arr[:-1])

    # seems to be faster than pd.read_parquet
    df = pq.read_table(path / file).to_pandas()

    out = Parallel(n_jobs=n_jobs)(
        delayed(_poly)(i.to_numpy())
        for _, i in df.groupby(XeniumKeys.CELL_ID)[[XeniumKeys.BOUNDARIES_VERTEX_X, XeniumKeys.BOUNDARIES_VERTEX_Y]]
    )
    geo_df = GeoDataFrame({"geometry": out})
    scale = NgffScale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]])
    return PolygonsModel.parse(geo_df, transform=scale)


def _get_points(path: Path, specs: dict[str, Any]) -> Table:
    table = pq.read_table(path / XeniumKeys.TRANSCRIPTS_FILE)
    arr = (
        table.select([XeniumKeys.TRANSCRIPTS_X, XeniumKeys.TRANSCRIPTS_Y, XeniumKeys.TRANSCRIPTS_Z])
        .to_pandas()
        .to_numpy()
    )
    annotations = table.select((XeniumKeys.OVERLAPS_NUCLEUS, XeniumKeys.QUALITY_VALUE, XeniumKeys.CELL_ID))
    annotations = annotations.add_column(
        3, XeniumKeys.FEATURE_NAME, table.column(XeniumKeys.FEATURE_NAME).cast("string").dictionary_encode()
    )

    transform = NgffScale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"], 1.0])
    points = PointsModel.parse(coords=arr, annotations=annotations, transform=transform)
    return points


def _get_tables(path: Path, specs: dict[str, Any], shape_size: Union[float, int]) -> tuple[AnnData, AnnData]:

    adata = _read_10x_h5(path / XeniumKeys.CELL_FEATURE_MATRIX_FILE)
    metadata = pd.read_parquet(path / XeniumKeys.CELL_METADATA_FILE)
    np.testing.assert_array_equal(metadata.cell_id.astype(str).values, adata.obs_names.values)

    circ = metadata[[XeniumKeys.CELL_X, XeniumKeys.CELL_Y]].to_numpy()
    metadata.drop([XeniumKeys.CELL_X, XeniumKeys.CELL_Y], axis=1, inplace=True)
    adata.obs = metadata
    transform = NgffScale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]])
    circles = ShapesModel.parse(circ, shape_type="circle", shape_size=shape_size, transform=transform)
    table = TableModel.parse(adata, region="/polygons/cell_boundaries", instance_key="cell_id")
    return table, circles


def _get_images(
    path: Path,
    file: str,
    specs: dict[str, Any],
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    image = imread(path / file, **imread_kwargs)
    transform = NgffScale([1.0, 1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]])
    return Image2DModel.parse(image, transform=transform, **image_models_kwargs)


def xenium(
    path: str | Path,
    # dataset_id: str,
    n_jobs: int = 1,
    nucleus_boundaries: bool = True,
    cell_boundaries: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    shape_size: Union[int, float] = 1,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read a *10X Genomics Xenium* dataset into a SpatialData object.

    .. seealso::

        - `10X Genomics Xenium file format  <https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    n_jobs
        Number of jobs to use for parallel processing.
    nucleus_boundaries
        Whether to read nucleus boundaries.
    cell_boundaries
        Whether to read cell boundaries.
    transcripts
        Whether to read transcripts.
    morphology_mip
        Whether to read morphology mip.
    morphology_focus
        Whether to read morphology focus.
    shape_size
        Size of the shape to use for the cell boundaries.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    with open(path / XeniumKeys.XENIUM_SPECS) as f:
        specs = json.load(f)

    polygons = {}
    if nucleus_boundaries:
        polygons["nucleus_boundaries"] = _get_polygons(
            path,
            XeniumKeys.NUCLEUS_BOUNDARIES_FILE,
            specs,
            n_jobs,
        )
    if cell_boundaries:
        polygons["cell_boundaries"] = _get_polygons(
            path,
            XeniumKeys.CELL_BOUNDARIES_FILE,
            specs,
            n_jobs,
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

    circles = {}
    table, circles["circles"] = _get_tables(path, specs, shape_size)

    return SpatialData(images=images, polygons=polygons, points=points, shapes=circles, table=table)
