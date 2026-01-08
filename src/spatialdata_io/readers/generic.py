from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dask_image.imread import imread
from spatialdata._docs import docstring_parameter
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from spatialdata.transformations import Identity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopandas import GeoDataFrame
    from xarray import DataArray

VALID_IMAGE_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
VALID_SHAPE_TYPES = [".geojson"]

__all__ = ["generic", "geojson", "image", "VALID_IMAGE_TYPES", "VALID_SHAPE_TYPES"]


@docstring_parameter(
    valid_image_types=", ".join(VALID_IMAGE_TYPES),
    valid_shape_types=", ".join(VALID_SHAPE_TYPES),
    default_coordinate_system=DEFAULT_COORDINATE_SYSTEM,
)
def generic(
    input: Path,
    data_axes: Sequence[str] | None = None,
    coordinate_system: str | None = None,
) -> DataArray | GeoDataFrame:
    """Read a generic shapes or image file and save it as SpatialData zarr.

    Supported image types: {valid_image_types}.
    Supported shape types: {valid_shape_types} (only Polygons and MultiPolygons are supported).

    Parameters
    ----------
    input
        Path to the input file.
    data_axes
        Axes of the data for image files, required for image files.
    coordinate_system
        Coordinate system of the spatial element; if None, the default coordinate system ({default_coordinate_system})
        is used.

    Returns
    -------
    Parsed spatial element.
    """
    if isinstance(input, str):
        input = Path(input)
    if coordinate_system is None:
        coordinate_system = DEFAULT_COORDINATE_SYSTEM
    if input.suffix in VALID_SHAPE_TYPES:
        if data_axes is not None:
            warnings.warn("data_axes is not used for geojson files", UserWarning, stacklevel=2)
        return geojson(input, coordinate_system=coordinate_system)
    elif input.suffix in VALID_IMAGE_TYPES:
        if data_axes is None:
            raise ValueError("data_axes must be provided for image files")
        return image(input, data_axes=data_axes, coordinate_system=coordinate_system)
    else:
        raise ValueError(f"Invalid file type. Must be one of {VALID_SHAPE_TYPES + VALID_IMAGE_TYPES}")


def geojson(input: Path, coordinate_system: str) -> GeoDataFrame:
    """Reads a GeoJSON file and returns a parsed GeoDataFrame spatial element."""
    return ShapesModel.parse(input, transformations={coordinate_system: Identity()})


def image(input: Path, data_axes: Sequence[str], coordinate_system: str) -> DataArray:
    """Reads an image file and returns a parsed Image2D spatial element."""
    # this function is just a draft, the more general one will be available when
    # https://github.com/scverse/spatialdata-io/pull/234 is merged
    image = imread(input)
    if len(image.shape) == len(data_axes) + 1 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    return Image2DModel.parse(image, dims=data_axes, transformations={coordinate_system: Identity()})
