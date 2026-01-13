from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import dask.array as da
import numpy as np
import tifffile
from dask_image.imread import imread
from geopandas import GeoDataFrame
from spatialdata._docs import docstring_parameter
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from spatialdata.transformations import Identity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopandas import GeoDataFrame
    from numpy.typing import NDArray
    from xarray import DataArray

from ._utils._image import _compute_chunks, _read_chunks

VALID_IMAGE_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
VALID_SHAPE_TYPES = [".geojson"]
DEFAULT_CHUNKSIZE = (1000, 1000)

__all__ = ["generic", "geojson", "image", "VALID_IMAGE_TYPES", "VALID_SHAPE_TYPES"]

T = TypeVar("T", bound=np.generic)  # Restrict to NumPy scalar types


class DaskArray(Protocol[T]):
    dtype: np.dtype[T]


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
            logger.warning("data_axes is not used for geojson files")
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


def _tiff_to_chunks(input: Path, axes_dim_mapping: dict[str, int]) -> list[list[DaskArray[np.int_]]]:
    """Chunkwise reader for tiff files.

    Parameters
    ----------
    input
        Path to image
    axes_dim_mapping
        Mapping between dimension name (x, y, c) and index

    Returns
    -------
    list[list[DaskArray]]
    """
    # Lazy file reader
    slide = tifffile.memmap(input)

    # Transpose to cyx order
    slide = np.transpose(slide, (axes_dim_mapping["c"], axes_dim_mapping["y"], axes_dim_mapping["x"]))

    # Get dimensions in (x, y)
    slide_dimensions = slide.shape[2], slide.shape[1]

    # Get number of channels (c)
    n_channel = slide.shape[0]

    # Compute chunk coords
    chunk_coords = _compute_chunks(slide_dimensions, chunk_size=DEFAULT_CHUNKSIZE, min_coordinates=(0, 0))

    # Define reader func
    def _reader_func(slide: NDArray[np.int_], x0: int, y0: int, width: int, height: int) -> NDArray[np.int_]:
        return np.array(slide[:, y0 : y0 + height, x0 : x0 + width])

    return _read_chunks(_reader_func, slide, coords=chunk_coords, n_channel=n_channel, dtype=slide.dtype)


def _dask_image_imread(input: Path, data_axes: Sequence[str]) -> da.Array:
    image = imread(input)
    if len(image.shape) == len(data_axes) + 1 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    return image


def image(input: Path, data_axes: Sequence[str], coordinate_system: str) -> DataArray:
    """Reads an image file and returns a parsed Image2D spatial element."""
    # Map passed data axes to position of dimension
    axes_dim_mapping = {axes: ndim for ndim, axes in enumerate(data_axes)}

    if input.suffix in [".tiff", ".tif"]:
        try:
            chunks = _tiff_to_chunks(input, axes_dim_mapping=axes_dim_mapping)
            image = da.block(chunks, allow_unknown_chunksizes=True)
            data_axes = ["c", "y", "x"]

        # Edge case: Compressed images are not memory-mappable
        except ValueError as e:
            # TODO: change to logger warning
            logger.warning(
                f"Exception occurred: {str(e)}\nPossible troubleshooting: image data "
                "is not memory-mappable, potentially due to compression. Trying to "
                "load the image into memory at once",
            )
            image = _dask_image_imread(input=input, data_axes=data_axes)

    elif input.suffix in [".png", ".jpg", ".jpeg"]:
        image = _dask_image_imread(input=input, data_axes=data_axes)
    else:
        raise NotImplementedError(f"File format {input.suffix} not implemented")

    return Image2DModel.parse(image, dims=data_axes, transformations={coordinate_system: Identity()})
