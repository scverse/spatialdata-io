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
    from spatialdata.models.models import Chunks_t
    from xarray import DataArray


from spatialdata_io.readers._utils._image import (
    _compute_chunks,
    _read_chunks,
    normalize_chunks,
)

VALID_IMAGE_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
VALID_SHAPE_TYPES = [".geojson"]

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


def _tiff_to_chunks(
    input: Path,
    axes_dim_mapping: dict[str, int],
    chunks_cyx: dict[str, int],
) -> list[list[DaskArray[np.number]]]:
    """Chunkwise reader for tiff files.

    Creates spatial tiles from a TIFF file. Each tile contains all channels.
    Channel chunking is handled downstream by Image2DModel.parse().

    Parameters
    ----------
    input
        Path to image
    axes_dim_mapping
        Mapping between dimension name (c, y, x) and index
    chunks_cyx
        Chunk size dict with 'c', 'y', and 'x' keys. The 'y' and 'x' values
        are used for spatial tiling. The 'c' value is passed through for
        downstream rechunking.

    Returns
    -------
    list[list[DaskArray]]
        2D list of dask arrays representing spatial tiles, each with shape (n_channels, height, width).
    """
    # Lazy file reader
    slide = tifffile.memmap(input)

    # Transpose to cyx order
    slide = np.transpose(slide, (axes_dim_mapping["c"], axes_dim_mapping["y"], axes_dim_mapping["x"]))

    # Get dimensions in (y, x)
    slide_dimensions = slide.shape[1], slide.shape[2]

    # Get number of channels (all channels are included in each spatial tile)
    n_channel = slide.shape[0]

    # Compute chunk coords using (y, x) tuple
    chunk_coords = _compute_chunks(slide_dimensions, chunk_size=(chunks_cyx["y"], chunks_cyx["x"]))

    # Define reader func - reads all channels for each spatial tile
    def _reader_func(slide: np.memmap, y0: int, x0: int, height: int, width: int) -> NDArray[np.number]:
        return np.array(slide[:, y0 : y0 + height, x0 : x0 + width])

    return _read_chunks(_reader_func, slide, coords=chunk_coords, n_channel=n_channel, dtype=slide.dtype)


def _dask_image_imread(input: Path, data_axes: Sequence[str], chunks_cyx: dict[str, int]) -> da.Array:
    """Read image using dask-image and rechunk.

    Parameters
    ----------
    input
        Path to image file.
    data_axes
        Axes of the input data.
    chunks_cyx
        Chunk size dict with 'c', 'y', 'x' keys.

    Returns
    -------
    Dask array with (c, y, x) axes order.
    """
    if set(data_axes) != {"c", "y", "x"}:
        raise NotImplementedError(f"Only 'c', 'y', 'x' axes are supported, got {data_axes}")
    im = imread(input)

    # dask_image.imread may add an extra leading dimension for frames/pages
    # If image has one extra dimension with size 1, squeeze it out
    if im.ndim == len(data_axes) + 1 and im.shape[0] == 1:
        im = im[0]

    if im.ndim != len(data_axes):
        raise ValueError(f"Expected image with {len(data_axes)} dimensions, got {im.ndim}")

    im = im.transpose(*[data_axes.index(ax) for ax in ["c", "y", "x"]])
    return im.rechunk((chunks_cyx["c"], chunks_cyx["y"], chunks_cyx["x"]))


def image(
    input: Path,
    data_axes: Sequence[str],
    coordinate_system: str,
    use_tiff_memmap: bool = True,
    chunks: Chunks_t | None = None,
    scale_factors: Sequence[int] | None = None,
) -> DataArray:
    """Reads an image file and returns a parsed Image2D spatial element.

    Parameters
    ----------
    input
        Path to the image file.
    data_axes
        Axes of the data (e.g., ('c', 'y', 'x') or ('y', 'x', 'c')).
    coordinate_system
        Coordinate system of the spatial element.
    use_tiff_memmap
        Whether to use memory-mapped reading for TIFF files.
    chunks
        Chunk size specification. Can be:
        - int: Applied to all dimensions
        - tuple: Chunk sizes matching the order of output axes (c, y, x)
        - dict: Mapping of axis names to chunk sizes (e.g., {'c': 1, 'y': 1000, 'x': 1000})
        If None, uses a default (DEFAULT_CHUNK_SIZE) for all axes.
    scale_factors
        Scale factors for building a multiscale image pyramid. Passed to Image2DModel.parse().

    Returns
    -------
    Parsed Image2D spatial element.
    """
    # Map passed data axes to position of dimension
    axes_dim_mapping = {axes: ndim for ndim, axes in enumerate(data_axes)}

    chunks_dict = normalize_chunks(chunks, axes=data_axes)

    im = None
    if input.suffix in [".tiff", ".tif"] and use_tiff_memmap:
        try:
            im_chunks = _tiff_to_chunks(input, axes_dim_mapping=axes_dim_mapping, chunks_cyx=chunks_dict)
            im = da.block(im_chunks, allow_unknown_chunksizes=True)

        # Edge case: Compressed images are not memory-mappable
        except ValueError as e:
            logger.warning(
                f"Exception occurred: {str(e)}\nPossible troubleshooting: image data "
                "is not memory-mappable, potentially due to compression. Trying to "
                "load the image into memory at once",
            )
            use_tiff_memmap = False

    if input.suffix in [".tiff", ".tif"] and not use_tiff_memmap or input.suffix in [".png", ".jpg", ".jpeg"]:
        im = _dask_image_imread(input=input, data_axes=data_axes, chunks_cyx=chunks_dict)

    if im is None:
        raise NotImplementedError(f"File format {input.suffix} not implemented")

    # the output axes are always cyx
    return Image2DModel.parse(
        im,
        dims=("c", "y", "x"),
        transformations={coordinate_system: Identity()},
        scale_factors=scale_factors,
        chunks=chunks_dict,
    )
