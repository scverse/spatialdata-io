from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

import dask.array as da
import numpy as np
from dask import delayed
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from spatialdata._docs import docstring_parameter
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from spatialdata.transformations import Identity
from tifffile import memmap as tiffmmemap
from xarray import DataArray

VALID_IMAGE_TYPES = [".tif", ".tiff"]
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
    """
    Read a generic shapes or image file and save it as SpatialData zarr.

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
    """Reads a GeoJSON file and returns a parsed GeoDataFrame spatial element"""
    return ShapesModel.parse(input, transformations={coordinate_system: Identity()})


def _compute_chunk_sizes_positions(size: int, chunk: int, min_coord: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Calculate chunk sizes and positions for a given dimension and chunk size"""
    # All chunks have the same size except for the last one
    positions = np.arange(min_coord, size, chunk)
    lengths = np.full_like(positions, chunk, dtype=int)

    if positions[-1] + chunk > size:
        lengths[-1] = size - positions[-1]

    return positions, lengths


def _compute_chunks(
    dimensions: tuple[int, int],
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int] = (0, 0),
) -> NDArray[np.int_]:
    """Create all chunk specs for a given image and chunk size.

    Creates specifications (x, y, width, height) with (x, y) being the upper left corner
    of chunks of size chunk_size. Chunks at the edges correspond to the remainder of
    chunk size and dimensions

    Parameters
    ----------
    dimensions : tuple[int, int]
        Size of the image in (width, height).
    chunk_size : tuple[int, int]
        Size of individual tiles in (width, height).
    min_coordinates : tuple[int, int], optional
        Minimum coordinates (x, y) in the image, defaults to (0, 0).

    Returns
    -------
    np.ndarray
        Array of shape (n_tiles_x, n_tiles_y, 4). Each entry defines a tile
        as (x, y, width, height).
    """
    x_positions, widths = _compute_chunk_sizes_positions(dimensions[1], chunk_size[1], min_coord=min_coordinates[1])
    y_positions, heights = _compute_chunk_sizes_positions(dimensions[0], chunk_size[0], min_coord=min_coordinates[0])

    # Generate the tiles
    tiles = np.array(
        [
            [[x, y, w, h] for x, w in zip(x_positions, widths, strict=True)]
            for y, h in zip(y_positions, heights, strict=True)
        ],
        dtype=int,
    )
    return tiles


def _read_chunks(
    func: Callable[..., NDArray[np.int_]],
    slide: Any,
    coords: NDArray[np.int_],
    n_channel: int,
    dtype: np.number,
    **func_kwargs: Any,
) -> list[list[da.array]]:
    """Abstract method to tile a large microscopy image.

    Parameters
    ----------
    func
        Function to retrieve a rectangular tile from the slide image. Must take the
        arguments:

            - slide Full slide image
            - x0: x (col) coordinate of upper left corner of chunk
            - y0: y (row) coordinate of upper left corner of chunk
            - width: Width of chunk
            - height: Height of chunk

        and should return the chunk as numpy array of shape (c, y, x)
    slide
        Slide image in lazyly loaded format compatible with func
    coords
        Coordinates of the upper left corner of the image in format (n_row_x, n_row_y, 4)
        where the last dimension defines the rectangular tile in format (x, y, width, height).
        n_row_x represents the number of chunks in x dimension and n_row_y the number of chunks
        in y dimension.
    n_channel
        Number of channels in array
    dtype
        Data type of image
    func_kwargs
        Additional keyword arguments passed to func

    Returns
    -------
    list[list[da.array]]
        List (length: n_row_x) of lists (length: n_row_y) of chunks.
        Represents all chunks of the full image.
    """
    func_kwargs = func_kwargs if func_kwargs else {}

    # Collect each delayed chunk as item in list of list
    # Inner list becomes dim=-1 (cols/x)
    # Outer list becomes dim=-2 (rows/y)
    # see dask.array.block
    chunks = [
        [
            da.from_delayed(
                delayed(func)(
                    slide,
                    x0=coords[y, x, 0],
                    y0=coords[y, x, 1],
                    width=coords[y, x, 2],
                    height=coords[y, x, 3],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, *coords[y, x, [3, 2]]),
            )
            for x in range(coords.shape[1])
        ]
        for y in range(coords.shape[0])
    ]
    return chunks


def _tiff_to_chunks(input: Path, axes_dim_mapping: dict[str, int]) -> list[list[DaskArray[np.int_]]]:
    """Chunkwise reader for tiff files.

    Parameters
    ----------
    input
        Path to image

    Returns
    -------
    list[list[DaskArray]]
    """
    # Lazy file reader
    slide = tiffmmemap(input)

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


def image(input: Path, data_axes: Sequence[str], coordinate_system: str) -> DataArray:
    """Reads an image file and returns a parsed Image2DModel"""
    # Map passed data axes to position of dimension
    axes_dim_mapping = {axes: ndim for ndim, axes in enumerate(data_axes)}

    if input.suffix in [".tiff", ".tif"]:
        chunks = _tiff_to_chunks(input, axes_dim_mapping=axes_dim_mapping)
    else:
        raise NotImplementedError(f"File format {input.suffix} not implemented")

    img = da.block(chunks, allow_unknown_chunksizes=True)

    return Image2DModel.parse(img, dims=data_axes, transformations={coordinate_system: Identity()})
