from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray


def _compute_chunk_sizes_positions(size: int, chunk: int, min_coord: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Calculate chunk sizes and positions for a given dimension and chunk size."""
    # All chunks have the same size except for the last one
    positions = np.arange(min_coord, min_coord + size, chunk)
    lengths = np.minimum(chunk, min_coord + size - positions)

    return positions, lengths


def _compute_chunks(
    shape: tuple[int, int],
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int] = (0, 0),
) -> NDArray[np.int_]:
    """Create all chunk specs for a given image and chunk size.

    Creates specifications (x, y, width, height) with (x, y) being the upper left corner
    of chunks of size chunk_size. Chunks at the edges correspond to the remainder of
    chunk size and dimensions

    Parameters
    ----------
    shape : tuple[int, int]
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
    # TODO: check x -> 1 and y -> 0?
    x_positions, widths = _compute_chunk_sizes_positions(shape[1], chunk_size[1], min_coord=min_coordinates[1])
    y_positions, heights = _compute_chunk_sizes_positions(shape[0], chunk_size[0], min_coord=min_coordinates[0])

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
    # TODO: expand type hints for ...
    # TODO: really only np.int_? Not float? Consider using np.number
    func: Callable[..., NDArray[np.int_]],
    slide: Any,
    coords: NDArray[np.int_],
    n_channel: int,
    dtype: np.dtype[Any],
    **func_kwargs: Any,
) -> list[list[da.Array]]:
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
        Slide image in lazily loaded format compatible with func
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
    # TODO: check, wasn't it x, y and not y, x?
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
                # TODO: double check the [3, 2] with debugger
                shape=(n_channel, *coords[y, x, [3, 2]]),
            )
            # TODO: seems inconsistent with coords docstring
            for x in range(coords.shape[1])
        ]
        # TODO: seems inconsistent with coords docstring
        for y in range(coords.shape[0])
    ]
    return chunks


# TODO: do a asv debugging for peak mem
