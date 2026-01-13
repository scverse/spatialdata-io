from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray


def _compute_chunk_sizes_positions(size: int, chunk: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Calculate chunk sizes and positions for a given dimension and chunk size."""
    # All chunks have the same size except for the last one
    positions = np.arange(0, size, chunk)
    lengths = np.minimum(chunk, size - positions)

    return positions, lengths


def _compute_chunks(
    shape: tuple[int, int],
    chunk_size: tuple[int, int],
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

    Returns
    -------
    np.ndarray
        Array of shape (n_tiles_x, n_tiles_y, 4). Each entry defines a tile
        as (x, y, width, height).
    """
    x_positions, widths = _compute_chunk_sizes_positions(shape[0], chunk_size[0])
    y_positions, heights = _compute_chunk_sizes_positions(shape[1], chunk_size[1])

    # Generate the tiles
    tiles = np.array(
        [
            [[x, y, w, h] for y, h in zip(y_positions, heights, strict=True)]
            for x, w in zip(x_positions, widths, strict=True)
        ],
        dtype=int,
    )
    return tiles


def _read_chunks(
    func: Callable[..., NDArray[np.number]],
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
        (Outer) list (length: n_row_y) of (inner) lists (length: n_row_x) of chunks with axes
        (c, y, x). Represents all chunks of the full image.

    Notes
    -------
    As seen in _compute_chunks(), since coords are in format (x, y, width, height), the
    inner list there (dim=-1) runs over the y values and the outer list (dim=-2) runs
    over the x values. In _read_chunks() we have the more common (y, x) format, where
    the inner list (dim=-1) runs over the x values and the outer list (dim=-2) runs over
    the y values.

    The above can be confusing, and a way to address this is to define coords to be
    in format (y, x, height, width) instead of (x, y, width, height).
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
                    x0=coords[x, y, 0],
                    y0=coords[x, y, 1],
                    width=coords[x, y, 2],
                    height=coords[x, y, 3],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, *coords[y, x, [3, 2]]),
            )
            for x in range(coords.shape[0])
        ]
        for y in range(coords.shape[1])
    ]
    return chunks


# TODO: do a asv debugging for peak mem
