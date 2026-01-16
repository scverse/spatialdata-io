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

    For each chunk in the final image with the chunk coords (block row=chunk_y_position, block col=chunk_x_position), this function
    creates specifications (y, x, height, width) with (y, x) being the upper left corner
    of chunks of size chunk_size. Chunks at the edges correspond to the remainder of
    chunk size and dimensions

    Parameters
    ----------
    shape : tuple[int, int]
        Size of the image in (image height, image width).
    chunk_size : tuple[int, int]
        Size of individual tiles in (height, width).

    Returns
    -------
    np.ndarray
        Array of shape (n_tiles_y, n_tiles_x, 4). Each entry defines a tile
        as (y, x, height, width).
    """
    y_positions, heights = _compute_chunk_sizes_positions(shape[0], chunk_size[0])
    x_positions, widths = _compute_chunk_sizes_positions(shape[1], chunk_size[1])

    # Generate the tiles
    # Each entry defines the chunk dimensions for a tile
    # The order of the chunk definitions (chunk_index_y=outer, chunk_index_x=inner) follows the dask.block convention
    tiles = np.array(
        [
            [[y, x, h, w] for x, w in zip(x_positions, widths, strict=True)]
            for y, h in zip(y_positions, heights, strict=True)
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
        Function to retrieve a single rectangular tile from the slide image. Must take the
        arguments:

            - slide Full slide image
            - y0: y (row) coordinate of upper left corner of chunk
            - x0: x (col) coordinate of upper left corner of chunk
            - height: Height of chunk
            - width: Width of chunk

        and should return the chunk as numpy array of shape (c, y, x)
    slide
        Slide image in lazily loaded format compatible with `func`
    coords
        Coordinates of the upper left corner of the image in format (n_row_y, n_row_x, 4)
        where the last dimension defines the rectangular tile in format (y, x, height, width).
        n_row_y represents the number of chunks in y dimension (block rows) and n_row_x the number of chunks
        in x dimension (block columns).
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
    As shown in `_compute_chunks()`, `coords` are in the form `(x, y,
    width, height)`. In that function, the inner list (dim = -1) iterates over `y`
    values, and the outer list (dim = -2) iterates over `x` values. In `_read_chunks(
    )`, we use the more common `(y, x)` ordering: the inner list (dim = -1) iterates
    over `x` values, and the outer list (dim = -2) iterates over `y` values.

    This mismatch can be confusing. A straightforward fix that one could perform would
    be to standardize `coords` to `(y, x, height, width)` instead of
    `(x, y, width, height)`.
    """
    # TODO: standardize `coords` as explained in the docstring above, then remove that
    # part from the docstring
    func_kwargs = func_kwargs if func_kwargs else {}

    # Collect each delayed chunk (c, y, x) as item in list of list
    # Inner list becomes dim=-1 (chunk columns/x)
    # Outer list becomes dim=-2 (chunk rows/y)
    # see dask.array.block
    chunks = [
        [
            da.from_delayed(
                delayed(func)(
                    slide,
                    x0=coords[chunk_y, chunk_x, 1],
                    y0=coords[chunk_y, chunk_x, 0],
                    width=coords[chunk_y, chunk_x, 3],
                    height=coords[chunk_y, chunk_x, 2],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, *coords[chunk_y, chunk_x, [3, 2]]),
            )
            for chunk_x in range(coords.shape[1])
        ]
        for chunk_y in range(coords.shape[0])
    ]
    return chunks


# TODO: do a asv debugging for peak mem
