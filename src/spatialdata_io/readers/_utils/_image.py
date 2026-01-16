from collections.abc import Callable, Mapping, Sequence
from typing import Any

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray
from spatialdata.models.models import Chunks_t

__all__ = ["Chunks_t", "_compute_chunks", "_read_chunks", "normalize_chunks"]

_Y_IDX = 0
"""Index of y coordinate in in chunk coordinate array format: (y, x, height, width)"""

_X_IDX = 1
"""Index of x coordinate in chunk coordinate array format: (y, x, height, width)"""

_HEIGHT_IDX = 2
"""Index of height specification in chunk coordinate array format: (y, x, height, width)"""

_WIDTH_IDX = 3
"""Index of width specification in chunk coordinate array format: (y, x, height, width)"""

DEFAULT_CHUNK_SIZE = 1000


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

    Creates chunk specifications for tiling an image. Returns an array where position [i, j]
    contains the spec for the chunk at block row i, block column j.
    Each chunk specification consists of (y, x, height, width) with (y, x) being the upper left
    corner of chunks of size chunk_size.
    Chunks at the edges correspond to the remainder of chunk size and dimensions

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

            - slide: Full slide image
            - y0: y (row) coordinate of upper left corner of chunk
            - x0: x (col) coordinate of upper left corner of chunk
            - height: Height of chunk
            - width: Width of chunk

        and should return the chunk as numpy array of shape (c, y, x)
    slide
        Slide image in lazily loaded format compatible with `func`
    coords
        Coordinates of the upper left corner of each chunk image in format (n_row_y, n_row_x, 4)
        where the last dimension defines the rectangular tile in format (y, x, height, width), as returned
        by :func:`_compute_chunks`.
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
    """
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
                    y0=coords[chunk_y, chunk_x, _Y_IDX],
                    x0=coords[chunk_y, chunk_x, _X_IDX],
                    height=coords[chunk_y, chunk_x, _HEIGHT_IDX],
                    width=coords[chunk_y, chunk_x, _WIDTH_IDX],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, coords[chunk_y, chunk_x, _HEIGHT_IDX], coords[chunk_y, chunk_x, _WIDTH_IDX]),
            )
            for chunk_x in range(coords.shape[1])
        ]
        for chunk_y in range(coords.shape[0])
    ]
    return chunks


def normalize_chunks(
    chunks: Chunks_t | None,
    axes: Sequence[str],
) -> dict[str, int]:
    """Normalize chunk specification to dict format.

    This function converts various chunk formats to a dict mapping dimension names
    to chunk sizes. The dict format is preferred because it's explicit about which
    dimension gets which chunk size and is compatible with spatialdata.

    Parameters
    ----------
    chunks
        Chunk specification. Can be:
        - None: Uses DEFAULT_CHUNK_SIZE for all axes
        - int: Applied to all axes
        - tuple[int, ...]: Chunk sizes in order corresponding to axes
        - dict: Mapping of axis names to chunk sizes (validated against axes)
    axes
        Tuple of axis names that defines the expected dimensions (e.g., ('c', 'y', 'x')).

    Returns
    -------
    dict[str, int]
        Dict mapping axis names to chunk sizes.

    Raises
    ------
    ValueError
        If chunks format is not supported or incompatible with axes.
    """
    if chunks is None:
        return dict.fromkeys(axes, DEFAULT_CHUNK_SIZE)

    if isinstance(chunks, int):
        return dict.fromkeys(axes, chunks)

    if isinstance(chunks, Mapping):
        chunks_dict = dict(chunks)
        missing = set(axes) - set(chunks_dict.keys())
        if missing:
            raise ValueError(f"chunks dict missing keys for axes {missing}, got: {list(chunks_dict.keys())}")
        return {ax: chunks_dict[ax] for ax in axes}

    if isinstance(chunks, tuple):
        if len(chunks) != len(axes):
            raise ValueError(f"chunks tuple length {len(chunks)} doesn't match axes {axes} (length {len(axes)})")
        if not all(isinstance(c, int) for c in chunks):
            raise ValueError(f"All elements in chunks tuple must be int, got: {chunks}")
        return dict(zip(axes, chunks, strict=True))

    raise ValueError(f"Unsupported chunks type: {type(chunks)}. Expected int, tuple, dict, or None.")


##
