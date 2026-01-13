import numpy as np
import pytest
from numpy.typing import NDArray

from spatialdata_io.readers._utils._image import (
    _compute_chunk_sizes_positions,
    _compute_chunks,
)


@pytest.mark.parametrize(
    ("size", "chunk", "min_coordinate", "positions", "lengths"),
    [
        (300, 100, 0, np.array([0, 100, 200]), np.array([100, 100, 100])),
        (300, 200, 0, np.array([0, 200]), np.array([200, 100])),
        # TODO: why negative coordinates if used only for 0, 0?
        (300, 100, -100, np.array([-100, 0, 100]), np.array([100, 100, 100])),
        (300, 200, -100, np.array([-100, 100]), np.array([200, 100])),
    ],
)
def test_compute_chunk_sizes_positions(
    size: int,
    chunk: int,
    min_coordinate: int,
    positions: NDArray[np.number],
    lengths: NDArray[np.number],
) -> None:
    computed_positions, computed_lengths = _compute_chunk_sizes_positions(size, chunk, min_coordinate)
    assert (positions == computed_positions).all()
    assert (lengths == computed_lengths).all()


@pytest.mark.parametrize(
    ("dimensions", "chunk_size", "min_coordinates", "result"),
    [
        # Regular grid 2x2
        (
            (2, 2),
            (1, 1),
            (0, 0),
            np.array([[[0, 0, 1, 1], [1, 0, 1, 1]], [[0, 1, 1, 1], [1, 1, 1, 1]]]),
        ),
        # Different tile sizes
        (
            (3, 3),
            (2, 2),
            (0, 0),
            np.array([[[0, 0, 2, 2], [2, 0, 1, 2]], [[0, 2, 2, 1], [2, 2, 1, 1]]]),
        ),
        (
            (2, 2),
            (1, 1),
            (-1, 0),
            np.array([[[0, -1, 1, 1], [1, -1, 1, 1]], [[0, 0, 1, 1], [1, 0, 1, 1]]]),
        ),
    ],
)
def test_compute_chunks(
    dimensions: tuple[int, int],
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int],
    result: NDArray[np.number],
) -> None:
    tiles = _compute_chunks(dimensions, chunk_size, min_coordinates)

    assert (tiles == result).all()
