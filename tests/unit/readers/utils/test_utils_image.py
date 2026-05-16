import numpy as np
import pytest
from numpy.typing import NDArray

from spatialdata_io.readers._utils._image import (
    _compute_chunk_sizes_positions,
    _compute_chunks,
)


@pytest.mark.parametrize(
    ("size", "chunk", "expected_positions", "expected_lengths"),
    [
        (300, 100, np.array([0, 100, 200]), np.array([100, 100, 100])),
        (300, 200, np.array([0, 200]), np.array([200, 100])),
    ],
)
def test_compute_chunk_sizes_positions(
    size: int,
    chunk: int,
    expected_positions: NDArray[np.number],
    expected_lengths: NDArray[np.number],
) -> None:
    computed_positions, computed_lengths = _compute_chunk_sizes_positions(size, chunk)
    assert (expected_positions == computed_positions).all()
    assert (expected_lengths == computed_lengths).all()


@pytest.mark.parametrize(
    ("dimensions", "chunk_size", "result"),
    [
        # Regular grid 2x2
        (
            (2, 2),
            (1, 1),
            np.array(
                [
                    [[0, 0, 1, 1], [0, 1, 1, 1]],
                    [[1, 0, 1, 1], [1, 1, 1, 1]],
                ]
            ),
        ),
        # Different tile sizes
        (
            (300, 300),
            (100, 200),
            np.array(
                [
                    [[0, 0, 100, 200], [0, 200, 100, 100]],
                    [[100, 0, 100, 200], [100, 200, 100, 100]],
                    [[200, 0, 100, 200], [200, 200, 100, 100]],
                ]
            ),
        ),
    ],
)
def test_compute_chunks(
    dimensions: tuple[int, int],
    chunk_size: tuple[int, int],
    result: NDArray[np.number],
) -> None:
    tiles = _compute_chunks(dimensions, chunk_size)

    assert (tiles == result).all()
