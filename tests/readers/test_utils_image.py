import numpy as np
import pytest
from numpy.typing import NDArray

from spatialdata_io.readers._utils._image import (
    DEFAULT_CHUNK_SIZE,
    Chunks_t,
    _compute_chunk_sizes_positions,
    _compute_chunks,
    normalize_chunks,
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


@pytest.mark.parametrize(
    "chunks, axes, expected",
    [
        # 2D (y, x)
        (None, ("y", "x"), {"y": DEFAULT_CHUNK_SIZE, "x": DEFAULT_CHUNK_SIZE}),
        (256, ("y", "x"), {"y": 256, "x": 256}),
        ((200, 100), ("x", "y"), {"y": 100, "x": 200}),
        ({"y": 300, "x": 400}, ("x", "y"), {"y": 300, "x": 400}),
        # 2D with channel (c, y, x)
        (None, ("c", "y", "x"), {"c": DEFAULT_CHUNK_SIZE, "y": DEFAULT_CHUNK_SIZE, "x": DEFAULT_CHUNK_SIZE}),
        (256, ("c", "y", "x"), {"c": 256, "y": 256, "x": 256}),
        ((1, 100, 200), ("c", "y", "x"), {"c": 1, "y": 100, "x": 200}),
        ({"c": 1, "y": 300, "x": 400}, ("c", "y", "x"), {"c": 1, "y": 300, "x": 400}),
        # 3D (z, y, x)
        ((10, 100, 200), ("z", "y", "x"), {"z": 10, "y": 100, "x": 200}),
        ({"z": 10, "y": 300, "x": 400}, ("z", "y", "x"), {"z": 10, "y": 300, "x": 400}),
    ],
)
def test_normalize_chunks_valid(chunks: Chunks_t, axes: tuple[str, ...], expected: dict[str, int]) -> None:
    assert normalize_chunks(chunks, axes=axes) == expected


@pytest.mark.parametrize(
    "chunks, axes, match",
    [
        ({"y": 100}, ("y", "x"), "missing keys for axes"),
        ((1, 2, 3), ("y", "x"), "doesn't match axes"),
        ((1.5, 2), ("y", "x"), "must be int"),
        ("invalid", ("y", "x"), "Unsupported chunks type"),
    ],
)
def test_normalize_chunks_errors(chunks: Chunks_t, axes: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_chunks(chunks, axes=axes)
