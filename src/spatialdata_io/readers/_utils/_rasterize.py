"""Lazy, chunk-wise rasterisation of cell polygons into a labels array."""

from __future__ import annotations

import hashlib
from itertools import product
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import shapely
from dask.array.core import normalize_chunks
from dask.base import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["rasterize_polygons_lazy"]

_MAX_ID = np.iinfo(np.uint32).max
# the polygon pixels tested at once, bounding the memory of a block's temporaries (~25 MB)
_BATCH_PIXELS = 1 << 18
_TOKEN_BATCH = 1 << 16


def rasterize_polygons_lazy(
    polygons: Sequence[shapely.Polygon | shapely.MultiPolygon],
    ids: Sequence[int],
    shape: tuple[int, int],
    chunks: tuple[int, int] = (4096, 4096),
    scale: float = 1.0,
) -> da.Array:
    """Rasterise polygons into a lazy ``uint32`` labels array, one chunk at a time.

    A pixel ``(row, col)`` gets the id of a polygon that contains its centre ``(col + 0.5, row + 0.5)`` or has it on
    its boundary: pixel ``(0, 0)`` covers ``[0, 1) x [0, 1)``, as in QuPath. Holes are not part of a polygon. Where
    polygons overlap or touch, the one with the higher id wins, independently of their order, so cells that share an
    edge leave no gap. Only the polygons that may
    intersect each chunk are found here; a chunk draws them when it is computed.

    Parameters
    ----------
    polygons
        The polygons, in pixel units after multiplying by ``scale``.
    ids
        The label of each polygon, between 1 and ``2**32 - 1``.
    shape
        The ``(y, x)`` shape of the labels.
    chunks
        The ``(y, x)`` chunk size.
    scale
        The number of pixels per polygon coordinate unit, e.g. ``1 / pixel_size_um`` for coordinates in µm.

    Returns
    -------
    The ``(y, x)`` labels, 0 for the background.
    """
    if len(polygons) != len(ids):
        raise ValueError(f"Got {len(polygons)} polygons but {len(ids)} ids.")
    labels = np.asarray(ids, dtype=np.int64)
    if len(labels) and (labels.min() < 1 or labels.max() > _MAX_ID):
        raise ValueError(f"Polygon ids must be between 1 and {_MAX_ID}; got {labels.min()} to {labels.max()}.")
    geometries = np.asarray(polygons, dtype=object)
    if scale != 1.0:
        geometries = shapely.transform(geometries, lambda xy: xy * scale)
    labels = labels.astype(np.uint32)
    sizes = normalize_chunks(chunks, shape, dtype=np.uint32)
    name = f"rasterize-polygons-{_token(geometries, labels, sizes)}"
    # every task holds just the polygons that may intersect its chunk, so the polygons are shipped once in all
    starts = [np.cumsum((0, *axis[:-1])) for axis in sizes]
    blocks = list(product(*(range(len(axis)) for axis in sizes)))
    boxes = [
        shapely.box(starts[1][j], starts[0][i], starts[1][j] + sizes[1][j], starts[0][i] + sizes[0][i])
        for i, j in blocks
    ]
    block_of, polygon = shapely.STRtree(geometries).query(boxes)
    counts = np.bincount(block_of, minlength=len(blocks))
    by_block = np.split(polygon[np.argsort(block_of, kind="stable")], np.cumsum(counts)[:-1])
    graph = {
        (name, i, j): (
            _draw_block,
            geometries[rows],
            labels[rows],
            (int(starts[0][i]), int(starts[1][j])),
            (sizes[0][i], sizes[1][j]),
        )
        for (i, j), rows in zip(blocks, by_block, strict=True)
    }
    return da.Array(graph, name, sizes, meta=np.empty((0, 0), dtype=np.uint32))


def _token(geometries: np.ndarray, labels: np.ndarray, *args: Any) -> str:
    """A deterministic token of the polygons, hashing their WKB in batches to bound the memory."""
    wkb = hashlib.blake2b(digest_size=16)
    for start in range(0, len(geometries), _TOKEN_BATCH):
        wkb.update(b"".join(shapely.to_wkb(geometries[start : start + _TOKEN_BATCH])))
    return tokenize(wkb.hexdigest(), labels, *args)


def _draw_block(
    geometries: np.ndarray, labels: np.ndarray, origin: tuple[int, int], shape: tuple[int, int]
) -> np.ndarray:
    """The labels of the block of ``shape`` at ``origin``, the highest id of the polygons covering each pixel centre."""
    out = np.zeros(shape, dtype=np.uint32)
    if not len(geometries):
        return out
    # private copies in block coordinates, prepared without racing other threads over the shared polygons
    geometries = shapely.transform(geometries, lambda xy: xy - origin[::-1])
    shapely.prepare(geometries)
    # the pixel window of each polygon in the block; pixel centres are at +0.5
    minx, miny, maxx, maxy = shapely.bounds(geometries).T
    r0, r1 = np.clip([np.floor(miny - 0.5), np.ceil(maxy - 0.5) + 1], 0, shape[0]).astype(np.int64)
    c0, c1 = np.clip([np.floor(minx - 0.5), np.ceil(maxx - 0.5) + 1], 0, shape[1]).astype(np.int64)
    width = c1 - c0
    sizes = np.maximum(r1 - r0, 0) * np.maximum(width, 0)
    firsts = np.cumsum(sizes) - sizes
    flat = out.reshape(-1)
    # all candidate pixels of a batch of polygons are tested in one GEOS call, which releases the GIL
    start = 0
    while start < len(geometries):
        stop = max(int(np.searchsorted(firsts, firsts[start] + _BATCH_PIXELS)), start + 1)
        owner = np.repeat(np.arange(start, stop), sizes[start:stop])
        offset = np.arange(len(owner)) - (firsts[owner] - firsts[start])
        rows = r0[owner] + offset // width[owner]
        cols = c0[owner] + offset % width[owner]
        inside = shapely.intersects_xy(geometries[owner], cols + 0.5, rows + 0.5)
        np.maximum.at(flat, rows[inside] * shape[1] + cols[inside], labels[owner[inside]])
        start = stop
    return out
