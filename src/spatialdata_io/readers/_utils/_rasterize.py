"""Lazy, chunk-wise rasterisation of cell polygons into a labels array."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import shapely
from dask.base import tokenize
from skimage.draw import polygon as draw_polygon

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["rasterize_polygons_lazy"]

_MAX_ID = np.iinfo(np.uint32).max


def rasterize_polygons_lazy(
    polygons: Sequence[shapely.Polygon | shapely.MultiPolygon],
    ids: Sequence[int],
    shape: tuple[int, int],
    chunks: tuple[int, int] = (4096, 4096),
    scale: float = 1.0,
) -> da.Array:
    """Rasterise polygons into a lazy ``uint32`` labels array, one chunk at a time.

    A pixel ``(row, col)`` gets the id of a polygon that contains its centre ``(col + 0.5, row + 0.5)``: pixel
    ``(0, 0)`` covers ``[0, 1) x [0, 1)``, as in QuPath. Holes are not part of a polygon. Where polygons overlap,
    the one with the higher id wins, independently of their order. Only a spatial index of the polygons is built
    here; each chunk draws just the polygons that intersect it when it is computed.

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
    # drawing in ascending id order lets the higher id win overlaps; the stable sort keeps ties in input order
    order = np.argsort(labels, kind="stable")
    geometries = np.asarray(polygons, dtype=object)[order]
    if scale != 1.0:
        geometries = shapely.transform(geometries, lambda xy: xy * scale)
    labels = labels[order].astype(np.uint32)
    template = da.empty(shape, chunks=chunks, dtype=np.uint32)
    # dask would otherwise tokenize the geometries one by one; their vectorised WKB is much faster to hash
    wkb = np.frombuffer(b"".join(shapely.to_wkb(geometries)), dtype=np.uint8)
    token = tokenize(wkb, labels, template.shape, template.chunks)
    return da.map_blocks(
        _draw_block,
        template,
        dtype=np.uint32,
        name=f"rasterize-polygons-{token}",
        tree=shapely.STRtree(geometries),
        labels=labels,
        meta=np.empty((0, 0), dtype=np.uint32),
    )


def _draw_block(
    block: np.ndarray, tree: shapely.STRtree, labels: np.ndarray, block_info: dict[Any, Any] | None = None
) -> np.ndarray:
    (y0, y1), (x0, x1) = block_info[0]["array-location"]  # type: ignore[index]
    out = np.zeros(block.shape, dtype=np.uint32)
    # the tree indices follow the ascending ids
    for i in np.sort(tree.query(shapely.box(x0, y0, x1, y1))):
        for part in shapely.get_parts(tree.geometries[i]):
            if isinstance(part, shapely.Polygon) and not part.is_empty:
                _draw_polygon(out, part, labels[i], y0, x0)
    return out


def _draw_polygon(out: np.ndarray, polygon: shapely.Polygon, value: int, y0: int, x0: int) -> None:
    """Set the pixels of ``out`` (the block at ``(y0, x0)``) whose centres are inside ``polygon`` to ``value``."""
    minx, miny, maxx, maxy = polygon.bounds
    # the pixel window of the polygon in the block; pixel centres are at +0.5
    r0, c0 = max(int(np.floor(miny - 0.5)) - y0, 0), max(int(np.floor(minx - 0.5)) - x0, 0)
    r1, c1 = min(int(np.ceil(maxy - 0.5)) + 1 - y0, out.shape[0]), min(int(np.ceil(maxx - 0.5)) + 1 - x0, out.shape[1])
    if r0 >= r1 or c0 >= c1:
        return
    inside = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    # holes are cleared from this polygon only, so cells lying in a hole keep their pixels
    for ring, fill in [(polygon.exterior, True), *((ring, False) for ring in polygon.interiors)]:
        xy = shapely.get_coordinates(ring)
        inside[draw_polygon(xy[:, 1] - 0.5 - (y0 + r0), xy[:, 0] - 0.5 - (x0 + c0), shape=inside.shape)] = fill
    out[r0:r1, c0:c1][inside] = value
