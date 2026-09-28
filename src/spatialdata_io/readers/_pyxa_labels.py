"""3D cell labels for the Pyxa reader, rasterized lazily from the segmentation polygons.

The polygons (one per cell per z-plane, in pixel units) are drawn onto the mosaic image's voxel grid,
so labels and image overlay voxel for voxel at every pyramid level. Ring decoding is eager; drawing is
one ``dask.delayed`` task per tile and runs only when the labels are computed or written.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from spatialdata._logging import logger
from spatialdata.transformations import Scale, Sequence, Translation

from spatialdata_io._constants._constants import PyxaKeys

# PIL draws labels into signed 32-bit ("I" mode) images
_MAX_LABEL = 2**31 - 1


@dataclass(frozen=True)
class _MosaicGrid:
    """The mosaic's voxel grid: every level's (z, y, x) shape and level 0's frame in micrometers."""

    shapes: tuple[tuple[int, int, int], ...]
    scale: tuple[float, float, float]
    translation: tuple[float, float, float]

    @property
    def transformation(self) -> Sequence:
        axes = ("z", "y", "x")
        return Sequence([Scale(list(self.scale), axes=axes), Translation(list(self.translation), axes=axes)])

    def step(self, level: int) -> tuple[int, int, int]:
        """Level-0 voxels per voxel of ``level``, per axis (nearest-neighbour stride)."""
        n0, n = self.shapes[0], self.shapes[level]
        return tuple(max(1, round(a / b)) for a, b in zip(n0, n, strict=True))  # type: ignore[return-value]


def _label_ids(cell_ids: pd.Index) -> tuple[np.ndarray, str]:
    """Integer label per cell, and the rule used.

    The trailing integer of each ``cell_id`` (after any prefix, e.g. ``Region_17`` -> 17) when every
    cell has one and they are unique, > 0 (0 is background) and < 2^31; otherwise 1..n in table order.
    """
    n = len(cell_ids)
    digits = pd.Series(np.asarray(cell_ids, dtype=str)).str.extract(r"(\d+)$", expand=False)
    if digits.isna().any():
        why = "no trailing integer in some cell_id"
    elif (digits.str.len() > 10).any() or (numbers := digits.astype(np.int64)).max() > _MAX_LABEL:
        why = "a trailing integer is not below 2^31"
    elif numbers.min() < 1:
        why = "a trailing integer is 0 (the background)"
    elif not numbers.is_unique:
        why = "trailing integers are not unique"
    else:
        return numbers.to_numpy(dtype=np.uint32), "trailing integer of cell_id"
    return np.arange(1, n + 1, dtype=np.uint32), f"1..n in table order ({why})"


@dataclass(frozen=True)
class _Rings:
    """Polygon exterior rings in level-0 voxel index space (voxel centres at integer coordinates)."""

    label: np.ndarray
    plane: np.ndarray
    length: np.ndarray
    coords: np.ndarray
    bounds: np.ndarray

    def __len__(self) -> int:
        return len(self.label)


def _rings_from_row_group(
    path: Path,
    row_group: int,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    simplify: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """One parquet row group's rings on the grid, and how many polygon parts were dropped and why."""
    sz, sy, sx = grid.scale
    tz, ty, tx = grid.translation
    nz = grid.shapes[0][0]
    table = pq.ParquetFile(path).read_row_group(
        row_group, columns=[PyxaKeys.CELL_ID.value, PyxaKeys.Z_INDEX.value, "geometry"]
    )
    cell_id = pc.cast(table.column(PyxaKeys.CELL_ID.value), "string").to_numpy(zero_copy_only=False)
    row = labels.index.get_indexer(cell_id)
    zindex = table.column(PyxaKeys.Z_INDEX.value).to_numpy()
    geoms = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    parts, part_of = shapely.get_parts(geoms, return_index=True)
    rings = shapely.get_exterior_ring(parts)
    rings = shapely.transform(
        rings,
        lambda c: np.column_stack(((c[:, 0] * xy_size - tx) / sx, (c[:, 1] * xy_size - ty) / sy)),
    )
    rings = shapely.simplify(rings, simplify)
    # plane k holds Z_um = (ZIndex + 0.5) * z_size, the reader's plane centre
    plane = np.rint(((zindex[part_of] + 0.5) * z_size - tz) / sz).astype(np.int32)
    in_table = row[part_of] >= 0
    on_grid = (plane >= 0) & (plane < nz)
    empty = shapely.is_empty(rings) | (shapely.get_num_coordinates(rings) < 3)
    keep = in_table & on_grid & ~empty
    rings = rings[keep]
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    out = {
        "label": labels.to_numpy(dtype=np.uint32)[row[part_of][keep]],
        "plane": plane[keep],
        "length": np.bincount(ring_of, minlength=len(rings)).astype(np.int64),
        "coords": coords.astype(np.float32),
        "bounds": shapely.bounds(rings).astype(np.float32).reshape(-1, 4),
    }
    dropped = {
        "not in the table": int((~in_table).sum()),
        "off the mosaic's z range": int((in_table & ~on_grid).sum()),
        "empty": int((in_table & on_grid & empty).sum()),
    }
    return out, dropped


def _read_rings(
    path: Path,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    *,
    simplify: float = 0.25,
) -> _Rings:
    """Every segmentation polygon's exterior ring on the mosaic's level-0 grid, with its label and plane.

    ``labels`` maps ``cell_id`` to the cell's label. Row groups are decoded in a thread pool (pyarrow
    and shapely release the GIL). Rings are simplified to ``simplify`` voxels: the polygons are traced
    on a finer pixel grid than the mosaic's, and the staircase vertices add nothing at its resolution.
    Holes are ignored (exteriors are filled).
    """
    n_groups = pq.ParquetFile(path).metadata.num_row_groups
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda i: _rings_from_row_group(path, i, labels, grid, xy_size, z_size, simplify), range(n_groups)
            )
        )
    parts = [r[0] for r in results]
    dropped: dict[str, int] = {}
    for _, d in results:
        for why, count in d.items():
            dropped[why] = dropped.get(why, 0) + count
    if parts:
        rings = _Rings(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    else:
        # no row groups at all (e.g. a crop with no cells): nothing to concatenate over
        rings = _Rings(
            label=np.empty(0, dtype=np.uint32),
            plane=np.empty(0, dtype=np.int32),
            length=np.empty(0, dtype=np.int64),
            coords=np.empty((0, 2), dtype=np.float32),
            bounds=np.empty((0, 4), dtype=np.float32),
        )
    summary = ", ".join(f"{count} {why}" for why, count in dropped.items() if count)
    logger.info(
        f"{path.name}: {len(rings)} polygon rings on the mosaic grid" + (f"; dropped {summary}" if summary else "")
    )
    return rings


# one drawing task: a whole number of storage chunks, so tiles never share a chunk
TILE = (32, 1024, 1024)
# storage chunks, as the mosaic's: an inspect window reads only the chunks it covers
CHUNKS = (32, 256, 256)


@dataclass(frozen=True)
class _Tile:
    """The rings to draw into one tile of one pyramid level."""

    origin: tuple[int, int, int]
    shape: tuple[int, int, int]
    step: tuple[int, int]
    label: np.ndarray
    plane: np.ndarray
    offsets: np.ndarray
    coords: np.ndarray


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Indices of the concatenated slices ``[s, s + n)`` for each (s, n), in order."""
    new_starts = np.cumsum(lengths) - lengths
    return np.arange(int(lengths.sum())) - np.repeat(new_starts, lengths) + np.repeat(starts, lengths)


def _plan_tiles(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    tile: tuple[int, int, int] = TILE,
) -> list[_Tile]:
    """Group the rings of one level (``shape``, ``step`` from level 0) by tile.

    The level keeps the planes a nearest-neighbour stride of level 0 keeps; a ring crossing tiles goes
    to each. Within a tile, rings are ordered by label, so where they overlap the higher label wins.
    """
    nz, ny, nx = shape
    dz, dy, dx = step
    tz, ty, tx = tile
    keep = np.flatnonzero(rings.plane % dz == 0)
    plane = rings.plane[keep] // dz
    on = plane < nz
    keep, plane = keep[on], plane[on]
    # bounds in level-voxel index space; a voxel v is drawn when its centre lies in the ring
    bounds = rings.bounds[keep] / np.array([dx, dy, dx, dy], dtype=np.float32)
    n_ty, n_tx = -(-ny // ty), -(-nx // tx)
    y_lo = np.clip(np.floor(bounds[:, 1] / ty), 0, n_ty - 1).astype(np.int64)
    y_hi = np.clip(np.floor(bounds[:, 3] / ty), 0, n_ty - 1).astype(np.int64)
    x_lo = np.clip(np.floor(bounds[:, 0] / tx), 0, n_tx - 1).astype(np.int64)
    x_hi = np.clip(np.floor(bounds[:, 2] / tx), 0, n_tx - 1).astype(np.int64)
    idx, keys = [], []
    for oy in range(int((y_hi - y_lo).max(initial=0)) + 1):
        for ox in range(int((x_hi - x_lo).max(initial=0)) + 1):
            hit = np.flatnonzero((y_lo + oy <= y_hi) & (x_lo + ox <= x_hi))
            idx.append(hit)
            keys.append(((plane[hit] // tz) * n_ty + y_lo[hit] + oy) * n_tx + x_lo[hit] + ox)
    if not idx:
        return []
    idx_arr, keys_arr = np.concatenate(idx), np.concatenate(keys)
    if keys_arr.size == 0:
        # no ring falls in this level at all (e.g. an empty _Rings): nothing to draw, no tiles
        return []
    order = np.lexsort((rings.label[keep][idx_arr], keys_arr))
    idx_arr, keys_arr = idx_arr[order], keys_arr[order]

    ring = keep[idx_arr]
    starts = np.cumsum(rings.length) - rings.length
    lengths = rings.length[ring]
    coords = rings.coords[_ragged_gather(starts[ring], lengths)]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    tiles = []
    edges = np.flatnonzero(np.diff(keys_arr)) + 1
    for lo, hi in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(keys_arr)]]), strict=True):
        kz, rest = divmod(int(keys_arr[lo]), n_ty * n_tx)
        ky, kx = divmod(rest, n_tx)
        origin = (kz * tz, ky * ty, kx * tx)
        c0, c1 = int(offsets[lo]), int(offsets[hi])
        tiles.append(
            _Tile(
                origin=origin,
                shape=(min(tz, nz - origin[0]), min(ty, ny - origin[1]), min(tx, nx - origin[2])),
                step=(dy, dx),
                label=rings.label[ring[lo:hi]],
                plane=plane[idx_arr[lo:hi]] - origin[0],
                offsets=offsets[lo : hi + 1] - c0,
                coords=coords[c0:c1],
            )
        )
    return tiles


def _rasterize_tile(tile: _Tile) -> np.ndarray:
    """Fill the tile's rings plane by plane into a ``uint32`` block (0 = background)."""
    from PIL import Image, ImageDraw

    depth, height, width = tile.shape
    block = np.zeros(tile.shape, dtype=np.uint32)
    dy, dx = tile.step
    _, y0, x0 = tile.origin
    # level voxel index space, then PIL pixel space: pixel p's centre is at p + 0.5
    xy = np.column_stack((tile.coords[:, 0] / dx + 0.5 - x0, tile.coords[:, 1] / dy + 0.5 - y0))
    for plane in np.unique(tile.plane):
        image = Image.new("I", (width, height))
        draw = ImageDraw.Draw(image)
        for r in np.flatnonzero(tile.plane == plane):
            draw.polygon(xy[tile.offsets[r] : tile.offsets[r + 1]].ravel().tolist(), fill=int(tile.label[r]))
        block[plane] = np.asarray(image, dtype=np.int32)
    return block


def _labels_level(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    *,
    tile: tuple[int, int, int] = TILE,
    chunks: tuple[int, int, int] = CHUNKS,
) -> da.Array:
    """One pyramid level as a lazy array: a ``dask.delayed`` drawing task per tile, zeros where no ring falls."""
    by_origin = {t.origin: t for t in _plan_tiles(rings, shape, step, tile)}
    nz, ny, nx = shape
    tz, ty, tx = tile
    blocks = []
    for z in range(0, nz, tz):
        rows = []
        for y in range(0, ny, ty):
            row = []
            for x in range(0, nx, tx):
                size = (min(tz, nz - z), min(ty, ny - y), min(tx, nx - x))
                t = by_origin.get((z, y, x))
                if t is None:
                    row.append(da.zeros(size, dtype=np.uint32, chunks=size))
                else:
                    # looked up at call time, so the drawing function can be patched in tests
                    row.append(da.from_delayed(dask.delayed(_draw)(t), shape=size, dtype=np.uint32))
            rows.append(row)
        blocks.append(rows)
    return da.block(blocks).rechunk(tuple(min(c, s) for c, s in zip(chunks, shape, strict=True)))


def _draw(tile: _Tile) -> np.ndarray:
    return _rasterize_tile(tile)
