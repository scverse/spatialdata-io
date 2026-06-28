"""Stitching and rasterization helpers for CosMx multi-FOV data.

All functions in this module produce zarr-backed dask arrays using
``zarr.storage.LocalStore(tempfile.mkdtemp(...))`` so that the resulting
arrays are compatible with ``SpatialData.write()`` and do not duplicate
data in memory.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from skimage.draw import polygon
from spatialdata._logging import logger
from tqdm.auto import tqdm

from ._cosmx_io import (
    COSMX_FOV_SIZE_PX,
    _read_fov_image,
)
from ._cosmx_utils import FOV_DIR_RE, MORPH_FOV_RE, compute_with_limit

# ---------------------------------------------------------------------------
# Tile clipping helper
# ---------------------------------------------------------------------------


def _clip_tile_to_canvas(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    canvas_h: int,
    canvas_w: int,
) -> tuple[slice, slice, slice, slice] | None:
    """Clip a tile's placement to the canvas bounds.

    Returns ``(global_y, global_x, src_y, src_x)`` slices suitable for
    ``da.store(tile[src_y, src_x], zarr, regions=(global_y, global_x))``,
    or ``None`` if the tile falls entirely outside the canvas.
    """
    gy0, gx0 = max(0, y0), max(0, x0)
    gy1, gx1 = min(canvas_h, y1), min(canvas_w, x1)
    if gy0 >= gy1 or gx0 >= gx1:
        return None
    sy0, sx0 = gy0 - y0, gx0 - x0
    sy1 = sy0 + (gy1 - gy0)
    sx1 = sx0 + (gx1 - gx0)
    return slice(gy0, gy1), slice(gx0, gx1), slice(sy0, sy1), slice(sx0, sx1)


def _open_temp_zarr(prefix: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: Any) -> zarr.Array:
    """Allocate a fresh temp-backed zarr array for scratch stitching output."""
    store = zarr.storage.LocalStore(tempfile.mkdtemp(prefix=prefix))
    return zarr.create_array(store=store, shape=shape, chunks=chunks, dtype=dtype)


# ---------------------------------------------------------------------------
# FOV grid normalisation
# ---------------------------------------------------------------------------


def _snap_fov_grid(
    fov_locs: pd.DataFrame,
    fov_arrays: dict[int, da.Array] | dict[int, np.ndarray] | None = None,
    *,
    fov_size: float = COSMX_FOV_SIZE_PX,
    tol: int = 8,
) -> pd.DataFrame:
    """Normalize FOV boxes to a 0-based canvas using the CosMx spec size.

    CosMx local coordinates span 0 .. *fov_size* (default 4256) for both
    axes.  Polygon CSVs are written in that frame and the rasterizer uses
    ``fov_locs["xmax"]`` / ``fov_locs["ymax"]`` to build the canvas.
    Forcing every FOV to the same extent prevents a single
    cropped / malformed FOV from shrinking the canvas.

    A warning is emitted when the actual image size deviates from the spec
    by more than *tol* pixels.

    Parameters
    ----------
    fov_locs
        DataFrame indexed by FOV id with at least ``xmin`` and ``ymin``.
    fov_arrays
        Optional mapping of FOV id to the image array for that FOV
        (used only for sanity-checking dimensions).
    fov_size
        Expected tile width **and** height in pixels.
    tol
        Tolerance in pixels before a size-mismatch warning is raised.

    Returns
    -------
    pd.DataFrame
        A copy of *fov_locs* with added columns ``x0``, ``x1``, ``y0``,
        ``y1`` (canvas coordinates) and updated ``xmax`` / ``ymax``.
    """
    if fov_locs.empty:
        return fov_locs

    fov_locs = fov_locs.copy()

    x_min = float(fov_locs["xmin"].min())
    y_min = float(fov_locs["ymin"].min())

    wanted_w = int(round(fov_size))
    wanted_h = int(round(fov_size))

    for f in fov_locs.index:
        x0 = int(round(float(fov_locs.at[f, "xmin"]) - x_min))
        y0 = int(round(float(fov_locs.at[f, "ymin"]) - y_min))

        w = wanted_w
        h = wanted_h

        arr = fov_arrays.get(int(f)) if fov_arrays is not None else None
        if arr is not None:
            if arr.ndim == 3:
                _, ah, aw = arr.shape
            else:
                ah, aw = arr.shape
            if abs(aw - wanted_w) > tol or abs(ah - wanted_h) > tol:
                logger.warning(
                    "FOV %s has image %sx%s but CosMx expects %sx%s. Forcing spec size.",
                    f,
                    aw,
                    ah,
                    wanted_w,
                    wanted_h,
                )

        x1 = x0 + w
        y1 = y0 + h

        fov_locs.at[f, "x0"] = x0
        fov_locs.at[f, "x1"] = x1
        fov_locs.at[f, "y0"] = y0
        fov_locs.at[f, "y1"] = y1

        fov_locs.at[f, "xmax"] = float(fov_locs.at[f, "xmin"]) + w
        fov_locs.at[f, "ymax"] = float(fov_locs.at[f, "ymin"]) + h

    return fov_locs


def _label_tile_flip(fov_locs: pd.DataFrame, fov: int, flip_image: bool) -> bool:
    """Per-FOV label-tile flip: the inverse of ``flip_y``.

    Transcripts are placed in coordinate space via ``flip_y`` (see
    :func:`spatialdata_io.readers._cosmx_io.place_local_in_fov_grid`), so a label
    raster co-registers iff it is flipped ``not flip_y``.  When ``flip_y`` is
    unavailable (column absent, or FOV missing from the table) fall back to
    ``flip_image``.  Single source of truth for both label stitchers (#39 / #41).
    """
    if "flip_y" in fov_locs.columns and fov in fov_locs.index:
        return not bool(fov_locs.at[fov, "flip_y"])
    return flip_image


# ---------------------------------------------------------------------------
# Multi-FOV image stitcher
# ---------------------------------------------------------------------------


def _read_stitched_image(
    images_dir: Path,
    fov_locs: pd.DataFrame,
    protein_dir_dict: dict[int, Path],
    morphology_coords: list[str],
    flip_image: bool,
    *,
    fovs_filter: set[int] | None = None,
    selected_channels: list[str] | None = None,
    n_workers: int | None = None,
    tighten_to_seen: bool = False,
    **imread_kwargs: Any,
) -> tuple[da.Array, list[str], float, float] | tuple[None, list[str], float, float]:
    """Stitch per-FOV morphology (and optional protein) TIFFs into one image.

    Each FOV tile is rechunked to align with the zarr target, clipped to
    the canvas bounds, and written via ``da.store``.  The result is a
    dask array backed by a temporary ``zarr.storage.LocalStore``.

    Parameters
    ----------
    images_dir
        Directory containing per-FOV Morphology2D TIFFs.
    fov_locs
        FOV positions table (indexed by FOV id).
    protein_dir_dict
        Mapping of FOV id to the directory holding protein TIFFs for
        that FOV (may be empty).
    morphology_coords
        Channel names parsed from the TIFF metadata.
    flip_image
        Whether to flip each tile vertically before stitching.
    fovs_filter
        If given, only these FOV ids are stitched.
    selected_channels
        Restrict the output to these channel names.
    n_workers
        Parallelism cap passed to :func:`compute_with_limit`.
    tighten_to_seen
        Snap the canvas over only image-bearing FOVs, so phantom / image-less
        FOVs neither anchor the origin nor inflate it (issue #37).
    **imread_kwargs
        Forwarded to :func:`dask_image.imread.imread`.

    Returns
    -------
    tuple[da.Array, list[str], float, float]
        ``(stitched_image, channel_names, origin_x, origin_y)``; origin is the
        canvas min over image-bearing FOVs.
    """
    tif_re = MORPH_FOV_RE
    all_tifs = list(images_dir.glob("*.TIF"))

    fov_images: dict[int, da.Array] = {}
    fov_channels: dict[int, list[str]] = {}
    seen_fovs: list[int] = []

    for img_path in all_tifs:
        m = tif_re.match(img_path.name)
        if not m:
            continue
        fov = int(m.group(1))
        if fovs_filter is not None and fov not in fovs_filter:
            continue
        if fov not in fov_locs.index:
            logger.warning("Image for FOV %d ignored (no entry in positions table).", fov)
            continue

        img, c_names = _read_fov_image(
            img_path,
            protein_dir_dict.get(fov),
            morphology_coords,
            selected_channels=selected_channels,
            **imread_kwargs,
        )
        if flip_image:
            img = img[:, ::-1, :]

        fov_images[fov] = img
        fov_channels[fov] = c_names
        seen_fovs.append(fov)

        _, h, w = img.shape
        fov_locs.loc[fov, "xmax"] = float(fov_locs.loc[fov, "xmin"]) + w
        fov_locs.loc[fov, "ymax"] = float(fov_locs.loc[fov, "ymin"]) + h

    if not seen_fovs:
        logger.warning("No matching FOV images found to stitch — skipping images.")
        return None, [], 0.0, 0.0

    # Snap only image-bearing FOVs when tightening (issue #37).
    snap_locs = fov_locs.loc[sorted(seen_fovs)].copy() if tighten_to_seen else fov_locs
    fov_locs = _snap_fov_grid(snap_locs, fov_images)

    used_ox = float(fov_locs.loc[seen_fovs, "xmin"].min())
    used_oy = float(fov_locs.loc[seen_fovs, "ymin"].min())

    H = int(fov_locs.loc[seen_fovs, "y1"].max())
    W = int(fov_locs.loc[seen_fovs, "x1"].max())

    if selected_channels is not None:
        all_channels = [ch for ch in selected_channels if any(ch in v for v in fov_channels.values())]
        if not all_channels:
            raise ValueError(f"None of the requested channels are present in the selected FOVs: {selected_channels}.")
    else:
        all_channels = sorted(set().union(*(set(v) for v in fov_channels.values())))

    channel_to_idx = {ch: i for i, ch in enumerate(all_channels)}
    sample_dtype = fov_images[seen_fovs[0]].dtype

    cy = min(1024, H)
    cx = min(1024, W)

    z_img = _open_temp_zarr("cosmx_stitch_", (len(all_channels), H, W), (1, cy, cx), sample_dtype)

    store_ops: list[dask.delayed] = []
    lock = SerializableLock()  # shared across tiles: serialises writes to shared zarr chunks

    for fov in tqdm(seen_fovs, desc="Reading FOVs", unit="fov"):
        img = fov_images[fov].rechunk((1, cy, cx))
        c_here = fov_channels[fov]

        y0 = int(fov_locs.loc[fov, "y0"])
        y1 = int(fov_locs.loc[fov, "y1"])
        x0 = int(fov_locs.loc[fov, "x0"])
        x1 = int(fov_locs.loc[fov, "x1"])

        slices = _clip_tile_to_canvas(y0, y1, x0, x1, H, W)
        if slices is None:
            continue
        gl_y, gl_x, src_y, src_x = slices

        for local_ci, ch in enumerate(c_here):
            global_ci = channel_to_idx[ch]

            src = img[local_ci : local_ci + 1, src_y, src_x]
            region = (slice(global_ci, global_ci + 1), gl_y, gl_x)

            op = da.store(src, z_img, regions=region, lock=lock, compute=False)
            store_ops.append(op)

    with ProgressBar():
        logger.info("Stitching FOVs")
        compute_with_limit(*store_ops, n_workers=n_workers)

    stitched = da.from_zarr(z_img)
    return stitched, all_channels, used_ox, used_oy


# ---------------------------------------------------------------------------
# Cell-label TIFF stitcher
# ---------------------------------------------------------------------------


def _read_stitched_cell_labels_from_dir(
    cell_labels_dir: Path,
    fov_locs: pd.DataFrame,
    *,
    fovs: set[int] | None = None,
    flip_image: bool = False,
    n_workers: int | None = None,
    fov_local_to_global: dict[int, np.ndarray] | None = None,
) -> tuple[da.Array | None, set[int], pd.DataFrame]:
    """Read flat ``CellLabels_Fxxx.tif`` images and stitch them.

    The grid is snapped only to the FOVs for which a TIFF was actually
    found, which removes blank space caused by FOVs listed in the
    positions file but absent from disk.

    Parameters
    ----------
    cell_labels_dir
        Directory containing ``CellLabels_F*.tif`` files.
    fov_locs
        FOV positions table (indexed by FOV id).
    fovs
        Optional subset of FOV ids to include.
    flip_image
        Fallback tile flip, used only when ``flip_y`` is unavailable for the FOV.
    n_workers
        Parallelism cap for :func:`compute_with_limit`.
    fov_local_to_global
        Per-FOV look-up table mapping local cell ids to global ids.

    Returns
    -------
    tuple[da.Array, set[int], pd.DataFrame]
        ``(stitched_labels, present_cell_ids, fov_locs_used)``
    """
    from spatialdata_io.readers._cosmx_utils import find_cell_label_tifs

    fov_tif_map = find_cell_label_tifs(cell_labels_dir)

    fov_tiles: dict[int, da.Array] = {}
    seen_fovs: list[int] = []

    for fov, img_path in fov_tif_map.items():
        if fovs is not None and fov not in fovs:
            continue
        if fov not in fov_locs.index:
            logger.warning("Cell labels for FOV %d ignored (no entry in positions table).", fov)
            continue

        arr = tifffile.imread(img_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D label image for {img_path}, got shape {arr.shape}")

        if _label_tile_flip(fov_locs, fov, flip_image):  # see :func:`_label_tile_flip`
            arr = arr[::-1, :]

        if fov_local_to_global is not None and fov in fov_local_to_global:
            lut = fov_local_to_global[fov]
            if arr.max() >= len(lut):
                raise ValueError(
                    f"FOV {fov}: label image contains id {int(arr.max())} but mapping has length {len(lut)}."
                )
            arr = lut[arr]

        tile = da.from_array(arr, chunks=arr.shape)

        h, w = arr.shape
        fov_locs.loc[fov, "xmax"] = float(fov_locs.loc[fov, "xmin"]) + w
        fov_locs.loc[fov, "ymax"] = float(fov_locs.loc[fov, "ymin"]) + h

        fov_tiles[fov] = tile
        seen_fovs.append(fov)

    if not seen_fovs:
        logger.warning("No matching CellLabels TIFs found in %s — skipping labels.", cell_labels_dir)
        return None, set(), fov_locs

    stitched_fov_locs = fov_locs.loc[sorted(seen_fovs)].copy()
    stitched_fov_locs = _snap_fov_grid(stitched_fov_locs, fov_tiles)

    total_y = int(stitched_fov_locs["y1"].max())
    total_x = int(stitched_fov_locs["x1"].max())

    cy = min(1024, total_y)
    cx = min(1024, total_x)

    z_arr = _open_temp_zarr("cosmx_cell_labels_", (total_y, total_x), (cy, cx), next(iter(fov_tiles.values())).dtype)

    store_ops: list[dask.delayed] = []
    lock = SerializableLock()  # shared across tiles: serialises writes to shared zarr chunks

    for fov, tile in fov_tiles.items():
        tile = tile.rechunk((cy, cx))

        y0 = int(stitched_fov_locs.loc[fov, "y0"])
        y1 = int(stitched_fov_locs.loc[fov, "y1"])
        x0 = int(stitched_fov_locs.loc[fov, "x0"])
        x1 = int(stitched_fov_locs.loc[fov, "x1"])

        slices = _clip_tile_to_canvas(y0, y1, x0, x1, total_y, total_x)
        if slices is None:
            continue
        gl_y, gl_x, src_y, src_x = slices

        op = da.store(tile[src_y, src_x], z_arr, regions=(gl_y, gl_x), lock=lock, compute=False)
        store_ops.append(op)

    with ProgressBar():
        compute_with_limit(*store_ops, n_workers=n_workers)

    stitched = da.from_zarr(z_arr)

    uniq_da = da.unique(stitched)
    uniq_np = np.asarray(uniq_da.compute(), dtype=int)
    present_ids = {int(x) for x in uniq_np if x != 0}

    return stitched, present_ids, stitched_fov_locs


# ---------------------------------------------------------------------------
# Polygon → label rasterization
# ---------------------------------------------------------------------------


def _polygons_to_label_raster(
    poly_df: pd.DataFrame,
    *,
    chunks: tuple[int, int] = (2048, 2048),
    n_jobs: int | None = None,
    canvas_min_x: float | None = None,
    canvas_min_y: float | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> xr.DataArray:
    """Rasterize polygons into a uint32 label image.

    Operates tile-by-tile to keep peak memory bounded, writing directly
    into a zarr ``LocalStore``.  ``skimage.draw.polygon`` is used
    for the actual rasterization.

    Parameters
    ----------
    poly_df
        DataFrame with a ``geometry`` column of :class:`shapely.geometry.Polygon`
        objects and, optionally, a ``global_cell_id`` column.
    chunks
        Tile size ``(height, width)`` for the zarr output.
    n_jobs
        Parallelism cap for :func:`compute_with_limit`.
    canvas_min_x, canvas_min_y
        Origin of the rasterization canvas (defaults to polygon bounds).
    canvas_width, canvas_height
        Canvas extent in pixels (defaults to polygon bounds).

    Returns
    -------
    xr.DataArray
        2-D label image with dims ``("y", "x")`` and ``uint32`` dtype,
        backed by a dask array.
    """
    if poly_df.empty:
        raise ValueError("poly_df is empty, cannot rasterize polygons.")

    geom_arr = poly_df.geometry.to_list()
    bounds = np.array([g.bounds for g in geom_arr], dtype=float)

    if canvas_min_x is None or canvas_min_y is None or canvas_width is None or canvas_height is None:
        min_x = float(bounds[:, 0].min())
        min_y = float(bounds[:, 1].min())
        width = int(math.ceil(float(bounds[:, 2].max() - min_x)))
        height = int(math.ceil(float(bounds[:, 3].max() - min_y)))
    else:
        min_x = float(canvas_min_x)
        min_y = float(canvas_min_y)
        width = int(canvas_width)
        height = int(canvas_height)

    cy, cx = chunks
    n_tiles_y = int(math.ceil(height / cy))
    n_tiles_x = int(math.ceil(width / cx))

    if "global_cell_id" in poly_df.columns:
        label_ids = poly_df["global_cell_id"].to_numpy(dtype=np.uint32, copy=False)
    else:
        label_ids = np.arange(len(poly_df), dtype=np.uint32) + 1

    z_arr = _open_temp_zarr("cosmx_labels_", (height, width), chunks, "uint32")

    def _burn_row(ty: int) -> None:
        y0 = ty * cy
        y1 = min((ty + 1) * cy, height)

        row_min_y = min_y + y0
        row_max_y = min_y + y1

        row_mask = (bounds[:, 1] < row_max_y) & (bounds[:, 3] > row_min_y)
        row_idxs = np.nonzero(row_mask)[0]

        if row_idxs.size == 0:
            for tx in range(n_tiles_x):
                x0 = tx * cx
                x1 = min((tx + 1) * cx, width)
                z_arr[y0:y1, x0:x1] = 0
            return

        row_geoms = [geom_arr[i] for i in row_idxs]
        row_labels = label_ids[row_idxs]
        row_bounds = bounds[row_idxs]

        for tx in range(n_tiles_x):
            x0 = tx * cx
            x1 = min((tx + 1) * cx, width)

            tile_min_x = min_x + x0
            tile_max_x = min_x + x1

            mask_x = (row_bounds[:, 0] < tile_max_x) & (row_bounds[:, 2] > tile_min_x)
            idxs = np.nonzero(mask_x)[0]

            if idxs.size == 0:
                z_arr[y0:y1, x0:x1] = 0
                continue

            tile_h = y1 - y0
            tile_w = x1 - x0
            tile_canvas = np.zeros((tile_h, tile_w), dtype=np.uint32)

            for j in idxs:
                geom = row_geoms[j]
                if geom is None or geom.is_empty:
                    continue

                # Handle both Polygon and MultiPolygon geometries.
                parts = geom.geoms if hasattr(geom, "geoms") else [geom]
                for part in parts:
                    if part.is_empty:
                        continue
                    xs, ys = part.exterior.coords.xy
                    xs = np.asarray(xs, dtype=float) - tile_min_x
                    ys = np.asarray(ys, dtype=float) - row_min_y

                    ys = ys - (y0 - ty * cy)

                    rr, cc = polygon(ys, xs, shape=tile_canvas.shape)
                    tile_canvas[rr, cc] = row_labels[j]

            z_arr[y0:y1, x0:x1] = tile_canvas

    tasks: list[dask.delayed] = []
    for ty in range(n_tiles_y):
        tasks.append(dask.delayed(_burn_row)(ty))

    if tasks:
        compute_with_limit(*tasks, n_workers=n_jobs)

    darr = da.from_zarr(z_arr)

    xs = np.arange(min_x, min_x + width, dtype=float)
    ys = np.arange(min_y, min_y + height, dtype=float)

    return xr.DataArray(
        darr,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
    )


# ---------------------------------------------------------------------------
# Canvas from FOV positions (for polygon rasterization)
# ---------------------------------------------------------------------------


def _canvas_from_fov_locs_for_polygons(
    fov_locs: pd.DataFrame,
    *,
    origin_x: float,
    origin_y: float,
    fovs: set[int] | None = None,
    fov_size: float = COSMX_FOV_SIZE_PX,
) -> tuple[float, float, int, int]:
    """Compute canvas dimensions from FOV positions in polygon space.

    The result is in the same coordinate frame as the polygons (i.e.
    after shifting by ``-origin_x``, ``-origin_y``).

    Parameters
    ----------
    fov_locs
        FOV positions table.
    origin_x, origin_y
        Global origin subtracted from FOV coordinates.
    fovs
        Optional FOV id subset.
    fov_size
        Expected tile extent in pixels.

    Returns
    -------
    tuple[float, float, int, int]
        ``(canvas_min_x, canvas_min_y, width, height)``
    """
    locs = fov_locs
    if fovs is not None:
        locs = locs.loc[sorted(fovs)].copy()

    xmins = (locs["xmin"].astype(float) - origin_x).to_numpy()
    ymins = (locs["ymin"].astype(float) - origin_y).to_numpy()

    if "xmax" in locs.columns:
        xmaxs = (locs["xmax"].astype(float) - origin_x).to_numpy()
    else:
        xmaxs = xmins + fov_size

    if "ymax" in locs.columns:
        ymaxs = (locs["ymax"].astype(float) - origin_y).to_numpy()
    else:
        ymaxs = ymins + fov_size

    min_x = float(xmins.min())
    min_y = float(ymins.min())
    width = int(math.ceil(float(xmaxs.max() - min_x)))
    height = int(math.ceil(float(ymaxs.max() - min_y)))
    return min_x, min_y, width, height


# ---------------------------------------------------------------------------
# Legacy label stitcher (CellStatsDir layout)
# ---------------------------------------------------------------------------


def _find_dir(path: Path, name: str) -> Path:
    """Locate a sub-directory by *name* under *path*.

    Raises
    ------
    FileNotFoundError
        If no directory with *name* is found, or multiple matches exist.
    """
    direct = path / name
    if direct.is_dir():
        return direct

    paths = list(path.rglob(f"**/{name}"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Found {len(paths)} path(s) with name {name} inside {path}")
    return paths[0]


def stitch_segmentation_label_image(
    path: str | Path,
    fov_position_file: str | Path,
    cell_info_file: str | Path,
    dataset_id: str | None = None,
    seg_dir_name: str = "CellStatsDir",
    label_prefix: str = "CellLabels",
    flip_image: bool = False,
    n_workers: int | None = None,
    fovs: set[int] | None = None,
) -> tuple[da.Array, pd.DataFrame]:
    """Stitch per-FOV label TIFFs from a CellStatsDir layout.

    Each ``FOVxxx/CellLabels_Fxxx.tif`` is read, local cell ids are
    mapped to unique global ids via *cell_info_file*, and the tiles are
    stitched into a single zarr-backed dask array.

    Parameters
    ----------
    path
        Root directory of the CosMx dataset.
    fov_position_file
        Path to the FOV positions CSV.
    cell_info_file
        Path to the CSV containing ``fov`` and ``cellID`` columns.
    dataset_id
        Optional dataset ID override.
    seg_dir_name
        Name of the segmentation directory to search for.
    label_prefix
        Filename prefix for the label TIFFs.
    flip_image
        Fallback tile flip, used only when ``flip_y`` is unavailable for the FOV.
    n_workers
        Parallelism cap for :func:`compute_with_limit`.
    fovs
        Optional set of FOV IDs to include. If ``None``, all FOVs are used.

    Returns
    -------
    tuple[da.Array, pd.DataFrame]
        ``(stitched_labels, cell_info_df)`` where *cell_info_df* has an
        added ``global_cell_id`` column.
    """
    from ._cosmx_discovery import _infer_dataset_id
    from ._cosmx_io import _read_fov_locs

    path = Path(path)
    dataset_id = _infer_dataset_id(path, dataset_id)
    fov_locs = _read_fov_locs(Path(fov_position_file))

    df = pd.read_csv(cell_info_file)
    if fovs is not None:
        df = df[df["fov"].isin(fovs)].reset_index(drop=True)
    mapping: dict[int, np.ndarray] = {}
    offset = 0
    for fov in sorted(df["fov"].unique()):
        local_ids = np.unique(df.loc[df["fov"] == fov, "cellID"].astype(int))
        if local_ids.size == 0:
            continue
        max_local = int(local_ids.max())
        arr_map = np.zeros(max_local + 1, dtype=int)
        for lid in local_ids:
            offset += 1
            arr_map[lid] = offset
        mapping[fov] = arr_map

    df = df.copy()
    df["global_cell_id"] = df.apply(lambda row: mapping[row["fov"]][int(row["cellID"])], axis=1)

    seg_root = _find_dir(path, seg_dir_name)
    folder_re = FOV_DIR_RE

    fov_tiles: dict[int, da.Array] = {}
    for sub in seg_root.iterdir():
        if not sub.is_dir():
            continue
        m = folder_re.match(sub.name)
        if m is None:
            continue
        fov = int(m.group(1))
        if fovs is not None and fov not in fovs:
            continue

        tif_path = sub / f"{label_prefix}_F{fov:03d}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Expected {tif_path} for FOV {fov}")

        arr = tifffile.imread(tif_path)
        if fov not in mapping:
            logger.warning("Segmentation labels for FOV %d ignored (no entry in %s).", fov, cell_info_file)
            continue
        lut = mapping[fov]
        if int(arr.max()) >= len(lut):
            raise ValueError(f"FOV {fov}: label image contains id {int(arr.max())} but mapping has length {len(lut)}.")
        arr = lut[arr]
        if _label_tile_flip(fov_locs, fov, flip_image):  # see :func:`_label_tile_flip`
            arr = arr[::-1, :]

        tile = da.from_array(arr, chunks=arr.shape)

        y_size, x_size = arr.shape
        fov_locs.loc[fov, "xmax"] = fov_locs.loc[fov, "xmin"] + x_size
        fov_locs.loc[fov, "ymax"] = fov_locs.loc[fov, "ymin"] + y_size

        fov_tiles[fov] = tile

    fov_locs = _snap_fov_grid(fov_locs, fov_tiles)

    total_y = int(fov_locs["y1"].max())
    total_x = int(fov_locs["x1"].max())

    cy = min(1024, total_y)
    cx = min(1024, total_x)

    z_arr = _open_temp_zarr("cosmx_seg_labels_", (total_y, total_x), (cy, cx), next(iter(fov_tiles.values())).dtype)

    store_ops: list[dask.delayed] = []
    lock = SerializableLock()  # shared across tiles: serialises writes to shared zarr chunks

    for fov, tile in fov_tiles.items():
        tile = tile.rechunk((cy, cx))

        y0 = int(fov_locs.loc[fov, "y0"])
        y1 = int(fov_locs.loc[fov, "y1"])
        x0 = int(fov_locs.loc[fov, "x0"])
        x1 = int(fov_locs.loc[fov, "x1"])

        slices = _clip_tile_to_canvas(y0, y1, x0, x1, total_y, total_x)
        if slices is None:
            continue
        gl_y, gl_x, src_y, src_x = slices

        op = da.store(tile[src_y, src_x], z_arr, regions=(gl_y, gl_x), lock=lock, compute=False)
        store_ops.append(op)

    with ProgressBar():
        compute_with_limit(*store_ops, n_workers=n_workers)

    stitched = da.from_zarr(z_arr)
    return stitched, df


# ---------------------------------------------------------------------------
# FOV preview plot
# ---------------------------------------------------------------------------


def _plot_fov_preview(
    fov_locs: pd.DataFrame,
    selected_fovs: set[int] | None = None,
    *,
    fov_size: float = COSMX_FOV_SIZE_PX,
) -> None:
    """Render a preview of the FOV layout using matplotlib.

    Each FOV is drawn as a square with the real CosMx tile size (default
    4256 x 4256 px) and labelled with its FOV id.  The Y axis is inverted
    to match the CosMx top-left origin convention.

    Parameters
    ----------
    fov_locs
        FOV positions table with ``xmin`` and ``ymin`` columns.
    selected_fovs
        If given, only these FOV ids are drawn.
    fov_size
        Tile size in pixels.

    Raises
    ------
    ValueError
        If *fov_locs* is empty or missing required columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if fov_locs is None or fov_locs.empty:
        raise ValueError("preview_fovs=True but FOV locations dataframe is empty.")

    df = fov_locs.reset_index().rename(columns={"index": "fov"})
    if selected_fovs is not None:
        df = df[df["fov"].isin(selected_fovs)].copy()

    if df.empty:
        raise ValueError("preview_fovs=True but no matching FOVs after applying the user subset.")

    if "xmin" not in df.columns or "ymin" not in df.columns:
        raise ValueError("preview_fovs=True but FOV positions do not have xmin/ymin columns.")

    x0s = df["xmin"].astype(float).to_numpy()
    y0s = df["ymin"].astype(float).to_numpy()
    fov_ids = df["fov"].astype(int).to_numpy()

    x_min = float(x0s.min())
    x_max = float((x0s + fov_size).max())
    y_min = float(y0s.min())
    y_max = float((y0s + fov_size).max())
    n = len(fov_ids)

    if n <= 40:
        fontsize = 12
    elif n <= 120:
        fontsize = 10
    elif n <= 300:
        fontsize = 8
    else:
        fontsize = 6

    fig, ax = plt.subplots(figsize=(8, 6))

    for x, y, _f in zip(x0s, y0s, fov_ids, strict=False):
        rect = Rectangle(
            (x, y),
            fov_size,
            fov_size,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(rect)

    text_kwargs = {
        "ha": "center",
        "va": "center",
        "color": "black",
        "fontsize": fontsize,
    }
    for x, y, f in zip(x0s, y0s, fov_ids, strict=False):
        cx = x + fov_size / 2.0
        cy = y + fov_size / 2.0
        ax.text(cx, cy, str(f), **text_kwargs)

    ax.set_xlim(x_min - fov_size * 0.05, x_max + fov_size * 0.05)
    ax.set_ylim(y_max + fov_size * 0.05, y_min - fov_size * 0.05)
    ax.set_aspect("equal")

    ax.set_xlabel("Global X (px)")
    ax.set_ylabel("Global Y (px)")
    ax.set_title("FOV Locations")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    plt.show()
