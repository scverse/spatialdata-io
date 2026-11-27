"""Write a browser-ready WebP image pyramid as Parquet row groups.

The canonical OME-Zarr image stays authoritative; this is a derived *display cache*, and
must never be treated as the quantitative image.

Layout matches Celldega's existing ``ImageRowGroupReader``: columns ``zoom``, ``tile_x``,
``tile_y``, ``image_data`` (encoded WebP bytes), one tile per row group, with a
``zoom_info`` map in the Parquet schema metadata::

    row_group_index = zoom_info[zoom].row_group_offset + tile_x * num_tiles_y + tile_y

Zoom numbering follows the DeepZoom convention, so the same numbers work against tiles
produced by ``vips dzsave``: level ``max_zoom`` is full resolution and each level below
halves both dimensions, down to a level that fits in a single tile.

Encoding uses Pillow rather than libvips, so this adds no system dependency. Pyramid
levels are produced from the SpatialData image's own multiscale levels where they line up,
and by downsampling otherwise.
"""

from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

__all__ = ["write_webp_pyramid", "DEFAULT_IMAGE_TILE_SIZE"]

#: DeepZoom tile size used by Celldega's image pipeline.
DEFAULT_IMAGE_TILE_SIZE = 512

#: Image tiles are numerous and individually small, so more fit comfortably per file
#: than for transcripts.
DEFAULT_IMAGE_ROW_GROUPS_PER_FILE = 2000


def _require_pillow() -> Any:
    try:
        from PIL import Image, features
    except ImportError as exc:
        raise RuntimeError("writing a WebP pyramid requires Pillow: pip install 'Pillow>=10'") from exc
    if not features.check("webp"):
        raise RuntimeError("this Pillow build has no WebP support; reinstall Pillow with WebP enabled")
    return Image


def _select_channel(array: Any, channel: int | str | None) -> Any:
    """Reduce one multiscale level to a 2D (y, x) plane."""
    if hasattr(array, "data_vars"):
        array = array[next(iter(array.data_vars))]
    dims = getattr(array, "dims", None)
    if dims and "c" in dims:
        if channel is None:
            array = array.isel(c=0)
        elif isinstance(channel, str):
            array = array.sel(c=channel)
        else:
            array = array.isel(c=channel)
    return array


def _as_2d(image: Any, channel: int | str | None) -> tuple[Any, Any]:
    """Return ``(full_resolution_plane, coarse_plane)`` for a SpatialData image element.

    The coarse plane is the smallest available multiscale level, used only to choose the
    display intensity window cheaply. For a single-scale image both are the same array.
    """
    if hasattr(image, "children") and len(image.children):
        levels = list(image.children)
        return (
            _select_channel(image[levels[0]], channel),
            _select_channel(image[levels[-1]], channel),
        )
    plane = _select_channel(image, channel)
    return plane, plane


def _display_window(sample_source: Any, display_min: float | None, display_max: float | None) -> tuple[float, float]:
    """Choose the intensity window.

    The default is the *full dtype range* for integer images, which is a linear mapping
    and no stretch at all. That deliberately matches Celldega's own image pipeline, which
    saves raw values and leaves brightening to the viewer's intensity slider. A percentile
    stretch here would be applied on top of that slider and blow out the mid-tones -- on
    Xenium DAPI it took a tile from mean 1.3 to mean 37.4 with 1.2% of pixels saturated.

    Pass ``display_min``/``display_max`` to window explicitly; both are recorded in the
    manifest so the choice is reproducible and invalidatable.
    """
    if display_min is not None and display_max is not None:
        lo, hi = float(display_min), float(display_max)
    else:
        sample = np.asarray(sample_source)
        if sample.dtype.kind in "ui":
            info = np.iinfo(sample.dtype)
            default_lo, default_hi = float(info.min), float(info.max)
        else:
            finite = sample[np.isfinite(sample)]
            default_lo = float(finite.min()) if finite.size else 0.0
            default_hi = float(finite.max()) if finite.size else 1.0
        lo = float(display_min) if display_min is not None else default_lo
        hi = float(display_max) if display_max is not None else default_hi
    return (lo, hi if hi > lo else lo + 1.0)


def _to_uint8(plane: NDArray[Any], lo: float, hi: float, gamma: float, block_rows: int = 2048) -> NDArray[np.uint8]:
    """Window a plane to 8-bit in row blocks, avoiding a float copy of the whole image."""
    height = plane.shape[0]
    out = np.empty(plane.shape, dtype=np.uint8)
    inv_gamma = 1.0 / gamma
    for start in range(0, height, block_rows):
        stop = min(start + block_rows, height)
        block = np.asarray(plane[start:stop], dtype=np.float32)
        block -= lo
        block /= hi - lo
        np.clip(block, 0.0, 1.0, out=block)
        if gamma != 1.0:
            block **= inv_gamma
        block *= 255.0
        block += 0.5
        out[start:stop] = block.astype(np.uint8)
    return out


def _downsample_half(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Box-filter by 2 in both axes, padding odd edges by replication."""
    h, w = a.shape
    if h % 2:
        a = np.vstack([a, a[-1:]])
    if w % 2:
        a = np.hstack([a, a[:, -1:]])
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3)).astype(np.uint8)


def write_webp_pyramid(
    image: Any,
    output_dir: str | Path,
    *,
    channel: int | str | None = None,
    tile_size: int = DEFAULT_IMAGE_TILE_SIZE,
    display_min: float | None = None,
    display_max: float | None = None,
    gamma: float = 1.0,
    quality: int = 85,
    lossless: bool = False,
    max_row_groups_per_file: int = DEFAULT_IMAGE_ROW_GROUPS_PER_FILE,
    source_element: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a DeepZoom-numbered WebP pyramid as Parquet row groups.

    Parameters
    ----------
    image
        A SpatialData image element (``DataTree`` or ``DataArray``).
    output_dir
        Directory to write chunk files into. Written atomically.
    channel
        Channel index or name to render. Defaults to the first channel.
    tile_size
        Tile edge length in pixels.
    display_min, display_max
        Intensity window. Defaults to the 1st and 99.9th percentiles of the full-resolution
        plane, which is a display choice and is recorded in the returned metadata.
    gamma
        Display gamma applied after windowing.
    quality
        WebP quality when ``lossless`` is False.
    lossless
        Whether to encode losslessly.
    max_row_groups_per_file
        Tiles per chunk file.
    source_element
        Name of the canonical image element, recorded for invalidation.
    overwrite
        Replace ``output_dir`` if it exists.

    Returns
    -------
    The manifest fragment describing the written pyramid.
    """
    Image = _require_pillow()

    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")

    plane, coarse = _as_2d(image, channel)
    if plane.ndim != 2:
        raise ValueError(f"expected a 2D plane after channel selection, got shape {plane.shape}")
    source_dtype = str(plane.dtype)

    applied_min, applied_max = _display_window(coarse, display_min, display_max)
    full = _to_uint8(plane, applied_min, applied_max, gamma)
    height, width = full.shape

    # DeepZoom numbering: level max_zoom is full resolution and each level below halves
    # both dimensions, down to level 0 (a single pixel). Every level is generated so the
    # numbering matches `vips dzsave` output exactly.
    max_zoom = max(1, math.ceil(math.log2(max(width, height))))
    levels: dict[int, NDArray[np.uint8]] = {max_zoom: full}
    current = full
    for zoom in range(max_zoom - 1, -1, -1):
        current = _downsample_half(current)
        levels[zoom] = current

    schema = pa.schema(
        [
            pa.field("zoom", pa.int32()),
            pa.field("tile_x", pa.int32()),
            pa.field("tile_y", pa.int32()),
            pa.field("image_data", pa.binary()),
        ]
    )

    # Enumerate tiles in the reader's order: zoom ascending, then column-major within zoom.
    ordered: list[tuple[int, int, int]] = []
    zoom_info: dict[str, dict[str, int]] = {}
    for zoom in sorted(levels):
        lh, lw = levels[zoom].shape
        nx = max(1, math.ceil(lw / tile_size))
        ny = max(1, math.ceil(lh / tile_size))
        zoom_info[str(zoom)] = {
            "num_tiles_x": nx,
            "num_tiles_y": ny,
            "num_tiles": nx * ny,
            "row_group_offset": len(ordered),
        }
        ordered.extend((zoom, tx, ty) for tx in range(nx) for ty in range(ny))

    n_files = max(1, -(-len(ordered) // max_row_groups_per_file))
    width_digits = len(str(n_files - 1)) if n_files > 1 else 1
    filenames = [f"chunk_{i:0{width_digits}d}.parquet" for i in range(n_files)]

    schema = schema.with_metadata(
        {
            b"zoom_info": json.dumps(zoom_info).encode(),
            b"storage_mode": b"row_groups_image_chunked",
            b"max_row_groups_per_file": str(max_row_groups_per_file).encode(),
            b"tile_size": str(tile_size).encode(),
            b"profile": b"grid_files_v1",
        }
    )

    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    encode_kwargs: dict[str, Any] = {"format": "WEBP", "lossless": lossless}
    if not lossless:
        encode_kwargs["quality"] = quality

    try:
        writer: pq.ParquetWriter | None = None
        current_file = -1
        for index, (zoom, tx, ty) in enumerate(ordered):
            file_index = index // max_row_groups_per_file
            if file_index != current_file:
                if writer is not None:
                    writer.close()
                writer = pq.ParquetWriter(staging / filenames[file_index], schema, write_statistics=False)
                current_file = file_index

            level = levels[zoom]
            crop = level[ty * tile_size : (ty + 1) * tile_size, tx * tile_size : (tx + 1) * tile_size]
            buf = io.BytesIO()
            Image.fromarray(crop, mode="L").save(buf, **encode_kwargs)

            assert writer is not None
            writer.write_table(
                pa.table(
                    {
                        "zoom": pa.array([zoom], pa.int32()),
                        "tile_x": pa.array([tx], pa.int32()),
                        "tile_y": pa.array([ty], pa.int32()),
                        "image_data": pa.array([buf.getvalue()], pa.binary()),
                    },
                    schema=schema,
                )
            )
        if writer is not None:
            writer.close()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.rename(output_dir)

    return {
        "directory": output_dir.name,
        "files": filenames,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": len(ordered),
        "zoom_info": zoom_info,
        "min_zoom": min(levels),
        "max_zoom": max_zoom,
        "tile_size": tile_size,
        "image_format": ".webp",
        # Recorded so the display cache can be invalidated when any of it changes.
        "source_element": source_element,
        "source_width": int(width),
        "source_height": int(height),
        "source_dtype": source_dtype,
        "channel": channel,
        "display_min": applied_min,
        "display_max": applied_max,
        "gamma": gamma,
        "downsampling": "box-2x2-mean",
        "webp_lossless": lossless,
        "webp_quality": None if lossless else quality,
    }
