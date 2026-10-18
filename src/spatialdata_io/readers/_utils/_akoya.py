"""Channel, transformation, image and metadata helpers shared by the Akoya readers."""

from __future__ import annotations

import io
import json
import math
import threading
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.base import tokenize
from spatialdata import sanitize_name
from spatialdata._logging import logger
from spatialdata.models import Image2DModel
from spatialdata.transformations import BaseTransformation, Identity, Translation

from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs

if TYPE_CHECKING:
    from pathlib import Path

    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = [
    "apply_channel_names",
    "build_image_element",
    "dedupe_channel_names",
    "lazy_tiff_array",
    "normalize_regions",
    "read_json_object",
    "read_tiff",
    "region_transformations",
    "scans_to_attrs",
    "select_regions",
    "spatial_chunks",
]


def dedupe_channel_names(names: Sequence[str]) -> list[str]:
    """Make channel names unique as ``name``, ``name (2)``, ``name (3)``; the first occurrence is unchanged.

    A suffix is skipped if it would repeat a name of the list, e.g. ``["A", "A", "A (2)"]`` -> ``A``, ``A (3)``,
    ``A (2)``.
    """
    taken = set(names)
    seen: Counter[str] = Counter()
    deduped = []
    for name in names:
        seen[name] += 1
        if seen[name] == 1:
            deduped.append(name)
            continue
        while (unique := f"{name} ({seen[name]})") in taken:
            seen[name] += 1
        taken.add(unique)
        deduped.append(unique)
    return deduped


def apply_channel_names(scan: AkoyaScan, channel_names: Sequence[str] | None) -> list[str]:
    """Return the de-duplicated channel names of ``scan``: the user override if given, else the metadata names."""
    if channel_names is None:
        return dedupe_channel_names([channel.name for channel in scan.channels])
    if isinstance(channel_names, Mapping | Set):
        raise TypeError(f"`channel_names` must be a sequence of names, got {type(channel_names).__name__}.")
    if len(channel_names) != len(scan.channels):
        raise ValueError(
            f"`channel_names` has {len(channel_names)} entries, but region '{scan.region}' ({scan.path}) has "
            f"{len(scan.channels)} channels: expected {len(scan.channels)}."
        )
    return dedupe_channel_names(list(channel_names))


def region_transformations(
    scans: Sequence[AkoyaScan], regions_without_image: Iterable[str] = ()
) -> dict[str, dict[str, BaseTransformation]]:
    """Compute the coordinate systems and transformations of every region, by region id.

    Akoya scans are already stitched, so scans of the same sample space are only placed: if there are several
    scans and all have a stage position and a pixel size, the same slide and pixel sizes equal within 0.1%, each is
    translated by its stage position (in pixels, relative to the top-left one) into the ``"global"`` coordinate
    system. Otherwise the regions are unrelated and each gets its own coordinate system, named after the region,
    with ``Identity``; a read of a single region uses ``"global"``.

    Parameters
    ----------
    scans
        The scans read.
    regions_without_image
        Regions read without an image (e.g. only their cells); they are unrelated to the scans.

    Returns
    -------
    ``{region: {coordinate_system: transformation}}``, to be passed to every element of the region.
    """
    imaged = [scan.region for scan in scans]
    others = sorted(set(regions_without_image) - set(imaged))
    if len(imaged) + len(others) == 1:
        return {region: {"global": Identity()} for region in [*imaged, *others]}

    result: dict[str, dict[str, BaseTransformation]] = {}
    if _same_sample_space(scans):
        min_x = min(scan.stage_position_um[0] for scan in scans)  # type: ignore[index]
        min_y = min(scan.stage_position_um[1] for scan in scans)  # type: ignore[index]
        for scan in scans:
            x, y = scan.stage_position_um  # type: ignore[misc]
            pixel_size = scan.pixel_size_um
            translation = [(x - min_x) / pixel_size, (y - min_y) / pixel_size]  # type: ignore[operator]
            result[scan.region] = {"global": Translation(translation, axes=("x", "y"))}
    else:
        if len(imaged) > 1:
            logger.info(
                "Regions %s do not share a sample space (missing stage position or pixel size, different slides "
                "or pixel sizes), so each is stored in its own coordinate system, named after the region.",
                imaged,
            )
        result = {region: {region: Identity()} for region in imaged}
    return result | {region: {region: Identity()} for region in others}


def _same_sample_space(scans: Sequence[AkoyaScan]) -> bool:
    """Whether several scans can be placed by their stage positions in one pixel coordinate system."""
    if len(scans) < 2 or not all(scan.stage_position_um is not None and scan.pixel_size_um for scan in scans):
        return False
    slides = {scan.slide_id for scan in scans}
    pixel_size = scans[0].pixel_size_um
    return (
        len(slides) == 1
        and None not in slides
        and all(math.isclose(scan.pixel_size_um, pixel_size, rel_tol=1e-3) for scan in scans)  # type: ignore[arg-type]
    )


def build_image_element(
    image: da.Array,
    scan: AkoyaScan,
    c_coords: list[str],
    transformations: Mapping[str, BaseTransformation],
    image_models_kwargs: Mapping[str, Any],
) -> DataArray | DataTree:
    """Parse a lazy ``(c, y, x)`` array, or ``(y, x, 3)`` for brightfield RGB scans, into an ``Image2DModel``.

    ``transformations`` are the region's, see :func:`region_transformations`.
    """
    image_models_kwargs, _ = _initialize_raster_models_kwargs(image_models_kwargs, {})
    if scan.rgb:
        image = image.transpose(2, 0, 1)
        c_coords = ["r", "g", "b"]
    return Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        c_coords=c_coords,
        rgb=scan.rgb,
        transformations=dict(transformations),
        **image_models_kwargs,
    )


def scans_to_attrs(scans: Sequence[AkoyaScan], channel_names: Mapping[str, list[str]]) -> dict[str, Any]:
    """Build the JSON-serialisable per-region metadata stored in ``sdata.attrs["akoya"]``."""
    regions: dict[str, Any] = {}
    for scan in scans:
        names = channel_names[scan.region]
        regions[scan.region] = {
            "generation": scan.generation,
            "software": scan.software,
            "slide_id": scan.slide_id,
            "pixel_size_um": scan.pixel_size_um,
            "nominal_pixel_size_um": scan.nominal_pixel_size_um,
            "stage_position_um": list(scan.stage_position_um) if scan.stage_position_um is not None else None,
            "path": str(scan.path),
            "channels": [
                {**asdict(channel), "name_deduped": name} for channel, name in zip(scan.channels, names, strict=True)
            ],
            "raw": scan.raw,
        }
    return {"regions": regions}


def normalize_regions(regions: str | Sequence[str] | None) -> list[str] | None:
    """Validate the ``regions`` argument of a reader: a region id or a sequence of them, without duplicates.

    The region ids are sanitised with :func:`spatialdata.sanitize_name`, like those of the datasets, so raw names
    (e.g. ``"Slide 1"`` for the region ``Slide_1``) select their region. Raises a `TypeError` for a mapping and a
    `ValueError` for an empty sequence.
    """
    if regions is None:
        return None
    if isinstance(regions, str):
        return [sanitize_name(regions)]
    if isinstance(regions, Mapping):
        raise TypeError(f"`regions` must be a sequence of region ids, got {type(regions).__name__}.")
    if not regions:
        raise ValueError("`regions` is empty; pass `None` to read all regions.")
    return list(dict.fromkeys(sanitize_name(str(region)) for region in regions))


def select_regions(available: Sequence[str], regions: str | Sequence[str] | None) -> list[str]:
    """Return the requested regions (all if ``None``), raising if one is not available; see :func:`normalize_regions`."""
    regions = normalize_regions(regions)
    if regions is None:
        return list(available)
    unknown = [region for region in regions if region not in available]
    if unknown:
        raise ValueError(f"Unknown region(s) {unknown}. Available regions: {list(available)}.")
    return regions


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from ``path``; warn and return ``None`` if the file is not valid UTF-8 JSON or not an object.

    A UTF-8 byte order mark, as written by some Windows tools, is ignored.
    """
    try:
        content = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        warnings.warn(f"Ignoring {path}: it is not valid JSON ({e}).", UserWarning, stacklevel=2)
        return None
    if not isinstance(content, dict):
        warnings.warn(f"Ignoring {path}: it is not a JSON object.", UserWarning, stacklevel=2)
        return None
    return content


def read_tiff(path: Path, chunks: tuple[int, int]) -> tuple[da.Array, tuple[int, ...], list[str] | None]:
    """Lazily read a tiff as ``(c, y, x)``, all non-spatial axes flattened into ``c``.

    Uncompressed contiguous data (e.g. ImageJ hyperstacks and per-marker images, often a single strip per plane)
    is memory-mapped and split into ``chunks``, so a chunk never needs a whole plane in memory; other layouts keep
    their native tiles via zarr. Returns the image, the shape of the flattened axes and the ImageJ ``Labels`` (if
    every plane has a non-blank one). The array opens the file only when its chunks are read and can be pickled,
    e.g. for ``dask.distributed``.
    """
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes, offset, shape = series.axes, series.dataoffset, series.shape
        # tifffile reports the dtype in native byte order; a memory map needs the file's (big-endian for ImageJ)
        dtype = series.dtype.newbyteorder(tf.byteorder)
        metadata = tf.imagej_metadata or {}
    labels = list(metadata["Labels"]) if isinstance(metadata.get("Labels"), list | tuple) else None
    # ImageJ writes an empty label for every slice without one, so partial labels do not name the channels
    if labels is not None and not all(isinstance(label, str) and label.strip() for label in labels):
        labels = None
    if offset is not None:
        source = _ContiguousTiff(path, offset, shape, dtype)
        tiles = dict.fromkeys(range(source.ndim), 1) | {axes.index("Y"): chunks[0], axes.index("X"): chunks[1]}
        tile_chunks = tuple(tiles[i] for i in range(source.ndim))
        image = da.from_array(source, chunks=tile_chunks, name=f"{source.name}-{tokenize(tile_chunks)}")
    else:
        image = lazy_tiff_array(path, series=0, chunks=chunks)
    image = da.moveaxis(image, [axes.index("Y"), axes.index("X")], [-2, -1])
    leading_shape = tuple(image.shape[:-2])
    return image.reshape((math.prod(leading_shape), *image.shape[-2:])), leading_shape, labels


class _ContiguousTiff:
    """Array-like view of uncompressed contiguous tiff data, memory-mapped anew for every chunk.

    Handing an ``np.memmap`` to :func:`dask.array.from_array` would read the whole file (dask hashes and slices
    numpy arrays eagerly), so dask only gets this lightweight object and a name derived from the file and chunks.
    """

    def __init__(self, path: Path, offset: int, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
        self.path = path
        self.offset = offset
        self.shape = shape
        self.ndim = len(shape)
        self._file_dtype = dtype
        self.dtype = dtype.newbyteorder("=")
        self.name = f"akoya-tiff-{tokenize(str(path), path.stat().st_mtime_ns, offset, shape, str(dtype))}"

    def __getitem__(self, key: Any) -> np.ndarray:
        data = np.memmap(self.path, dtype=self._file_dtype, mode="r", offset=self.offset, shape=self.shape)
        # a copy, as a view would keep the memory map and its file descriptor open as long as the chunk lives
        return np.array(data[key], dtype=self.dtype)


def lazy_tiff_array(
    path: Path, *, series: int = 0, page: int | None = None, chunks: tuple[int, int] = (4096, 4096)
) -> da.Array:
    """Lazily read the level 0 of a tiff series, or one page, in chunks of whole native tiles or strips.

    The array can be pickled (e.g. for ``dask.distributed`` or process-based schedulers): its chunks are read by a
    :class:`_TiffZarrSource`, which opens the file in each process that reads them.

    Parameters
    ----------
    path
        The tiff file.
    series
        The index of the series to read, if ``page`` is `None`.
    page
        The index of the page to read instead of a series.
    chunks
        The ``(y, x)`` chunk size to aim for: the native tiles or strips are grouped into chunks of about as many
        pixels (the default is that of the raster models), so the graph does not get one task per strip.

    Returns
    -------
    The array, with the shape of the series or page.
    """
    source = _TiffZarrSource(path, series=series, page=page)
    y, x = source.axes.index("Y"), source.axes.index("X")
    grouped = list(source.chunks)
    native_y, native_x = grouped[y], grouped[x]
    grouped[x] = min(source.shape[x], max(native_x, chunks[1] // native_x * native_x))
    grouped[y] = min(source.shape[y], max(native_y, chunks[0] * chunks[1] // grouped[x] // native_y * native_y))
    return da.from_array(
        source,
        chunks=tuple(grouped),
        name=f"{source.name}-{tokenize(grouped)}",
        meta=np.empty((0,) * source.ndim, dtype=source.dtype),
    )


class _TiffZarrSource:
    """Picklable array-like view of a tiff series (level 0) or page, read chunk-wise through tifffile's zarr store.

    Neither the ``TiffFile`` nor its zarr store are pickled, but opened again in each process that reads chunks, so
    dask graphs built on it work across processes. The file is read through a :class:`_TransientFile`, so the parsed
    tiff stays cached without holding a file descriptor.
    """

    def __init__(self, path: Path, *, series: int = 0, page: int | None = None) -> None:
        self.path = path
        self.series = series
        self.page = page
        self._lock = threading.Lock()
        self._array: zarr.Array | None = None
        array = self._open()
        self.shape = tuple(array.shape)
        self.chunks = tuple(array.chunks)
        self.dtype = array.dtype
        self.ndim = len(self.shape)
        token = tokenize(str(path), path.stat().st_mtime_ns, series, page, self.shape, str(self.dtype))
        self.name = f"akoya-tiff-zarr-{token}"

    def _open(self) -> zarr.Array:
        with self._lock:
            if self._array is None:
                tiff = tifffile.TiffFile(io.BufferedReader(_TransientFile(self.path)))
                data = tiff.pages[self.page] if self.page is not None else tiff.series[self.series]
                self.axes = data.axes
                self._array = zarr.open_array(data.aszarr(level=0), mode="r")
            return self._array

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._open()[key])

    def __getstate__(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key not in ("_lock", "_array")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state, _lock=threading.Lock(), _array=None)


class _TransientFile(io.RawIOBase):
    """Read-only binary file that opens ``path`` for every read, so it holds no file descriptor in between."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.name = str(path)
        self._size = path.stat().st_size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._position = (0, self._position, self._size)[whence] + offset
        return self._position

    def readinto(self, buffer: Any) -> int:
        with open(self.name, "rb") as f:
            f.seek(self._position)
            n = f.readinto(buffer)
        self._position += n
        return n


def spatial_chunks(chunks: Any, default: Sequence[int]) -> tuple[int, int]:
    """The ``(y, x)`` sizes of a raster model ``chunks`` argument, ``default`` when it does not give plain sizes."""
    if isinstance(chunks, int):
        return chunks, chunks
    if isinstance(chunks, Mapping) and isinstance(chunks.get("y"), int) and isinstance(chunks.get("x"), int):
        return chunks["y"], chunks["x"]
    if isinstance(chunks, list | tuple) and len(chunks) >= 2 and all(isinstance(c, int) for c in chunks[-2:]):
        return chunks[-2], chunks[-1]
    return default[-2], default[-1]
