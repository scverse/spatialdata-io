"""Channel, transformation, image and metadata helpers shared by the Akoya readers."""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.base import tokenize
from dask_image.imread import imread
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
    "normalize_regions",
    "read_json_object",
    "read_tiff",
    "region_transformations",
    "scans_to_attrs",
    "select_regions",
    "spatial_chunks",
]


def dedupe_channel_names(names: Sequence[str]) -> list[str]:
    """Make channel names unique as ``name``, ``name (2)``, ``name (3)``; the first occurrence is unchanged."""
    seen: Counter[str] = Counter()
    deduped = []
    for name in names:
        seen[name] += 1
        deduped.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return deduped


def apply_channel_names(scan: AkoyaScan, channel_names: Sequence[str] | None) -> list[str]:
    """Return the channel names of ``scan``: the user override if given, else the de-duplicated metadata names."""
    if channel_names is None:
        return dedupe_channel_names([channel.name for channel in scan.channels])
    if len(channel_names) != len(scan.channels):
        raise ValueError(
            f"`channel_names` has {len(channel_names)} entries, but region '{scan.region}' ({scan.path}) has "
            f"{len(scan.channels)} channels: expected {len(scan.channels)}."
        )
    return list(channel_names)


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

    Raises a `TypeError` for a mapping, e.g. ``imread_kwargs`` passed positionally to ``codex()``, whose third
    parameter used to be ``imread_kwargs``, and a `ValueError` for an empty sequence.
    """
    if regions is None:
        return None
    if isinstance(regions, str):
        return [regions]
    if isinstance(regions, Mapping):
        raise TypeError(
            f"`regions` must be a sequence of region ids, got {type(regions).__name__}; `imread_kwargs` is no "
            "longer the third positional argument of `codex()` — pass it by keyword."
        )
    if not regions:
        raise ValueError("`regions` is empty; pass `None` to read all regions.")
    return list(dict.fromkeys(regions))


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
    """Read a JSON object from ``path``; warn and return ``None`` if the file is not valid JSON or not an object."""
    try:
        content = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        warnings.warn(f"Ignoring {path}: it is not valid JSON ({e}).", UserWarning, stacklevel=2)
        return None
    if not isinstance(content, dict):
        warnings.warn(f"Ignoring {path}: it is not a JSON object.", UserWarning, stacklevel=2)
        return None
    return content


def read_tiff(
    path: Path, imread_kwargs: Mapping[str, Any], chunks: tuple[int, int]
) -> tuple[da.Array, tuple[int, ...], list[str] | None, bool]:
    """Lazily read a tiff as ``(c, y, x)``, all non-spatial axes flattened into ``c``.

    Uncompressed contiguous data (e.g. ImageJ hyperstacks and per-marker images, often a single strip per plane)
    is memory-mapped and split into ``chunks``, so a chunk never needs a whole plane in memory; other layouts keep
    their native tiles via zarr. Returns the image, the shape of the flattened axes, the ImageJ ``Labels`` (if
    any) and whether tifffile, rather than the :func:`dask_image.imread.imread` fallback, read the file. The
    file handle stays open for as long as the array needs it.
    """
    tf = tifffile.TiffFile(path)
    series = tf.series[0]
    axes = series.axes
    metadata = tf.imagej_metadata or {}
    labels = list(metadata["Labels"]) if isinstance(metadata.get("Labels"), list | tuple) else None
    read_by_tifffile = True
    if series.dataoffset is not None:
        tf.close()
        source = _ContiguousTiff(path, series.dataoffset, series.shape, series.dtype)
        tiles = dict.fromkeys(range(source.ndim), 1) | {axes.index("Y"): chunks[0], axes.index("X"): chunks[1]}
        image = da.from_array(source, chunks=tuple(tiles[i] for i in range(source.ndim)), name=source.name)
    else:
        try:
            image = da.from_zarr(zarr.open(series.aszarr(level=0), mode="r"))
        except ValueError as e:  # e.g. a compression tifffile cannot decode chunk-wise
            tf.close()
            logger.info("Reading %s with dask_image.imread because tifffile cannot read it lazily: %s", path, e)
            image = imread(str(path), **imread_kwargs)
            axes = "Q" * (image.ndim - 2) + "YX"
            labels = None
            read_by_tifffile = False
    image = da.moveaxis(image, [axes.index("Y"), axes.index("X")], [-2, -1])
    leading_shape = tuple(image.shape[:-2])
    return image.reshape((math.prod(leading_shape), *image.shape[-2:])), leading_shape, labels, read_by_tifffile


class _ContiguousTiff:
    """Array-like view of uncompressed contiguous tiff data, memory-mapped anew for every chunk.

    Handing an ``np.memmap`` to :func:`dask.array.from_array` would read the whole file (dask hashes and slices
    numpy arrays eagerly), so dask only gets this lightweight object and a name derived from the file.
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
        return np.asarray(data[key], dtype=self.dtype)


def spatial_chunks(chunks: Any, default: Sequence[int]) -> tuple[int, int]:
    """The ``(y, x)`` sizes of a raster model ``chunks`` argument, ``default`` when it does not give plain sizes."""
    if isinstance(chunks, int):
        return chunks, chunks
    if isinstance(chunks, Mapping) and isinstance(chunks.get("y"), int) and isinstance(chunks.get("x"), int):
        return chunks["y"], chunks["x"]
    if isinstance(chunks, list | tuple) and len(chunks) >= 2 and all(isinstance(c, int) for c in chunks[-2:]):
        return chunks[-2], chunks[-1]
    return default[-2], default[-1]
