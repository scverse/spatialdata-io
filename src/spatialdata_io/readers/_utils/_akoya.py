"""Channel, transformation, image and metadata helpers shared by the Akoya readers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from spatialdata._logging import logger
from spatialdata.models import Image2DModel
from spatialdata.transformations import BaseTransformation, Identity, Translation

from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs

if TYPE_CHECKING:
    import dask.array as da
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = [
    "apply_channel_names",
    "build_image_element",
    "dedupe_channel_names",
    "region_transformations",
    "scans_to_attrs",
    "select_regions",
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


def region_transformations(scans: Sequence[AkoyaScan]) -> dict[str, BaseTransformation]:
    """Compute the transformation to the ``"global"`` coordinate system of every region.

    Regions are placed with a pixel ``Translation`` derived from their stage positions only if every scan has a
    stage position and a pixel size and all scans share the same slide. Otherwise all regions get ``Identity``.
    """
    regions = [scan.region for scan in scans]
    placeable = (
        len(scans) > 1
        and all(scan.stage_position_um is not None and scan.pixel_size_um for scan in scans)
        and len({scan.slide_id for scan in scans}) == 1
        and scans[0].slide_id is not None
    )
    if not placeable:
        if len(scans) > 1:
            logger.info(
                "Regions %s have no common stage frame (missing stage position or pixel size, or different "
                "slides) and are all placed at the origin of 'global', so they overlap. Use `regions=` to read a "
                "single region or `spatialdata.transform` to arrange them.",
                regions,
            )
        return {region: Identity() for region in regions}

    positions = [scan.stage_position_um for scan in scans]
    min_x = min(position[0] for position in positions if position is not None)
    min_y = min(position[1] for position in positions if position is not None)
    result: dict[str, BaseTransformation] = {}
    for scan in scans:
        x, y = scan.stage_position_um  # type: ignore[misc]
        pixel_size = scan.pixel_size_um
        result[scan.region] = Translation([(x - min_x) / pixel_size, (y - min_y) / pixel_size], axes=("x", "y"))  # type: ignore[operator]
    return result


def build_image_element(
    image: da.Array,
    scan: AkoyaScan,
    c_coords: list[str],
    transformation: BaseTransformation,
    image_models_kwargs: Mapping[str, Any],
) -> DataArray | DataTree:
    """Parse a lazy ``(c, y, x)`` array, or ``(y, x, 3)`` for brightfield RGB scans, into an ``Image2DModel``."""
    image_models_kwargs, _ = _initialize_raster_models_kwargs(image_models_kwargs, {})
    if scan.rgb:
        image = image.transpose(2, 0, 1)
        c_coords = ["r", "g", "b"]
    return Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        c_coords=c_coords,
        rgb=scan.rgb,
        transformations={"global": transformation},
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


def select_regions(available: Sequence[str], regions: Sequence[str] | None) -> list[str]:
    """Return the requested regions (all if ``None``), raising if one is not available."""
    if regions is None:
        return list(available)
    unknown = [region for region in regions if region not in available]
    if unknown:
        raise ValueError(f"Unknown region(s) {unknown}. Available regions: {list(available)}.")
    return list(regions)
