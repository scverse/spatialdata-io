"""PerkinElmer/Akoya QPTIFF metadata and lazy level-0 image access, shared by the Akoya readers."""

from __future__ import annotations

import json
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import dask.array as da
import tifffile
from spatialdata import sanitize_name

from spatialdata_io._constants._constants import PhenocyclerKeys
from spatialdata_io.readers._utils._akoya import lazy_tiff_array

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "AkoyaChannel",
    "AkoyaScan",
    "qpi_pixel_size_um",
    "qptiff_base_name",
    "qptiff_regions",
    "qptiff_regions_with_duplicates",
    "read_qptiff_image",
    "read_qptiff_metadata",
]

_FULL_RESOLUTION = "FullResolution"
_SCAN_SUFFIX = re.compile(r"_Scan\d+$")
_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")
# micrometres per unit of the TIFF ResolutionUnit; XPosition/YPosition are expressed in the same unit
_UM_PER_RESOLUTION_UNIT = {tifffile.RESUNIT.CENTIMETER: 1e4, tifffile.RESUNIT.INCH: 25400.0}


@dataclass
class AkoyaChannel:
    """One image channel of an Akoya acquisition.

    ``name`` is the marker identity before de-duplication (``Biomarker`` > ``Name`` > ``channel_{i}``).
    ``cycle`` and ``channel`` (both 1-based) come from the scan profile and stay ``None`` when it does not
    describe this channel. ``extra`` holds further JSON-serialisable metadata.
    """

    name: str
    fluor: str | None = None
    cycle: int | None = None
    channel: int | None = None
    exposure_ms: float | None = None
    color: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AkoyaScan:
    """Metadata of one Akoya image (a QPTIFF scan or a legacy CODEX Processor region).

    ``generation`` is one of ``"fusion1"``, ``"fusion2"``, ``"cim_qptiff"``, ``"qpi_other"`` or
    ``"legacy_processor"``. ``pixel_size_um`` is the calibrated pixel size, ``nominal_pixel_size_um`` the one
    the acquisition software reports. ``stage_position_um`` is the ``(x, y)`` stage position of the upper-left
    corner. ``raw`` keeps the unparsed source metadata (JSON-serialisable).
    """

    path: Path
    region: str
    generation: str
    software: str | None
    slide_id: str | None
    pixel_size_um: float | None
    nominal_pixel_size_um: float | None
    stage_position_um: tuple[float, float] | None
    channels: list[AkoyaChannel]
    rgb: bool
    raw: dict[str, Any]


def read_qptiff_metadata(path: Path, region: str | None = None) -> AkoyaScan:
    """Read the metadata of a PerkinElmer/Akoya QPTIFF.

    Channels are the ``FullResolution`` pages in file order; each page's own XML is authoritative for the
    channel identity, while the page-0 ``ScanProfile`` (Fusion JSON or CIM ``ExperimentV4``) only adds cycle,
    channel and reagent information.

    Parameters
    ----------
    path
        Path to the ``.qptiff`` file.
    region
        Region name; defaults to the file name without the ``.qptiff``/``.er.qptiff`` suffix.

    Returns
    -------
    The scan metadata.
    """
    with _open_qpi(path) as tf:
        page0 = _page(tf, 0)
        page_xml = [_parse_description(_page(tf, i).description) for i in range(len(tf.pages))]
        full_res = [i for i, xml in enumerate(page_xml) if _text(xml, "ImageType") == _FULL_RESOLUTION]
        if not full_res:
            raise ValueError(f"{path} has no page with <ImageType>{_FULL_RESOLUTION}</ImageType>.")
        rgb = len(full_res) == 1 and _page(tf, full_res[0]).samplesperpixel == 3
        pixel_size_um = qpi_pixel_size_um(page0)
        stage_position_um = _stage_position_um(page0)
        page0_description = page0.description

    root = page_xml[0]
    software = _text(root, "AcquisitionSoftware")
    experiment_v4 = root.find("ScanProfile/ExperimentV4")
    scan_profile = _scan_profile(root)
    channels = [_channel(page_xml[page_index], i, page_index) for i, page_index in enumerate(full_res)]
    if isinstance(scan_profile, dict):
        _enrich(channels, _fusion_profile_records(scan_profile))
    elif experiment_v4 is not None:
        _enrich(channels, _experiment_v4_records(experiment_v4))

    if pixel_size_um is None:
        warnings.warn(
            f"Could not determine the pixel size of {path}; pixel_size_um is None.", UserWarning, stacklevel=2
        )

    return AkoyaScan(
        path=path,
        region=region if region is not None else qptiff_base_name(path),
        generation=_generation(software, experiment_v4 is not None),
        software=software,
        slide_id=_text(root, "SlideID"),
        pixel_size_um=pixel_size_um,
        nominal_pixel_size_um=_nominal_pixel_size_um(root),
        stage_position_um=stage_position_um,
        channels=channels,
        rgb=rgb,
        raw={
            "page0_xml": page0_description,
            "scan_profile": scan_profile,
            "image_types": dict(Counter(_text(xml, "ImageType") or "unknown" for xml in page_xml)),
        },
    )


def read_qptiff_image(path: Path) -> da.Array:
    """Lazily read the full-resolution image of a PerkinElmer/Akoya QPTIFF.

    The array keeps the native tile chunks of the file, opens the file only when its chunks are read and can be
    pickled, e.g. for ``dask.distributed``.

    Parameters
    ----------
    path
        Path to the ``.qptiff`` file.

    Returns
    -------
    ``(c, y, x)`` array, or ``(y, x, 3)`` for a brightfield RGB scan.
    """
    with _open_qpi(path) as tf:
        full_res = [
            i
            for i in range(len(tf.pages))
            if _text(_parse_description(_page(tf, i).description), "ImageType") == _FULL_RESOLUTION
        ]
        if not full_res:
            raise ValueError(f"{path} has no page with <ImageType>{_FULL_RESOLUTION}</ImageType>.")
        first = _page(tf, full_res[0])
        page_shape = first.shape
        if len(full_res) == 1 and first.samplesperpixel == 3:
            expected_shape: tuple[int, ...] = page_shape
        else:
            expected_shape = (len(full_res), *page_shape)

    image = lazy_tiff_array(path, series=0)
    if image.shape == expected_shape:
        return image
    # tifffile groups pages into series by shape heuristics; fall back to the pages the XML says are full-res
    pages = [lazy_tiff_array(path, page=i) for i in full_res]
    return pages[0] if len(expected_shape) == len(page_shape) else da.stack(pages)


def qpi_pixel_size_um(page: tifffile.TiffPage) -> float | None:
    """Return the pixel size of a QPTIFF page in micrometres, or ``None`` when it is unknown.

    The calibrated ``XResolution`` is used when ``ResolutionUnit`` is centimetre. Otherwise (e.g. the
    meaningless 96 dpi written when the resolution is unknown) the nominal ``PixelSizeMicrons`` or the CIM
    ``Resolution_nm`` from the page XML is returned, if present.
    """
    unit = page.tags.get("ResolutionUnit")
    resolution = page.tags.get("XResolution")
    if unit is not None and resolution is not None and unit.value == tifffile.RESUNIT.CENTIMETER:
        numerator, denominator = resolution.value
        if numerator > 0 and denominator > 0:
            return 1e4 * denominator / numerator
    return _nominal_pixel_size_um(_parse_description(page.description))


def _open_qpi(path: Path) -> tifffile.TiffFile:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist or is not a file.")
    tf = tifffile.TiffFile(path)
    if not tf.is_qpi:
        tf.close()
        raise ValueError(
            f"{path} is not a PerkinElmer/Akoya QPTIFF (Software tag does not start with 'PerkinElmer-QPI')."
        )
    return tf


def _page(tf: tifffile.TiffFile, index: int) -> tifffile.TiffPage:
    page = tf.pages[index]
    if not isinstance(page, tifffile.TiffPage):  # only frames when ``useframes`` is set, which we never do
        raise TypeError(f"Expected a TiffPage at index {index}, got {type(page).__name__}.")
    return page


def _parse_description(description: str) -> ET.Element:
    # the declaration claims utf-16 but the tag is ASCII, which ElementTree rejects
    return ET.fromstring(_XML_DECLARATION.sub("", description, count=1))


def _text(element: ET.Element, tag: str) -> str | None:
    """Stripped text of ``tag`` (a path relative to ``element``), ``None`` when missing or empty."""
    found = element.find(tag)
    if found is None or found.text is None or not found.text.strip():
        return None
    return found.text.strip()


def _float(text: str | None) -> float | None:
    try:
        return None if text is None else float(text)
    except ValueError:
        return None


def _bool(text: str | None) -> bool | None:
    return None if text is None else text.lower() == "true"


def qptiff_base_name(path: Path) -> str:
    """The file name without the ``.er.qptiff``/``.qptiff`` suffix (in any case)."""
    for suffix in (PhenocyclerKeys.ER_QPTIFF_SUFFIX.value, PhenocyclerKeys.QPTIFF_SUFFIX.value):
        if path.name.lower().endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _is_extended_range(path: Path) -> bool:
    return path.name.lower().endswith(PhenocyclerKeys.ER_QPTIFF_SUFFIX.value)


def qptiff_regions(files: Sequence[Path]) -> dict[str, Path]:
    """The region id of every qptiff scan among ``files``, the same for both Akoya readers.

    Files of the same scan (folder and base name) are read once, preferring the extended-range ``.er.qptiff``.
    The region id is the base name without a trailing ``_Scan<N>``, unless that makes two scans share a region id.

    Parameters
    ----------
    files
        The qptiff files, in the order of the result.

    Returns
    -------
    The scan file of every region.
    """
    return qptiff_regions_with_duplicates(files)[0]


def qptiff_regions_with_duplicates(files: Sequence[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    """Like :func:`qptiff_regions`, and the 8-bit ``.qptiff`` of every region that also has an ``.er.qptiff``.

    Returns
    -------
    The scan file of every region (the ``.er.qptiff`` if there is one), and the 8-bit file of the same scan by
    region id for the regions that have both.
    """
    scans: dict[tuple[Path, str], Path] = {}
    eight_bit: dict[tuple[Path, str], Path] = {}
    for file in files:
        key = (file.parent, qptiff_base_name(file))
        if _is_extended_range(file):
            if key in scans and not _is_extended_range(scans[key]):
                eight_bit.setdefault(key, scans[key])
            scans[key] = file
        elif key not in scans:
            scans[key] = file
        else:
            eight_bit.setdefault(key, file)
    # an 8-bit file is only a duplicate if the scan is read from an extended-range file
    eight_bit = {key: file for key, file in eight_bit.items() if _is_extended_range(scans[key])}
    bases = [base for _, base in scans]
    # region ids must be valid element names; inForm names its files e.g. `<sample>_[13860,52919]_component_data.tif`
    stripped = [sanitize_name(_SCAN_SUFFIX.sub("", base) or base) for base in bases]
    # stripping `_Scan<N>` must not merge two scans into one region
    regions = [
        sanitize_name(base) if stripped.count(region) > 1 else region
        for base, region in zip(bases, stripped, strict=True)
    ]
    if len(set(regions)) < len(regions):
        duplicated = sorted({region for region in regions if regions.count(region) > 1})
        found = [str(file) for file, region in zip(scans.values(), regions, strict=True) if region in duplicated]
        raise ValueError(
            f"Several qptiff files map to the same region(s) {duplicated}: {found}. Pass the path of a single qptiff "
            "file or of a directory that contains each region once."
        )
    region_of = dict(zip(scans, regions, strict=True))
    return dict(zip(regions, scans.values(), strict=True)), {region_of[key]: file for key, file in eight_bit.items()}


def _generation(software: str | None, has_experiment_v4: bool) -> str:
    software = software or ""
    if software.startswith("Fusion 1"):
        return "fusion1"
    if software.startswith("Fusion 2"):
        return "fusion2"
    if "instrument manager" in software.lower() or has_experiment_v4:
        return "cim_qptiff"
    return "qpi_other"


def _nominal_pixel_size_um(root: ET.Element) -> float | None:
    # Vectra/Opal: ScanProfile/.../ScanResolution/PixelSizeMicrons; CIM: ScanProfile/ExperimentV4/Resolution_nm
    pixel_size = _float(_text(root, ".//PixelSizeMicrons"))
    if pixel_size is not None:
        return pixel_size
    resolution_nm = _float(_text(root, ".//Resolution_nm"))
    return None if resolution_nm is None else resolution_nm / 1000


def _stage_position_um(page: tifffile.TiffPage) -> tuple[float, float] | None:
    unit = page.tags.get("ResolutionUnit")
    um_per_unit = _UM_PER_RESOLUTION_UNIT.get(unit.value) if unit is not None else None
    positions = [page.tags.get(name) for name in ("XPosition", "YPosition")]
    if um_per_unit is None or positions[0] is None or positions[1] is None:
        return None
    (x_num, x_den), (y_num, y_den) = positions[0].value, positions[1].value
    if x_den == 0 or y_den == 0:
        return None
    return x_num / x_den * um_per_unit, y_num / y_den * um_per_unit


def _scan_profile(root: ET.Element) -> dict[str, Any] | str | None:
    """Page-0 ``ScanProfile``: parsed JSON (Fusion PhenoCycler mode) or the XML string (Opal, Vectra, CIM)."""
    element = root.find("ScanProfile")
    if element is None:
        return None
    if len(element):
        return "".join(ET.tostring(child, encoding="unicode") for child in element)
    text = (element.text or "").strip()
    if not text:
        return None
    try:
        profile = json.loads(text)
    except json.JSONDecodeError:
        return text
    return profile if isinstance(profile, dict) else text


def _channel(xml: ET.Element, index: int, page_index: int) -> AkoyaChannel:
    biomarker = _text(xml, "Biomarker")
    fluor = _text(xml, "Name")
    exposure_us = _float(_text(xml, "ExposureTime"))
    extra: dict[str, Any] = {"page_index": page_index}
    for key, tag in (
        ("is_unmixed_component", "IsUnmixedComponent"),
        ("autofluorescence_subtracted", "AutofluorescenceSubtracted"),
    ):
        value = _bool(_text(xml, tag))
        if value is not None:
            extra[key] = value
    return AkoyaChannel(
        name=biomarker or fluor or f"channel_{index}",
        fluor=fluor if biomarker else None,
        exposure_ms=None if exposure_us is None else exposure_us / 1000,
        color=_text(xml, "Color"),
        extra=extra,
    )


def _fusion_profile_records(profile: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(marker, fields)`` per item of the Fusion ``experimentDescription.wells``; one well per cycle.

    Wells and items that are not JSON objects, and ``wells``/``items`` that are not lists, are skipped.
    """
    records = []
    description = profile.get("experimentDescription")
    wells = description.get("wells") if isinstance(description, dict) else None
    for cycle, well in enumerate(_objects(wells), start=1):
        for channel, item in enumerate(_objects(well.get("items")), start=1):
            marker = item.get("markerName")
            if not isinstance(marker, str) or _is_blank(marker):
                continue
            # Fusion writes the string "None" for reagent fields that do not apply (e.g. DAPI)
            extra = {
                key: item[key]
                for key in ("reporter", "barcode", "clone", "dye")
                if item.get(key) not in (None, "", "None")
            }
            if item.get("channel"):
                extra["channel_name"] = item["channel"]
            extra["well"] = well.get("wellName")
            exposure = item.get("exposure")
            records.append(
                (
                    marker,
                    {
                        "cycle": cycle,
                        "channel": channel,
                        "exposure_ms": float(exposure) if isinstance(exposure, int | float) else None,
                        "extra": extra,
                    },
                )
            )
    return records


def _objects(value: Any) -> list[dict[str, Any]]:
    """The entries of ``value`` if it is a list, else none; entries that are not objects become empty objects.

    They are kept as empty objects so that the cycle and channel indices still count every entry.
    """
    return [entry if isinstance(entry, dict) else {} for entry in value] if isinstance(value, list) else []


def _experiment_v4_records(experiment: ET.Element) -> list[tuple[str, dict[str, Any]]]:
    """``(marker, fields)`` per ``Cycles/Cycle/Channels/Channel`` of a CIM ``ExperimentV4``."""
    records = []
    for cycle in experiment.iterfind("Cycles/Cycle"):
        cycle_index = _float(_text(cycle, "Index"))
        for channel in cycle.iterfind("Channels/Channel"):
            marker = _text(channel, "MarkerName")
            if marker is None or _is_blank(marker) or _text(channel, "MarkerClass") == "Blank":
                continue
            channel_index = _float(_text(channel, "Index"))
            extra: dict[str, Any] = {}
            for key, tag in (("dye", "DyeName"), ("marker_class", "MarkerClass")):
                value = _text(channel, tag)
                if value is not None:
                    extra[key] = value
            wavelength = _float(_text(channel, "Wavelength_nm"))
            if wavelength is not None:
                extra["wavelength_nm"] = wavelength
            records.append(
                (
                    marker,
                    {
                        "cycle": None if cycle_index is None else int(cycle_index),
                        "channel": None if channel_index is None else int(channel_index),
                        "exposure_ms": _float(_text(channel, "ExposureTime_ms")),
                        "extra": extra,
                    },
                )
            )
    return records


def _is_blank(marker: str) -> bool:
    """Whether a scan-profile marker is a blank/empty slot (Fusion ``--``, CIM ``BLANK``/``EMPTY``)."""
    return marker.strip() in {"", "--"} or marker.strip().casefold() in {"blank", "empty"}


def _name_key(name: str) -> str:
    # CIM sanitises page Biomarker names (``F4/80`` -> ``F4-80``), so compare alphanumerics only (in any script)
    return "".join(char for char in name.casefold() if char.isalnum())


def _enrich(channels: list[AkoyaChannel], records: list[tuple[str, dict[str, Any]]]) -> None:
    """Fill cycle/channel/extra of ``channels`` from the non-blank scan-profile ``records``.

    Record ``k`` enriches page ``k`` when there are as many records as pages and no pair contradicts the names.
    Otherwise channels are joined on the normalised name with occurrence index, but only for names that occur
    as often in the pages as in the profile; any other channel is left unchanged rather than guessed. The page
    XML stays authoritative: names and order never change and the page exposure wins.
    """
    for channel, (_, record) in _positional_pairs(channels, records) or _name_pairs(channels, records):
        channel.cycle = record["cycle"]
        channel.channel = record["channel"]
        if channel.exposure_ms is None:
            channel.exposure_ms = record["exposure_ms"]
        channel.extra.update({k: v for k, v in record["extra"].items() if k not in channel.extra})


def _positional_pairs(
    channels: list[AkoyaChannel], records: list[tuple[str, dict[str, Any]]]
) -> list[tuple[AkoyaChannel, tuple[str, dict[str, Any]]]]:
    if len(channels) != len(records):
        return []
    # Fusion orders the pages of a cycle differently from the well items; a page whose marker sits at another
    # position in the profile, or a record naming the marker of another page, means the orders differ and the
    # positional pairing would be wrong
    page_keys = {_name_key(channel.name) for channel in channels}
    record_keys = {_name_key(name) for name, _ in records}
    for channel, (name, _) in zip(channels, records, strict=True):
        key, record_key = _name_key(channel.name), _name_key(name)
        if key != record_key and (key in record_keys or record_key in page_keys):
            return []
    return list(zip(channels, records, strict=True))


def _name_pairs(
    channels: list[AkoyaChannel], records: list[tuple[str, dict[str, Any]]]
) -> list[tuple[AkoyaChannel, tuple[str, dict[str, Any]]]]:
    channels_by_key: dict[str, list[AkoyaChannel]] = {}
    records_by_key: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for channel in channels:
        channels_by_key.setdefault(_name_key(channel.name), []).append(channel)
    for record in records:
        records_by_key.setdefault(_name_key(record[0]), []).append(record)
    pairs: list[tuple[AkoyaChannel, tuple[str, dict[str, Any]]]] = []
    for key, keyed_channels in channels_by_key.items():
        keyed_records = records_by_key.get(key, [])
        if len(keyed_records) == len(keyed_channels):
            pairs.extend(zip(keyed_channels, keyed_records, strict=True))
    return pairs
