"""Synthetic Akoya (PhenoCycler-Fusion qptiff, CODEX processor) inputs for the reader tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import tifffile

from spatialdata_io.readers._utils import _akoya

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_XML_DECLARATION = '<?xml version="1.0" encoding="utf-16"?>'


def qpi_xml(
    *,
    image_type: str,
    name: str,
    biomarker: str | None = None,
    software: str = "Fusion 2.3.1",
    description_version: int = 6,
    exposure_us: int = 450000,
    color: str = "255,255,0",
    scan_profile: str | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    """Build a ``PerkinElmer-QPI-ImageDescription`` document (CRLF line endings, utf-16 declaration)."""
    fields = [
        ("DescriptionVersion", str(description_version)),
        ("AcquisitionSoftware", software),
        ("ImageType", image_type),
        ("Identifier", "00000000-0000-0000-0000-000000000000"),
        ("SlideID", "SLIDE1"),
        ("IsUnmixedComponent", "False"),
        ("ExposureTime", str(exposure_us)),
        ("SignalUnits", "64"),
        ("Name", name),
        ("Color", color),
    ]
    if biomarker is not None:
        fields.append(("Biomarker", biomarker))
    fields.append(("ScaleFactor", "2.0001"))
    fields.extend((extra or {}).items())
    if scan_profile is not None:
        fields.append(("ScanProfile", scan_profile))
    lines = [_XML_DECLARATION, "<PerkinElmer-QPI-ImageDescription>"]
    lines += [f"  <{tag}>{text}</{tag}>" for tag, text in fields]
    lines.append("</PerkinElmer-QPI-ImageDescription>")
    return "\r\n".join(lines)


def fusion_scan_profile_json(wells: list[list[tuple[str, str]]]) -> str:
    """Build a Fusion ``ScanProfile`` JSON (formatVersion 2); each well is a list of ``(markerName, channel)``."""
    channels = sorted({channel for well in wells for _, channel in well})
    profile = {
        "binning": 2,
        "experimentDescription": {
            "name": "synthetic",
            "formatVersion": 2,
            "tissueType": "HumanFFPE",
            "channels": [{"name": c, "defaultExposure": 100} for c in channels],
            "wells": [
                {
                    "wellName": f"A{i + 1}",
                    "items": [
                        {
                            "id": "00000000-0000-0000-0000-000000000000",
                            "inventoryItemType": "Marker",
                            "markerName": marker,
                            "channel": channel,
                            "dye": channel,
                            "exposure": 100,
                        }
                        for marker, channel in well
                    ],
                }
                for i, well in enumerate(wells)
            ],
        },
    }
    return json.dumps(profile, separators=(",", ":"))


def cim_experiment_v4_xml(cycles: list[list[str]], resolution_nm: float = 377.39) -> str:
    """Build a CODEX Instrument Manager ``ExperimentV4`` document; ``cycles`` lists marker names per channel."""
    cycle_xml = []
    for i, markers in enumerate(cycles, start=1):
        channels = "".join(
            f"<Channel><Index>{j}</Index><MarkerName>{marker}</MarkerName>"
            f"<MarkerClass>{'Nuclear' if marker == 'DAPI' else 'Blank' if marker == 'BLANK' else 'Other'}</MarkerClass>"
            f"<ExposureTime_ms>100</ExposureTime_ms><Wavelength_nm>{358 + 100 * j}</Wavelength_nm></Channel>"
            for j, marker in enumerate(markers, start=1)
        )
        cycle_xml.append(f"<Cycle><Index>{i}</Index><Channels>{channels}</Channels></Cycle>")
    return (
        "<ExperimentV4>"
        "<Name>synthetic</Name>"
        f"<NumPlanes>{len(cycles)}</NumPlanes>"
        f"<Resolution_nm>{resolution_nm}</Resolution_nm>"
        f"<Cycles>{''.join(cycle_xml)}</Cycles>"
        "</ExperimentV4>"
    )


def write_qptiff(
    path: Path,
    *,
    biomarkers: list[str | None],
    names: list[str] | None = None,
    shape: tuple[int, int] = (256, 320),
    dtype: str = "uint8",
    levels: int = 2,
    software: str = "Fusion 2.3.1",
    description_version: int = 6,
    scan_profile: str | None = None,
    resolution_px_per_cm: float | None = 1e4 / 0.5069,
    resolution_unit: str = "CENTIMETER",
    stage_position_cm: tuple[float, float] | None = None,
    label_page: bool = True,
    rgb: bool = False,
    bigtiff: bool = True,
    tile: tuple[int, int] | None = (64, 64),
    slide_id: str = "SLIDE1",
) -> np.ndarray:
    """Write a synthetic QPTIFF and return the level-0 data, ``(c, y, x)`` or ``(y, x, 3)`` when ``rgb``.

    Page order mirrors real files: full-resolution pages, an RGB thumbnail, the reduced-resolution
    groups (the last one stripped), an RGB overview and an RGB label.
    """
    rng = np.random.default_rng(0)

    def random(size: tuple[int, ...]) -> np.ndarray:
        if np.dtype(dtype).kind == "f":  # unmixed component data
            return (rng.random(size) * 1000).astype(dtype)
        return rng.integers(0, np.iinfo(dtype).max, size, dtype=dtype)

    n_pages = 1 if rgb else len(biomarkers)
    names = (
        names if names is not None else [b if b is not None else f"Channel {i + 1}" for i, b in enumerate(biomarkers)]
    )
    resolution = None if resolution_px_per_cm is None else (resolution_px_per_cm, resolution_px_per_cm)
    shape_c = (*shape, 3) if rgb else shape

    def description(image_type: str, name: str, biomarker: str | None = None, *, first: bool = False) -> str:
        xml = qpi_xml(
            image_type=image_type,
            name=name,
            biomarker=biomarker,
            software=software,
            description_version=description_version,
            scan_profile=scan_profile if first else None,
        )
        return xml.replace("<SlideID>SLIDE1</SlideID>", f"<SlideID>{slide_id}</SlideID>")

    def write_page(
        tw: tifffile.TiffWriter, data: np.ndarray, desc: str, *, subfiletype: int = 0, **kwargs: Any
    ) -> None:
        tw.write(
            data,
            software="PerkinElmer-QPI",
            description=desc,
            resolution=resolution,
            resolutionunit=resolution_unit if resolution is not None else None,
            subfiletype=subfiletype,
            photometric="rgb" if data.ndim == 3 else "minisblack",
            metadata=None,
            **kwargs,
        )

    def rgb_page(side: tuple[int, int] = (64, 64)) -> np.ndarray:
        return rng.integers(0, 256, (*side, 3), dtype=np.uint8)

    extratags: list[Any] = []
    if stage_position_cm is not None:
        for code, value in zip((286, 287), stage_position_cm, strict=True):
            extratags.append((code, 5, 1, (round(value * 1e6), 1_000_000), False))

    data = random(shape_c if rgb else (n_pages, *shape))
    with tifffile.TiffWriter(path, bigtiff=bigtiff) as tw:
        for i in range(n_pages):
            page = data if rgb else data[i]
            write_page(
                tw,
                page,
                description("FullResolution", names[i], biomarkers[i], first=i == 0),
                tile=tile,
                extratags=extratags if i == 0 else None,
            )
        write_page(tw, rgb_page(), description("Thumbnail", "Thumbnail"))
        for level in range(1, levels):
            side = (shape[0] // 2**level, shape[1] // 2**level)
            is_last = level == levels - 1
            for i in range(n_pages):
                page = random((*side, 3) if rgb else side)
                write_page(
                    tw,
                    page,
                    description("ReducedResolution", names[i], biomarkers[i]),
                    subfiletype=1,
                    tile=None if is_last else tile,
                )
        write_page(tw, rgb_page(), description("Overview", "Overview"))
        if label_page:
            write_page(tw, rgb_page(), description("Label", "Label"))
    return data


def write_fcs(path: Path, df: pd.DataFrame, marker_names: dict[str, str] | None = None) -> None:
    """Write a minimal FCS 3.0 file: little-endian float32, ``$PnN`` = column name, ``$PnS`` = marker name."""
    marker_names = marker_names or {}
    values = np.ascontiguousarray(df.to_numpy(dtype="<f4"))
    data = values.tobytes()
    n_par = df.shape[1]

    def text_segment(data_begin: int, data_end: int) -> bytes:
        pairs: list[tuple[str, str]] = [
            ("$BEGINDATA", f"{data_begin:>12}"),
            ("$ENDDATA", f"{data_end:>12}"),
            ("$PAR", str(n_par)),
            ("$TOT", str(len(df))),
            ("$DATATYPE", "F"),
            ("$BYTEORD", "1,2,3,4"),
            ("$MODE", "L"),
            ("$NEXTDATA", "0"),
        ]
        for i, col in enumerate(df.columns, start=1):
            pairs += [(f"$P{i}N", str(col)), (f"$P{i}B", "32"), (f"$P{i}E", "0,0"), (f"$P{i}R", "262144")]
            if col in marker_names:
                pairs.append((f"$P{i}S", marker_names[col]))
        return ("/" + "/".join(f"{k}/{v}" for k, v in pairs) + "/").encode("ascii")

    text_start = 58
    text_len = len(text_segment(0, 0))  # offsets are fixed-width, so the length does not depend on their value
    text_end = text_start + text_len - 1
    data_begin = text_end + 1
    data_end = data_begin + len(data) - 1
    header = f"FCS3.0    {text_start:>8}{text_end:>8}{data_begin:>8}{data_end:>8}{0:>8}{0:>8}".encode("ascii")
    path.write_bytes(header.ljust(text_start) + text_segment(data_begin, data_end) + data)


def write_legacy_processor_tree(
    root: Path,
    *,
    regions: int = 1,
    cycles: int = 2,
    channels: int = 4,
    markers: list[str] | None = None,
    shape: tuple[int, int] = (128, 160),
    layout: Literal["per_marker", "hyperstack", "old_flat"] = "per_marker",
    fcs: bool = True,
    csv: bool = True,
    n_cells: int = 50,
) -> dict[str, Any]:
    """Write a CODEX Processor output tree and return ``{"markers", "images", "cells"}``.

    ``markers`` has one entry per (cycle, channel), cycle-major. ``images`` maps region name to
    ``(n_markers, y, x)`` for ``per_marker`` and ``old_flat`` and to ``(cycles, channels, y, x)`` for
    ``hyperstack``. ``cells`` is the table written for the last region.
    """
    n_markers = cycles * channels
    if markers is None:
        markers = ["DAPI" if c == 0 else f"MARKER{i}" for i in range(n_markers) for c in [i % channels]]
    if len(markers) != n_markers:
        raise ValueError(f"expected {n_markers} markers (cycles * channels), got {len(markers)}")
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    processed = root / "processed_2020-01-01"
    channel_cols = [f"cyc{c + 1:03d}_ch{ch + 1:03d}" for c in range(cycles) for ch in range(channels)]
    images: dict[str, np.ndarray] = {}
    cells = pd.DataFrame()

    if layout != "old_flat":
        (root / "channelNames.txt").write_text("\n".join(markers) + "\n")
        (root / "experiment.json").write_text(
            json.dumps({"name": "synthetic", "numCycles": cycles, "numChannels": channels, "regions": regions})
        )

    for r in range(1, regions + 1):
        name = f"reg{r:03d}"
        stack = rng.integers(0, 2**16, (n_markers, *shape), dtype=np.uint16)
        cells = pd.DataFrame(
            {
                "cell_id": np.arange(1, n_cells + 1, dtype=float),
                "region": float(r),
                "tile_num": 1.0,
                "x": rng.uniform(0, shape[1], n_cells),
                "y": rng.uniform(0, shape[0], n_cells),
                "z": 1.0,
                "size": rng.uniform(10, 100, n_cells),
                **{col: rng.uniform(0, 1000, n_cells) for col in channel_cols},
            }
        )
        if layout == "per_marker":
            images[name] = stack
            image_dir = processed / "stitched" / name
            image_dir.mkdir(parents=True, exist_ok=True)
            for i, marker in enumerate(markers):
                cyc, ch = divmod(i, channels)
                tifffile.imwrite(image_dir / f"{name}_cyc{cyc + 1:03d}_ch{ch + 1:03d}_{marker}.tif", stack[i])
            table_dir = processed / "segm" / "segm-1" / "fcs" / "compensated"
            table_stem = f"{name}_compensated"
        elif layout == "hyperstack":
            images[name] = stack.reshape(cycles, channels, *shape)
            (processed / "stitched").mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                processed / "stitched" / f"{name}_montage.tif", images[name], imagej=True, metadata={"axes": "TCYX"}
            )
            table_dir = processed / "segm" / "segm-1" / "fcs" / "compensated"
            table_stem = f"{name}_compensated"
        else:
            images[name] = stack
            tifffile.imwrite(root / f"exp_{name}.tif", stack[0])
            table_dir = root
            table_stem = f"exp_{name}"
        table_dir.mkdir(parents=True, exist_ok=True)
        if fcs:
            write_fcs(table_dir / f"{table_stem}.fcs", cells, dict(zip(channel_cols, markers, strict=True)))
        if csv and not (layout == "old_flat" and fcs):
            cells.to_csv(table_dir / f"{table_stem}.csv", index=False)
    return {"markers": markers, "images": images, "cells": cells}


def forbid_pixel_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every read of pixel data fail: the chunk reads of the lazy tiff arrays and tifffile's own decoding."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("pixel data read eagerly")

    monkeypatch.setattr(_akoya._TiffZarrSource, "__getitem__", fail)
    monkeypatch.setattr(_akoya._ContiguousTiff, "__getitem__", fail)
    monkeypatch.setattr(tifffile.TiffPage, "decode", property(lambda self: fail))
    monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    monkeypatch.setattr(tifffile.TiffFile, "asarray", fail)
