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


def fusion_scan_profile_json(wells: list[list[tuple[str, str]]]) -> str:
    """A Fusion ``ScanProfile`` JSON; each well is a list of ``(markerName, channel)``."""
    items = [[{"markerName": marker, "channel": channel, "dye": channel} for marker, channel in well] for well in wells]
    wells_json = [{"wellName": f"A{i + 1}", "items": well} for i, well in enumerate(items)]
    return json.dumps({"experimentDescription": {"formatVersion": 2, "wells": wells_json}})


def cim_experiment_v4_xml(cycles: list[list[str]]) -> str:
    """A CODEX Instrument Manager ``ExperimentV4`` document; ``cycles`` lists the marker names per channel."""
    cycle_xml = "".join(
        f"<Cycle><Index>{i}</Index><Channels>"
        + "".join(
            f"<Channel><Index>{j}</Index><MarkerName>{marker}</MarkerName>"
            f"<MarkerClass>{'Blank' if marker == 'BLANK' else 'Other'}</MarkerClass>"
            f"<ExposureTime_ms>100</ExposureTime_ms><Wavelength_nm>{358 + 100 * j}</Wavelength_nm></Channel>"
            for j, marker in enumerate(markers, start=1)
        )
        + "</Channels></Cycle>"
        for i, markers in enumerate(cycles, start=1)
    )
    return f"<ExperimentV4><Resolution_nm>377.39</Resolution_nm><Cycles>{cycle_xml}</Cycles></ExperimentV4>"


def write_qptiff(
    path: Path,
    *,
    biomarkers: list[str | None],
    names: list[str] | None = None,
    shape: tuple[int, int] = (256, 320),
    dtype: str = "uint8",
    levels: int = 2,
    software: str = "Fusion 2.3.1",
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

    Page order mirrors real files: full-resolution pages, an RGB thumbnail, the reduced-resolution groups (the last
    one stripped), an RGB overview and an RGB label.
    """
    rng = np.random.default_rng(0)

    def random(size: tuple[int, ...]) -> np.ndarray:
        if np.dtype(dtype).kind == "f":  # unmixed component data
            return (rng.random(size) * 1000).astype(dtype)
        return rng.integers(0, np.iinfo(dtype).max, size, dtype=dtype)

    n_pages = 1 if rgb else len(biomarkers)
    names = names or [b if b is not None else f"Channel {i + 1}" for i, b in enumerate(biomarkers)]
    resolution = None if resolution_px_per_cm is None else (resolution_px_per_cm, resolution_px_per_cm)

    def description(image_type: str, name: str, biomarker: str | None = None, profile: str | None = None) -> str:
        fields = {
            "AcquisitionSoftware": software,
            "ImageType": image_type,
            "SlideID": slide_id,
            "ExposureTime": "450000",
            "Name": name,
            "Color": "255,255,0",
            "Biomarker": biomarker,
            "ScanProfile": profile,
        }
        tags = "".join(f"<{tag}>{text}</{tag}>\r\n" for tag, text in fields.items() if text is not None)
        root = "PerkinElmer-QPI-ImageDescription"
        return f'<?xml version="1.0" encoding="utf-16"?>\r\n<{root}>\r\n{tags}</{root}>'

    def write_page(tw: tifffile.TiffWriter, data: np.ndarray, desc: str, subfiletype: int = 0, **kwargs: Any) -> None:
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

    def rgb_page() -> np.ndarray:
        return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    extratags: list[Any] = []
    for code, value in zip((286, 287), stage_position_cm or (), strict=False):
        extratags.append((code, 5, 1, (round(value * 1e6), 1_000_000), False))

    data = random((*shape, 3) if rgb else (n_pages, *shape))
    with tifffile.TiffWriter(path, bigtiff=bigtiff) as tw:
        for i in range(n_pages):
            desc = description("FullResolution", names[i], biomarkers[i], scan_profile if i == 0 else None)
            write_page(tw, data if rgb else data[i], desc, tile=tile, extratags=extratags if i == 0 else None)
        write_page(tw, rgb_page(), description("Thumbnail", "Thumbnail"))
        for level in range(1, levels):
            side = (shape[0] // 2**level, shape[1] // 2**level)
            for i in range(n_pages):
                page = random((*side, 3) if rgb else side)
                desc = description("ReducedResolution", names[i], biomarkers[i])
                write_page(tw, page, desc, subfiletype=1, tile=None if level == levels - 1 else tile)
        write_page(tw, rgb_page(), description("Overview", "Overview"))
        if label_page:
            write_page(tw, rgb_page(), description("Label", "Label"))
    return data


def write_fcs(path: Path, df: pd.DataFrame, marker_names: dict[str, str] | None = None) -> None:
    """Write a minimal FCS 3.0 file: little-endian float32, ``$PnN`` = column name, ``$PnS`` = marker name."""
    marker_names = marker_names or {}
    data = np.ascontiguousarray(df.to_numpy(dtype="<f4")).tobytes()

    def text_segment(data_begin: int, data_end: int) -> bytes:
        pairs = [("$BEGINDATA", f"{data_begin:>12}"), ("$ENDDATA", f"{data_end:>12}"), ("$PAR", str(df.shape[1]))]
        pairs += [("$TOT", str(len(df))), ("$DATATYPE", "F"), ("$BYTEORD", "1,2,3,4"), ("$MODE", "L")]
        pairs.append(("$NEXTDATA", "0"))
        for i, col in enumerate(df.columns, start=1):
            pairs += [(f"$P{i}N", str(col)), (f"$P{i}B", "32"), (f"$P{i}E", "0,0"), (f"$P{i}R", "262144")]
            if col in marker_names:
                pairs.append((f"$P{i}S", marker_names[col]))
        return ("/" + "/".join(f"{k}/{v}" for k, v in pairs) + "/").encode("ascii")

    text_start = 58
    text_end = text_start + len(text_segment(0, 0)) - 1  # the offsets are fixed-width
    data_begin, data_end = text_end + 1, text_end + len(data)
    header = f"FCS3.0    {text_start:>8}{text_end:>8}{data_begin:>8}{data_end:>8}{0:>8}{0:>8}".encode("ascii")
    path.write_bytes(header.ljust(text_start) + text_segment(data_begin, data_end) + data)


def write_legacy_processor_tree(
    root: Path,
    *,
    regions: int = 1,
    cycles: int = 2,
    channels: int = 4,
    shape: tuple[int, int] = (128, 160),
    layout: Literal["per_marker", "hyperstack", "old_flat"] = "per_marker",
    fcs: bool = True,
    csv: bool = True,
    n_cells: int = 50,
) -> dict[str, Any]:
    """Write a CODEX Processor output tree and return ``{"markers", "images", "cells"}``.

    ``images`` maps region name to ``(n_markers, y, x)`` for ``per_marker`` and ``old_flat`` and to
    ``(cycles, channels, y, x)`` for ``hyperstack``. ``cells`` is the table written for the last region.
    """
    n_markers = cycles * channels
    markers = ["DAPI" if i % channels == 0 else f"MARKER{i}" for i in range(n_markers)]
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    processed = root / "processed_2020-01-01"
    channel_cols = [f"cyc{c + 1:03d}_ch{ch + 1:03d}" for c in range(cycles) for ch in range(channels)]
    images: dict[str, np.ndarray] = {}
    cells = pd.DataFrame()

    if layout != "old_flat":
        (root / "channelNames.txt").write_text("\n".join(markers) + "\n")
        (root / "experiment.json").write_text(json.dumps({"numCycles": cycles, "numChannels": channels}))

    for r in range(1, regions + 1):
        name = f"reg{r:03d}"
        stack = rng.integers(0, 2**16, (n_markers, *shape), dtype=np.uint16)
        cells = pd.DataFrame(
            {
                "cell_id": np.arange(1, n_cells + 1, dtype=float),
                "region": float(r),
                "x": rng.uniform(0, shape[1], n_cells),
                "y": rng.uniform(0, shape[0], n_cells),
                "size": rng.uniform(10, 100, n_cells),
                **{col: rng.uniform(0, 1000, n_cells) for col in channel_cols},
            }
        )
        table_dir = processed / "segm" / "segm-1" / "fcs" / "compensated"
        table_stem = f"{name}_compensated"
        if layout == "per_marker":
            images[name] = stack
            image_dir = processed / "stitched" / name
            image_dir.mkdir(parents=True, exist_ok=True)
            for i, marker in enumerate(markers):
                cyc, ch = divmod(i, channels)
                tifffile.imwrite(image_dir / f"{name}_cyc{cyc + 1:03d}_ch{ch + 1:03d}_{marker}.tif", stack[i])
        elif layout == "hyperstack":
            images[name] = stack.reshape(cycles, channels, *shape)
            (processed / "stitched").mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                processed / "stitched" / f"{name}_montage.tif", images[name], imagej=True, metadata={"axes": "TCYX"}
            )
        else:
            images[name] = stack
            tifffile.imwrite(root / f"exp_{name}.tif", stack[0])
            table_dir, table_stem = root, f"exp_{name}"
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
