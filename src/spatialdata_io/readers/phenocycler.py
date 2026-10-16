from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._constants import PhenocyclerKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    build_image_element,
    read_json_object,
    region_transformations,
    scans_to_attrs,
    select_regions,
)
from spatialdata_io.readers._utils._cell_tables import (
    build_cells,
    read_region_labels,
    read_region_tables,
    resolve_per_region,
)
from spatialdata_io.readers._utils._qptiff import (
    _fusion_profile_records,
    _is_blank,
    read_qptiff_image,
    read_qptiff_metadata,
)
from spatialdata_io.readers._utils._utils import _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = ["discover_qptiffs", "phenocycler"]

_SCAN_SUFFIX = re.compile(r"_Scan\d+$")
# the Fusion writes per-cycle acquisition files to ``raw/images/phenocycler/`` (HuBMAP layout)
_ACQUISITION_DIR = ("raw", "images", "phenocycler")


@inject_docs(vx=PhenocyclerKeys)
def phenocycler(
    path: str | Path,
    regions: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    cells_table: str | Path | Mapping[str, str | Path] | None = None,
    cells_table_format: str = "auto",
    labels: str | Path | Mapping[str, str | Path] | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    cells_table_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *Akoya PhenoCycler-Fusion* 1.x/2.x data.

    This function reads the following files:

        - ``*{vx.QPTIFF_SUFFIX!r}`` / ``*{vx.ER_QPTIFF_SUFFIX!r}``: the scans, one region each. The images are read
          lazily; when both ``X{vx.QPTIFF_SUFFIX}`` and ``X{vx.ER_QPTIFF_SUFFIX}`` exist, the extended-range
          (higher bit depth) ``{vx.ER_QPTIFF_SUFFIX!r}`` is used.
        - ``*{vx.XPD_SUFFIX!r}``: optional experiment protocol (JSON) next to a scan.
        - ``*{vx.CHANNELS_CSV_SUFFIX!r}``: optional channel list next to a scan or, in the HuBMAP layout, in
          ``lab_processed/images/``.

    Sidecar files are matched to a scan by file-name prefix and stored as metadata only: the channel names always
    come from the per-page XML of the qptiff. A sidecar whose number of markers differs from the number of channels
    raises a warning. Acquisition files (``*.raw.qptiff``, ``*.qptiff.raw``, ``*.qptiff.intermediate`` and anything
    in ``raw/images/phenocycler/``) are ignored.

    The region name is the file name without the ``{vx.ER_QPTIFF_SUFFIX!r}``/``{vx.QPTIFF_SUFFIX!r}`` suffix and
    without a trailing ``_Scan<N>``, unless that makes two regions share a name. Each region is stored as the image
    ``<region>{vx.IMAGE_SUFFIX}``. Several regions scanned on the same slide are placed at their stage positions in
    the ``"global"`` coordinate system; otherwise all regions sit at its origin. The per-region metadata is stored
    in ``sdata.attrs[{vx.ATTRS_KEY!r}]``.

    A segmentation mask given in ``labels`` is stored lazily as the labels ``<region>{vx.LABELS_SUFFIX}`` with the
    transformation of the region's image. The cells of ``cells_table`` are stored in one table
    ``{vx.TABLE_KEY!r}`` (marker intensities dense float32 in ``X``, pixel centroids in ``obsm["spatial"]``,
    ``obs[{vx.REGION_KEY!r}]`` the annotated element, ``obs[{vx.INSTANCE_KEY!r}]`` the instance key), or in one
    table ``<region>_table`` per region if the regions have different markers. Cells annotate the labels of their
    region, whose values must be their instance ids; without a mask they are stored as circles
    ``<region>{vx.CELLS_SUFFIX}``, with a warning. The paths of the tables and masks are stored in
    ``sdata.attrs[{vx.ATTRS_KEY!r}]["cell_tables"]`` and ``["labels"]``.

    .. seealso::

        - `Akoya PhenoCycler-Fusion <https://www.akoyabio.com/phenocycler/>`_.

    Parameters
    ----------
    path
        Path to a ``{vx.QPTIFF_SUFFIX!r}`` file, or to a directory that is searched recursively for them.
    regions
        Regions to read. If `None`, all regions are read.
    channel_names
        Channel names to use instead of the ones stored in the qptiff; must have one entry per channel of every
        region read.
    cells_table
        A cell table: a path, or a mapping from region id to path, e.g. ``{{"A": "A_cells.csv"}}``. A single path
        for several regions needs a region column (``Image`` for QuPath, ``region`` for CODEX tables, else
        ``cells_table_kwargs["region_column"]``); its values are matched to the region ids, e.g.
        ``A_Scan1.qptiff`` to ``A``.
    cells_table_format
        The format of ``cells_table``: ``"auto"`` (detected from the header), ``"codex"`` (CODEX Processor),
        ``"qupath"`` (QuPath measurements; µm centroids are converted with the region's pixel size), ``"mcquant"``
        (MCMICRO) or ``"generic"`` (columns given by ``cells_table_kwargs``).
    labels
        A segmentation mask (TIFF/OME-TIFF with integer instance ids) per region: a path for a single region, or a
        mapping from region id to path.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.
    cells_table_kwargs
        Options for reading ``cells_table`` and ``labels``: ``region_column``; for ``"generic"`` tables ``x``,
        ``y``, ``instance_id``, ``markers`` (list or regular expression) and ``coordinates_unit`` (``"px"`` or
        ``"um"``); ``mask_channel`` (index or OME channel name of the mask in a multi-channel ``labels`` file).

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    regions = [regions] if isinstance(regions, str) else regions
    channel_names = [channel_names] if isinstance(channel_names, str) else channel_names
    path = Path(path)
    qptiffs = discover_qptiffs(path)
    selected = select_regions(list(qptiffs), regions)
    table_files = resolve_per_region(cells_table, selected)
    mask_files = resolve_per_region(labels, selected)

    scans = [read_qptiff_metadata(qptiffs[region], region=region) for region in selected]
    names = {}
    for scan in scans:
        names[scan.region] = apply_channel_names(scan, channel_names)
        _read_sidecars(scan)

    transformations = region_transformations(scans)
    images = {
        f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}": build_image_element(
            read_qptiff_image(scan.path),
            scan,
            names[scan.region],
            transformations[scan.region],
            image_models_kwargs,
        )
        for scan in scans
    }
    region_labels = read_region_labels(mask_files, transformations, labels_models_kwargs, cells_table_kwargs)
    cell_tables = read_region_tables(
        table_files,
        cells_table_format,
        available=list(qptiffs),
        selected=selected,
        pixel_sizes={scan.region: scan.pixel_size_um for scan in scans},
        cells_table_kwargs=cells_table_kwargs,
    )
    tables, shapes = build_cells(cell_tables, {}, region_labels, transformations) if cell_tables else ({}, {})
    sdata = SpatialData(
        images=images,
        labels={f"{region}{PhenocyclerKeys.LABELS_SUFFIX}": element for region, element in region_labels.items()},
        shapes=shapes,
        tables=tables,
    )
    attrs = scans_to_attrs(scans, names)
    if table_files:
        attrs["cell_tables"] = {key: str(file) for key, file in table_files.items()}
    if mask_files:
        attrs["labels"] = {region: str(file) for region, file in mask_files.items()}
    sdata.attrs[PhenocyclerKeys.ATTRS_KEY.value] = attrs
    return _set_reader_metadata(sdata, "phenocycler")


def discover_qptiffs(path: Path) -> dict[str, Path]:
    """Find the qptiff scans of a PhenoCycler-Fusion dataset.

    Parameters
    ----------
    path
        A ``.qptiff`` file, or a directory that is searched recursively.

    Returns
    -------
    The scan file of every region, sorted by path.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if path.is_file():
        return {_region(_base_name(path)): path}

    # one scan per (directory, base name), preferring the extended-range `.er.qptiff`
    scans: dict[tuple[Path, str], Path] = {}
    for qptiff in sorted(path.rglob(f"*{PhenocyclerKeys.QPTIFF_SUFFIX}")):
        if not qptiff.is_file() or _is_acquisition_file(qptiff):
            continue
        key = (qptiff.parent, _base_name(qptiff))
        if key not in scans or _is_extended_range(qptiff):
            scans[key] = qptiff
    if not scans:
        raise FileNotFoundError(
            f"No .qptiff files found in {path} (searched recursively for '*.qptiff', excluding acquisition files)."
        )

    bases = [base for _, base in scans]
    regions = [_region(base) for base in bases]
    # stripping `_Scan<N>` must not merge two scans into one region
    regions = [base if regions.count(region) > 1 else region for base, region in zip(bases, regions, strict=True)]
    if len(set(regions)) < len(regions):
        duplicated = sorted({region for region in regions if regions.count(region) > 1})
        raise ValueError(
            f"Several qptiff files in {path} map to the same region(s) {duplicated}. Pass the path of a single "
            "qptiff file or of a directory that contains each region once."
        )
    return dict(zip(regions, scans.values(), strict=True))


def _base_name(path: Path) -> str:
    """File name without the ``.er.qptiff``/``.qptiff`` suffix."""
    for suffix in (PhenocyclerKeys.ER_QPTIFF_SUFFIX, PhenocyclerKeys.QPTIFF_SUFFIX):
        if path.name.lower().endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _region(base_name: str) -> str:
    return _SCAN_SUFFIX.sub("", base_name) or base_name


def _is_extended_range(path: Path) -> bool:
    return path.name.lower().endswith(PhenocyclerKeys.ER_QPTIFF_SUFFIX)


def _is_acquisition_file(path: Path) -> bool:
    if path.name.lower().endswith(".raw" + PhenocyclerKeys.QPTIFF_SUFFIX):
        return True
    parts = path.parts
    n = len(_ACQUISITION_DIR)
    return any(parts[i : i + n] == _ACQUISITION_DIR for i in range(len(parts) - n))


def _read_sidecars(scan: AkoyaScan) -> None:
    """Store the ``.xpd`` and ``*channels.csv`` sidecars of ``scan`` in ``scan.raw`` and check their channel count."""
    base = _base_name(scan.path)
    xpd = _find_sidecar(scan.path, base, PhenocyclerKeys.XPD_SUFFIX)
    protocol = read_json_object(xpd) if xpd is not None else None
    if xpd is not None and protocol is not None:
        scan.raw["xpd"] = protocol
        # an .xpd has the layout of the Fusion scan profile's `experimentDescription`
        markers = [marker for marker, _ in _fusion_profile_records({"experimentDescription": protocol})]
        _check_channel_count(scan, xpd, markers)

    channels_csv = _find_sidecar(scan.path, base, PhenocyclerKeys.CHANNELS_CSV_SUFFIX)
    if channels_csv is not None:
        with channels_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        scan.raw["channels_csv"] = rows
        column = next((key for key in (rows[0] if rows else {}) if key.strip().casefold() == "marker"), None)
        if column is not None:
            markers = [row[column] for row in rows if row[column] and not _is_blank(row[column])]
            _check_channel_count(scan, channels_csv, markers)


def _find_sidecar(qptiff: Path, base: str, suffix: str) -> Path | None:
    """First file ending in ``suffix`` whose name starts with ``base``, next to the scan or in HuBMAP's processed dir."""
    pattern = re.compile(rf"{re.escape(base)}(?![0-9A-Za-z]).*{re.escape(suffix)}$")
    directories = [qptiff.parent]
    # HuBMAP keeps `raw/images/<scan>.qptiff` and `lab_processed/images/<scan>.qptiff.channels.csv`
    for ancestor in qptiff.parents:
        if (ancestor / "raw").is_dir() and qptiff.is_relative_to(ancestor / "raw"):
            directories.append(ancestor / "lab_processed" / "images")
            break
    for directory in directories:
        if not directory.is_dir():
            continue
        matches = sorted(p for p in directory.iterdir() if p.is_file() and pattern.match(p.name))
        if matches:
            if len(matches) > 1:
                logger.info("Found several sidecar files %s for %s; using %s.", matches, qptiff, matches[0])
            return matches[0]
    return None


def _check_channel_count(scan: AkoyaScan, sidecar: Path, markers: list[str]) -> None:
    # the Fusion acquires DAPI in every cycle but stores it as one page, so each marker is counted once
    n_markers = len(set(markers))
    n_channels = len({channel.name for channel in scan.channels})
    if n_markers != n_channels:
        warnings.warn(
            f"{sidecar.name} lists {n_markers} distinct markers, but {scan.path.name} has {n_channels} distinct "
            f"channel names ({len(scan.channels)} channels); using the channel names from the qptiff.",
            UserWarning,
            stacklevel=2,
        )
