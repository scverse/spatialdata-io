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
    normalize_regions,
    read_json_object,
    select_regions,
)
from spatialdata_io.readers._utils._cell_tables import build_spatialdata, read_region_tables, resolve_per_region
from spatialdata_io.readers._utils._qptiff import (
    _fusion_profile_records,
    _is_blank,
    qptiff_base_name,
    qptiff_regions_with_duplicates,
    read_qptiff_image,
    read_qptiff_metadata,
)
from spatialdata_io.readers._utils._utils import _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = ["phenocycler"]

# the Fusion writes per-cycle acquisition files to ``raw/images/phenocycler/`` (HuBMAP layout)
_ACQUISITION_DIR = ("raw", "images", "phenocycler")
# region id suffix of the 8-bit scan read next to its extended-range scan (``read_all_bit_depths``)
_EIGHT_BIT_SUFFIX = "_8bit"


@inject_docs(vx=PhenocyclerKeys)
def phenocycler(
    path: str | Path,
    *,
    regions: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    cells_table: str | Path | Mapping[str, str | Path] | None = None,
    cells_table_format: str = "auto",
    labels: str | Path | Mapping[str, str | Path] | None = None,
    read_all_bit_depths: bool = False,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    cells_table_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *Akoya PhenoCycler-Fusion* 1.x/2.x data, and PerkinElmer/Akoya Vectra and PhenoImager QPTIFFs.

    This function reads the following files:

        - ``*{vx.QPTIFF_SUFFIX!r}`` / ``*{vx.ER_QPTIFF_SUFFIX!r}``: the scans, one region each. The images are read
          lazily; when both ``X{vx.QPTIFF_SUFFIX}`` and ``X{vx.ER_QPTIFF_SUFFIX}`` exist, the extended-range
          (higher bit depth) ``{vx.ER_QPTIFF_SUFFIX!r}`` is used, unless ``read_all_bit_depths`` is set.
        - ``*{vx.XPD_SUFFIX!r}``: optional experiment protocol (JSON) next to a scan.
        - ``*{vx.CHANNELS_CSV_SUFFIX!r}``: optional channel list next to a scan or, in the HuBMAP layout, in
          ``lab_processed/images/``.

    The same QPTIFF container is written by Vectra Polaris and PhenoImager HT (also unmixed, uint16) scans, whose
    channels are read like the Fusion's, by brightfield (H&E) scans, read as the three channels ``r``, ``g`` and
    ``b``, and by inForm component data (float32, usually a ``.tif`` file, which is read if ``path`` is that single
    file). The image keeps the dtype of the file. These files have the generation ``"qpi_other"`` in the metadata.

    Sidecar files are matched to a scan by file-name prefix and stored as metadata only: the channel names always
    come from the per-page XML of the qptiff. A sidecar whose number of markers differs from the number of channels
    raises a warning. Acquisition files (``*.raw.qptiff``, ``*.qptiff.raw``, ``*.qptiff.intermediate`` and anything
    in a ``raw/images/phenocycler/`` folder below ``path``) are ignored. File suffixes are matched ignoring case.

    The region name is the file name without the ``{vx.ER_QPTIFF_SUFFIX!r}``/``{vx.QPTIFF_SUFFIX!r}`` suffix and
    without a trailing ``_Scan<N>``, unless that makes two regions share a name, made a valid element name by
    :func:`spatialdata.sanitize_name` (characters other than letters, digits, ``_``, ``.`` and ``-`` are replaced by
    ``_``, e.g. ``A_[1,2]`` becomes ``A__1_2_``). Each region is stored as the image
    ``<region>{vx.IMAGE_SUFFIX}``. A single region is stored in the ``"global"`` coordinate system. Several regions
    of the same sample space (scanned on the same slide, with stage positions and equal pixel sizes) are placed at
    their stage positions in ``"global"``; otherwise the regions are unrelated and each is stored in its own
    coordinate system, named after the region. A region with cells or a mask but no scan gets its own coordinate
    system too. All elements of a region share its coordinate system. The per-region metadata is stored in
    ``sdata.attrs[{vx.ATTRS_KEY!r}]``.

    A segmentation mask given in ``labels`` is stored lazily as the labels ``<region>{vx.LABELS_SUFFIX}`` with the
    transformation of the region's image. The cells of ``cells_table`` are stored in one table
    ``{vx.TABLE_KEY!r}`` (marker intensities dense float32 in ``X``, pixel centroids in ``obsm["spatial"]``,
    ``obs[{vx.REGION_KEY!r}]`` the annotated element, ``obs[{vx.INSTANCE_KEY!r}]`` the instance key), or in one
    table ``<region>_table`` per region if the regions have different markers. Columns of the cell table that clash
    with ``{vx.REGION_KEY!r}`` or ``{vx.INSTANCE_KEY!r}`` are kept as ``source_<name>``, and column names that
    spatialdata cannot write (e.g. QuPath's ``Centroid X µm``) are sanitised (``Centroid_X_µm``). Cells annotate
    the labels of their region, whose values must be their instance ids; without a mask they are stored as circles
    ``<region>{vx.CELLS_SUFFIX}``, with a warning.

    The cell outlines of a QuPath GeoJSON export (see ``cells_table_kwargs``) are rasterised lazily, one chunk at a
    time, into the labels of a region without a mask; a feature's ``id`` is matched to the table's ``Object ID``
    (else features and rows are matched in order), the labels hold the cells' instance ids, and where cells overlap
    the higher instance id wins; the table annotates the labels. The outlines are always kept as the shapes
    ``<region>{vx.POLYGONS_SUFFIX}`` in pixels, indexed by the same instance ids, also when a mask is given for the
    region (the mask is used for the labels, with a warning). The paths of the tables, masks and GeoJSON files are
    stored in ``sdata.attrs[{vx.ATTRS_KEY!r}]["cell_tables"]``, ``["labels"]`` and ``["geojson"]``.

    .. seealso::

        - `Akoya PhenoCycler-Fusion <https://www.akoyabio.com/phenocycler/>`_.

    Parameters
    ----------
    path
        Path to a ``{vx.QPTIFF_SUFFIX!r}`` file (or an inForm ``.tif`` QPI file), or to a directory that is searched
        recursively for ``{vx.QPTIFF_SUFFIX!r}`` files.
    regions
        Regions to read. If `None`, all regions are read.
    channel_names
        Channel names to use instead of the ones stored in the qptiff; must have one entry per channel of every
        region read.
    cells_table
        A cell table: a path, or a mapping from region id to path, e.g. ``{{"A": "A_cells.csv"}}``. A single path
        for several regions needs a region column (``Image`` for QuPath, ``region`` for CODEX tables, else
        ``cells_table_kwargs["region_column"]``); its values are matched to the region ids, also after
        :func:`spatialdata.sanitize_name`, e.g. ``A_Scan1.qptiff`` to ``A`` and ``A_[1,2].qptiff`` to ``A__1_2_``.
    cells_table_format
        The format of ``cells_table``: ``"auto"`` (detected from the header), ``"codex"`` (CODEX Processor),
        ``"qupath"`` (QuPath measurements; µm centroids are converted with the region's pixel size), ``"mcquant"``
        (MCMICRO), ``"sprm"`` (HuBMAP SPRM ``*-cell_channel_mean.csv``), ``"inform"`` (inForm
        ``*_cell_seg_data.txt``) or ``"generic"`` (columns given by ``cells_table_kwargs``).
    labels
        A segmentation mask (TIFF/OME-TIFF with integer instance ids) per region: a path for a single region, or a
        mapping from region id to path. A mask whose shape is the region image's divided by an integer factor ``f``
        (exactly, or rounded up) is scaled onto the image (``Scale([f, f])`` before the region's transformation);
        any other shape mismatch raises a warning. The pixel centroids of ``"mcquant"``, ``"sprm"`` and
        ``"generic"`` (``"px"``) tables are taken to be in mask pixels, so their centroids (``obsm["spatial"]``) and
        circle radii are scaled by ``f`` to image pixels; the source columns in ``obs`` are kept as read (mask
        pixels). To avoid this for a generic table in image pixels, use ``coordinates_unit="um"`` or a
        full-resolution mask.
    read_all_bit_depths
        If `True`, a scan that exists as both ``X{vx.QPTIFF_SUFFIX}`` (8-bit) and ``X{vx.ER_QPTIFF_SUFFIX}``
        (extended range, e.g. uint16) is read from both files: the extended-range file as region ``X`` and the
        8-bit file as the image ``X_8bit{vx.IMAGE_SUFFIX}`` (region ``X_8bit`` in ``sdata.attrs[{vx.ATTRS_KEY!r}]``).
        Both are the same physical scan, so the 8-bit image has exactly the transformations of region ``X``. Cell
        tables and masks annotate region ``X`` only. If `False`, only the extended-range file is read.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.
    cells_table_kwargs
        Options for reading ``cells_table`` and ``labels``: ``region_column``; for ``"generic"`` tables ``x``,
        ``y``, ``instance_id``, ``markers`` (list or regular expression; also for QuPath tables) and
        ``coordinates_unit`` (``"px"`` or ``"um"``; for QuPath ``X``/``Y`` and inForm ``Cell X Position`` columns
        without a unit, which are otherwise read in the unit of the table's other columns); with ``x`` and ``y``,
        ``cells_table_format="auto"`` reads a generic table. ``mask_channel`` (index or channel name of the mask in
        a multi-channel ``labels`` file: an OME channel name, or for an inForm ``*_binary_seg_maps.tif``
        ``"Tissue"``, ``"Nucleus"`` (the default there, its labels are the ``Cell ID`` of every cell),
        ``"Cytoplasm"`` or ``"Membrane"``); for QuPath tables ``geojson`` (the GeoJSON export of the cell
        outlines, a path or a mapping from region id to path like ``cells_table``; default ``<table stem>.geojson``
        or ``<table stem>_cells.geojson`` next to the table) and ``geojson_units`` (``"px"`` of the
        full-resolution image, default and what QuPath writes, or ``"um"``).

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    regions = normalize_regions(regions)
    channel_names = [channel_names] if isinstance(channel_names, str) else channel_names
    path = Path(path)
    qptiffs, eight_bit = _discover_qptiffs(path)
    selected = select_regions(list(qptiffs), regions)
    table_files = resolve_per_region(cells_table, selected, list(qptiffs))
    mask_files = resolve_per_region(labels, selected, list(qptiffs))

    scans = [read_qptiff_metadata(qptiffs[region], region=region) for region in selected]
    duplicates: dict[str, AkoyaScan] = {}
    if read_all_bit_depths:
        for region in selected:
            if region in eight_bit:
                duplicates[region] = read_qptiff_metadata(eight_bit[region], region=f"{region}{_EIGHT_BIT_SUFFIX}")
        if clash := sorted({scan.region for scan in duplicates.values()} & set(qptiffs)):
            raise ValueError(
                f"Cannot read the 8-bit scans as region(s) {clash}: a scan with that region id exists. Rename it or "
                "pass read_all_bit_depths=False."
            )
    names = {}
    for scan in [*scans, *duplicates.values()]:
        names[scan.region] = apply_channel_names(scan, channel_names)
        _read_sidecars(scan)
    cell_tables = read_region_tables(
        table_files,
        cells_table_format,
        available=list(qptiffs),
        selected=selected,
        pixel_sizes={scan.region: scan.pixel_size_um for scan in scans},
        cells_table_kwargs=cells_table_kwargs,
    )
    sdata = build_spatialdata(
        scans,
        {scan.region: read_qptiff_image(scan.path) for scan in [*scans, *duplicates.values()]},
        names,
        cell_tables,
        mask_files,
        table_paths={key: str(file) for key, file in table_files.items()} if table_files else None,
        image_models_kwargs=image_models_kwargs,
        labels_models_kwargs=labels_models_kwargs,
        cells_table_kwargs=cells_table_kwargs,
        duplicate_scans=duplicates,
    )
    return _set_reader_metadata(sdata, "phenocycler")


def _discover_qptiffs(path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Find the qptiff scans of a PhenoCycler-Fusion dataset: a ``.qptiff`` file, or a directory searched recursively.

    Returns the scan file of every region, sorted by path, and the 8-bit file of the regions that also have an
    extended-range one; see :func:`qptiff_regions_with_duplicates`.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if path.is_file():
        return qptiff_regions_with_duplicates([path])
    files = [
        qptiff
        for qptiff in sorted(path.rglob(f"*{PhenocyclerKeys.QPTIFF_SUFFIX}", case_sensitive=False))
        if qptiff.is_file() and not _is_acquisition_file(qptiff.relative_to(path))
    ]
    if not files:
        raise FileNotFoundError(
            f"No .qptiff files found in {path} (searched recursively for '*.qptiff', excluding acquisition files)."
        )
    return qptiff_regions_with_duplicates(files)


def _is_acquisition_file(path: Path) -> bool:
    """Whether ``path``, relative to the searched directory, is a per-cycle acquisition file of the Fusion."""
    if path.name.lower().endswith(".raw" + PhenocyclerKeys.QPTIFF_SUFFIX):
        return True
    parts = tuple(part.casefold() for part in path.parent.parts)
    n = len(_ACQUISITION_DIR)
    return any(parts[i : i + n] == _ACQUISITION_DIR for i in range(len(parts) - n + 1))


def _read_sidecars(scan: AkoyaScan) -> None:
    """Store the ``.xpd`` and ``*channels.csv`` sidecars of ``scan`` in ``scan.raw`` and check their channel count.

    The sidecars are metadata only, so one that cannot be read is ignored with a warning.
    """
    base = qptiff_base_name(scan.path)
    xpd = _find_sidecar(scan.path, base, PhenocyclerKeys.XPD_SUFFIX)
    protocol = read_json_object(xpd) if xpd is not None else None
    if xpd is not None and protocol is not None:
        scan.raw["xpd"] = protocol
        # an .xpd has the layout of the Fusion scan profile's `experimentDescription`
        markers = [marker for marker, _ in _fusion_profile_records({"experimentDescription": protocol})]
        if markers:
            _check_channel_count(scan, xpd, markers)
        elif protocol.get("wells") is not None:
            warnings.warn(
                f"{xpd.name} has no readable markers in its 'wells', so its channel count is not checked.",
                UserWarning,
                stacklevel=3,
            )

    channels_csv = _find_sidecar(scan.path, base, PhenocyclerKeys.CHANNELS_CSV_SUFFIX)
    if channels_csv is not None:
        try:
            with channels_csv.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except (UnicodeDecodeError, csv.Error) as e:
            warnings.warn(f"Ignoring {channels_csv}: it is not a UTF-8 CSV file ({e}).", UserWarning, stacklevel=3)
            return
        scan.raw["channels_csv"] = rows
        header = rows[0] if rows else {}
        column = next((key for key in header if isinstance(key, str) and key.strip().casefold() == "marker"), None)
        if column is not None:
            markers = [row[column] for row in rows if row[column] and not _is_blank(row[column])]
            _check_channel_count(scan, channels_csv, markers)


def _find_sidecar(qptiff: Path, base: str, suffix: str) -> Path | None:
    """First file ending in ``suffix`` whose name starts with ``base``, next to the scan or in HuBMAP processed dir."""
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
