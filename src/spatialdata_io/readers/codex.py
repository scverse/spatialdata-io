from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd
from spatialdata import SpatialData

from spatialdata_io._constants._constants import CodexKeys, PhenocyclerKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    build_image_element,
    read_json_object,
    read_tiff,
    region_transformations,
    scans_to_attrs,
    select_regions,
    spatial_chunks,
)
from spatialdata_io.readers._utils._cell_tables import (
    CellTable,
    build_cells,
    detect_table_format,
    polygon_labels,
    polygon_shapes,
    raster_shape,
    read_codex_table,
    read_region_labels,
    read_region_tables,
    resolve_per_region,
)
from spatialdata_io.readers._utils._qptiff import AkoyaChannel, AkoyaScan, read_qptiff_image, read_qptiff_metadata
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs, _set_reader_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LegacyLayout", "codex", "discover_legacy", "read_legacy_image"]

_REGION = re.compile(r"reg(\d+)", re.IGNORECASE)
_PER_MARKER = re.compile(r"reg(\d+)_cyc(\d+)_ch(\d+)_(.+)\.tiff?$", re.IGNORECASE)
_MONTAGE = re.compile(r"reg\d+_montage\.tiff?$", re.IGNORECASE)
_TILE_STACK = re.compile(r"reg\d+_X\d+_Y\d+_Z\d+\.tiff?$", re.IGNORECASE)
# raw Keyence acquisition folders, e.g. `cyc001_reg001/1_00001_Z001_CH1.tif`
_RAW_CYCLE_DIR = re.compile(r"cyc\d+_reg\d+", re.IGNORECASE)
_TIFF_SUFFIXES = (".tif", ".tiff")
_TABLE_SUFFIXES = (CodexKeys.FCS_FILE.value, CodexKeys.FCS_FILE_CSV.value)
# deep enough for `<experiment>/processed_<date>/segm/segm-1/fcs/compensated/<table>`
_MAX_DEPTH = 8
# `channelNames.txt`/`experiment.json` sit in the experiment folder, up to `processed_<date>/stitched/reg001/` above
_SIDECAR_ANCESTORS = 3
_PATTERNS = (
    "'regNNN_cycCCC_chHHH_<marker>.tif(f)' (per-marker stitched images)",
    "'regNNN_montage.tif', 'regNNN_X01_Y01_Z01.tif' or '*regNNN*.tif' (hyperstacks)",
    f"'*{CodexKeys.QPTIFF}' (CODEX Instrument Manager)",
    f"'*{CodexKeys.FCS_FILE}' or '*{CodexKeys.FCS_FILE_CSV}' (cell tables)",
)


@dataclass
class LegacyLayout:
    """Files of a CODEX Processor/Uploader output tree or of CODEX Instrument Manager scans.

    ``images`` maps each region to its per-marker images (sorted by cycle and channel), to its single hyperstack
    or to its single ``.qptiff``. ``tables`` maps each region, or ``"*"`` for a table without a region in its
    file name, to its ``.fcs``/``.csv`` cell table.
    """

    images: dict[str, list[Path]]
    tables: dict[str, Path]
    channel_names_txt: Path | None
    experiment_json: Path | None


@inject_docs(vx=CodexKeys)
def codex(
    path: str | Path,
    fcs: bool = True,
    regions: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    cells_table: str | Path | Mapping[str, str | Path] | None = None,
    cells_table_format: str = "auto",
    labels: str | Path | Mapping[str, str | Path] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    cells_table_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *CODEX* (CODEX Processor, CODEX Uploader and CODEX Instrument Manager) data.

    This function reads the following files, searching ``path`` recursively:

        - ``regNNN_cycCCC_chHHH_<marker>{vx.IMAGE_TIF}(f)``: per-marker stitched images of the CODEX Processor,
          stacked lazily into one image per region in (cycle, channel) order; the channel names come from the file
          names.
        - ``regNNN_montage{vx.IMAGE_TIF}``, ``regNNN_X01_Y01_Z01{vx.IMAGE_TIF}`` or ``*regNNN*{vx.IMAGE_TIF}``:
          hyperstacks (CODEX Uploader, older Processor versions). All axes except ``y`` and ``x`` are flattened
          into channels, in file order. The channel names come from the ImageJ ``Labels``, else from
          ``{vx.CHANNEL_NAMES_FILE!r}`` (one name per line, cycle-major), else they are ``cycCCC_chHHH``.
        - ``*{vx.QPTIFF!r}``: scans of the CODEX Instrument Manager, read lazily with their channel metadata.
        - ``{vx.EXPERIMENT_FILE!r}``: optional; the pixel size is taken from its ``xyResolution`` (nm).
        - ``*{vx.FCS_FILE!r}`` / ``*{vx.FCS_FILE_CSV!r}``: cell tables of the CODEX Processor, one per region (or
          one for all regions), e.g. ``segm/segm-1/fcs/compensated/regNNN_compensated.fcs``. Compensated tables
          are preferred over uncompensated ones. The ``cycCCC_chHHH`` marker columns are stored in ``X`` (dense
          float32) under their marker names, taken from the FCS ``$PnS`` keywords or the CSV header
          ``cycCCC_chHHH:<marker>``; all other columns go to ``obs``, the ``x``/``y`` pixel centroids to
          ``obsm["spatial"]``.

    Raw acquisition and intermediate folders (``cycCCC_regNNN/``, ``tiles/`` and, next to ``stitched/``,
    ``bestFocus/``) are ignored. The region id ``regNNN`` is taken from the file names; a ``{vx.QPTIFF!r}`` without
    one is named after its file. If a directory contains no such files, its top-level ``{vx.IMAGE_TIF}(f)`` files
    are read as hyperstacks named after their files. Each region is stored as the image ``<region>_image`` and its
    metadata in ``sdata.attrs["akoya"]``. A single region is stored in the ``"global"`` coordinate system. Several
    CIM scans of the same sample space (same slide, with stage positions and equal pixel sizes) are placed at their
    stage positions in ``"global"``; otherwise (e.g. CODEX Processor regions, which have no stage positions) each
    region is stored in its own coordinate system, named after the region. All elements of a region share its
    coordinate system. The cells of each region are stored as circles ``<region>_cells`` with the cells' area
    (``size`` column), annotated by one table ``"table"`` (``obs["region"]`` names the circles, ``obs["cell_id"]``
    is the instance key), or by one table ``<region>_table`` per region if the regions have different markers. A region with a segmentation mask in
    ``labels`` gets the labels ``<region>_labels`` instead of circles, and its cells annotate them; so does a
    region with the cell outlines of a QuPath GeoJSON export (see ``cells_table_kwargs``), rasterised lazily one
    chunk at a time with the cells' instance ids (the higher id wins where cells overlap); the outlines are also kept
    as the shapes ``<region>_polygons`` in pixels, indexed by the same instance ids. The paths of the tables,
    masks and GeoJSON files read are stored in ``sdata.attrs["akoya"]["cell_tables"]``, ``["labels"]`` and
    ``["geojson"]``.

    .. seealso::

        - `CODEX output <https://help.codex.bio/codex/processor/technical-notes/expected-output>`_.

    Parameters
    ----------
    path
        Path to a CODEX output directory, or to a single image (``{vx.QPTIFF!r}`` or ``{vx.IMAGE_TIF}(f)``).
    fcs
        Whether to prefer ``{vx.FCS_FILE!r}`` cell tables over ``{vx.FCS_FILE_CSV!r}`` ones when both exist. With
        `False`, a region that only has a ``{vx.FCS_FILE!r}`` table is read from it, with a warning.
    regions
        Regions to read, e.g. ``["reg001"]``, including only their cells. If `None`, all regions are read.
    channel_names
        Channel names to use instead of the discovered ones; must have one entry per channel of every region read.
    cells_table
        A cell table to read instead of the discovered ones (explicit beats discovered): a path, or a mapping from
        region id to path, e.g. ``{{"reg001": "reg001_cells.csv"}}``. A single path for several regions needs a
        region column (``region`` for CODEX tables, ``Image`` for QuPath, else
        ``cells_table_kwargs["region_column"]``). See ``cells_table_format`` for the formats.
    cells_table_format
        The format of ``cells_table``: ``"auto"`` (detected from the header), ``"codex"``, ``"qupath"`` (QuPath
        measurements; µm centroids are converted with the region's pixel size), ``"mcquant"`` (MCMICRO) or
        ``"generic"`` (columns given by ``cells_table_kwargs``).
    labels
        A segmentation mask (TIFF/OME-TIFF with integer instance ids) per region: a path for a single region, or a
        mapping from region id to path. Each mask is stored lazily as the labels ``<region>_labels``
        with the transformation of the region's image; the cells of the region annotate it instead of circles, and
        their instance ids must be the mask values.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`, which is only used for ``{vx.IMAGE_TIF}(f)``
        files that :mod:`tifffile` cannot read lazily (e.g. unsupported compression).
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.
    cells_table_kwargs
        Options for reading ``cells_table`` and ``labels``: ``region_column``; for ``"generic"`` tables ``x``,
        ``y``, ``instance_id``, ``markers`` (list or regular expression) and ``coordinates_unit`` (``"px"`` or
        ``"um"``); ``mask_channel`` (index or OME channel name of the mask in a multi-channel ``labels`` file);
        for QuPath tables ``geojson`` (the GeoJSON export of the cell outlines, a path or a mapping from region id
        to path like ``cells_table``; default ``<table stem>.geojson`` or ``<table stem>_cells.geojson`` next to
        the table) and ``geojson_units`` (``"px"`` of the full-resolution image, default and what QuPath writes,
        or ``"um"``).

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    regions = [regions] if isinstance(regions, str) else regions
    channel_names = [channel_names] if isinstance(channel_names, str) else channel_names
    path = Path(path)
    layout = discover_legacy(path, fcs)
    # regions that only have a discovered table exist only while the discovered tables are read
    table_regions = layout.tables.keys() - {"*"} if cells_table is None else set()
    available = sorted(layout.images.keys() | table_regions)
    selected = select_regions(available, regions)
    table_files = resolve_per_region(cells_table, selected, available)
    mask_files = resolve_per_region(labels, selected, available)
    experiment = read_json_object(layout.experiment_json) if layout.experiment_json is not None else None
    names_txt = _read_channel_names_txt(layout.channel_names_txt) if layout.channel_names_txt is not None else None

    scans: list[AkoyaScan] = []
    arrays: dict[str, da.Array] = {}
    for region in (region for region in selected if region in layout.images):
        files = layout.images[region]
        if _is_qptiff(files[0]):
            scans.append(read_qptiff_metadata(files[0], region=region))
            arrays[region] = read_qptiff_image(files[0])
            continue
        arrays[region], channels = read_legacy_image(
            files, layout.channel_names_txt, imread_kwargs, image_models_kwargs=image_models_kwargs
        )
        scans.append(
            AkoyaScan(
                path=files[0] if len(files) == 1 else files[0].parent,
                region=region,
                generation="legacy_processor",
                software=None,
                slide_id=None,
                pixel_size_um=_pixel_size_um(experiment),
                nominal_pixel_size_um=None,
                stage_position_um=None,
                channels=channels,
                rgb=False,
                raw={
                    "files": [str(file) for file in files],
                    "experiment_json": experiment,
                    "channel_names_txt": names_txt,
                },
            )
        )

    names = {scan.region: apply_channel_names(scan, channel_names) for scan in scans}
    if table_files:
        cell_tables = read_region_tables(
            table_files,
            cells_table_format,
            available=list(layout.images),
            selected=selected,
            pixel_sizes={scan.region: scan.pixel_size_um for scan in scans},
            cells_table_kwargs=cells_table_kwargs,
        )
        table_paths = {key: str(file) for key, file in table_files.items()}
    else:
        discovered = {key: file for key, file in layout.tables.items() if key == "*" or key in selected}
        read = _read_tables(discovered, layout, selected if regions is not None else None, fcs)
        cell_tables = list(read.values())
        table_paths = {key: str(discovered[key]) for key in read}
    # the regions of cells or masks without an image are unrelated to the scans
    cell_regions = {region for table in cell_tables for region in table.region.unique()}
    transformations = region_transformations(scans, cell_regions | (mask_files.keys() - {"*"}))
    images = {
        f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}": build_image_element(
            arrays[scan.region], scan, names[scan.region], transformations[scan.region], image_models_kwargs
        )
        for scan in scans
    }
    image_shapes = {scan.region: raster_shape(images[f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}"]) for scan in scans}
    region_labels = read_region_labels(
        mask_files, transformations, labels_models_kwargs, cells_table_kwargs, image_shapes
    )
    polygons, geojson_files = polygon_labels(
        cell_tables, region_labels, image_shapes, transformations, labels_models_kwargs
    )
    region_labels |= polygons
    tables, shapes = build_cells(cell_tables, {}, region_labels, transformations) if cell_tables else ({}, {})
    for region, element in polygon_shapes(cell_tables, transformations).items():
        shapes[f"{region}{PhenocyclerKeys.POLYGONS_SUFFIX}"] = element
    sdata = SpatialData(
        images=images,
        labels={f"{region}{PhenocyclerKeys.LABELS_SUFFIX}": element for region, element in region_labels.items()},
        shapes=shapes,
        tables=tables,
    )
    attrs = scans_to_attrs(scans, names)
    attrs["cell_tables"] = table_paths
    if mask_files:
        attrs["labels"] = {region: str(file) for region, file in mask_files.items()}
    if geojson_files:
        attrs["geojson"] = geojson_files
    sdata.attrs[PhenocyclerKeys.ATTRS_KEY.value] = attrs
    return _set_reader_metadata(sdata, "codex")


def _read_tables(
    files: Mapping[str, Path], layout: LegacyLayout, regions: Sequence[str] | None, fcs: bool
) -> dict[str, CellTable]:
    """Read the cell tables of ``files`` (by region, ``"*"`` for one without a region), keeping only ``regions``.

    Returns the tables that have cells of ``regions``. A table without a region in its file name that cannot be
    read, or whose cells cannot be assigned to a region, is skipped with a warning.
    """
    fallback = [str(file) for file in files.values() if file.suffix.lower() == CodexKeys.FCS_FILE.value]
    if not fcs and fallback:
        warnings.warn(
            f"`fcs=False`, but no CSV cell table was found for {fallback}; reading the FCS table(s) instead.",
            UserWarning,
            stacklevel=3,
        )
    only_image = next(iter(layout.images)) if len(layout.images) == 1 else None
    tables = {}
    for key, file in files.items():
        if key != "*":
            table = read_codex_table(file, region=key)
        else:
            # a table without a region in its file name may be any CSV: skip it rather than fail the whole read
            try:
                table = read_codex_table(file, region=only_image)
            except ValueError as e:
                warnings.warn(f"Ignoring the cell table {file}: {e}", UserWarning, stacklevel=3)
                continue
            # next to a single image named after its file (e.g. `Kidney.tif`), its cells belong to that image
            values = table.region.unique()
            if only_image is not None and len(values) == 1 and values[0] not in layout.images:
                table.region = pd.Series(only_image, index=table.region.index, name=table.region.name)
        if regions is not None:
            table = table.take(np.flatnonzero(table.region.isin(regions).to_numpy()))
        if len(table.region):
            tables[key] = table
    return tables


def discover_legacy(path: Path, fcs: bool) -> LegacyLayout:
    """Find the images, cell tables and sidecar files of a CODEX dataset.

    Parameters
    ----------
    path
        A CODEX output directory, searched recursively in sorted order, or a single image or table file.
    fcs
        Whether to prefer ``.fcs`` over ``.csv`` tables of the same region.

    Returns
    -------
    The discovered files.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    files = [path] if path.is_file() else _walk(path)
    images = _discover_images(files)
    if not images:
        # the original (2019) layout: one stitched `.tif` next to its table, without a region in the file name
        top_level = files if path.is_file() else [file for file in files if file.parent == path]
        images = {file.stem: [file] for file in top_level if _is_tiff(file)}
    tables = _discover_tables(files, fcs)
    if not images and not tables:
        raise FileNotFoundError(f"No CODEX images or cell tables found in {path}. Expected: {'; '.join(_PATTERNS)}.")
    return LegacyLayout(
        images=dict(sorted(images.items())),
        tables=dict(sorted(tables.items())),
        channel_names_txt=_find_sidecar(path, files, CodexKeys.CHANNEL_NAMES_FILE.value),
        experiment_json=_find_sidecar(path, files, CodexKeys.EXPERIMENT_FILE.value),
    )


def read_legacy_image(
    files: list[Path],
    channel_names_txt: Path | None,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[da.Array, list[AkoyaChannel]]:
    """Lazily read the image of one legacy CODEX region.

    Parameters
    ----------
    files
        The per-marker images of the region, or its single hyperstack, as returned by :func:`discover_legacy`.
    channel_names_txt
        Optional ``channelNames.txt`` naming the hyperstack channels when the file has no ImageJ ``Labels``.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread` for files :mod:`tifffile` cannot read lazily.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`; their ``chunks`` set the spatial chunks of
        uncompressed files, which are memory-mapped.

    Returns
    -------
    The ``(c, y, x)`` image and its channels.
    """
    chunks = _spatial_chunks(image_models_kwargs)
    markers = [(match, file) for file in files if (match := _PER_MARKER.search(file.name)) is not None]
    if len(markers) == len(files):
        image, channels, read_by_tifffile = _read_per_marker(markers, imread_kwargs, chunks)
        _warn_unused_imread_kwargs(imread_kwargs, read_by_tifffile)
        return image, channels
    if len(files) != 1:
        raise ValueError(f"Expected per-marker images or a single hyperstack, got {[str(f) for f in files]}.")

    image, leading_shape, labels, read_by_tifffile = read_tiff(files[0], imread_kwargs, chunks)
    _warn_unused_imread_kwargs(imread_kwargs, read_by_tifffile)
    n_planes = image.shape[0]
    # the last non-spatial axis holds the channels of a cycle; ImageJ axis labels are unreliable (cycles as Z)
    n_channels = leading_shape[-1] if leading_shape else 1
    names: list[str] | None = labels if labels is not None and len(labels) == n_planes else None
    if names is None and channel_names_txt is not None:
        names = _read_channel_names_txt(channel_names_txt)
        if len(names) != n_planes:
            warnings.warn(
                f"{channel_names_txt} lists {len(names)} channel names, but {files[0]} has {n_planes} channels; "
                "using 'cycCCC_chHHH' names instead.",
                UserWarning,
                stacklevel=2,
            )
            names = None
    channels = []
    for i in range(n_planes):
        cycle, channel = i // n_channels + 1, i % n_channels + 1
        name = names[i] if names is not None else f"cyc{cycle:03d}_ch{channel:03d}"
        channels.append(AkoyaChannel(name=name, cycle=cycle, channel=channel))
    return image, channels


def _read_per_marker(
    markers: list[tuple[re.Match[str], Path]], imread_kwargs: Mapping[str, Any], chunks: tuple[int, int]
) -> tuple[da.Array, list[AkoyaChannel], bool]:
    parsed = sorted((int(match.group(2)), int(match.group(3)), match.group(4), file) for match, file in markers)

    planes = []
    read_by_tifffile = False
    for _, _, _, file in parsed:
        plane, _, _, lazy = read_tiff(file, imread_kwargs, chunks)
        read_by_tifffile |= lazy
        if plane.shape[0] != 1:
            raise ValueError(f"{file} holds {plane.shape[0]} planes; expected a single 2-D marker image.")
        planes.append(plane[0])
    if len({(plane.shape, plane.dtype) for plane in planes}) > 1:
        found = {file.name: (plane.shape, str(plane.dtype)) for (*_, file), plane in zip(parsed, planes, strict=True)}
        raise ValueError(f"The per-marker images of one region must share shape and dtype, got {found}.")
    channels = [
        AkoyaChannel(name=marker, cycle=cycle, channel=channel, extra={"file": file.name})
        for cycle, channel, marker, file in parsed
    ]
    return da.stack(planes), channels, read_by_tifffile


def _spatial_chunks(image_models_kwargs: Mapping[str, Any]) -> tuple[int, int]:
    """The ``(y, x)`` chunk size requested in ``image_models_kwargs``, with the reader default when unspecified."""
    default, _ = _initialize_raster_models_kwargs({}, {})
    chunks = _initialize_raster_models_kwargs(image_models_kwargs, {})[0]["chunks"]
    return spatial_chunks(chunks, default["chunks"][-2:])


def _warn_unused_imread_kwargs(imread_kwargs: Mapping[str, Any], read_by_tifffile: bool) -> None:
    if imread_kwargs and read_by_tifffile:
        warnings.warn(
            "`imread_kwargs` is ignored for files read lazily with tifffile; it is only used by the "
            "dask_image.imread fallback for files tifffile cannot read lazily.",
            UserWarning,
            stacklevel=3,
        )


def _walk(root: Path) -> list[Path]:
    """All files below ``root`` in sorted order, skipping raw acquisition and intermediate processing folders."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        depth = len(directory.relative_to(root).parts)
        has_stitched = any(name.casefold() == "stitched" for name in dirnames)
        dirnames[:] = sorted(
            name for name in dirnames if depth < _MAX_DEPTH and not _is_ignored_dir(name, has_stitched)
        )
        files.extend(directory / name for name in sorted(filenames))
    return files


def _is_ignored_dir(name: str, has_stitched: bool) -> bool:
    folded = name.casefold()
    return (
        folded == "tiles"
        or (folded == "bestfocus" and has_stitched)
        or _RAW_CYCLE_DIR.fullmatch(name) is not None
        or name.startswith(".")
    )


def _discover_images(files: Sequence[Path]) -> dict[str, list[Path]]:
    per_marker: dict[str, dict[tuple[int, int], Path]] = {}
    stacks: dict[str, list[Path]] = {}
    qptiffs: dict[str, list[Path]] = {}
    for file in files:
        region = _region_id(file.name)
        if _is_qptiff(file):
            qptiffs.setdefault(region or file.stem, []).append(file)
        elif _is_tiff(file) and region is not None:
            if (match := _PER_MARKER.search(file.name)) is None:
                stacks.setdefault(region, []).append(file)
                continue
            markers = per_marker.setdefault(region, {})
            key = (int(match.group(2)), int(match.group(3)))
            if key in markers:
                raise ValueError(
                    f"Found several images for cycle {key[0]}, channel {key[1]} of region '{region}': "
                    f"{markers[key]} and {file}. Pass the path of a directory that contains each region once."
                )
            markers[key] = file

    images: dict[str, list[Path]] = {}
    # the Processor's stitched outputs come first: cell tables refer to their pixel space
    for region in sorted(per_marker.keys() | stacks.keys() | qptiffs.keys()):
        if region in per_marker:
            images[region] = [per_marker[region][key] for key in sorted(per_marker[region])]
        elif region in stacks:
            images[region] = [_single(region, _preferred_stacks(stacks[region]))]
        else:
            images[region] = [_single(region, qptiffs[region])]
    return images


def _preferred_stacks(files: list[Path]) -> list[Path]:
    for pattern in (_MONTAGE, _TILE_STACK):
        matching = [file for file in files if pattern.search(file.name)]
        if matching:
            return matching
    return files


def _single(region: str, files: list[Path]) -> Path:
    if len(files) > 1:
        raise ValueError(
            f"Found several images for region '{region}': {[str(f) for f in files]}. Pass the path of a single "
            "image or of a directory that contains each region once."
        )
    return files[0]


def _discover_tables(files: Sequence[Path], fcs: bool) -> dict[str, Path]:
    preferred = CodexKeys.FCS_FILE.value if fcs else CodexKeys.FCS_FILE_CSV.value
    candidates: dict[str, list[Path]] = {}
    for file in files:
        if file.suffix.lower() not in _TABLE_SUFFIXES:
            continue
        region = _region_id(file.name)
        # without a region in its name, a CSV may be any sidecar (e.g. an antibody list)
        if region is None and detect_table_format(file) != "codex":
            continue
        candidates.setdefault(region or "*", []).append(file)

    def rank(file: Path) -> tuple[bool, bool]:
        # the preferred format first, then compensated over uncompensated
        return file.suffix.lower() != preferred, any("uncompensated" in part.casefold() for part in file.parts)

    tables = {}
    for region, paths in candidates.items():
        best = min(rank(path) for path in paths)
        paths = [path for path in paths if rank(path) == best]
        if region == "*" and len(paths) > 1:
            raise ValueError(
                f"Found several cell tables without a region in their file name: {[str(p) for p in paths]}; "
                "the reader cannot choose between them. Pass the path of a directory that contains only one."
            )
        tables[region] = min(paths, key=lambda path: (len(path.parts), str(path)))
    return tables


def _find_sidecar(path: Path, files: Sequence[Path], name: str) -> Path | None:
    """The shallowest file called ``name`` in ``files``, else in the nearest ancestor of ``path`` holding one."""
    matches = sorted((file for file in files if file.name == name), key=lambda file: (len(file.parts), str(file)))
    if matches:
        return matches[0]
    start = path.parent if path.is_file() else path
    for directory in [start, *start.parents][: _SIDECAR_ANCESTORS + 1]:
        if (directory / name).is_file():
            return directory / name
    return None


def _read_channel_names_txt(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _pixel_size_um(experiment: Mapping[str, Any] | None) -> float | None:
    """Pixel size in µm from the ``experiment.json`` of the CODEX Processor (nm), ``None`` when it is missing."""
    for key in ("xyResolution", "per_pixel_XY_resolution", "Resolution_nm"):
        value = (experiment or {}).get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return value / 1000
    return None


def _region_id(name: str) -> str | None:
    match = _REGION.search(name)
    return f"reg{int(match.group(1)):03d}" if match else None


def _is_qptiff(path: Path) -> bool:
    return path.suffix.lower() == CodexKeys.QPTIFF.value


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in _TIFF_SUFFIXES
