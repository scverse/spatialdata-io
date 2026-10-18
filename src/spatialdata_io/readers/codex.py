from __future__ import annotations

import codecs
import re
import warnings
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
from spatialdata import SpatialData, sanitize_name

from spatialdata_io._constants._constants import CodexKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._akoya import (
    IMAGE_EXTENSION,
    apply_channel_names,
    normalize_regions,
    read_json_object,
    read_tiff,
    region_id,
    select_regions,
    spatial_chunks,
)
from spatialdata_io.readers._utils._cell_tables import (
    CellTable,
    build_spatialdata,
    is_codex_table,
    read_cell_table,
    read_region_tables,
    resolve_per_region,
)
from spatialdata_io.readers._utils._qptiff import (
    AkoyaChannel,
    AkoyaScan,
    qptiff_regions,
    read_qptiff_image,
    read_qptiff_metadata,
)
from spatialdata_io.readers._utils._utils import (
    _initialize_raster_models_kwargs,
    _set_reader_metadata,
    walk_visible,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["codex"]

_PER_MARKER = re.compile(r"reg(\d+)_cyc(\d+)_ch(\d+)_(.+)\.tiff?$", re.IGNORECASE)
_MONTAGE = re.compile(r"reg\d+_montage\.tiff?$", re.IGNORECASE)
_TILE_STACK = re.compile(r"reg\d+_X\d+_Y\d+_Z\d+\.tiff?$", re.IGNORECASE)
# raw Keyence acquisition folders, e.g. `cyc001_reg001/1_00001_Z001_CH1.tif`
_RAW_CYCLE_DIR = re.compile(r"cyc\d+_reg\d+", re.IGNORECASE)
_TIFF_SUFFIXES = (".tif", ".tiff")
# the segmentation mask of a region, e.g. HuBMAP's `reg001_mask.ome.tiff` next to `reg001_expr.ome.tiff`
_MASK = re.compile(r"[_-](?:mask|labels)(?:\.ome)?\.tiff?$", re.IGNORECASE)
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
class _LegacyLayout:
    """Files of a CODEX Processor/Uploader output tree or of CODEX Instrument Manager scans.

    ``images`` maps each region to its per-marker images (sorted by cycle and channel), to its single hyperstack
    or to its single ``.qptiff``. ``tables`` maps each region, or ``"*"`` for a table without a region in its
    file name, to its ``.fcs``/``.csv`` cell table. ``masks`` maps regions to their segmentation mask.
    """

    images: dict[str, list[Path]]
    tables: dict[str, Path]
    masks: dict[str, Path]
    channel_names_txt: Path | None
    experiment_json: Path | None


@inject_docs(vx=CodexKeys)
def codex(
    path: str | Path,
    fcs: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    *,
    regions: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    cells_table: str | Path | Mapping[str, str | Path] | None = None,
    labels: str | Path | Mapping[str, str | Path] | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
        - ``{vx.EXPERIMENT_FILE!r}``: optional; the pixel size is taken from its ``xyResolution`` (nm). Like
          ``{vx.CHANNEL_NAMES_FILE!r}``, it is found in ``path`` or up to three folders above it, ignoring the case
          of the file name (e.g. ``channelnames.txt``).
        - ``*{vx.FCS_FILE!r}`` / ``*{vx.FCS_FILE_CSV!r}``: cell tables of the CODEX Processor, one per region (or one
          for all regions), e.g. ``segm/segm-1/fcs/compensated/regNNN_compensated.fcs``. Compensated tables are
          preferred over uncompensated ones; other tables of the same region (e.g. of a second segmentation run) raise
          an error. The ``cycCCC_chHHH`` marker columns are stored in ``X`` (dense float32) under their marker names,
          taken from the FCS ``$PnS`` keywords or the CSV header ``cycCCC_chHHH:<marker>``; all other columns go to
          ``obs``, the ``x``/``y`` pixel centroids to ``obsm["spatial"]``. A table without ``cycCCC_chHHH`` columns
          reads its numeric columns that are not CODEX Processor columns (``x``, ``size``, ...) as the markers, with a
          warning listing them. Other CSV files (without ``x``/``y`` and ``cell_id`` or ``cycCCC_chHHH`` columns,
          e.g. antibody lists) and FCS files of other instruments are ignored.

    Raw acquisition and intermediate folders (``cycCCC_regNNN/``, ``tiles/`` and, next to ``stitched/``, ``bestFocus/``)
    are ignored, as are hidden files and folders (names starting with ``.``, e.g. macOS ``._*`` files, and
    ``__MACOSX/``); symlinked folders are followed. The region id ``regNNN`` is taken from the file names (``reg`` and
    digits not preceded by a letter, e.g. ``exp_reg001`` but not ``Treg2``); a ``{vx.QPTIFF!r}`` without one is named as
    :func:`spatialdata_io.phenocycler` names it: after its file, without the ``.er.qptiff`` or ``{vx.QPTIFF!r}`` suffix
    and a trailing ``_Scan<N>`` (unless that makes two regions share a name), and the extended-range ``.er.qptiff`` of a
    scan is preferred, also for one with a region id (``exp_reg001.er.qptiff`` and ``exp_reg001.qptiff`` are region
    ``reg001``). Several scans with the same region id raise an error. If a directory contains no such files, its top-level ``{vx.IMAGE_TIF}(f)`` files are read as
    hyperstacks named after their files without the extension (``.ome.tif`` included), made valid element names by
    :func:`spatialdata.sanitize_name` (e.g. ``Kidney sample.ome.tif`` becomes the region ``Kidney_sample``). Each region is stored as the image ``<region>_image``
    and its metadata in ``sdata.attrs["akoya"]``.

    A single region is stored in the ``"global"`` coordinate system. Several CIM scans of the same sample space
    (same slide, with stage positions and equal pixel sizes) are placed at their stage positions in ``"global"``;
    otherwise (e.g. CODEX Processor regions, which have no stage positions) each region is stored in its own
    coordinate system, named after the region. A region with cells or a mask but no image is unrelated to the
    scans, so it also gets its own coordinate system. All elements of a region share its coordinate system.

    The cells of each region are stored as circles ``<region>_cells`` with the cells' area (``size`` column),
    annotated by one table ``"table"`` (``obs["region"]`` names the annotated element, ``obs["cell_id"]`` is the
    instance key), or by one table ``<region>_table`` per region if the regions have different markers. Columns of
    the cell table that spatialdata cannot write are sanitised. A region with a segmentation mask (see ``labels``)
    gets the labels ``<region>_labels`` instead of circles, and its cells annotate them. The paths of the tables and
    masks read are stored in ``sdata.attrs["akoya"]["cell_tables"]`` and ``["labels"]``, by region id (``"*"`` for a
    file of several regions).

    .. seealso::

        - `CODEX output <https://help.codex.bio/codex/processor/technical-notes/expected-output>`_.

    Parameters
    ----------
    path
        Path to a CODEX output directory, or to a single image (``{vx.QPTIFF!r}`` or ``{vx.IMAGE_TIF}(f)``).
    fcs
        Whether to prefer ``{vx.FCS_FILE!r}`` cell tables over ``{vx.FCS_FILE_CSV!r}`` ones when both exist. With
        `False`, a region that only has a ``{vx.FCS_FILE!r}`` table is read from it, with a warning.
    imread_kwargs
        Ignored, with a warning: every image is read lazily with ``tifffile``. Kept for backward compatibility.
    regions
        Regions to read, e.g. ``["reg001"]``, including only their cells. If `None`, all regions are read.
    channel_names
        Channel names to use instead of the discovered ones; must have one entry per channel of every region read.
    cells_table
        A CODEX cell table to read instead of the discovered ones (explicit beats discovered): a path, or a mapping
        from region id to path, e.g. ``{{"reg001": "reg001_cells.csv"}}``. A single path for several regions needs a
        ``region`` column.
    labels
        A segmentation mask (TIFF/OME-TIFF with integer instance ids) per region: a path for a single region, or a
        mapping from region id to path; a mask inside ``path`` is not read as an image. If `None`, a TIFF named after
        a region image or with its region id, ending in ``_mask`` or ``_labels`` (e.g. ``reg001_mask.ome.tiff``,
        ``Kidney_labels.tif`` next to ``Kidney.tif``, or ``Tonsil_mask.tif`` or ``Tonsil_Scan1_mask.tif`` next to
        ``Tonsil_Scan1.qptiff``), is the mask of that region; such files are never read as images.
        Each mask is stored lazily as
        the labels ``<region>_labels`` with the transformation of the region's image; the cells of the region annotate
        it instead of circles, and their instance ids must be the mask values. A mask whose shape is the region image's
        divided by an integer factor ``f`` (exactly, or rounded up) is scaled onto the image (``Scale([f, f])`` before
        the region's transformation); the cell centroids stay in image pixels. Any other shape mismatch raises a
        warning. Of a mask file with several planes, the first is read.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if imread_kwargs:
        warnings.warn(
            "`imread_kwargs` is ignored: every image is read lazily with tifffile.", UserWarning, stacklevel=2
        )
    regions = normalize_regions(regions)
    channel_names = [channel_names] if isinstance(channel_names, str) else channel_names
    path = Path(path)
    # explicit beats discovered: a given table disables table discovery, and a given mask is no image
    masks = labels.values() if isinstance(labels, Mapping) else [] if labels is None else [labels]
    layout = _discover_legacy(
        path,
        fcs,
        find_tables=cells_table is None,
        find_masks=labels is None,
        masks={Path(mask).resolve() for mask in masks},
    )
    # the regions of a table without a region in its file name are known only once it is read
    regionless = _read_regionless_table(layout.tables["*"], layout) if "*" in layout.tables else None
    table_regions = set(regionless.region.unique()) if regionless is not None else set()
    available = sorted(layout.images.keys() | layout.tables.keys() - {"*"} | table_regions)
    selected = select_regions(available, regions)
    _warn_fcs_fallback([file for key, file in layout.tables.items() if key == "*" or key in selected], fcs)
    table_files = resolve_per_region(cells_table, selected, available)
    if labels is None:
        mask_files = {region: file for region, file in layout.masks.items() if region in selected}
    else:
        mask_files = resolve_per_region(labels, selected, available)
    experiment = read_json_object(layout.experiment_json) if layout.experiment_json is not None else None
    names_txt = _read_channel_names_txt(layout.channel_names_txt) if layout.channel_names_txt is not None else None

    scans: list[AkoyaScan] = []
    arrays: dict[str, da.Array] = {}
    for region in (region for region in selected if region in layout.images):
        files = layout.images[region]
        if _is_qptiff(files[0]):
            scans.append(read_qptiff_metadata(files[0], region=region))
            arrays[region] = read_qptiff_image(scans[-1])
            continue
        arrays[region], channels = _read_legacy_image(
            files, layout.channel_names_txt, image_models_kwargs=image_models_kwargs
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
    if cells_table is not None:
        cell_tables = read_region_tables(table_files, available=list(layout.images), selected=selected)
        table_paths = {key: str(file) for key, file in table_files.items()}
    else:
        named_files = {key: layout.tables[key] for key in selected if key in layout.tables}
        named = read_region_tables(named_files, available=available, selected=selected)
        read = dict(zip(named_files, named, strict=True))
        if regionless is not None:
            regionless = regionless.take(np.flatnonzero(regionless.region.isin(selected).to_numpy()))
            if len(regionless.region):
                read = {"*": regionless, **read}
        cell_tables = list(read.values())
        table_paths = {}
        for key, table in read.items():
            # by region id, like a given table; `"*"` only for a region-less table of several regions
            values = table.region.unique()
            region = values[0] if key == "*" and len(values) == 1 and values[0] not in read else key
            table_paths[region] = str(layout.tables[key])
    sdata = build_spatialdata(
        scans,
        arrays,
        names,
        cell_tables,
        mask_files,
        table_paths=table_paths,
        image_models_kwargs=image_models_kwargs,
        labels_models_kwargs=labels_models_kwargs,
    )
    return _set_reader_metadata(sdata, "codex")


def _warn_fcs_fallback(files: Iterable[Path], fcs: bool) -> None:
    fallback = [str(file) for file in files if file.suffix.lower() == CodexKeys.FCS_FILE.value]
    if not fcs and fallback:
        warnings.warn(
            f"`fcs=False`, but no CSV cell table was found for {fallback}; reading the FCS table(s) instead.",
            UserWarning,
            stacklevel=3,
        )


def _read_regionless_table(file: Path, layout: _LegacyLayout) -> CellTable:
    """Read the discovered cell table ``file`` that has no region in its file name."""
    only_image = next(iter(layout.images)) if len(layout.images) == 1 else None
    # next to a single image named after its file (e.g. `Kidney.tif`), a single region value names that image
    table = read_cell_table(
        file, default_region=only_image, regions=layout.images.keys() | layout.tables.keys() - {"*"}
    )
    # values that name no region of the dataset are regions without an image
    table.region = table.region.map({value: sanitize_name(value) for value in table.region.unique()})
    return table


def _discover_legacy(
    path: Path,
    fcs: bool,
    *,
    find_tables: bool = True,
    find_masks: bool = True,
    masks: Collection[Path] = frozenset(),
) -> _LegacyLayout:
    """Find the images, cell tables and sidecar files of a CODEX dataset.

    Parameters
    ----------
    path
        A CODEX output directory, searched recursively in sorted order, or a single image or table file.
    fcs
        Whether to prefer ``.fcs`` over ``.csv`` tables of the same region.
    find_tables
        Whether to discover cell tables.
    find_masks
        Whether to discover the segmentation masks of the regions.
    masks
        The resolved paths of the segmentation masks, which are not images.

    Returns
    -------
    The discovered files.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    files = [path] if path.is_file() else _walk(path)
    if masks:
        files = [file for file in files if file.resolve() not in masks]
    # masks named after their region are never images
    named_masks = [_is_tiff(file) and _MASK.search(file.name) is not None for file in files]
    image_files = [file for file, is_mask in zip(files, named_masks, strict=True) if not is_mask]
    mask_files = [file for file, is_mask in zip(files, named_masks, strict=True) if is_mask]
    images = _discover_images(image_files)
    if not images:
        # the original (2019) layout: one stitched `.tif` next to its table, without a region in the file name
        top_level = image_files if path.is_file() else [file for file in image_files if file.parent == path]
        images = {sanitize_name(IMAGE_EXTENSION.sub("", file.name)): [file] for file in top_level if _is_tiff(file)}
    tables = _discover_tables(files, fcs) if find_tables else {}
    if not images and not tables:
        raise FileNotFoundError(f"No CODEX images or cell tables found in {path}. Expected: {'; '.join(_PATTERNS)}.")
    return _LegacyLayout(
        images=dict(sorted(images.items())),
        tables=dict(sorted(tables.items())),
        masks=_discover_masks(mask_files, images) if find_masks else {},
        channel_names_txt=_find_sidecar(path, files, CodexKeys.CHANNEL_NAMES_FILE.value),
        experiment_json=_find_sidecar(path, files, CodexKeys.EXPERIMENT_FILE.value),
    )


def _read_legacy_image(
    files: list[Path],
    channel_names_txt: Path | None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[da.Array, list[AkoyaChannel]]:
    """Lazily read the image of one legacy CODEX region.

    Parameters
    ----------
    files
        The per-marker images of the region, or its single hyperstack, as returned by :func:`_discover_legacy`.
    channel_names_txt
        Optional ``channelNames.txt`` naming the hyperstack channels when the file has no ImageJ ``Labels``.
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
        return _read_per_marker(markers, chunks)
    if len(files) != 1:
        raise ValueError(f"Expected per-marker images or a single hyperstack, got {[str(f) for f in files]}.")

    image, leading_shape, labels = read_tiff(files[0], chunks)
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
    markers: list[tuple[re.Match[str], Path]], chunks: tuple[int, int]
) -> tuple[da.Array, list[AkoyaChannel]]:
    parsed = sorted((int(match.group(2)), int(match.group(3)), match.group(4), file) for match, file in markers)

    planes = []
    for _, _, _, file in parsed:
        plane, _, _ = read_tiff(file, chunks)
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
    return da.stack(planes), channels


def _spatial_chunks(image_models_kwargs: Mapping[str, Any]) -> tuple[int, int]:
    """The ``(y, x)`` chunk size requested in ``image_models_kwargs``, with the reader default when unspecified."""
    default, _ = _initialize_raster_models_kwargs({}, {})
    chunks = _initialize_raster_models_kwargs(image_models_kwargs, {})[0]["chunks"]
    return spatial_chunks(chunks, default["chunks"][-2:])


def _walk(root: Path) -> list[Path]:
    """All visible files below ``root`` in sorted order, skipping raw acquisition and intermediate processing folders."""
    files: list[Path] = []
    for directory, dirnames, filenames in walk_visible(root):
        depth = len(directory.relative_to(root).parts)
        has_stitched = any(name.casefold() == "stitched" for name in dirnames)
        dirnames[:] = [name for name in dirnames if depth < _MAX_DEPTH and not _is_ignored_dir(name, has_stitched)]
        files.extend(directory / name for name in filenames)
    return files


def _is_ignored_dir(name: str, has_stitched: bool) -> bool:
    folded = name.casefold()
    return folded == "tiles" or (folded == "bestfocus" and has_stitched) or _RAW_CYCLE_DIR.fullmatch(name) is not None


def _discover_images(files: Sequence[Path]) -> dict[str, list[Path]]:
    per_marker: dict[str, dict[tuple[int, int], Path]] = {}
    stacks: dict[str, list[Path]] = {}
    all_qptiffs = []
    for file in files:
        region = region_id(file.name)
        if _is_qptiff(file):
            all_qptiffs.append(file)
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

    # named like `phenocycler()` names them, e.g. `exp_reg001.qptiff` -> `reg001`, `Tonsil_Scan1.er.qptiff` -> `Tonsil`
    qptiffs = qptiff_regions(all_qptiffs)

    images: dict[str, list[Path]] = {}
    # the Processor's stitched outputs come first: cell tables refer to their pixel space
    for region in sorted(per_marker.keys() | stacks.keys() | qptiffs.keys()):
        if region in per_marker:
            images[region] = [per_marker[region][key] for key in sorted(per_marker[region])]
        elif region in stacks:
            images[region] = [_single(region, _preferred_stacks(stacks[region]))]
        else:
            images[region] = [qptiffs[region]]
    return images


def _discover_masks(files: Sequence[Path], images: Mapping[str, Sequence[Path]]) -> dict[str, Path]:
    """The masks among ``files`` (``<image>_mask.tif`` or ``_labels``, also ``.ome.tiff``) of the regions of ``images``.

    A mask belongs to the region in its file name (``regNNN``), else to the region it is named after: the region id
    (e.g. ``Tonsil_mask.tif``) or the file name of its image (``Tonsil_Scan1_mask.tif`` of ``Tonsil_Scan1.qptiff``).
    """
    by_image = {region: region for region in images} | {
        IMAGE_EXTENSION.sub("", file.name): region for region, paths in images.items() for file in paths
    }
    masks: dict[str, list[Path]] = {}
    for file in files:
        region = region_id(file.name) or by_image.get(_MASK.sub("", file.name))
        if region is None or region not in images:
            warnings.warn(
                f"Ignoring the mask {file}: it is named after no region image. Pass it as `labels` to read it.",
                UserWarning,
                stacklevel=4,
            )
            continue
        masks.setdefault(region, []).append(file)
    for region, paths in masks.items():
        if len(paths) > 1:
            raise ValueError(
                f"Found several masks for region '{region}': {[str(p) for p in paths]}. Pass the mask to read as "
                "`labels`."
            )
    return {region: paths[0] for region, paths in sorted(masks.items())}


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
        if file.suffix.lower() in _TABLE_SUFFIXES and is_codex_table(file):
            candidates.setdefault(region_id(file.name) or "*", []).append(file)

    def uncompensated(file: Path) -> bool:
        return any("uncompensated" in part.casefold() for part in file.parts)

    tables = {}
    for region, paths in candidates.items():
        best = min(map(uncompensated, paths))
        # the same table may be written as both .fcs and .csv: the preferred format is read
        same_table: dict[Path, list[Path]] = {}
        for path in sorted(paths):
            if uncompensated(path) == best:
                same_table.setdefault(path.with_suffix(""), []).append(path)
        if len(same_table) > 1:
            where = "without a region in their file name" if region == "*" else f"of region '{region}'"
            raise ValueError(
                f"Found several cell tables {where}: {[str(p) for group in same_table.values() for p in group]}; the reader cannot "
                "choose between them. Pass the path of a directory that contains only one, or pass the table as "
                "`cells_table`."
            )
        (formats,) = same_table.values()
        tables[region] = min(formats, key=lambda path: path.suffix.lower() != preferred)
    return tables


def _find_sidecar(path: Path, files: Sequence[Path], name: str) -> Path | None:
    """The shallowest file called ``name`` in ``files``, else in the nearest ancestor of ``path`` holding one.

    Names are compared case-insensitively: SPACEc and the HuBMAP codex-pipeline write ``channelnames.txt``.
    """
    folded = name.casefold()
    matches = sorted(
        (file for file in files if file.name.casefold() == folded), key=lambda file: (len(file.parts), str(file))
    )
    if matches:
        return matches[0]
    # absolute, so that the ancestors of e.g. `codex(".")` are searched too
    start = (path.parent if path.is_file() else path).absolute()
    for directory in [start, *start.parents][: _SIDECAR_ANCESTORS + 1]:
        try:
            found = sorted(file for file in directory.iterdir() if file.name.casefold() == folded and file.is_file())
        except OSError:  # e.g. an unreadable ancestor directory
            continue
        if found:
            return found[0]
    return None


def _read_channel_names_txt(path: Path) -> list[str]:
    data = path.read_bytes()
    # written on Windows: UTF-16 with a byte order mark, UTF-8 (with or without one) or the ANSI code page
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _pixel_size_um(experiment: Mapping[str, Any] | None) -> float | None:
    """Pixel size in µm from the ``experiment.json`` of the CODEX Processor (nm), ``None`` when it is missing."""
    for key in ("xyResolution", "per_pixel_XY_resolution", "Resolution_nm"):
        value = (experiment or {}).get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return value / 1000
    return None


def _is_qptiff(path: Path) -> bool:
    return path.suffix.lower() == CodexKeys.QPTIFF.value


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in _TIFF_SUFFIXES
