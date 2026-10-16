from __future__ import annotations

import math
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
import tifffile
import zarr
from dask.base import tokenize
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._constants import CodexKeys, PhenocyclerKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    build_image_element,
    read_json_object,
    region_transformations,
    scans_to_attrs,
    select_regions,
)
from spatialdata_io.readers._utils._cell_tables import CellTable, build_cells, read_codex_table
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
# marker columns of CODEX Processor tables: `cyc001_ch001` or `cyc001_ch001:DAPI1`
_MARKER_COLUMN = re.compile(r"cyc\d+_ch\d+", re.IGNORECASE)
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
    are read as hyperstacks named after their files. Each region is stored as the image ``<region>_image`` in the
    ``"global"`` coordinate system and its metadata in ``sdata.attrs["akoya"]``. The cells of each region are stored
    as circles ``<region>_cells`` with the cells' area (``size`` column), annotated by one table ``"table"``
    (``obs["region"]`` names the circles, ``obs["cell_id"]`` is the instance key), or by one table
    ``<region>_table`` per region if the regions have different markers. The paths of the tables read are stored in
    ``sdata.attrs["akoya"]["cell_tables"]``.

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
        Not implemented yet.
    cells_table_format
        Not implemented yet.
    labels
        Not implemented yet.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`, which is only used for ``{vx.IMAGE_TIF}(f)``
        files that :mod:`tifffile` cannot read lazily (e.g. unsupported compression).
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Not implemented yet.
    cells_table_kwargs
        Not implemented yet.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if cells_table is not None or labels is not None:
        raise NotImplementedError("Reading cell tables and labels with `codex` is added in a later task.")

    path = Path(path)
    layout = discover_legacy(path, fcs)
    selected = select_regions(sorted(layout.images.keys() | layout.tables.keys() - {"*"}), regions)
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
    transformations = region_transformations(scans)
    images = {
        f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}": build_image_element(
            arrays[scan.region], scan, names[scan.region], transformations[scan.region], image_models_kwargs
        )
        for scan in scans
    }
    table_files = {key: file for key, file in layout.tables.items() if key == "*" or key in selected}
    cell_tables = _read_tables(table_files, layout, selected if regions is not None else None, fcs)
    tables, shapes = build_cells(list(cell_tables.values()), {}, {}, transformations) if cell_tables else ({}, {})
    sdata = SpatialData(images=images, shapes=shapes, tables=tables)
    sdata.attrs[PhenocyclerKeys.ATTRS_KEY.value] = scans_to_attrs(scans, names)
    sdata.attrs[PhenocyclerKeys.ATTRS_KEY.value]["cell_tables"] = {key: str(table_files[key]) for key in cell_tables}
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

    image, leading_shape, labels, read_by_tifffile = _read_tiff(files[0], imread_kwargs, chunks)
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
        plane, _, _, lazy = _read_tiff(file, imread_kwargs, chunks)
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


def _read_tiff(
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
            image = da.from_zarr(zarr.open(series.aszarr(), mode="r"))
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
        self.name = f"codex-tiff-{tokenize(str(path), path.stat().st_mtime_ns, offset, shape, str(dtype))}"

    def __getitem__(self, key: Any) -> np.ndarray:
        data = np.memmap(self.path, dtype=self._file_dtype, mode="r", offset=self.offset, shape=self.shape)
        return np.asarray(data[key], dtype=self.dtype)


def _spatial_chunks(image_models_kwargs: Mapping[str, Any]) -> tuple[int, int]:
    """The ``(y, x)`` chunk size requested in ``image_models_kwargs``, with the reader default when unspecified."""
    default, _ = _initialize_raster_models_kwargs({}, {})
    chunks = _initialize_raster_models_kwargs(image_models_kwargs, {})[0]["chunks"]
    if isinstance(chunks, int):
        return chunks, chunks
    if isinstance(chunks, Mapping) and isinstance(chunks.get("y"), int) and isinstance(chunks.get("x"), int):
        return chunks["y"], chunks["x"]
    if isinstance(chunks, list | tuple) and len(chunks) >= 2 and all(isinstance(c, int) for c in chunks[-2:]):
        return chunks[-2], chunks[-1]
    return default["chunks"][-2], default["chunks"][-1]


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
        if region is None and not _is_cell_table(file):
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


def _is_cell_table(path: Path) -> bool:
    """Whether ``path`` is an FCS file or a CSV whose header has CODEX cell-table columns."""
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        return True
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    # the name part of the Processor header form `cyc001_ch001:DAPI1`
    columns = {column.strip().strip('"').partition(":")[0].casefold() for column in header.split(",")}
    return {"x", "y"} <= columns and (
        CodexKeys.INSTANCE_KEY.value in columns or any(_MARKER_COLUMN.fullmatch(column) for column in columns)
    )


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
