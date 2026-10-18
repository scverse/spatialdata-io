"""CODEX Processor cell tables and segmentation masks shared by the Akoya readers, and the assembly of their output."""

from __future__ import annotations

import csv
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import flowio  # readfcs' parser, for the TEXT segment only
import numpy as np
import pandas as pd
import readfcs
from spatialdata import SpatialData, sanitize_name, sanitize_table
from spatialdata._logging import logger
from spatialdata.models import Labels2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale, set_transformation
from spatialdata.transformations import Sequence as TransformationSequence
from xarray import DataTree

from spatialdata_io._constants._constants import CodexKeys, PhenocyclerKeys
from spatialdata_io.readers._utils._akoya import (
    build_image_element,
    dedupe_channel_names,
    read_tiff,
    region_transformations,
    scans_to_attrs,
    spatial_chunks,
)
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    import dask.array as da
    from geopandas import GeoDataFrame
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = [
    "CellTable",
    "build_cells",
    "build_spatialdata",
    "is_codex_table",
    "mask_downsample_factor",
    "raster_shape",
    "read_cell_table",
    "read_labels",
    "read_region_labels",
    "read_region_tables",
    "resolve_per_region",
]

# `cyc001_ch001` (FCS `$PnN`, plain CSV header) or the name part of the Processor CSV header `cyc001_ch001:DAPI1`
_MARKER_COLUMN = re.compile(r"cyc(\d+)_ch(\d+)", re.IGNORECASE)
# non-marker columns of CODEX Processor tables, for tables whose marker columns are named after the markers
_META_COLUMNS = frozenset(
    {"cell_id", "region", "tile_num", "x", "y", "z", "x_tile", "y_tile", "size", "tsne_x", "tsne_y", "homogeneity"}
)


@dataclass
class CellTable:
    """A CODEX cell table.

    Attributes
    ----------
    obs
        The non-marker columns, indexed by the string instance ids.
    X
        The marker intensities, ``(n_cells, n_markers)`` float32.
    var
        Per-marker metadata indexed by the de-duplicated marker names: ``source_column`` and, for
        ``cycCCC_chHHH`` columns, ``cycle`` and ``channel``.
    region
        The region id of every cell.
    instance_id
        The instance id of every cell (int64).
    centroids_px
        The ``(x, y)`` centroids in level-0 pixels of the region image.
    radius_px
        The radius of a circle with the cell's area, or `None` if the table has no area.
    row_instance_ids
        Whether the instance ids are row numbers because the table has no instance id column.
    """

    obs: pd.DataFrame
    X: np.ndarray
    var: pd.DataFrame
    region: pd.Series
    instance_id: pd.Series
    centroids_px: np.ndarray
    radius_px: np.ndarray | None
    row_instance_ids: bool = False

    def take(self, rows: np.ndarray) -> CellTable:
        """The cells at the positional indices ``rows``."""
        return CellTable(
            obs=self.obs.iloc[rows],
            X=self.X[rows],
            var=self.var,
            region=self.region.iloc[rows],
            instance_id=self.instance_id.iloc[rows],
            centroids_px=self.centroids_px[rows],
            radius_px=None if self.radius_px is None else self.radius_px[rows],
            row_instance_ids=self.row_instance_ids,
        )


def is_codex_table(path: Path) -> bool:
    """Whether ``path`` is a CODEX Processor ``.fcs`` or ``.csv`` cell table.

    Its columns (the ``$PnN`` channel names of an FCS file) must include ``x`` and ``y`` and ``cell_id`` or a
    ``cycCCC_chHHH`` column. Other CSV files (e.g. antibody lists) and FCS files of other instruments, also
    unreadable ones, are not.
    """
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        try:
            columns = list(flowio.read_multiple_data_sets(str(path), only_text=True)[0].pnn_labels)
        # flowio raises these, not only its own exceptions, for malformed files (e.g. truncated, without `$DATATYPE`)
        except (ValueError, KeyError, IndexError, EOFError, OSError, flowio.exceptions.FlowIOException):
            return False
    else:
        columns = _read_header(path)
    # the name part of the CODEX Processor header form `cyc001_ch001:DAPI1`
    names = {column.partition(":")[0].casefold() for column in columns}
    return {"x", "y"} <= names and (
        CodexKeys.INSTANCE_KEY.value in names or any(_MARKER_COLUMN.fullmatch(name) for name in names)
    )


def read_cell_table(path: Path, *, default_region: str | None, regions: Collection[str] | None = None) -> CellTable:
    """Read a CODEX Processor ``.fcs`` or ``.csv`` cell table.

    The markers are the ``cycCCC_chHHH`` columns, named by the FCS ``$PnS`` keywords or by the display part of the
    Processor CSV header ``cycCCC_chHHH:<marker>`` (in ``var`` with their ``cycle`` and ``channel``), else, with a
    `UserWarning`, the numeric columns that are not CODEX Processor columns (``x``, ``size``, ...). Centroids
    ``x``/``y`` in pixels, instance ids ``cell_id`` (else the row number in the file, from 1), the region from
    ``region`` (a numeric region ``r`` is the region id ``regRRR``), the radius from ``size``.

    Parameters
    ----------
    path
        The cell table.
    default_region
        The region id of all cells if the table has no region column.
    regions
        The region ids of the dataset, to which the values of the region column are matched, also after
        :func:`spatialdata.sanitize_name`; values that name none of them are kept. If the table has
        ``default_region`` and its region column holds a single value that names no region, that value names
        ``default_region``. If `None`, the values are the region ids.

    Returns
    -------
    The cell table.
    """
    df = _read_fcs(path) if path.suffix.lower() == CodexKeys.FCS_FILE.value else _read_text_table(path)
    headers = [str(column) for column in df.columns]
    split = [header.partition(":") for header in headers]
    # the source of an FCS column is its `$PnN`
    fcs = path.suffix.lower() == CodexKeys.FCS_FILE.value
    sources = [name if fcs else header for header, (name, _, _) in zip(headers, split, strict=True)]
    names = dedupe_channel_names([name for name, _, _ in split])
    displays = [display.strip() or name for name, (_, _, display) in zip(names, split, strict=True)]
    df.columns = pd.Index(names)
    columns = {column.casefold(): column for column in names}
    x, y = columns.get("x"), columns.get("y")
    if x is None or y is None:
        raise ValueError(f"{path} has no 'x' and 'y' centroid columns; found {names}.")
    instance_column = columns.get(CodexKeys.INSTANCE_KEY.value)
    region_column = columns.get(CodexKeys.REGION_KEY.value)
    area_column = columns.get("size")

    matches = [_MARKER_COLUMN.fullmatch(name) for name in names]
    if any(matches):
        is_marker = [match is not None for match in matches]
    else:
        is_marker = [
            name.casefold() not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df.iloc[:, i])
            for i, name in enumerate(names)
        ]
        taken = [name for name, marker in zip(names, is_marker, strict=True) if marker]
        warnings.warn(
            f"{path} has no 'cycCCC_chHHH' columns; reading the {len(taken)} numeric columns {taken} that are not "
            "CODEX Processor columns as the markers.",
            UserWarning,
            stacklevel=4,
        )
    positions = [i for i, marker in enumerate(is_marker) if marker]
    markers = [names[i] for i in positions]
    var = pd.DataFrame(
        {"source_column": [sources[i] for i in positions]},
        index=pd.Index(dedupe_channel_names([displays[i] for i in positions])),
    )
    if positions and all(matches[i] for i in positions):
        var["cycle"] = [int(matches[i].group(1)) for i in positions]  # type: ignore[union-attr]
        var["channel"] = [int(matches[i].group(2)) for i in positions]  # type: ignore[union-attr]

    if instance_column is not None:
        instance_id = pd.Series(_instance_ids(path, df[instance_column]))
    else:
        # the row numbers in the file; label images use 0 for the background, so instance ids start at 1
        instance_id = pd.Series(np.arange(1, len(df) + 1, dtype=np.int64))
    index = pd.Index(instance_id.astype(str))
    instance_id.index = index
    instance_id.name = CodexKeys.INSTANCE_KEY.value

    region = _region_values(df, region_column, default_region, regions)
    if region is None:
        raise ValueError(
            f"{path} has no region column, so its cells cannot be assigned to regions: a table for several regions "
            "needs a 'region' column. Alternatively, pass one table per region as a mapping {region: path}."
        )
    region = pd.Series(region.to_numpy(), index=index, name=CodexKeys.REGION_KEY.value)
    _check_unique_instance_ids(path, region, instance_id)

    # the instance ids and regions are stored under the keys of the table element
    obs = df.drop(columns=[*markers, *(c for c in (instance_column, region_column) if c is not None)])
    obs.index = index
    radius = None if area_column is None else np.sqrt(df[area_column].to_numpy(dtype=np.float64) / np.pi)
    return CellTable(
        obs=obs,
        X=df[markers].to_numpy(dtype=np.float32),
        var=var,
        region=region,
        instance_id=instance_id,
        centroids_px=df[[x, y]].to_numpy(dtype=np.float64),
        radius_px=radius,
        row_instance_ids=instance_column is None,
    )


def _region_values(
    df: pd.DataFrame, region_column: str | None, default_region: str | None, regions: Collection[str] | None
) -> pd.Series | None:
    """The region of every row of ``df``, see :func:`read_cell_table`; `None` if it is unknown."""
    if region_column is None:
        return None if default_region is None else pd.Series(default_region, index=df.index)
    values = df[region_column]
    if pd.api.types.is_numeric_dtype(values):
        values = values.map({value: f"reg{int(value):03d}" for value in values.dropna().unique()})
    values = values.astype(str)
    if regions is None:
        return values
    matched = {value: _match_region(value, regions) for value in values.unique()}
    if default_region is not None and len(matched) == 1 and None in matched.values():
        # the region column of a table for one region names another region (e.g. `1` of an image `Kidney.tif`)
        return pd.Series(default_region, index=df.index)
    return values.map({value: value if region is None else region for value, region in matched.items()})


def _match_region(value: str, regions: Collection[str]) -> str | None:
    """The region named by ``value``, also after :func:`spatialdata.sanitize_name` (``A [1]`` names ``A__1_``)."""
    return next((candidate for candidate in (value, sanitize_name(value)) if candidate in regions), None)


def _read_fcs(path: Path) -> pd.DataFrame:
    """The events of an FCS file, with the columns named ``$PnN:$PnS`` like the CODEX Processor CSV header."""
    fcs = readfcs.ReadFCS(path)
    df = fcs.data  # positional columns, named by `$PnN`
    labels = fcs.channels["PnS"] if "PnS" in fcs.channels.columns else [""] * df.shape[1]
    names = fcs.channels["PnN"]
    df.columns = pd.Index(
        [f"{name}:{label.strip()}" if label.strip() else name for name, label in zip(names, labels, strict=True)]
    )
    return df


def _instance_ids(path: Path, column: pd.Series) -> np.ndarray:
    """The integer instance ids in ``column``; raises for missing or non-integer ids, which no label can have."""
    if pd.api.types.is_integer_dtype(column):
        return column.to_numpy(dtype=np.int64)
    ids = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~(np.isfinite(ids) & (ids == np.round(ids)))
    if invalid.any():
        raise ValueError(
            f"{path} has {invalid.sum()} cell(s) whose instance id in '{column.name}' is missing or not an integer "
            f"(e.g. {column[invalid].tolist()[:5]}); every cell needs an integer instance id, the value of its label."
        )
    return ids.astype(np.int64)


def _check_unique_instance_ids(path: Path, region: pd.Series, instance_id: pd.Series) -> None:
    """Raise if two cells of the same region have the same instance id: they could not be told apart."""
    regions, ids = region.to_numpy(), instance_id.to_numpy()
    duplicated = pd.DataFrame({"region": regions, "id": ids}).duplicated().to_numpy()
    if duplicated.any():
        first = regions[duplicated][0]
        found = sorted(set(ids[duplicated & (regions == first)].tolist()))
        raise ValueError(
            f"{path} has {len(found)} duplicate instance id(s) {found[:5]} in region '{first}'; every cell of a "
            "region needs its own instance id."
        )


def _read_header(path: Path) -> list[str]:
    """The column names of a comma-separated table."""
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    return [column.strip() for column in next(csv.reader([header]))]


def _read_text_table(path: Path) -> pd.DataFrame:
    """Read a comma-separated table with unique column names."""
    df = pd.read_csv(path, engine="pyarrow")
    # pandas renames repeated names (DAPI.1; with the pyarrow engine since 3.1), so the names come from the header and
    # are made unique like the marker names
    df.columns = pd.Index(dedupe_channel_names(_read_header(path)))
    return df


def resolve_per_region(
    value: str | Path | Mapping[str, str | Path] | None,
    regions: Sequence[str],
    available: Sequence[str] = (),
) -> dict[str, Path]:
    """Resolve a ``cells_table`` or ``labels`` argument to one file per region.

    Parameters
    ----------
    value
        A path, a mapping from region id to path, or `None`. The keys are sanitised with
        :func:`spatialdata.sanitize_name`, like the region ids.
    regions
        The regions read.
    available
        All regions of the dataset; mapping keys that are one of them but not read are ignored.

    Returns
    -------
    ``{}`` for `None`; ``{region: path}`` for a path and a single region, else ``{"*": path}`` (a file for all
    regions); the mapping, in the order of ``regions``.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        by_region = {sanitize_name(str(key)): file for key, file in value.items()}
        if len(by_region) < len(value):
            raise ValueError(
                f"Several of the keys {list(value)} name the same region after `spatialdata.sanitize_name`; pass "
                "each region once."
            )
        unknown = sorted(set(by_region) - set(regions) - set(available))
        if unknown:
            raise ValueError(f"Unknown region(s) {unknown}. Available regions: {list(available or regions)}.")
        files = {region: Path(by_region[region]) for region in regions if region in by_region}
    else:
        files = {regions[0] if len(regions) == 1 else "*": Path(value)}
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(f"{file} does not exist or is not a file.")
    return files


def read_region_tables(
    files: Mapping[str, Path], *, available: Collection[str], selected: Sequence[str]
) -> list[CellTable]:
    """Read cell tables, as resolved by :func:`resolve_per_region`, and assign their cells to regions.

    A table given for one region holds the cells of that region; if its region column holds a single value that
    names no region, it holds only cells of that region. Cells whose region names no available region are dropped
    with a warning; cells of available regions that are not read are dropped silently.

    Parameters
    ----------
    files
        The table of each region, or ``{"*": path}`` for one table for all regions.
    available
        All region ids of the dataset.
    selected
        The region ids read.

    Returns
    -------
    The cell tables, one per file.
    """
    tables = []
    for key, path in files.items():
        table = read_cell_table(path, default_region=None if key == "*" else key, regions=available)
        tables.append(_assign_regions(table, path, key, available, selected))
    return tables


def _assign_regions(
    table: CellTable, path: Path, key: str, available: Collection[str], selected: Sequence[str]
) -> CellTable:
    values = list(table.region.unique())
    unmatched = sorted(value for value in values if value not in available)
    if unmatched:
        warnings.warn(
            f"Ignoring the cells of {unmatched} in {path}: they match none of the regions {sorted(available)}.",
            UserWarning,
            stacklevel=4,
        )
    keep = table.region.isin(selected if key == "*" else [key]).to_numpy()
    if not keep.any():
        target = f"region '{key}'" if key != "*" else f"the regions read {list(selected)}"
        raise ValueError(f"{path} has no cells of {target}; its region column holds {values}.")
    return table.take(np.flatnonzero(keep))


def read_labels(
    path: Path, labels_models_kwargs: Mapping[str, Any], transformations: Mapping[str, BaseTransformation]
) -> DataArray | DataTree:
    """Lazily read a segmentation mask as a ``Labels2DModel``.

    Parameters
    ----------
    path
        A TIFF or OME-TIFF with integer instance ids (0 = background). Of a file with several planes, the first is
        read.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`; their ``chunks`` also set the
        spatial chunks of uncompressed files, which are memory-mapped.
    transformations
        The transformations of the region's image, by coordinate system.

    Returns
    -------
    The labels element.
    """
    _, labels_models_kwargs = _initialize_raster_models_kwargs({}, labels_models_kwargs)
    _, defaults = _initialize_raster_models_kwargs({}, {})
    image, _, _ = read_tiff(path, spatial_chunks(labels_models_kwargs["chunks"], defaults["chunks"]))
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"{path} has dtype {image.dtype}; a segmentation mask must have an integer dtype.")
    return Labels2DModel.parse(image[0], dims=("y", "x"), transformations=dict(transformations), **labels_models_kwargs)


def read_region_labels(
    files: Mapping[str, Path],
    transformations: Mapping[str, Mapping[str, BaseTransformation]],
    labels_models_kwargs: Mapping[str, Any],
    image_shapes: Mapping[str, tuple[int, int]],
) -> dict[str, DataArray | DataTree]:
    """Read the mask of each region, as resolved by :func:`resolve_per_region`, with the region's transformations.

    A mask whose ``(y, x)`` shape differs from ``image_shapes`` of its region is scaled onto the image if it is
    downsampled by an integer factor, see :func:`mask_downsample_factor`: its transformations become
    ``Sequence([Scale([f, f]), t])`` for each region transformation ``t``. Any other mismatch raises a warning: its
    pixels would not align with the image.

    Returns
    -------
    The labels elements by region id.
    """
    if "*" in files:
        raise ValueError(
            f"A single labels mask {files['*']} was given for several regions; a mask belongs to one region image, "
            "so pass a mapping {region: path}."
        )
    labels = {}
    for region, path in files.items():
        region_transformations = _transformations_of(transformations, region)
        labels[region] = read_labels(path, labels_models_kwargs, region_transformations)
        shape, image_shape = raster_shape(labels[region]), image_shapes.get(region)
        if image_shape is None or shape == tuple(image_shape):
            continue
        factor = mask_downsample_factor(shape, image_shape)
        if factor is None:
            warnings.warn(
                f"The mask {path} has shape {shape}, but the image of region '{region}' has shape "
                f"{tuple(image_shape)} (y, x); the labels will not align with the image.",
                UserWarning,
                stacklevel=3,
            )
            continue
        logger.info(
            "The mask %s has shape %s, the image of region '%s' %s (y, x); the labels are scaled by %d to align.",
            path,
            shape,
            region,
            tuple(image_shape),
            factor,
        )
        scale = Scale([factor, factor], axes=("y", "x"))
        set_transformation(
            labels[region],
            {
                cs: TransformationSequence([scale, transformation])
                for cs, transformation in region_transformations.items()
            },
            set_all=True,
        )
    return labels


def mask_downsample_factor(mask_shape: tuple[int, int], image_shape: tuple[int, int]) -> int | None:
    """The integer factor ``f > 1`` by which a mask of ``(y, x)`` shape ``mask_shape`` downsamples an image.

    The mask must be ``image_shape / f`` along both axes, exactly or rounded up (each image size within ``f - 1``
    pixels of ``f`` times the mask size); `None` if there is no such factor.
    """
    candidates = {-(-image // mask) for image, mask in zip(image_shape, mask_shape, strict=True) if mask > 0}
    for factor in sorted(candidates):
        if factor > 1 and all(-(-image // factor) == mask for image, mask in zip(image_shape, mask_shape, strict=True)):
            return factor
    return None


def _transformations_of(
    transformations: Mapping[str, Mapping[str, BaseTransformation]], region: str
) -> dict[str, BaseTransformation]:
    """A copy of the transformations of ``region``, ``Identity`` to ``"global"`` if it has none."""
    return dict(transformations.get(region, {"global": Identity()}))


def raster_shape(element: DataArray | DataTree) -> tuple[int, int]:
    """The level-0 ``(y, x)`` shape of a raster element."""
    data = element["scale0"]["image"] if isinstance(element, DataTree) else element
    return int(data.sizes["y"]), int(data.sizes["x"])


def build_cells(
    tables: Sequence[CellTable],
    labels: Mapping[str, DataArray | DataTree],
    transformations: Mapping[str, Mapping[str, BaseTransformation]],
) -> tuple[dict[str, ad.AnnData], dict[str, GeoDataFrame]]:
    """Build the table elements and, for regions without labels, circle shapes from cell tables.

    Parameters
    ----------
    tables
        The cell tables, each covering one or more regions.
    labels
        The labels elements by region id; their regions' cells annotate ``<region>_labels`` instead of the circles
        ``<region>_cells`` built here.
    transformations
        The transformations of each region, see :func:`region_transformations`; missing regions get ``Identity``
        to ``"global"``.

    Returns
    -------
    The table elements, ``{"table": ...}`` if all tables have the same ``var`` apart from its ``source_column``
    (dropped if it differs), else ``{"<region>_table": ...}`` per region, and the circle shapes by element name. The
    circles have the cells' area (radius 1 px without a positive one).
    """
    _check_disjoint_regions(tables)
    row_ids = sorted(
        region for table in tables if table.row_instance_ids for region in table.region.unique() if region in labels
    )
    if row_ids:
        warnings.warn(
            f"The cell table of region(s) {row_ids} has no instance id column, so its row numbers are used as "
            "instance ids; they may not match the values of the mask.",
            UserWarning,
            stacklevel=3,
        )
    # tables with other markers are not concatenated; the source columns only record where a marker was read from
    # (e.g. `cyc001_ch001:DAPI1` of a CSV, `cyc001_ch001` of an FCS)
    panels = [_panel(table.var) for table in tables]
    pieces = _split_by_region(tables) if any(not panel.equals(panels[0]) for panel in panels) else None

    def element(region: str) -> str:
        suffix = PhenocyclerKeys.LABELS_SUFFIX if region in labels else PhenocyclerKeys.CELLS_SUFFIX
        return f"{region}{suffix}"

    shapes = {}
    for table in tables:
        for region, rows in _region_rows(table).items():
            if region in labels:
                continue
            radius = np.ones(len(rows)) if table.radius_px is None else table.radius_px[rows]
            if (no_area := ~(radius > 0)).any():
                logger.info(
                    "%d cells of region '%s' have no positive area; their circles get radius 1.", no_area.sum(), region
                )
                radius = np.where(no_area, 1.0, radius)
            shapes[element(region)] = ShapesModel.parse(
                table.centroids_px[rows],
                geometry=0,
                radius=radius,
                index=pd.Index(table.instance_id.to_numpy()[rows]),
                transformations=_transformations_of(transformations, region),
            )
    if shapes:
        circle_regions = sorted(name.removesuffix(PhenocyclerKeys.CELLS_SUFFIX) for name in shapes)
        warnings.warn(
            f"No segmentation mask found for region(s) {circle_regions}; cells are represented as circles.",
            UserWarning,
            stacklevel=3,
        )

    if pieces is None:
        return {PhenocyclerKeys.TABLE_KEY.value: _to_anndata(tables, element)}, shapes
    return {f"{region}_table": _to_anndata([piece], element) for region, piece in sorted(pieces.items())}, shapes


def build_spatialdata(
    scans: Sequence[AkoyaScan],
    images: Mapping[str, da.Array],
    channel_names: Mapping[str, list[str]],
    cell_tables: Sequence[CellTable],
    mask_files: Mapping[str, Path],
    *,
    table_paths: Mapping[str, str] | None,
    image_models_kwargs: Mapping[str, Any],
    labels_models_kwargs: Mapping[str, Any],
    duplicate_scans: Mapping[str, AkoyaScan] | None = None,
) -> SpatialData:
    """Assemble the regions of an Akoya read into a `SpatialData` with their metadata in ``attrs["akoya"]``.

    Each region gets the image ``<region>_image``, the labels ``<region>_labels`` from its mask (see
    :func:`read_region_labels`) and the cells (see :func:`build_cells`), all in the coordinate systems of
    :func:`region_transformations`; regions with cells or a mask but no image are unrelated to the scans. The cell
    centroids are in image pixels, also when the mask is downsampled.

    Parameters
    ----------
    scans
        The scans read.
    images
        The lazy image of each scan, by region id, see :func:`build_image_element`.
    channel_names
        The channel names of each scan, by region id.
    cell_tables
        The cell tables read.
    mask_files
        The mask of each region, as resolved by :func:`resolve_per_region`.
    table_paths
        The paths of the cell tables, stored in ``attrs["akoya"]["cell_tables"]`` unless `None`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`.
    duplicate_scans
        Further scans of the same physical scan as a region (e.g. its 8-bit file), by that region's id. Each is
        stored as its own image ``<duplicate.region>_image`` with exactly the transformations of the region; their
        images and channel names are looked up in ``images`` and ``channel_names`` by ``duplicate.region``.

    Returns
    -------
    The `SpatialData` object.
    """
    cell_regions = {region for table in cell_tables for region in table.region.unique()}
    transformations = region_transformations(scans, cell_regions | (mask_files.keys() - {"*"}))
    image_elements = {
        f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}": build_image_element(
            images[scan.region], scan, channel_names[scan.region], transformations[scan.region], image_models_kwargs
        )
        for scan in scans
    }
    for region, duplicate in (duplicate_scans or {}).items():
        image_elements[f"{duplicate.region}{PhenocyclerKeys.IMAGE_SUFFIX}"] = build_image_element(
            images[duplicate.region],
            duplicate,
            channel_names[duplicate.region],
            transformations[region],
            image_models_kwargs,
        )
    image_shapes = {
        scan.region: raster_shape(image_elements[f"{scan.region}{PhenocyclerKeys.IMAGE_SUFFIX}"]) for scan in scans
    }
    labels = read_region_labels(mask_files, transformations, labels_models_kwargs, image_shapes)
    tables, shapes = build_cells(cell_tables, labels, transformations) if cell_tables else ({}, {})
    sdata = SpatialData(
        images=image_elements,
        labels={f"{region}{PhenocyclerKeys.LABELS_SUFFIX}": element for region, element in labels.items()},
        shapes=shapes,
        tables=tables,
    )
    attrs = scans_to_attrs([*scans, *(duplicate_scans or {}).values()], channel_names)
    if table_paths is not None:
        attrs["cell_tables"] = dict(table_paths)
    if mask_files:
        attrs["labels"] = {region: str(file) for region, file in mask_files.items()}
    sdata.attrs[PhenocyclerKeys.ATTRS_KEY.value] = attrs
    return sdata


def _region_rows(table: CellTable) -> dict[str, np.ndarray]:
    """The row indices of each region of ``table``, in order of first appearance."""
    codes, regions = pd.factorize(table.region)
    return {str(region): np.flatnonzero(codes == i) for i, region in enumerate(regions)}


def _check_disjoint_regions(tables: Sequence[CellTable]) -> None:
    seen: set[str] = set()
    for table in tables:
        regions = set(table.region.unique())
        if overlap := seen & regions:
            raise ValueError(f"Several cell tables contain the cells of region(s) {sorted(overlap)}.")
        seen |= regions


def _panel(var: pd.DataFrame) -> pd.DataFrame:
    """The marker metadata of ``var`` without the provenance column ``source_column``."""
    return var.drop(columns="source_column", errors="ignore")


def _split_by_region(tables: Sequence[CellTable]) -> dict[str, CellTable]:
    return {region: table.take(rows) for table in tables for region, rows in _region_rows(table).items()}


def _to_anndata(tables: Sequence[CellTable], element: Callable[[str], str]) -> ad.AnnData:
    """Concatenate cell tables with the same markers into one table annotating the elements of their regions."""
    region_key, instance_key = CodexKeys.REGION_KEY.value, CodexKeys.INSTANCE_KEY.value
    obs = pd.concat([table.obs for table in tables]) if len(tables) > 1 else tables[0].obs.copy()
    regions = pd.concat([table.region for table in tables]) if len(tables) > 1 else tables[0].region
    if not obs.index.is_unique:
        obs.index = pd.Index(regions.to_numpy(dtype=object) + "_" + obs.index.to_numpy(dtype=object))
    X = np.concatenate([table.X for table in tables]) if len(tables) > 1 else tables[0].X
    var = tables[0].var.copy()
    if any(not table.var.equals(tables[0].var) for table in tables):
        # the same markers read from columns of other names
        var = _panel(var)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm[CodexKeys.SPATIAL_KEY.value] = np.concatenate([table.centroids_px for table in tables])
    # spatialdata only writes alphanumeric column names and `_.-`
    if any(sanitize_name(str(column), is_dataframe_column=True) != column for column in adata.obs.columns):
        sanitize_table(adata, inplace=True)
    elements = regions.map({region: element(region) for region in regions.unique()}).to_numpy()
    adata.obs[region_key] = pd.Categorical(elements, categories=sorted(set(elements)))
    adata.obs[instance_key] = np.concatenate([table.instance_id.to_numpy() for table in tables])
    return TableModel.parse(
        adata,
        region=list(adata.obs[region_key].cat.categories),
        region_key=region_key,
        instance_key=instance_key,
    )
