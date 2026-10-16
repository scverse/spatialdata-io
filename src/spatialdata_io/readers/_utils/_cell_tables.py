"""Cell-table adapters shared by the Akoya readers: read a table into a `CellTable` and build the table elements."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import readfcs
import shapely
import tifffile
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
from spatialdata_io.readers._utils._rasterize import rasterize_polygons_lazy
from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import dask.array as da
    from geopandas import GeoDataFrame
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray

    from spatialdata_io.readers._utils._qptiff import AkoyaScan

__all__ = [
    "CellTable",
    "build_cells",
    "build_spatialdata",
    "detect_table_format",
    "mask_downsample_factor",
    "polygon_labels",
    "polygon_shapes",
    "raster_shape",
    "read_cell_table",
    "read_codex_table",
    "read_labels",
    "read_qupath_polygons",
    "read_region_labels",
    "read_region_tables",
    "resolve_per_region",
]

TABLE_FORMATS = ("codex", "qupath", "mcquant", "sprm", "inform", "generic")
# the keys of `cells_table_kwargs`; see `read_cell_table` and `read_labels`
CELLS_TABLE_KWARGS = (
    "x",
    "y",
    "instance_id",
    "region_column",
    "markers",
    "coordinates_unit",
    "mask_channel",
    "geojson",
    "geojson_units",
)

# `cyc001_ch001` (FCS `$PnN`, plain CSV header) or the name part of the Processor CSV header `cyc001_ch001:DAPI1`
_MARKER_COLUMN = re.compile(r"cyc(\d+)_ch(\d+)", re.IGNORECASE)
# non-marker columns of CODEX Processor tables, for tables whose marker columns are named after the markers
_META_COLUMNS = frozenset(
    {"cell_id", "region", "tile_num", "x", "y", "z", "x_tile", "y_tile", "size", "tsne_x", "tsne_y", "homogeneity"}
)
# QuPath measurement columns, e.g. `CD3: Cell: Mean`; the per-marker intensity is taken from the first compartment
_QUPATH_MARKER = re.compile(r"(.+): (Cell|Nucleus|Cytoplasm|Membrane): Mean")
_QUPATH_COMPARTMENTS = ("Cell", "Nucleus", "Cytoplasm", "Membrane")
_QUPATH_IMAGE = "Image"
_QUPATH_ID = "Object ID"
# the QuPath object types of cells; other features of a GeoJSON export (e.g. annotations) are not cells
_QUPATH_CELL_TYPES = frozenset({"cell", "detection", "PathCellObject", "PathDetectionObject"})
# inForm `*_cell_seg_data.txt` columns: `Cell X Position (microns)`, `Entire Cell <marker> Mean (<unit>)`, ...
_INFORM_POSITION = re.compile(r"Cell X Position(?: \((.*)\))?")
_INFORM_MARKER = re.compile(r"Entire Cell (.+) Mean(?: \(.*\))?")
_INFORM_AREA = re.compile(r"Entire Cell Area \((.*)\)")
_INFORM_REGION = "Sample Name"
_INFORM_ID = "Cell ID"
# HuBMAP SPRM outputs: `<image>-cell_channel_mean.csv` with the centroids in `<image>-cell_centers.csv`
_SPRM_MEANS_SUFFIX = "-cell_channel_mean.csv"
_SPRM_CENTERS_SUFFIX = "-cell_centers.csv"
# a matched polygon must lie within this distance (px) of its cell's centroid, else the match is considered wrong
_CENTROID_TOLERANCE_PX = 1.0
# the largest fraction of matched polygons that may be far from their centroids before the match is rejected
_MAX_MISPLACED_FRACTION = 0.05


@dataclass
class CellTable:
    """A cell table in reader-independent form.

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
        The ``(x, y)`` centroids in level-0 pixels of the region image, or `None` if the table has none (SPRM without
        ``*-cell_centers.csv``).
    radius_px
        The radius of a circle with the cell's area, or `None` if the table has no area.
    mask_pixels
        Whether ``centroids_px`` and ``radius_px`` are in pixels of the segmentation mask (pixel coordinates of
        mask-based pipelines: mcquant, SPRM, ``"generic"`` in ``"px"``) rather than of the image; they are scaled
        to image pixels when the mask is a downsample of the image, see :func:`build_spatialdata`.
    row_instance_ids
        Whether the instance ids are row numbers because the table has no instance id column.
    polygons_px
        The outline of every cell in level-0 pixels (`None` for a cell without one), or `None` if the table has no
        polygons.
    geojson
        The file of ``polygons_px``.
    """

    obs: pd.DataFrame
    X: np.ndarray
    var: pd.DataFrame
    region: pd.Series
    instance_id: pd.Series
    centroids_px: np.ndarray | None
    radius_px: np.ndarray | None
    row_instance_ids: bool = False
    polygons_px: np.ndarray | None = None
    geojson: Path | None = None
    mask_pixels: bool = False

    def take(self, rows: np.ndarray) -> CellTable:
        """The cells at the positional indices ``rows``."""
        return CellTable(
            obs=self.obs.iloc[rows],
            X=self.X[rows],
            var=self.var,
            region=self.region.iloc[rows],
            instance_id=self.instance_id.iloc[rows],
            centroids_px=None if self.centroids_px is None else self.centroids_px[rows],
            radius_px=None if self.radius_px is None else self.radius_px[rows],
            row_instance_ids=self.row_instance_ids,
            polygons_px=None if self.polygons_px is None else self.polygons_px[rows],
            geojson=self.geojson,
            mask_pixels=self.mask_pixels,
        )


def read_codex_table(path: Path, region: str | None = None) -> CellTable:
    """Read a CODEX Processor ``.fcs`` or ``.csv`` cell table.

    Marker columns are the ``cycCCC_chHHH`` columns; their names come from the FCS ``$PnS`` keywords or from the
    display part of the Processor CSV header ``cycCCC_chHHH:<marker>``. A table without such columns has one
    marker column per numeric column that is not a known CODEX Processor column (``x``, ``size``, ...).

    Parameters
    ----------
    path
        The ``.fcs`` or ``.csv`` file.
    region
        The region id of all cells if the table has no ``region`` column. A numeric ``region`` column ``r`` is
        mapped to the region id ``regRRR``.

    Returns
    -------
    The cell table.
    """
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        fcs = readfcs.ReadFCS(path)
        df = fcs.data  # positional columns, named by `$PnN`
        names = list(fcs.channels["PnN"])
        labels = list(fcs.channels["PnS"]) if "PnS" in fcs.channels.columns else [""] * len(names)
    else:
        df = pd.read_csv(path, engine="pyarrow")
        names = [str(column) for column in df.columns]
        labels = [""] * len(names)
    sources = names
    # the Processor CSV header form `cyc001_ch001:DAPI1` (also `cell_id:cell_id`)
    split = [name.partition(":") for name in names]
    df.columns = pd.Index([name for name, _, _ in split])
    displays = [label.strip() or display or name for (name, _, display), label in zip(split, labels, strict=True)]

    marker_matches = [_MARKER_COLUMN.fullmatch(column) for column in df.columns]
    if any(marker_matches):
        is_marker = [match is not None for match in marker_matches]
    else:
        is_marker = [
            column.casefold() not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
            for column in df.columns
        ]
    markers = [i for i, marker in enumerate(is_marker) if marker]
    var = pd.DataFrame(
        {"source_column": [sources[i] for i in markers]},
        index=pd.Index(dedupe_channel_names([displays[i] for i in markers])),
    )
    if any(marker_matches):
        var["cycle"] = [int(marker_matches[i].group(1)) for i in markers]  # type: ignore[union-attr]
        var["channel"] = [int(marker_matches[i].group(2)) for i in markers]  # type: ignore[union-attr]
    X = df.iloc[:, markers].to_numpy(dtype=np.float32)
    obs = df.iloc[:, [i for i, marker in enumerate(is_marker) if not marker]]

    columns = {column.casefold(): column for column in obs.columns}
    if "x" not in columns or "y" not in columns:
        raise ValueError(f"{path} has no 'x' and 'y' centroid columns; found {list(obs.columns)}.")
    centroids = obs[[columns["x"], columns["y"]]].to_numpy(dtype=np.float64)
    radius = np.sqrt(obs[columns["size"]].to_numpy(dtype=np.float64) / np.pi) if "size" in columns else None

    instance_key = CodexKeys.INSTANCE_KEY.value
    if instance_key in columns:
        instance_id = obs[columns[instance_key]].astype(np.int64)
    else:
        # label images use 0 for the background, so instance ids start at 1
        instance_id = pd.Series(np.arange(1, len(obs) + 1, dtype=np.int64))
    instance_id.index = pd.Index(instance_id.astype(str))
    obs.index = instance_id.index
    instance_id.name = instance_key

    region_key = CodexKeys.REGION_KEY.value
    if region_key in columns:
        regions = obs[columns[region_key]]
        if pd.api.types.is_numeric_dtype(regions):
            ids = {value: f"reg{int(value):03d}" for value in regions.unique()}
            regions = regions.map(ids)
        region_ids = regions.astype(str)
    elif region is not None:
        region_ids = pd.Series(region, index=obs.index)
    else:
        raise ValueError(
            f"{path} has no '{region_key}' column and no region id in its file name, so its cells cannot be "
            "assigned to a region; a table for several regions needs a region column."
        )
    region_ids.name = region_key
    _check_unique_instance_ids(path, region_ids, instance_id)
    return CellTable(
        obs=_drop_key_columns(obs, columns.get(instance_key), columns.get(region_key)),
        X=X,
        var=var,
        region=region_ids,
        instance_id=instance_id,
        centroids_px=centroids,
        radius_px=radius,
        row_instance_ids=instance_key not in columns,
    )


def detect_table_format(path: Path) -> str:
    """Detect the format of a cell table from its file type and header.

    Parameters
    ----------
    path
        The ``.fcs``, ``.csv``, ``.tsv`` or ``.txt`` (tab-separated) cell table.

    Returns
    -------
    ``"codex"`` (FCS, or ``x``/``y`` with ``cell_id`` or ``cycCCC_chHHH`` columns), ``"qupath"`` (``Centroid X µm``
    or ``Object ID``), ``"mcquant"`` (``CellID`` and ``X_centroid``), ``"inform"`` (``Cell X Position``, optionally
    followed by its unit in parentheses),
    ``"sprm"`` (first column ``ID``), else ``"generic"``.
    """
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        return "codex"
    columns, _ = _read_header(path)
    # the name part of the CODEX Processor header form `cyc001_ch001:DAPI1`
    names = {column.partition(":")[0].casefold() for column in columns}
    if {"x", "y"} <= names and (
        CodexKeys.INSTANCE_KEY.value in names or any(_MARKER_COLUMN.fullmatch(name) for name in names)
    ):
        return "codex"
    if {"Centroid X µm", "Centroid X px", _QUPATH_ID} & set(columns):
        return "qupath"
    if {"CellID", "X_centroid"} <= set(columns):
        return "mcquant"
    if any(_INFORM_POSITION.fullmatch(column) for column in columns):
        return "inform"
    if columns[:1] == ["ID"]:
        return "sprm"
    return "generic"


def read_cell_table(
    path: Path,
    fmt: str,
    *,
    pixel_size_um: float | Mapping[str, float | None] | None,
    default_region: str | None,
    cells_table_kwargs: Mapping[str, Any],
) -> CellTable:
    """Read a cell table of any supported format.

    Parameters
    ----------
    path
        The cell table.
    fmt
        One of ``"auto"`` (see :func:`detect_table_format`), ``"codex"``, ``"qupath"``, ``"mcquant"``,
        ``"sprm"``, ``"inform"`` or ``"generic"``:

            - ``"codex"``: see :func:`read_codex_table`.
            - ``"qupath"``: a QuPath measurement export. Centroids ``Centroid X µm``/``Centroid Y µm`` (divided by
              ``pixel_size_um``) or ``Centroid X px``/``Centroid Y px``; instance ids from a numeric ``Object ID``,
              else the row number (from 1); the region from the ``Image`` column. The markers are the
              ``<marker>: <compartment>: Mean`` columns, one per marker from the first of the ``Cell``,
              ``Nucleus``, ``Cytoplasm`` and ``Membrane`` compartments (in ``var["compartment"]``). The radius
              comes from ``Cell: Area µm^2`` (or ``px^2``). The cell outlines are read from a QuPath GeoJSON
              export, see :func:`read_qupath_polygons`: ``cells_table_kwargs["geojson"]``, else
              ``<table stem>.geojson`` or ``<table stem>_cells.geojson`` next to the table, if it exists.
            - ``"mcquant"``: an MCMICRO ``mcquant`` table. Instance ids ``CellID``, pixel centroids
              ``X_centroid``/``Y_centroid``, the markers are the columns between ``CellID`` and ``X_centroid``, the
              radius comes from ``Area``.
            - ``"sprm"``: a HuBMAP SPRM ``<image>-cell_channel_mean.csv`` (validated on synthetic data only). Instance
              ids ``ID`` (the values of the mask, ``pipeline_output/mask/*.ome.tiff`` channel 0, passed as ``labels``),
              one marker per remaining column. Pixel centroids come from ``<image>-cell_centers.csv`` (``ID,x,y``)
              next to the table, which must hold every cell; without it a `UserWarning` is raised and the table has
              no ``obsm["spatial"]`` (the centroids are never computed from the mask), so its region needs a mask.
            - ``"inform"``: a PerkinElmer/Akoya inForm ``*_cell_seg_data.txt`` (tab-separated; validated on synthetic
              data only). Instance ids ``Cell ID``, centroids ``Cell X Position``/``Cell Y Position`` in the unit
              named in the header, ``(microns)`` (divided by ``pixel_size_um``) or ``(pixels)``; pixels if the
              header names none. The region comes from ``Sample Name``. The markers are the
              ``Entire Cell <marker> Mean (...)`` columns; the radius comes from ``Entire Cell Area (...)`` if its
              unit equals that of the positions.
            - ``"generic"``: columns given by ``cells_table_kwargs``.

    pixel_size_um
        The pixel size of the region image, to convert µm centroids to pixels; for a table of several regions, the
        pixel size of each region, by region id: every cell is converted with the pixel size of the region its
        region column names (see :func:`read_region_tables`), cells of other regions get NaN coordinates.
    default_region
        The region id of all cells if the table has no region column.
    cells_table_kwargs
        Optional keys: ``region_column`` (the column holding the region of each cell; QuPath default ``Image``),
        and for ``"generic"`` tables ``x`` and ``y`` (required centroid columns), ``instance_id`` (else the row
        number, from 1), ``markers`` (a list of columns or a regular expression matching the full column name;
        default: all other numeric columns) and ``coordinates_unit`` (``"px"``, default, or ``"um"``); for
        ``"qupath"`` tables ``geojson`` (the GeoJSON file of the cell outlines) and ``geojson_units`` (the unit of
        its coordinates, ``"px"`` of the full-resolution image, default and what QuPath writes, or ``"um"``). The
        key ``mask_channel`` is used by :func:`read_labels`.

    Returns
    -------
    The cell table. Its regions are the values of the region column, or ``default_region``.
    """
    unknown = sorted(set(cells_table_kwargs) - set(CELLS_TABLE_KWARGS))
    if unknown:
        raise ValueError(f"`cells_table_kwargs` has unknown key(s) {unknown}; expected some of {CELLS_TABLE_KWARGS}.")
    if fmt == "auto":
        fmt = detect_table_format(path)
        logger.info("Reading %s as a %r cell table.", path, fmt)
    if fmt != "qupath" and cells_table_kwargs.get("geojson") is not None:
        raise ValueError(f"`cells_table_kwargs['geojson']` is only read with QuPath cell tables; {path} is {fmt!r}.")
    if fmt == "codex":
        return read_codex_table(path, region=default_region)
    if fmt not in TABLE_FORMATS:
        raise ValueError(f"Unknown cell table format {fmt!r}; expected 'auto' or one of {TABLE_FORMATS}.")

    _, sep = _read_header(path)
    df = pd.read_csv(path, sep=sep, engine="pyarrow")
    df.columns = pd.Index([str(column) for column in df.columns])
    reader = {
        "qupath": _qupath_roles,
        "mcquant": _mcquant_roles,
        "sprm": _sprm_roles,
        "inform": _inform_roles,
        "generic": _generic_roles,
    }[fmt]
    roles = reader(path, df, cells_table_kwargs)
    mask_pixels = fmt in {"mcquant", "sprm"} or (fmt == "generic" and not roles["in_um"])
    units = cells_table_kwargs.get("geojson_units", "px")
    row_pixel_sizes = None
    if roles["in_um"] or (fmt == "qupath" and units == "um"):
        row_pixel_sizes = _row_pixel_sizes(path, df, roles["region_column"], pixel_size_um)
    if roles.pop("in_um"):
        if row_pixel_sizes is None:
            raise ValueError(
                f"{path} has centroids in µm, but no pixel size is known to convert them to pixels (the region's "
                "pixel size is missing)."
            )
        roles["px_per_unit"] = 1 / row_pixel_sizes
    table = _from_frame(path, df, default_region=default_region, **roles)
    table.mask_pixels = mask_pixels
    if fmt == "sprm":
        table.centroids_px = _read_sprm_centers(path, table)
    if fmt == "qupath":
        geojson = cells_table_kwargs.get("geojson")
        geojson = _find_geojson(path) if geojson is None else Path(geojson)
        if geojson is not None:
            table.polygons_px = read_qupath_polygons(geojson, table, path, units=units, pixel_size_um=row_pixel_sizes)
            table.geojson = geojson
    return table


def _row_pixel_sizes(
    path: Path, df: pd.DataFrame, region_column: str | None, pixel_size_um: float | Mapping[str, float | None] | None
) -> float | np.ndarray | None:
    """The pixel size of every row of ``df``, or one for all rows; `None` if it is unknown.

    For pixel sizes by region id, each row gets that of the region its region column names, NaN if it names none of
    them (its cells are not read). Raises if a region named by the table has no pixel size.
    """
    if not isinstance(pixel_size_um, Mapping):
        return pixel_size_um
    if region_column is None or region_column not in df.columns:
        sizes = set(pixel_size_um.values())
        return sizes.pop() if len(sizes) == 1 else None
    values = df[region_column].astype(str)
    regions = {value: _match_region(value, list(pixel_size_um)) for value in values.unique()}
    unknown = sorted({region for region in regions.values() if region is not None and pixel_size_um[region] is None})
    if unknown:
        raise ValueError(
            f"{path} has coordinates in µm, but the pixel size of region(s) {unknown} is unknown, so they cannot be "
            "converted to pixels."
        )
    sizes_by_value = {value: np.nan if region is None else pixel_size_um[region] for value, region in regions.items()}
    return values.map(sizes_by_value).to_numpy(dtype=np.float64)


def _find_geojson(table_path: Path) -> Path | None:
    """The GeoJSON next to a QuPath table: ``<stem>.geojson``, else ``<stem>_cells.geojson``.

    Other GeoJSON files, e.g. HuBMAP's ``*tissue-boundary.geojson`` tissue outline, are not cells.
    """
    candidates = (table_path.with_name(f"{table_path.stem}{suffix}.geojson") for suffix in ("", "_cells"))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def read_qupath_polygons(
    geojson: Path,
    table: CellTable,
    table_path: Path,
    *,
    units: str = "px",
    pixel_size_um: float | np.ndarray | None = None,
) -> np.ndarray:
    """Read the cell outlines of a QuPath GeoJSON export and match them to the cells of its measurement table.

    A feature belongs to the cell whose ``Object ID`` is the feature ``id`` (the object's UUID, QuPath >= 0.4);
    features of other objects (e.g. annotations) are skipped. If the table has no ``Object ID`` column or the
    features have no ids, the cell and detection features (``objectType``) are matched to the table rows in order,
    and their number must equal the number of rows.

    Every matched polygon must contain its cell's centroid or lie within 1 px of it; if more than 5% do not, the
    polygons belong to other cells (or ``units`` is wrong) and a `ValueError` is raised.

    Parameters
    ----------
    geojson
        A GeoJSON ``FeatureCollection`` or JSON array of features (QuPath *Export as GeoJSON*).
    table
        The cell table read from ``table_path``.
    table_path
        The QuPath measurement table, for error messages.
    units
        The unit of the coordinates: ``"px"`` (of the full-resolution image, what QuPath writes) or ``"um"``.
    pixel_size_um
        The pixel size, to convert ``"um"`` coordinates, or the pixel size of every row of ``table``.

    Returns
    -------
    The polygon of every row of ``table`` in pixels, `None` for rows without one.
    """
    if units not in ("px", "um"):
        raise ValueError(f"`cells_table_kwargs['geojson_units']` must be 'px' or 'um', got {units!r}.")
    if units == "um" and pixel_size_um is None:
        raise ValueError(f"{geojson} has coordinates in µm, but no pixel size is known to convert them to pixels.")
    if not geojson.is_file():
        raise FileNotFoundError(f"{geojson} does not exist or is not a file.")
    ids, geometries, kinds = _read_geojson(geojson)

    polygons = np.full(len(table.obs), None, dtype=object)
    if _QUPATH_ID in table.obs.columns and ids is not None:
        object_ids = pd.Index(table.obs[_QUPATH_ID].astype(str))
        if not object_ids.is_unique:
            raise ValueError(f"The '{_QUPATH_ID}' column of {table_path} is not unique.")
        rows = object_ids.get_indexer(pd.Index(ids).astype(str))
        found = rows >= 0
        if not found.any():
            raise ValueError(
                f"{geojson} does not hold the cells of {table_path}: none of the feature ids is an '{_QUPATH_ID}' "
                "of the table."
            )
        if not pd.Index(rows[found]).is_unique:
            raise ValueError(f"{geojson} has several features with the same id.")
        polygons[rows[found]] = geometries[found]
        logger.info("Matched %d of the %d features of %s to cells by id.", found.sum(), len(geometries), geojson)
        if found.sum() < len(polygons):
            warnings.warn(
                f"{len(polygons) - found.sum()} of the {len(polygons)} cells of {table_path} have no polygon in "
                f"{geojson}; they are in the table but not in the labels and polygons.",
                UserWarning,
                stacklevel=5,
            )
    else:
        cells = geometries if kinds is None else geometries[np.isin(kinds, list(_QUPATH_CELL_TYPES))]
        if len(cells) != len(polygons):
            raise ValueError(
                f"{geojson} has {len(cells)} cell polygons without ids, but {table_path} has {len(polygons)} rows; "
                f"without ids ('{_QUPATH_ID}'), polygons are matched to cells in order."
            )
        polygons[:] = cells
    if units == "um":
        _scale_polygons(polygons, 1 / np.broadcast_to(np.asarray(pixel_size_um, dtype=np.float64), polygons.shape))
    if table.centroids_px is not None:
        _check_centroids(polygons, table.centroids_px, geojson, table_path)
    return polygons


def _scale_polygons(polygons: np.ndarray, factors: np.ndarray) -> None:
    """Scale every polygon by its factor in place; polygons with a NaN factor (cells that are not read) are dropped."""
    polygons[np.isnan(factors)] = None
    for factor in np.unique(factors[~np.isnan(factors)]):
        rows = np.flatnonzero((factors == factor) & pd.notna(polygons))
        polygons[rows] = shapely.transform(polygons[rows], partial(np.multiply, factor))


def _check_centroids(polygons: np.ndarray, centroids_px: np.ndarray, geojson: Path, table_path: Path) -> None:
    """Raise if too many polygons are far from the centroids of the cells they were matched to."""
    # cells of regions that are not read have no pixel coordinates
    matched = pd.notna(polygons) & np.isfinite(centroids_px).all(axis=1)
    if not matched.any():
        return
    distance = shapely.distance(polygons[matched], shapely.points(centroids_px[matched]))
    misplaced = int(np.count_nonzero(~(distance <= _CENTROID_TOLERANCE_PX)))
    if misplaced > _MAX_MISPLACED_FRACTION * matched.sum():
        raise ValueError(
            f"{misplaced} of the {matched.sum()} polygons of {geojson} are more than {_CENTROID_TOLERANCE_PX:g} px "
            f"from the centroid of the cell of {table_path} they were matched to: the polygons are not those of "
            "these cells, are in another order, or `cells_table_kwargs['geojson_units']` is wrong."
        )
    if misplaced:
        logger.info(
            "%d polygons of %s are more than %g px from their cell's centroid.",
            misplaced,
            geojson,
            _CENTROID_TOLERANCE_PX,
        )


def _read_geojson(path: Path) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """The ids (`None` if there are none), geometries and QuPath object types (`None` if unknown) of the features."""
    with path.open("rb") as f:
        is_array = f.read(1024).lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"[")
    if is_array:
        # QuPath's default export without the FeatureCollection option, which GDAL does not read
        features = json.loads(path.read_text(encoding="utf-8-sig"))
        geometries = np.array(
            [shapely.geometry.shape(f["geometry"]) if f.get("geometry") else None for f in features], dtype=object
        )
        ids = [f.get("id") for f in features]
        types = [(f.get("properties") or {}).get("objectType") for f in features]
    else:
        frame = gpd.read_file(path)
        geometries = frame.geometry.to_numpy()
        ids = list(frame["id"]) if "id" in frame.columns else [None] * len(frame)
        types = list(frame["objectType"]) if "objectType" in frame.columns else [None] * len(frame)
    if all(kind is None for kind in types) and any(i in _QUPATH_CELL_TYPES for i in ids):
        # QuPath < 0.4 names the object type in the feature id, e.g. `PathCellObject`
        types, ids = ids, [None] * len(ids)
    has_ids = any(i is not None and not pd.isna(i) for i in ids)
    has_types = any(kind is not None for kind in types)
    return (
        np.asarray(ids, dtype=object) if has_ids else None,
        geometries,
        np.asarray(types, dtype=object) if has_types else None,
    )


def _qupath_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    unit = next((unit for unit in ("µm", "px") if f"Centroid X {unit}" in df.columns), None)
    if unit is None:
        raise ValueError(f"{path} has no 'Centroid X µm' or 'Centroid X px' column; found {list(df.columns)}.")
    compartments: dict[str, dict[str, str]] = {}
    for column in df.columns:
        if (match := _QUPATH_MARKER.fullmatch(column)) is not None:
            compartments.setdefault(match.group(1), {})[match.group(2)] = column
    chosen = {marker: next(c for c in _QUPATH_COMPARTMENTS if c in found) for marker, found in compartments.items()}
    markers = [compartments[marker][compartment] for marker, compartment in chosen.items()]
    if not markers:
        warnings.warn(
            f"{path} has no '<marker>: <compartment>: Mean' columns, so its table has no marker intensities.",
            UserWarning,
            stacklevel=5,
        )
    var = pd.DataFrame(
        {"source_column": markers, "compartment": list(chosen.values())},
        index=pd.Index(dedupe_channel_names(list(chosen))),
    )
    object_id = df.get(_QUPATH_ID)
    area = f"Cell: Area {unit}^2"
    return {
        "markers": markers,
        "var": var,
        "x": f"Centroid X {unit}",
        "y": f"Centroid Y {unit}",
        "in_um": unit == "µm",
        "instance_column": _QUPATH_ID if object_id is not None and pd.api.types.is_integer_dtype(object_id) else None,
        "region_column": kwargs.get("region_column", _QUPATH_IMAGE if _QUPATH_IMAGE in df.columns else None),
        "area_column": area if area in df.columns else None,
    }


def _mcquant_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    columns = list(df.columns)
    if "CellID" not in columns or "X_centroid" not in columns:
        raise ValueError(f"{path} has no 'CellID' and 'X_centroid' columns; found {columns}.")
    markers = columns[columns.index("CellID") + 1 : columns.index("X_centroid")]
    return {
        "markers": markers,
        "var": pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(markers))),
        "x": "X_centroid",
        "y": "Y_centroid",
        "in_um": False,
        "instance_column": "CellID",
        "region_column": kwargs.get("region_column"),
        "area_column": "Area" if "Area" in columns else None,
    }


def _sprm_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if "ID" not in df.columns:
        raise ValueError(f"{path} has no 'ID' column; found {list(df.columns)}.")
    if not pd.api.types.is_integer_dtype(df["ID"]):
        raise ValueError(f"The 'ID' column of {path} must hold integer cell ids, got dtype {df['ID'].dtype}.")
    region_column = kwargs.get("region_column")
    markers = [c for c in df.columns if c not in ("ID", region_column)]
    return {
        "markers": markers,
        "var": pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(markers))),
        "x": None,
        "y": None,
        "in_um": False,
        "instance_column": "ID",
        "region_column": region_column,
        "area_column": None,
    }


def _read_sprm_centers(path: Path, table: CellTable) -> np.ndarray | None:
    """The ``(x, y)`` pixel centroids of the cells of an SPRM table from its ``*-cell_centers.csv``, if it exists."""
    name = path.name
    centers = path.with_name(
        f"{name.removesuffix(_SPRM_MEANS_SUFFIX)}{_SPRM_CENTERS_SUFFIX}"
        if name.endswith(_SPRM_MEANS_SUFFIX)
        else f"{path.stem}{_SPRM_CENTERS_SUFFIX}"
    )
    if not centers.is_file():
        warnings.warn(
            f"No {centers.name} next to {path}, so the cells of this SPRM table have no centroids and the table has "
            "no 'spatial' coordinates; the cells need a segmentation mask (`labels`).",
            UserWarning,
            stacklevel=4,
        )
        return None
    frame = pd.read_csv(centers)
    if not {"ID", "x", "y"} <= set(frame.columns):
        raise ValueError(f"{centers} must have the columns 'ID', 'x' and 'y'; found {list(frame.columns)}.")
    if not frame["ID"].is_unique:
        raise ValueError(f"The 'ID' column of {centers} is not unique.")
    rows = pd.Index(frame["ID"]).get_indexer(pd.Index(table.instance_id))
    if (rows < 0).any():
        raise ValueError(f"{centers} has no centroid for {int((rows < 0).sum())} of the {len(rows)} cells of {path}.")
    return frame[["x", "y"]].to_numpy(dtype=np.float64)[rows]


def _inform_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    positions = [m for column in df.columns if (m := _INFORM_POSITION.fullmatch(column)) is not None]
    y = next((c for c in df.columns if c.startswith("Cell Y Position")), None)
    if not positions or y is None:
        raise ValueError(f"{path} has no 'Cell X Position' and 'Cell Y Position' columns; found {list(df.columns)}.")
    x = positions[0].group(0)
    unit = (positions[0].group(1) or "pixels").casefold()
    if unit not in ("microns", "pixels"):
        raise ValueError(f"The unit '{unit}' of the column '{x}' of {path} must be 'microns' or 'pixels'.")
    markers, names = [], []
    for column in df.columns:
        if (match := _INFORM_MARKER.fullmatch(column)) is not None:
            markers.append(column)
            names.append(match.group(1))
    if not markers:
        warnings.warn(
            f"{path} has no 'Entire Cell <marker> Mean' columns, so its table has no marker intensities.",
            UserWarning,
            stacklevel=5,
        )
    area = next((c for c in df.columns if (m := _INFORM_AREA.fullmatch(c)) is not None and m.group(1)), None)
    if area is not None:
        area_unit = "microns" if "micron" in area else "pixels"
        area = area if area_unit == unit else None
    return {
        "markers": markers,
        "var": pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(names))),
        "x": x,
        "y": y,
        "in_um": unit == "microns",
        "instance_column": _INFORM_ID if _INFORM_ID in df.columns else None,
        "region_column": kwargs.get("region_column", _INFORM_REGION if _INFORM_REGION in df.columns else None),
        "area_column": area,
    }


def _generic_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if "x" not in kwargs or "y" not in kwargs:
        raise ValueError(
            f"Reading {path} as a 'generic' cell table needs the centroid columns as `cells_table_kwargs` 'x' and "
            f"'y'; found the columns {list(df.columns)}."
        )
    unit = kwargs.get("coordinates_unit", "px")
    if unit not in ("px", "um"):
        raise ValueError(f"`cells_table_kwargs['coordinates_unit']` must be 'px' or 'um', got {unit!r}.")
    roles = {kwargs["x"], kwargs["y"], kwargs.get("instance_id"), kwargs.get("region_column")}
    selection = kwargs.get("markers")
    if selection is None:
        markers = [c for c in df.columns if c not in roles and pd.api.types.is_numeric_dtype(df[c])]
    elif isinstance(selection, str):
        markers = [c for c in df.columns if c not in roles and re.fullmatch(selection, c)]
    else:
        markers = [str(marker) for marker in selection]
    return {
        "markers": markers,
        "var": pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(markers))),
        "x": kwargs["x"],
        "y": kwargs["y"],
        "in_um": unit == "um",
        "instance_column": kwargs.get("instance_id"),
        "region_column": kwargs.get("region_column"),
        "area_column": None,
    }


def _from_frame(
    path: Path,
    df: pd.DataFrame,
    *,
    markers: list[str],
    var: pd.DataFrame,
    x: str | None,
    y: str | None,
    instance_column: str | None,
    region_column: str | None,
    area_column: str | None,
    default_region: str | None,
    px_per_unit: float | np.ndarray = 1.0,
) -> CellTable:
    """Build a `CellTable` from a data frame and the roles of its columns; ``px_per_unit`` may be given per row."""
    roles = [x, y, instance_column, region_column, area_column, *markers]
    missing = [column for column in roles if column is not None and column not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}; found {list(df.columns)}.")

    if instance_column is not None:
        instance_id = pd.Series(df[instance_column].to_numpy(dtype=np.int64))
    else:
        # label images use 0 for the background, so instance ids start at 1
        instance_id = pd.Series(np.arange(1, len(df) + 1, dtype=np.int64))
    index = pd.Index(instance_id.astype(str))
    instance_id.index = index
    instance_id.name = CodexKeys.INSTANCE_KEY.value

    if region_column is not None:
        region = pd.Series(df[region_column].astype(str).to_numpy(), index=index)
    elif default_region is not None:
        region = pd.Series(default_region, index=index)
    else:
        raise ValueError(
            f"{path} has no region column, so its cells cannot be assigned to regions: a table for several regions "
            "needs a region column (`cells_table_kwargs['region_column']`). Alternatively, pass one table per "
            "region as a mapping {region: path}."
        )
    region.name = CodexKeys.REGION_KEY.value
    _check_unique_instance_ids(path, region, instance_id)

    obs = _drop_key_columns(df.drop(columns=markers), instance_column, region_column)
    obs.index = index
    radius = None
    if area_column is not None:
        radius = np.sqrt(df[area_column].to_numpy(dtype=np.float64) * px_per_unit**2 / np.pi)
    return CellTable(
        obs=obs,
        X=df[markers].to_numpy(dtype=np.float32),
        var=var,
        region=region,
        instance_id=instance_id,
        centroids_px=None
        if x is None or y is None
        else df[[x, y]].to_numpy(dtype=np.float64) * np.reshape(px_per_unit, (-1, 1)),
        radius_px=radius,
        row_instance_ids=instance_column is None,
    )


def _check_unique_instance_ids(path: Path, region: pd.Series, instance_id: pd.Series) -> None:
    """Raise if two cells of the same region have the same instance id: they could not be told apart."""
    regions, ids = region.to_numpy(), instance_id.to_numpy()
    duplicated = pd.DataFrame({"region": regions, "id": ids}).duplicated().to_numpy()
    if duplicated.any():
        first = regions[duplicated][0]
        found = sorted(set(ids[duplicated & (regions == first)].tolist()))
        raise ValueError(
            f"{path} has {len(found)} duplicate instance id(s) {found[:5]} in region '{first}'; every cell of a "
            "region needs its own instance id. Choose another instance id column, or remove the duplicate rows."
        )


def _drop_key_columns(obs: pd.DataFrame, instance_column: str | None, region_column: str | None) -> pd.DataFrame:
    """Drop the instance id and region columns if they are named like the keys whose values replace them.

    Any other column named like a key keeps its values as ``source_<name>`` in the table element.
    """
    consumed = {CodexKeys.INSTANCE_KEY.value: instance_column, CodexKeys.REGION_KEY.value: region_column}
    return obs.drop(columns=[key for key, column in consumed.items() if column == key])


def _read_header(path: Path) -> tuple[list[str], str]:
    """The column names of a delimited text table and its delimiter (tab if the header has one, else comma)."""
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    sep = "\t" if "\t" in header else ","
    return [column.strip().strip('"') for column in header.split(sep)], sep


def resolve_per_region(
    value: str | Path | Mapping[str, str | Path] | None,
    regions: Sequence[str],
    available: Sequence[str] = (),
) -> dict[str, Path]:
    """Resolve a ``cells_table`` or ``labels`` argument to one file per region.

    Parameters
    ----------
    value
        A path, a mapping from region id to path, or `None`.
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
        unknown = sorted(set(value) - set(regions) - set(available))
        if unknown:
            raise ValueError(f"Unknown region(s) {unknown}. Available regions: {list(available or regions)}.")
        files = {region: Path(value[region]) for region in regions if region in value}
    else:
        files = {regions[0] if len(regions) == 1 else "*": Path(value)}
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(f"{file} does not exist or is not a file.")
    return files


def read_region_tables(
    files: Mapping[str, Path],
    fmt: str,
    *,
    available: Sequence[str],
    selected: Sequence[str],
    pixel_sizes: Mapping[str, float | None],
    cells_table_kwargs: Mapping[str, Any],
) -> list[CellTable]:
    """Read user-supplied cell tables, as resolved by :func:`resolve_per_region`, and assign their cells to regions.

    A region column value names the region whose id it equals, else the longest region id it starts with followed
    by a non-alphanumeric character (``A_Scan1.qptiff`` -> ``A``). A table given for one region holds the cells of
    that region; if its region column matches no region, it holds only cells of that region. Cells of a table for
    all regions (``"*"``) whose region matches no available region are dropped with a warning; cells of available
    regions that are not read are dropped silently.

    Parameters
    ----------
    files
        The table of each region, or ``{"*": path}`` for one table for all regions.
    fmt
        The table format, see :func:`read_cell_table`.
    available
        All region ids of the dataset.
    selected
        The region ids read.
    pixel_sizes
        The pixel size in µm of each region with an image.
    cells_table_kwargs
        See :func:`read_cell_table`; ``geojson`` may also be a mapping from region id to path, like ``files``.

    Returns
    -------
    The cell tables, one per file.
    """
    geojson = cells_table_kwargs.get("geojson")
    geojson_files = resolve_per_region(geojson, selected, available) if geojson is not None else {}
    if unknown := sorted(geojson_files.keys() - files.keys()):
        raise ValueError(
            f"`cells_table_kwargs['geojson']` is given for {unknown}, but the cell tables are given for "
            f"{list(files)}; pass the GeoJSON files in the same form as the tables."
        )
    tables = []
    for key, path in files.items():
        # a table for several regions converts the µm coordinates of each cell with the pixel size of its region
        size = {region: pixel_sizes.get(region) for region in selected} if key == "*" else pixel_sizes.get(key)
        table = read_cell_table(
            path,
            fmt,
            pixel_size_um=size,
            default_region=None if key == "*" else key,
            cells_table_kwargs={**cells_table_kwargs, "geojson": geojson_files.get(key)},
        )
        tables.append(_assign_regions(table, path, key, available, selected))
    return tables


def _assign_regions(
    table: CellTable, path: Path, key: str, available: Sequence[str], selected: Sequence[str]
) -> CellTable:
    values = list(table.region.unique())
    matched = {value: _match_region(value, available) for value in values}
    if key != "*" and all(region is None for region in matched.values()):
        # the region column of a table for one region names its source image only
        matched = dict.fromkeys(values, key)
    unmatched = sorted(value for value, region in matched.items() if region is None)
    if unmatched:
        warnings.warn(
            f"Ignoring the cells of {unmatched} in {path}: they match none of the regions {list(available)}.",
            UserWarning,
            stacklevel=4,
        )
    table.region = table.region.map(matched)
    keep = table.region.isin(selected if key == "*" else [key]).to_numpy()
    if not keep.any():
        target = f"region '{key}'" if key != "*" else f"the regions read {list(selected)}"
        raise ValueError(f"{path} has no cells of {target}; its region column holds {values}.")
    return table.take(np.flatnonzero(keep))


def _match_region(value: str, regions: Sequence[str]) -> str | None:
    """The region named by ``value``: equal to it, else the longest one it starts with before a non-alphanumeric."""
    if value in regions:
        return value
    prefixes = [r for r in regions if value.startswith(r) and not value[len(r)].isalnum()]
    return max(prefixes, key=len, default=None)


def read_labels(
    path: Path,
    labels_models_kwargs: Mapping[str, Any],
    transformations: Mapping[str, BaseTransformation],
    *,
    mask_channel: int | str | None = None,
) -> DataArray | DataTree:
    """Lazily read a segmentation mask as a ``Labels2DModel``.

    Parameters
    ----------
    path
        A TIFF or OME-TIFF with integer instance ids (0 = background).
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`; their ``chunks`` also set the
        spatial chunks of uncompressed files, which are memory-mapped.
    transformations
        The transformations of the region's image, by coordinate system.
    mask_channel
        The mask to read from a file with several: its index or OME channel name (e.g. ``"cells"``). If `None`,
        the first one.

    Returns
    -------
    The labels element.
    """
    labels_models_kwargs, chunks = _labels_models_kwargs(labels_models_kwargs)
    image, _, _, _ = read_tiff(path, {}, chunks)
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"{path} has dtype {image.dtype}; a segmentation mask must have an integer dtype.")

    n_masks = image.shape[0]
    if isinstance(mask_channel, str):
        names = _ome_channel_names(path)
        if mask_channel not in names:
            raise ValueError(f"{path} has no mask channel {mask_channel!r}; its OME channel names are {names}.")
        index = names.index(mask_channel)
    else:
        index = 0 if mask_channel is None else mask_channel
        if mask_channel is None and n_masks > 1:
            logger.info("%s holds %d masks; reading the first. Set `mask_channel` to choose another.", path, n_masks)
    if not 0 <= index < n_masks:
        raise ValueError(f"{path} holds {n_masks} mask channel(s); `mask_channel` {mask_channel!r} is out of range.")
    return Labels2DModel.parse(
        image[index], dims=("y", "x"), transformations=dict(transformations), **labels_models_kwargs
    )


def _labels_models_kwargs(labels_models_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, int]]:
    """``labels_models_kwargs`` with the reader defaults, and their ``(y, x)`` chunk size."""
    _, labels_models_kwargs = _initialize_raster_models_kwargs({}, labels_models_kwargs)
    _, defaults = _initialize_raster_models_kwargs({}, {})
    return labels_models_kwargs, spatial_chunks(labels_models_kwargs["chunks"], defaults["chunks"])


def _ome_channel_names(path: Path) -> list[str]:
    with tifffile.TiffFile(path) as tf:
        if not tf.ome_metadata:
            return []
        images = tifffile.xml2dict(tf.ome_metadata)["OME"]["Image"]
    pixels = (images[0] if isinstance(images, list) else images)["Pixels"]
    channels = pixels.get("Channel", [])
    return [str(channel.get("Name", "")) for channel in (channels if isinstance(channels, list) else [channels])]


def read_region_labels(
    files: Mapping[str, Path],
    transformations: Mapping[str, Mapping[str, BaseTransformation]],
    labels_models_kwargs: Mapping[str, Any],
    cells_table_kwargs: Mapping[str, Any],
    image_shapes: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, DataArray | DataTree], dict[str, int]]:
    """Read the mask of each region, as resolved by :func:`resolve_per_region`, with the region's transformations.

    A mask whose ``(y, x)`` shape differs from ``image_shapes`` of its region is scaled onto the image if it is
    downsampled by an integer factor, see :func:`mask_downsample_factor`: its transformations become
    ``Sequence([Scale([f, f]), t])`` for each region transformation ``t``. Any other mismatch raises a warning: its
    pixels would not align with the image.

    Returns
    -------
    The labels elements by region id, and the downsampling factor of the scaled masks by region id.
    """
    if "*" in files:
        raise ValueError(
            f"A single labels mask {files['*']} was given for several regions; a mask belongs to one region image, "
            "so pass a mapping {region: path}."
        )
    labels, factors = {}, {}
    for region, path in files.items():
        region_transformations = _transformations_of(transformations, region)
        labels[region] = read_labels(
            path, labels_models_kwargs, region_transformations, mask_channel=cells_table_kwargs.get("mask_channel")
        )
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
        factors[region] = factor
    return labels, factors


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


def _to_image_pixels(tables: Sequence[CellTable], factors: Mapping[str, int]) -> list[CellTable]:
    """``tables`` with mask-pixel centroids and radii scaled to image pixels by their region's mask factor."""
    scaled = []
    for table in tables:
        if table.mask_pixels and factors:
            row_factors = table.region.map(factors).fillna(1).to_numpy(dtype=np.float64)
            table = replace(
                table,
                centroids_px=None if table.centroids_px is None else table.centroids_px * row_factors[:, None],
                radius_px=None if table.radius_px is None else table.radius_px * row_factors,
            )
        scaled.append(table)
    return scaled


def polygon_labels(
    tables: Sequence[CellTable],
    labels: Mapping[str, DataArray | DataTree],
    image_shapes: Mapping[str, tuple[int, int]],
    transformations: Mapping[str, Mapping[str, BaseTransformation]],
    labels_models_kwargs: Mapping[str, Any],
) -> tuple[dict[str, DataArray | DataTree], dict[str, str]]:
    """Lazily rasterise the cell polygons of each region into labels whose values are the cells' instance ids.

    Where cells overlap, the one with the higher instance id wins. Regions with a mask in ``labels`` keep it, with
    a warning; regions without an image are skipped, their cells become circles. In both cases the polygons are
    still kept as shapes, see :func:`polygon_shapes`.

    Parameters
    ----------
    tables
        The cell tables; those with ``polygons_px`` are rasterised.
    labels
        The labels read from masks, by region id.
    image_shapes
        The level-0 ``(y, x)`` shape of each region's image; the labels get the same shape.
    transformations
        The transformations of each region's image, see :func:`region_transformations`; missing regions are in
        ``"global"``.
    labels_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Labels2DModel`; their ``chunks`` are the chunks
        that are rasterised one at a time.

    Returns
    -------
    The labels elements by region id, and the GeoJSON file of each.
    """
    labels_models_kwargs, chunks = _labels_models_kwargs(labels_models_kwargs)
    elements, sources = {}, {}
    for table in tables:
        if table.polygons_px is None:
            continue
        for region, rows in _region_rows(table).items():
            if region in labels:
                row_ids = " Its cells' instance ids are row numbers and may not match the mask's values."
                warnings.warn(
                    f"Region '{region}' has both a segmentation mask and the cell polygons {table.geojson}; the mask "
                    f"is used for the labels and the polygons are kept as '{region}{PhenocyclerKeys.POLYGONS_SUFFIX}'."
                    f"{row_ids if table.row_instance_ids else ''}",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            if region not in image_shapes:
                logger.info("Region '%s' has no image, so its cell polygons are not rasterised.", region)
                continue
            rows = rows[pd.notna(table.polygons_px[rows])]
            data = rasterize_polygons_lazy(
                table.polygons_px[rows], table.instance_id.to_numpy()[rows], image_shapes[region], chunks
            )
            elements[region] = Labels2DModel.parse(
                data,
                dims=("y", "x"),
                transformations=_transformations_of(transformations, region),
                **labels_models_kwargs,
            )
            sources[region] = str(table.geojson)
    return elements, sources


def polygon_shapes(
    tables: Sequence[CellTable], transformations: Mapping[str, Mapping[str, BaseTransformation]]
) -> dict[str, GeoDataFrame]:
    """The cell polygons of each region as shapes indexed by the cells' instance ids, the ids of their labels.

    Cells without a polygon are left out.

    Parameters
    ----------
    tables
        The cell tables; those with ``polygons_px`` are kept.
    transformations
        The transformations of each region, see :func:`region_transformations`; missing regions are in
        ``"global"``.

    Returns
    -------
    The shapes elements by region id.
    """
    shapes = {}
    for table in tables:
        if table.polygons_px is None:
            continue
        for region, rows in _region_rows(table).items():
            rows = rows[pd.notna(table.polygons_px[rows])]
            if len(rows) == 0:
                continue
            geometry = gpd.GeoDataFrame(
                geometry=list(table.polygons_px[rows]), index=pd.Index(table.instance_id.to_numpy()[rows])
            )
            shapes[region] = ShapesModel.parse(geometry, transformations=_transformations_of(transformations, region))
    return shapes


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
    region_elements: Mapping[str, str],
    labels: Mapping[str, DataArray | DataTree],
    transformations: Mapping[str, Mapping[str, BaseTransformation]],
) -> tuple[dict[str, ad.AnnData], dict[str, GeoDataFrame]]:
    """Build the table elements and, for regions without labels, circle shapes from cell tables.

    Parameters
    ----------
    tables
        The cell tables, each covering one or more regions.
    region_elements
        The element each region's cells annotate. Regions missing from it annotate ``<region>_labels`` if they are
        in ``labels``, else the circles ``<region>_cells`` built here.
    labels
        The labels elements by region id; their regions get no circles.
    transformations
        The transformations of each region, see :func:`region_transformations`; missing regions get ``Identity``
        to ``"global"``.

    Returns
    -------
    The table elements, ``{"table": ...}`` if all tables have the same markers, else ``{"<region>_table": ...}``
    per region, and the circle shapes by element name. The circles have the cells' area (radius 1 px without one).
    """
    _check_disjoint_regions(tables)
    row_ids = sorted(
        region
        for table in tables
        # labels rasterised from a table's polygons have its instance ids by construction
        if table.row_instance_ids and table.polygons_px is None
        for region in table.region.unique()
        if region in labels
    )
    if row_ids:
        warnings.warn(
            f"The cell table of region(s) {row_ids} has no instance id column, so its row numbers are used as "
            "instance ids; they may not match the values of the mask.",
            UserWarning,
            stacklevel=3,
        )
    pieces = _split_by_region(tables) if len({tuple(table.var.index) for table in tables}) > 1 else None

    def element(region: str) -> str:
        if region in region_elements:
            return region_elements[region]
        suffix = PhenocyclerKeys.LABELS_SUFFIX if region in labels else PhenocyclerKeys.CELLS_SUFFIX
        return f"{region}{suffix}"

    shapes = {}
    circle_regions = []
    for table in tables:
        for region, rows in _region_rows(table).items():
            if region in labels:
                continue
            if table.centroids_px is None:
                raise ValueError(
                    f"The cells of region '{region}' have no centroids and no segmentation mask, so they cannot be "
                    "placed; pass a mask as `labels` or a table with centroids."
                )
            circle_regions.append(region)
            radius = 1.0 if table.radius_px is None else table.radius_px[rows]
            shapes[element(region)] = ShapesModel.parse(
                table.centroids_px[rows],
                geometry=0,
                radius=radius,
                index=pd.Index(table.instance_id.to_numpy()[rows]),
                transformations=_transformations_of(transformations, region),
            )
    if circle_regions:
        warnings.warn(
            f"No segmentation mask found for region(s) {sorted(circle_regions)}; cells are represented as circles.",
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
    cells_table_kwargs: Mapping[str, Any],
    duplicate_scans: Mapping[str, AkoyaScan] | None = None,
) -> SpatialData:
    """Assemble the regions of an Akoya read into a `SpatialData` with their metadata in ``attrs["akoya"]``.

    Each region gets the image ``<region>_image``, the labels ``<region>_labels`` from its mask or its cell polygons
    (see :func:`polygon_labels`), the polygons ``<region>_polygons`` and the cells (see :func:`build_cells`), all in
    the coordinate systems of :func:`region_transformations`; regions with cells or a mask but no image are
    unrelated to the scans. A mask downsampled from its region image by an integer factor is scaled onto the image
    (see :func:`read_region_labels`), and the centroids and radii in mask pixels of its cells (``mask_pixels``, see
    :class:`CellTable`) are scaled by the same factor, so that the table and circles are in image pixels.

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
    cells_table_kwargs
        See :func:`read_labels` (``mask_channel``).
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
    labels, factors = read_region_labels(
        mask_files, transformations, labels_models_kwargs, cells_table_kwargs, image_shapes
    )
    cell_tables = _to_image_pixels(cell_tables, factors)
    polygons, geojson_files = polygon_labels(cell_tables, labels, image_shapes, transformations, labels_models_kwargs)
    labels |= polygons
    tables, shapes = build_cells(cell_tables, {}, labels, transformations) if cell_tables else ({}, {})
    for region, element in polygon_shapes(cell_tables, transformations).items():
        shapes[f"{region}{PhenocyclerKeys.POLYGONS_SUFFIX}"] = element
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
    if geojson_files:
        attrs["geojson"] = geojson_files
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


def _split_by_region(tables: Sequence[CellTable]) -> dict[str, CellTable]:
    return {region: table.take(rows) for table in tables for region, rows in _region_rows(table).items()}


def _to_anndata(tables: Sequence[CellTable], element: Callable[[str], str]) -> ad.AnnData:
    """Concatenate cell tables with the same markers into one table annotating the elements of their regions."""
    region_key, instance_key = CodexKeys.REGION_KEY.value, CodexKeys.INSTANCE_KEY.value
    obs = pd.concat([table.obs for table in tables]) if len(tables) > 1 else tables[0].obs.copy()
    # the region and instance keys are set below; the adapters dropped the columns they were read from, and other
    # columns that would clash with them (also case-insensitively, as spatialdata requires, e.g. `Cell ID` ->
    # `Cell_ID` vs `cell_id`) keep their values under `source_<name>`
    reserved = {region_key, instance_key}
    obs = obs.rename(
        columns={
            column: f"source_{column}"
            for column in obs.columns
            if sanitize_name(str(column), is_dataframe_column=True).casefold() in reserved
        }
    )
    regions = pd.concat([table.region for table in tables]) if len(tables) > 1 else tables[0].region
    if not obs.index.is_unique:
        obs.index = pd.Index(regions.to_numpy(dtype=object) + "_" + obs.index.to_numpy(dtype=object))
    X = np.concatenate([table.X for table in tables]) if len(tables) > 1 else tables[0].X
    adata = ad.AnnData(X=X, obs=obs, var=tables[0].var.copy())
    centroids = [table.centroids_px for table in tables if table.centroids_px is not None]
    if len(centroids) == len(tables):
        adata.obsm[CodexKeys.SPATIAL_KEY.value] = np.concatenate(centroids)
    elif centroids:
        warnings.warn(
            "Some cell tables have no centroids, so the table has no 'spatial' coordinates.", UserWarning, stacklevel=4
        )
    # e.g. QuPath's `Centroid X µm` -> `Centroid_X_µm`: spatialdata only writes alphanumeric names and `_.-`
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
