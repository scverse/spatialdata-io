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
import flowio  # readfcs' parser, for the TEXT segment only
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
    "check_cells_table_options",
    "detect_table_format",
    "mask_downsample_factor",
    "polygon_labels",
    "polygon_shapes",
    "raster_shape",
    "read_cell_table",
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
    "coordinates_space",
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
# QuPath mean intensities: `CD3: Cell: Mean` (also `CD3..Cell..Mean` of R's `make.names` and `CD3_Cell_Mean`),
# `Cell: CD3: Mean` (QuPath >= 0.6) and `Cell: CD3 mean` (built-in cell detection); the per-marker intensity is
# taken from the first compartment. Detections without compartments have `CD3: Mean`.
_QUPATH_MARKER = re.compile(
    r"(?P<marker>.+?)(?:: |\.\.|_)(?P<compartment>Cell|Nucleus|Cytoplasm|Membrane)(?:: |\.\.|_)Mean"
)
_QUPATH_COMPARTMENT_MARKER = re.compile(
    r"(?P<compartment>Cell|Nucleus|Cytoplasm|Membrane): (?P<marker>.+?)(?:: M| m)ean"
)
_QUPATH_DETECTION_MARKER = re.compile(r"(?P<marker>.+?): Mean")
_QUPATH_COMPARTMENTS = ("Cell", "Nucleus", "Cytoplasm", "Membrane")
# other QuPath measurements, e.g. `Nucleus: Area µm^2` or `Nucleus/Cell area ratio`, are not markers
_QUPATH_MEASUREMENT = re.compile(r"(?:Cell|Nucleus|Cytoplasm|Membrane)\b|.*: ")
# the unit of a QuPath measurement, e.g. `Cell: Area µm^2`
_QUPATH_UNIT = re.compile(r" (µm|um|px)(?:\^2)?$")
# QuPath shape and detection measurements without a unit, e.g. of a detection export without compartments
_QUPATH_UNITLESS_MEASUREMENTS = frozenset({"Circularity", "Detection probability", "Eccentricity", "Solidity"})
_QUPATH_IMAGE = "Image"
# the extensions of image files, also named in the region column of a table (e.g. QuPath's `Image`)
_IMAGE_SUFFIX = re.compile(r"(?:\.er)?\.(?:qptiff|ome\.tiff?|tiff?)$", re.IGNORECASE)
_QUPATH_ID = "Object ID"
# QuPath's centroid columns by unit; R's `make.names` writes `Centroid X µm` as `Centroid.X.um`
_QUPATH_CENTROIDS = {"um": ("Centroid X µm", "Centroid Y µm"), "px": ("Centroid X px", "Centroid Y px")}
# the QuPath object types of cells; other features of a GeoJSON export (e.g. annotations) are not cells
_QUPATH_CELL_TYPES = frozenset({"cell", "detection", "PathCellObject", "PathDetectionObject"})
# inForm `*_cell_seg_data.txt` columns: `Cell X Position (microns)`, `Entire Cell <marker> Mean (<unit>)`, ...
_INFORM_POSITION = re.compile(r"Cell X Position(?: \((.*)\))?")
_INFORM_MARKER = re.compile(r"Entire Cell (.+) Mean(?: \(.*\))?")
_INFORM_AREA = re.compile(r"Entire Cell Area \((.*)\)")
# the unit of an inForm measurement, e.g. `Tissue Category Area (square microns)`
_INFORM_UNIT = re.compile(r"\((?:square )?(microns|pixels)\)$")
# the pages of an inForm `*_binary_seg_maps.tif`, named by the root tag of their XML description: the tissue class
# map `<TissueClassMap>` (named `Tissue`), and a `<SegmentationImage>` per compartment (named after the compartment)
_INFORM_XML_ROOT = re.compile(r"<([A-Za-z][\w-]*)")
_INFORM_SEGMENTATION = "SegmentationImage"
_INFORM_COMPARTMENT = re.compile(r"<CompartmentType>\s*(\w+)\s*</CompartmentType>")
_INFORM_PAGE_NAMES = {"TissueClassMap": "Tissue"}
# the default mask of an inForm segmentation map: the nuclei carry the `Cell ID` of every cell
_INFORM_DEFAULT_MASK = "Nucleus"
_INFORM_REGION = "Sample Name"
_INFORM_ID = "Cell ID"
# HuBMAP SPRM outputs: `<image>-cell_channel_mean.csv` with the centroids in `<image>-cell_centers.csv`
_SPRM_MEANS_SUFFIX = "-cell_channel_mean.csv"
_SPRM_CENTERS_SUFFIX = "-cell_centers.csv"
# a matched polygon must lie within this distance (px) of its cell's centroid, else the match is considered wrong
_CENTROID_TOLERANCE_PX = 1.0
# the largest fraction of matched polygons that may be far from their centroids before the match is rejected
_MAX_MISPLACED_FRACTION = 0.05
# the largest fraction of a region's cells that may lie outside its image without a warning
_MAX_OUTSIDE_FRACTION = 0.05


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
        mask-based pipelines: mcquant, SPRM, ``"generic"`` with ``coordinates_space="mask"``) rather than of the
        image; they are scaled to image pixels when the mask is a downsample of the image, see
        :func:`build_spatialdata`.
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


def detect_table_format(path: Path) -> str:
    """Detect the format of a cell table from its file type and header.

    Parameters
    ----------
    path
        The ``.fcs``, ``.csv``, ``.tsv`` or ``.txt`` cell table. A text table is tab-separated if its header has a
        tab, else semicolon-separated (with decimal commas or points) if it has more semicolons than commas, else
        comma-separated.

    Returns
    -------
    ``"codex"`` (``x``/``y`` with ``cell_id`` or ``cycCCC_chHHH`` columns, also as the channels of an FCS file),
    ``"qupath"`` (``Centroid X µm``, ``Centroid X px`` or ``Object ID``, also in R's ``make.names`` form, e.g.
    ``Centroid.X.um``), ``"mcquant"`` (``CellID`` and ``X_centroid``), ``"inform"`` (``Cell X Position``,
    optionally followed by its unit in parentheses), ``"sprm"`` (first column ``ID``, no ``x`` or ``y``), else
    ``"generic"``.

    Raises
    ------
    ValueError
        If ``path`` is an FCS file that cannot be parsed or is not a CODEX table (its ``$PnN`` channel names).
    """
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        try:
            columns = list(flowio.read_multiple_data_sets(str(path), only_text=True)[0].pnn_labels)
        except (ValueError, flowio.exceptions.FlowIOException) as e:
            raise ValueError(f"{path} is not a readable FCS file: {e}") from e
        if _detect_header_format(columns) != "codex":
            raise ValueError(f"{path} is not a CODEX cell table: its channels are {columns}.")
        return "codex"
    return _detect_header_format(_read_header(path)[0])


def _detect_header_format(columns: list[str]) -> str:
    # the name part of the CODEX Processor header form `cyc001_ch001:DAPI1`
    names = {column.partition(":")[0].casefold() for column in columns}
    if {"x", "y"} <= names and (
        CodexKeys.INSTANCE_KEY.value in names or any(_MARKER_COLUMN.fullmatch(name) for name in names)
    ):
        return "codex"
    qupath = {_QUPATH_ID, *(name for names in _QUPATH_CENTROIDS.values() for name in names)}
    if qupath & set(_qupath_names(columns)):
        return "qupath"
    if {"CellID", "X_centroid"} <= set(columns):
        return "mcquant"
    if any(_INFORM_POSITION.fullmatch(column) for column in columns):
        return "inform"
    # SPRM means have no centroids, which are in `*-cell_centers.csv`
    if columns[:1] == ["ID"] and not {"x", "y"} & names:
        return "sprm"
    return "generic"


def read_cell_table(
    path: Path,
    fmt: str,
    *,
    pixel_size_um: float | Mapping[str, float | None] | None,
    default_region: str | None,
    cells_table_kwargs: Mapping[str, Any],
    regions: Mapping[str, Sequence[Path]] | None = None,
) -> CellTable:
    """Read a cell table of any supported format.

    Parameters
    ----------
    path
        The cell table.
    fmt
        One of ``"auto"`` (see :func:`detect_table_format`; ``"generic"`` if ``cells_table_kwargs`` has ``x`` and
        ``y``), ``"codex"``, ``"qupath"``, ``"mcquant"``,
        ``"sprm"``, ``"inform"`` or ``"generic"``:

            - ``"codex"``: a CODEX Processor ``.fcs`` or ``.csv`` table. The markers are ``markers``, else the
              ``cycCCC_chHHH`` columns, named by the FCS ``$PnS`` keywords or by the display part of the Processor
              CSV header ``cycCCC_chHHH:<marker>`` (in ``var`` with their ``cycle`` and ``channel``), else, with a
              `UserWarning`, the numeric columns that are not CODEX Processor columns (``x``, ``size``, ...).
              Centroids ``x``/``y`` in
              pixels, instance ids ``cell_id``, the region from ``region`` (a numeric region ``r`` is the region id
              ``regRRR``), the radius from ``size``.
            - ``"qupath"``: a QuPath measurement export, also with R's ``make.names`` column names
              (``Centroid.X.um``, ``Object.ID``). Centroids ``Centroid X µm``/``Centroid Y µm`` (divided by
              ``pixel_size_um``) or ``Centroid X px``/``Centroid Y px``, else ``X``/``Y`` in ``coordinates_unit``
              or, without it, in the unit of the other measurements (e.g. ``Cell: Area µm^2``) with a
              `UserWarning`, raising if that is not unique.
              Instance ids from a numeric ``Object ID``, else the row number (from 1); the region from the
              ``Image`` column. The markers are ``markers``, else the ``<marker>: <compartment>: Mean`` columns
              (also ``<marker>..<compartment>..Mean`` and ``<marker>_<compartment>_Mean``), ``<compartment>:
              <marker>: Mean`` (QuPath >= 0.6) and ``<compartment>: <marker> mean`` (built-in cell detection)
              columns, one per marker from the first of the ``Cell``, ``Nucleus``, ``Cytoplasm`` and ``Membrane``
              compartments (in ``var["compartment"]``), else the ``<marker>: Mean`` columns of detections
              without compartments, else, with a `UserWarning`, the numeric columns that are not QuPath
              measurements or empty (e.g. ``CD44`` of a detection export; not morphology such as ``Area µm^2``,
              ``Max diameter µm`` or ``Circularity``). The radius comes from ``Cell: Area µm^2`` (or
              ``px^2``) in the unit of the centroids. A table split into centroids and measurements (two files)
              must be joined on the object id first. The cell outlines are read from a QuPath GeoJSON export, see
              :func:`read_qupath_polygons`: ``cells_table_kwargs["geojson"]`` (``False`` for none), else
              ``<table stem>.geojson`` or ``<table stem>_cells.geojson`` next to the table, if it exists and holds
              any of the table's cells.
            - ``"mcquant"``: an MCMICRO ``mcquant`` table. Instance ids ``CellID``, pixel centroids
              ``X_centroid``/``Y_centroid``, the markers are the columns between ``CellID`` and ``X_centroid``, the
              radius comes from ``Area``.
            - ``"sprm"``: a HuBMAP SPRM ``<image>-cell_channel_mean.csv`` (validated on synthetic data only). Instance
              ids ``ID`` (the values of the mask, ``pipeline_output/mask/*.ome.tiff`` channel 0, passed as ``labels``),
              one marker per remaining column. Pixel centroids come from ``<image>-cell_centers.csv`` (``ID,x,y``)
              next to the table, which must hold every cell; without it a `UserWarning` is raised and the table has
              no ``obsm["spatial"]`` (the centroids are never computed from the mask), so its region needs a mask.
            - ``"inform"``: a PerkinElmer/Akoya inForm ``*_cell_seg_data.txt`` (tab-separated; spaces around the
              column names are removed). Instance ids ``Cell ID``, centroids ``Cell X Position``/``Cell Y
              Position`` in the unit named in their header, ``(microns)`` (divided by ``pixel_size_um``) or
              ``(pixels)``; if it names none, in ``coordinates_unit`` or, without it, in the unit of the other
              columns (e.g. ``Tissue Category Area (square microns)``; inForm writes all of them in one unit),
              raising if they name both, pixels if they name none. The region comes from ``Sample Name``. The
              markers are the ``Entire Cell <marker> Mean (...)`` columns; the radius comes from
              ``Entire Cell Area (...)`` if its unit equals that of the positions. Positions in microns may be
              relative to the slide rather than to the field image, see :func:`build_spatialdata`.
            - ``"generic"``: columns given by ``cells_table_kwargs``.

    pixel_size_um
        The pixel size of the region image, to convert µm centroids to pixels; for a table of several regions, the
        pixel size of each region, by region id: every cell is converted with the pixel size of its region, cells
        of other regions get NaN coordinates.
    default_region
        The region id of all cells if the table has no region column.
    regions
        The region ids of the dataset with the image files of each, to which the values of the region column are
        matched, see :func:`_match_region`; values that name none of them are kept. If the table has
        ``default_region`` and its region column holds a single value that names no region, that value names
        ``default_region`` (e.g. the source image of a table for one region). If `None`, the values are the
        region ids.
    cells_table_kwargs
        Optional keys: ``region_column`` (the column holding the region of each cell; QuPath default ``Image``),
        and for ``"generic"`` tables ``x`` and ``y`` (required centroid columns), ``instance_id`` (else the row
        number, from 1), ``markers`` (a list of columns or a regular expression matching the full column name;
        default: all other numeric columns; also for ``"qupath"`` and ``"codex"`` tables) and ``coordinates_unit`` (``"px"``,
        default, or ``"um"``; for ``"qupath"`` and ``"inform"`` tables the unit of centroid columns whose name
        has none) and ``coordinates_space`` (the pixels of ``"px"`` centroids: ``"image"``, default, those of the
        full-resolution region image, or ``"mask"``, those of the region's mask, scaled to image pixels like the
        ``"mcquant"`` and ``"sprm"`` centroids if the mask is downsampled); for ``"qupath"`` tables ``geojson`` (the GeoJSON file of the cell outlines) and
        ``geojson_units`` (the unit of its coordinates, ``"px"`` of the full-resolution image, default and what
        QuPath writes, or ``"um"``). The key ``mask_channel`` is used by :func:`read_labels`.

    Returns
    -------
    The cell table. Its regions are the (matched) values of the region column, or ``default_region``.
    """
    check_cells_table_options(fmt, cells_table_kwargs)
    if fmt == "auto":
        # centroid columns given by the user make a generic table
        fmt = "generic" if {"x", "y"} <= cells_table_kwargs.keys() else detect_table_format(path)
        logger.info("Reading %s as a %r cell table.", path, fmt)
    if fmt != "qupath" and cells_table_kwargs.get("geojson") not in (None, False):
        raise ValueError(f"`cells_table_kwargs['geojson']` is only read with QuPath cell tables; {path} is {fmt!r}.")
    if fmt != "generic" and cells_table_kwargs.get("coordinates_space") is not None:
        raise ValueError(
            f"`cells_table_kwargs['coordinates_space']` is only used with generic cell tables; {path} is {fmt!r}."
        )

    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        df = _read_fcs(path)
    else:
        df = _read_text_table(path)
    reader = {
        "codex": _codex_roles,
        "qupath": _qupath_roles,
        "mcquant": _mcquant_roles,
        "sprm": _sprm_roles,
        "inform": _inform_roles,
        "generic": _generic_roles,
    }[fmt]
    roles = reader(path, df, cells_table_kwargs)
    mask_pixels = fmt in {"mcquant", "sprm"}
    if fmt == "generic":
        mask_pixels = _coordinates_space(path, roles["in_um"], cells_table_kwargs) == "mask"
    if roles["region_column"] in df.columns and (empty := df[roles["region_column"]].isna().to_numpy()).any():
        warnings.warn(
            f"Ignoring the {empty.sum()} cell(s) of {path} without a region in '{roles['region_column']}'.",
            UserWarning,
            stacklevel=4,
        )
        df = df.loc[~empty].reset_index(drop=True)
    region = _region_values(df, roles["region_column"], default_region, regions)
    units = cells_table_kwargs.get("geojson_units", "px")
    row_pixel_sizes = None
    if roles["in_um"] or (fmt == "qupath" and units == "um"):
        row_pixel_sizes = _row_pixel_sizes(path, region, pixel_size_um)
    if roles.pop("in_um"):
        if row_pixel_sizes is None:
            raise ValueError(
                f"{path} has centroids in µm, but no pixel size is known to convert them to pixels (the region's "
                "pixel size is missing)."
            )
        roles["px_per_unit"] = 1 / row_pixel_sizes
    table = _from_frame(path, df, region=region, **roles)
    table.mask_pixels = mask_pixels
    if fmt == "sprm":
        table.centroids_px = _read_sprm_centers(path, table)
    if fmt == "qupath":
        option = cells_table_kwargs.get("geojson")
        geojson = _find_geojson(path) if option is None else None if option is False else Path(option)
        if geojson is not None:
            table.polygons_px = read_qupath_polygons(
                geojson, table, path, units=units, pixel_size_um=row_pixel_sizes, discovered=option is None
            )
            table.geojson = None if table.polygons_px is None else geojson
    return table


def _region_values(
    df: pd.DataFrame,
    region_column: str | None,
    default_region: str | None,
    regions: Mapping[str, Sequence[Path]] | None,
) -> pd.Series | None:
    """The region of every row of ``df``, see :func:`read_cell_table`; `None` if it is unknown."""
    if region_column is None or region_column not in df.columns:
        return None if default_region is None else pd.Series(default_region, index=df.index)
    values = df[region_column].astype(str)
    if regions is None:
        return values
    matched = {value: _match_region(value, regions) for value in values.unique()}
    if default_region is not None and len(matched) == 1 and None in matched.values():
        # the region column of a table for one region names its source image only
        return pd.Series(default_region, index=df.index)
    return values.map({value: value if region is None else region for value, region in matched.items()})


def check_cells_table_options(fmt: str, cells_table_kwargs: Mapping[str, Any]) -> None:
    """Raise for an unknown cell table format or key of ``cells_table_kwargs``, also when no table is read."""
    if fmt != "auto" and fmt not in TABLE_FORMATS:
        raise ValueError(f"Unknown cell table format {fmt!r}; expected 'auto' or one of {TABLE_FORMATS}.")
    unknown = sorted(set(cells_table_kwargs) - set(CELLS_TABLE_KWARGS))
    if unknown:
        raise ValueError(f"`cells_table_kwargs` has unknown key(s) {unknown}; expected some of {CELLS_TABLE_KWARGS}.")


def _row_pixel_sizes(
    path: Path, region: pd.Series | None, pixel_size_um: float | Mapping[str, float | None] | None
) -> float | np.ndarray | None:
    """The pixel size of every row, by its ``region``, or one for all rows; `None` if it is unknown.

    For pixel sizes by region id, rows of other regions get NaN (their cells are not read). Raises if a region of
    the rows has no pixel size.
    """
    if not isinstance(pixel_size_um, Mapping):
        return pixel_size_um
    if region is None:
        sizes = set(pixel_size_um.values())
        return sizes.pop() if len(sizes) == 1 else None
    unknown = sorted(r for r in region.unique() if r in pixel_size_um and pixel_size_um[r] is None)
    if unknown:
        raise ValueError(
            f"{path} has coordinates in µm, but the pixel size of region(s) {unknown} is unknown, so they cannot be "
            "converted to pixels."
        )
    sizes_by_region = {r: pixel_size_um.get(r, np.nan) for r in region.unique()}
    return region.map(sizes_by_region).to_numpy(dtype=np.float64)


def _qupath_names(columns: Sequence[str]) -> dict[str, str]:
    """The columns by their QuPath name: as written, or undoing R's ``make.names``.

    ``make.names`` replaces ``: ``, `` `` and ``^`` by dots: ``Centroid.X.um`` is ``Centroid X µm`` and
    ``Cell..Area.µm.2`` is ``Cell: Area µm^2``.
    """
    names = {}
    for column in columns:
        name = re.sub(r"\.2$", "^2", column.replace("..", ": ")).replace(".", " ")
        names[re.sub(r" um(?=\^2$|$)", " µm", name)] = column
    return names | {column: column for column in columns}


def _object_id_column(columns: Sequence[str]) -> str | None:
    """The QuPath ``Object ID`` column (also R's ``Object.ID``), or `None`."""
    return _qupath_names(columns).get(_QUPATH_ID)


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
    discovered: bool = False,
) -> np.ndarray | None:
    """Read the cell outlines of a QuPath GeoJSON export and match them to the cells of its measurement table.

    A feature belongs to the cell whose ``Object ID`` (or R's ``Object.ID``) is the feature ``id`` (the object's
    UUID, QuPath >= 0.4); features of other objects (e.g. annotations) are skipped, and so are cell features without
    a table row (e.g. cells filtered out of the table), with a `UserWarning`. If the table has no ``Object ID``
    column or the features have no ids, the cell and detection features (``objectType``) are matched to the table
    rows in order, and their number must equal the number of rows.

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
    discovered
        Whether ``geojson`` was found next to the table rather than given: if it holds none of the table's cells
        (e.g. QuPath's export of an annotation), it is ignored with a `UserWarning` instead of raising.

    Returns
    -------
    The polygon of every row of ``table`` in pixels, `None` for rows without one; `None` for an ignored
    ``geojson``.
    """
    if units not in ("px", "um"):
        raise ValueError(f"`cells_table_kwargs['geojson_units']` must be 'px' or 'um', got {units!r}.")
    if units == "um" and pixel_size_um is None:
        raise ValueError(f"{geojson} has coordinates in µm, but no pixel size is known to convert them to pixels.")
    if not geojson.is_file():
        raise FileNotFoundError(f"{geojson} does not exist or is not a file.")
    ids, geometries, kinds = _read_geojson(geojson)

    polygons = np.full(len(table.obs), None, dtype=object)
    id_column = _object_id_column(list(table.obs.columns))
    if id_column is not None and ids is not None:
        object_ids = pd.Index(table.obs[id_column].astype(str))
        if not object_ids.is_unique:
            raise ValueError(f"The '{id_column}' column of {table_path} is not unique.")
        rows = object_ids.get_indexer(pd.Index(ids).astype(str))
        found = rows >= 0
        if not found.any() and discovered:
            _ignore_geojson(geojson, table_path)
            return None
        if not found.any():
            raise ValueError(
                f"{geojson} does not hold the cells of {table_path}: none of the feature ids is an '{id_column}' "
                "of the table."
            )
        if not pd.Index(rows[found]).is_unique:
            raise ValueError(f"{geojson} has several features with the same id.")
        polygons[rows[found]] = geometries[found]
        logger.info("Matched %d of the %d features of %s to cells by id.", found.sum(), len(geometries), geojson)
        is_cell = np.ones(len(ids), dtype=bool) if kinds is None else np.isin(kinds, list(_QUPATH_CELL_TYPES))
        if unmatched := int(np.count_nonzero(is_cell & ~found)):
            warnings.warn(
                f"{unmatched} of the {is_cell.sum()} cell features of {geojson} have no row in {table_path} (e.g. "
                "cells filtered out of the table); they are ignored.",
                UserWarning,
                stacklevel=5,
            )
        if found.sum() < len(polygons):
            warnings.warn(
                f"{len(polygons) - found.sum()} of the {len(polygons)} cells of {table_path} have no polygon in "
                f"{geojson}; they are in the table but not in the labels and polygons.",
                UserWarning,
                stacklevel=5,
            )
    else:
        cells = geometries if kinds is None else geometries[np.isin(kinds, list(_QUPATH_CELL_TYPES))]
        if len(cells) == 0 and discovered:
            _ignore_geojson(geojson, table_path)
            return None
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


def _ignore_geojson(geojson: Path, table_path: Path) -> None:
    warnings.warn(
        f"Ignoring {geojson}: it holds none of the cells of {table_path}. Pass the cell outlines as "
        "`cells_table_kwargs['geojson']`.",
        UserWarning,
        stacklevel=6,
    )


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
        # QuPath's default export without the FeatureCollection option, which GDAL does not read; each feature is
        # reduced as soon as it is parsed, so that its other properties are not all held at once
        with path.open("rb") as f:
            features = json.load(f, object_hook=_reduce_feature)
        ids = [feature[0] for feature in features]
        geometries = np.array([feature[1] for feature in features], dtype=object)
        types = [feature[2] for feature in features]
    else:
        # QuPath's other properties (measurements, nucleus outlines) would take several times the memory
        frame = gpd.read_file(path, columns=["id", "objectType"])
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


def _reduce_feature(obj: dict[str, Any]) -> Any:
    """The id, geometry and QuPath object type of a parsed GeoJSON feature; other JSON objects unchanged."""
    if "geometry" not in obj:
        return obj
    geometry = shapely.geometry.shape(obj["geometry"]) if obj.get("geometry") else None
    return obj.get("id"), geometry, (obj.get("properties") or {}).get("objectType")


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


def _codex_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The roles of a CODEX table; renames the columns of ``df`` to the name part of ``cycCCC_chHHH:<marker>``."""
    split = [str(column).partition(":") for column in df.columns]
    # the source of an FCS column is its `$PnN`
    fcs = path.suffix.lower() == CodexKeys.FCS_FILE.value
    sources = [name if fcs else str(column) for column, (name, _, _) in zip(df.columns, split, strict=True)]
    names = dedupe_channel_names([name for name, _, _ in split])
    displays = [display.strip() or name for name, (_, _, display) in zip(names, split, strict=True)]
    df.columns = pd.Index(names)
    columns = {column.casefold(): column for column in names}
    x, y = kwargs.get("x", columns.get("x")), kwargs.get("y", columns.get("y"))
    if x is None or y is None:
        raise ValueError(f"{path} has no 'x' and 'y' centroid columns; found {names}.")
    instance_column = kwargs.get("instance_id", columns.get(CodexKeys.INSTANCE_KEY.value))
    region_column = kwargs.get("region_column", columns.get(CodexKeys.REGION_KEY.value))
    if region_column in df.columns and pd.api.types.is_numeric_dtype(df[region_column]):
        values = df[region_column]
        df[region_column] = values.map({value: f"reg{int(value):03d}" for value in values.dropna().unique()})
    area = columns.get("size")

    matches = [_MARKER_COLUMN.fullmatch(name) for name in names]
    if kwargs.get("markers") is not None:
        selected = set(_select_markers(df, kwargs["markers"], {x, y, instance_column, region_column, area}))
        is_marker = [name in selected for name in names]
    elif any(matches):
        is_marker = [match is not None for match in matches]
    else:
        is_marker = [
            name.casefold() not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df.iloc[:, i])
            for i, name in enumerate(names)
        ]
        taken = [name for name, marker in zip(names, is_marker, strict=True) if marker]
        warnings.warn(
            f"{path} has no 'cycCCC_chHHH' columns; reading the {len(taken)} numeric columns {taken} that are not "
            "CODEX Processor columns as the markers. Set `cells_table_kwargs['markers']` to choose the marker columns.",
            UserWarning,
            stacklevel=5,
        )
    positions = [i for i, marker in enumerate(is_marker) if marker]
    var = pd.DataFrame(
        {"source_column": [sources[i] for i in positions]},
        index=pd.Index(dedupe_channel_names([displays[i] for i in positions])),
    )
    if positions and all(matches[i] for i in positions):
        var["cycle"] = [int(matches[i].group(1)) for i in positions]  # type: ignore[union-attr]
        var["channel"] = [int(matches[i].group(2)) for i in positions]  # type: ignore[union-attr]
    return {
        "markers": [names[i] for i in positions],
        "var": var,
        "x": x,
        "y": y,
        "in_um": False,
        "instance_column": instance_column,
        "region_column": region_column,
        "area_column": area,
    }


def _qupath_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    columns = _qupath_names(list(df.columns))
    unit = next((unit for unit, (x_name, _) in _QUPATH_CENTROIDS.items() if x_name in columns), None)
    x_name, y_name = ("X", "Y") if unit is None else _QUPATH_CENTROIDS[unit]
    x, y = columns.get(x_name), columns.get(y_name)
    if x is None or y is None:
        raise ValueError(
            f"{path} has no centroid columns ('Centroid X µm' and 'Centroid Y µm', 'Centroid X px' and "
            f"'Centroid Y px', or 'X' and 'Y'); found {list(df.columns)}. If the export splits the centroids and "
            "the measurements into two files, join them on the object id first."
        )
    if unit is None:
        unit = _qupath_xy_unit(path, list(columns), kwargs)
    object_id = _object_id_column(list(df.columns))
    instance_column = object_id if object_id is not None and pd.api.types.is_integer_dtype(df[object_id]) else None
    region_column = kwargs.get("region_column", _QUPATH_IMAGE if _QUPATH_IMAGE in df.columns else None)
    area = columns.get(f"Cell: Area {'µm' if unit == 'um' else 'px'}^2")
    roles = {x, y, object_id, region_column, area}

    intensities: dict[str, dict[str | None, str]] = {}
    for column in df.columns:
        if (found := _qupath_intensity(column)) is not None:
            intensities.setdefault(found[0], {}).setdefault(found[1], column)
    # `<marker>: Mean` columns are the markers only of detections without compartments
    with_compartments = {marker: found for marker, found in intensities.items() if set(found) - {None}}
    compartments = with_compartments or intensities
    if kwargs.get("markers") is not None:
        markers = _select_markers(df, kwargs["markers"], roles)
        var = pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(markers)))
    elif compartments:
        chosen = {
            marker: next(c for c in (*_QUPATH_COMPARTMENTS, None) if c in found)
            for marker, found in compartments.items()
        }
        markers = [compartments[marker][compartment] for marker, compartment in chosen.items()]
        var = pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(list(chosen))))
        if with_compartments:
            var["compartment"] = list(chosen.values())
    else:
        # e.g. a detection export with the measurements named after the markers only; its morphology (e.g.
        # `Area µm^2`, `Max diameter µm`, `Circularity`), other measurements (e.g. `DAPI: Median`) and empty text
        # columns (e.g. `Name`, read as all-NaN numbers) are not markers
        candidates = [
            c
            for c in df.columns
            if not any(_QUPATH_MEASUREMENT.match(name) for name in _qupath_names([c]))
            and not _is_qupath_morphology(c)
            and df[c].notna().any()
        ]
        markers = _select_markers(df[candidates], None, roles)
        var = pd.DataFrame({"source_column": markers}, index=pd.Index(dedupe_channel_names(markers)))
        if markers:
            shown = f"{markers[:10]}{' ...' if len(markers) > 10 else ''}"
            message = f"reading the {len(markers)} other numeric columns {shown} as the markers"
        else:
            message = "its table has no marker intensities"
        warnings.warn(
            f"{path} has no QuPath mean intensity columns (e.g. '<marker>: <compartment>: Mean'); {message}. Set "
            "`cells_table_kwargs['markers']` to choose the marker columns.",
            UserWarning,
            stacklevel=5,
        )
    return {
        "markers": markers,
        "var": var,
        "x": x,
        "y": y,
        "in_um": unit == "um",
        "instance_column": instance_column,
        "region_column": region_column,
        "area_column": area,
    }


def _qupath_intensity(column: str) -> tuple[str, str | None] | None:
    """The marker and compartment (`None` without one) of a QuPath mean intensity column, also in R's form."""
    for name in (column, *_qupath_names([column])):
        for pattern in (_QUPATH_MARKER, _QUPATH_COMPARTMENT_MARKER):
            if (match := pattern.fullmatch(name)) is not None:
                return match["marker"], match["compartment"]
    for name in (column, *_qupath_names([column])):
        if (match := _QUPATH_DETECTION_MARKER.fullmatch(name)) is not None:
            return match["marker"], None
    return None


def _is_qupath_morphology(column: str) -> bool:
    """Whether ``column`` (also in R's ``make.names`` form) is a QuPath measurement with a unit or a shape measure."""
    return any(_QUPATH_UNIT.search(name) or name in _QUPATH_UNITLESS_MEASUREMENTS for name in _qupath_names([column]))


def _qupath_xy_unit(path: Path, columns: list[str], kwargs: Mapping[str, Any]) -> str:
    """The unit of the ``X``/``Y`` centroids of a QuPath table: ``coordinates_unit``, else that of its measurements.

    QuPath writes all measurements of an image in one unit, µm if the image is calibrated, else pixels.
    """
    if kwargs.get("coordinates_unit") is not None:
        return _coordinates_unit(kwargs)
    units = {"px" if m.group(1) == "px" else "um" for column in columns if (m := _QUPATH_UNIT.search(column))}
    if len(units) != 1:
        found = f"in {sorted(units)}" if units else "in no unit"
        raise ValueError(
            f"The centroid columns 'X' and 'Y' of {path} name no unit, and its other measurements are {found}; set "
            "`cells_table_kwargs['coordinates_unit']` to 'px' or 'um'."
        )
    unit = units.pop()
    warnings.warn(
        f"The centroid columns 'X' and 'Y' of {path} name no unit; reading them in {unit}, the unit of its other "
        "measurements. Set `cells_table_kwargs['coordinates_unit']` to 'px' or 'um' to choose it.",
        UserWarning,
        stacklevel=6,
    )
    return unit


def _coordinates_unit(kwargs: Mapping[str, Any]) -> str:
    """``cells_table_kwargs['coordinates_unit']``: ``"px"`` (the default) or ``"um"``."""
    unit = kwargs.get("coordinates_unit", "px")
    if unit not in ("px", "um"):
        raise ValueError(f"`cells_table_kwargs['coordinates_unit']` must be 'px' or 'um', got {unit!r}.")
    return str(unit)


def _coordinates_space(path: Path, in_um: bool, kwargs: Mapping[str, Any]) -> str:
    """``cells_table_kwargs['coordinates_space']`` of a generic table: ``"image"`` (the default) or ``"mask"``."""
    space = kwargs.get("coordinates_space", "image")
    if space not in ("image", "mask"):
        raise ValueError(f"`cells_table_kwargs['coordinates_space']` must be 'image' or 'mask', got {space!r}.")
    if space == "mask" and in_um:
        raise ValueError(
            f"`cells_table_kwargs['coordinates_space']` is 'mask', but the centroids of {path} are in µm; mask "
            "pixels need `cells_table_kwargs['coordinates_unit']` 'px'."
        )
    return str(space)


def _select_markers(df: pd.DataFrame, selection: str | Sequence[str] | None, exclude: set[str | None]) -> list[str]:
    """The marker columns: ``selection`` (columns, or a regular expression matching the full column name).

    Without ``selection``, all numeric columns. Columns in ``exclude`` are only selected if listed in ``selection``.
    """
    if selection is None:
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if isinstance(selection, str):
        return [c for c in df.columns if c not in exclude and re.fullmatch(selection, c)]
    return [str(marker) for marker in selection]


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
    unit = (positions[0].group(1) or _inform_position_unit(path, list(df.columns), kwargs)).casefold()
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


def _inform_position_unit(path: Path, columns: list[str], kwargs: Mapping[str, Any]) -> str:
    """The unit of ``Cell X Position`` without one in its name: ``coordinates_unit``, else that of the other columns.

    inForm writes all columns in one unit, but names it only for some, e.g. ``Tissue Category Area (pixels)`` or
    ``Distance from Process Region Edge (microns)``; a table that names none is in pixels.
    """
    if kwargs.get("coordinates_unit") is not None:
        return {"px": "pixels", "um": "microns"}[_coordinates_unit(kwargs)]
    units = {m.group(1) for column in columns if (m := _INFORM_UNIT.search(column)) is not None}
    if len(units) > 1:
        raise ValueError(
            f"The columns 'Cell X Position' and 'Cell Y Position' of {path} name no unit, and its other columns are "
            f"in {sorted(units)}; set `cells_table_kwargs['coordinates_unit']` to 'px' or 'um'."
        )
    unit = units.pop() if units else "pixels"
    logger.info("Reading the cell positions of %s in %s, the unit of its other columns.", path, unit)
    return unit


def _generic_roles(path: Path, df: pd.DataFrame, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if "x" not in kwargs or "y" not in kwargs:
        raise ValueError(
            f"Reading {path} as a 'generic' cell table needs the centroid columns as `cells_table_kwargs` 'x' and "
            f"'y'; found the columns {list(df.columns)}."
        )
    unit = _coordinates_unit(kwargs)
    roles = {kwargs["x"], kwargs["y"], kwargs.get("instance_id"), kwargs.get("region_column")}
    markers = _select_markers(df, kwargs.get("markers"), roles)
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
    region: pd.Series | None,
    px_per_unit: float | np.ndarray = 1.0,
) -> CellTable:
    """Build a `CellTable` from a data frame, the roles of its columns and the region of every row.

    ``px_per_unit`` may be given per row.
    """
    roles = [x, y, instance_column, region_column, area_column, *markers]
    missing = [column for column in roles if column is not None and column not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}; found {list(df.columns)}.")

    if instance_column is not None:
        instance_id = pd.Series(_instance_ids(path, df[instance_column]))
    else:
        # label images use 0 for the background, so instance ids start at 1
        instance_id = pd.Series(np.arange(1, len(df) + 1, dtype=np.int64))
    index = pd.Index(instance_id.astype(str))
    instance_id.index = index
    instance_id.name = CodexKeys.INSTANCE_KEY.value

    if region is None:
        raise ValueError(
            f"{path} has no region column, so its cells cannot be assigned to regions: a table for several regions "
            "needs a region column (`cells_table_kwargs['region_column']`). Alternatively, pass one table per "
            "region as a mapping {region: path}."
        )
    region = pd.Series(region.to_numpy(), index=index, name=CodexKeys.REGION_KEY.value)
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


def _instance_ids(path: Path, column: pd.Series) -> np.ndarray:
    """The integer instance ids in ``column``; raises for missing or non-integer ids, which no label can have."""
    if pd.api.types.is_integer_dtype(column):
        return column.to_numpy(dtype=np.int64)
    ids = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~(np.isfinite(ids) & (ids == np.round(ids)))
    if invalid.any():
        raise ValueError(
            f"{path} has {invalid.sum()} cell(s) whose instance id in '{column.name}' is missing or not an integer "
            f"(e.g. {column[invalid].tolist()[:5]}); every cell needs an integer instance id, the value of its "
            "label. Choose another instance id column, or remove these rows."
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
            "region needs its own instance id. Choose another instance id column, or remove the duplicate rows."
        )


def _drop_key_columns(obs: pd.DataFrame, instance_column: str | None, region_column: str | None) -> pd.DataFrame:
    """Drop the instance id and region columns if they are named like the keys whose values replace them.

    Any other column named like a key keeps its values as ``source_<name>`` in the table element.
    """
    consumed = {CodexKeys.INSTANCE_KEY.value: instance_column, CodexKeys.REGION_KEY.value: region_column}
    return obs.drop(columns=[key for key, column in consumed.items() if column == key])


def _read_header(path: Path) -> tuple[list[str], str]:
    """The column names of a delimited text table and its delimiter.

    The delimiter is a tab if the header has one, else a semicolon if it has more semicolons than commas, else a comma.
    """
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    sep = "\t" if "\t" in header else ";" if header.count(";") > header.count(",") else ","
    return [column.strip().strip('"') for column in header.split(sep)], sep


def _read_text_table(path: Path) -> pd.DataFrame:
    """Read a delimited text table with the delimiter of :func:`_read_header`, with unique column names.

    Semicolon-separated tables (European locales) are read with decimal commas; their columns with decimal points,
    which that leaves as text, are converted to numbers too.
    """
    names, sep = _read_header(path)
    if sep == ";":
        # the C engine keeps the header's repeated names apart only by renaming them, so the names come from the header
        df = pd.read_csv(path, sep=sep, decimal=",", engine="c", header=None, skiprows=1)
        df.columns = pd.Index(names)
        for i in range(df.shape[1]):
            if not pd.api.types.is_numeric_dtype(df.iloc[:, i]):
                try:
                    df.isetitem(i, pd.to_numeric(df.iloc[:, i]).to_numpy())
                except (ValueError, TypeError):
                    pass
    else:
        df = pd.read_csv(path, sep=sep, engine="pyarrow")
    # inForm writes some column names with trailing spaces; pyarrow keeps repeated names, which are made unique like
    # the marker names
    df.columns = pd.Index(dedupe_channel_names([str(column).strip() for column in df.columns]))
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
    files: Mapping[str, Path],
    fmt: str,
    *,
    available: Mapping[str, Sequence[Path]],
    selected: Sequence[str],
    pixel_sizes: Mapping[str, float | None],
    cells_table_kwargs: Mapping[str, Any],
) -> list[CellTable]:
    """Read user-supplied cell tables, as resolved by :func:`resolve_per_region`, and assign their cells to regions.

    The values of a region column are matched to the regions, see :func:`_match_region`. A table given for one
    region holds the cells of that region; if its region column holds a single value that names no region, it
    holds only cells of that region. Cells whose region names no available region are dropped with a warning;
    cells of available regions that are not read are dropped silently.

    Parameters
    ----------
    files
        The table of each region, or ``{"*": path}`` for one table for all regions.
    fmt
        The table format, see :func:`read_cell_table`.
    available
        All region ids of the dataset, with the image files of each.
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
    geojson_files = {} if geojson is False else resolve_per_region(geojson, selected, list(available))
    if unknown := sorted(geojson_files.keys() - files.keys()):
        raise ValueError(
            f"`cells_table_kwargs['geojson']` is given for {unknown}, but the cell tables are given for "
            f"{list(files)}; pass the GeoJSON files in the same form as the tables."
        )
    tables = []
    # the µm coordinates of each cell are converted with the pixel size of its region
    sizes = {region: pixel_sizes.get(region) for region in selected}
    for key, path in files.items():
        table = read_cell_table(
            path,
            fmt,
            pixel_size_um=sizes,
            default_region=None if key == "*" else key,
            cells_table_kwargs={
                **cells_table_kwargs,
                "geojson": False if geojson is False else geojson_files.get(key),
            },
            regions=available,
        )
        tables.append(_assign_regions(table, path, key, available, selected))
    return tables


def _assign_regions(
    table: CellTable, path: Path, key: str, available: Mapping[str, Sequence[Path]], selected: Sequence[str]
) -> CellTable:
    values = list(table.region.unique())
    unmatched = sorted(value for value in values if value not in available)
    if unmatched:
        warnings.warn(
            f"Ignoring the cells of {unmatched} in {path}: they match none of the regions {list(available)}.",
            UserWarning,
            stacklevel=4,
        )
    keep = table.region.isin(selected if key == "*" else [key]).to_numpy()
    if not keep.any():
        target = f"region '{key}'" if key != "*" else f"the regions read {list(selected)}"
        raise ValueError(f"{path} has no cells of {target}; its region column holds {values}.")
    return table.take(np.flatnonzero(keep))


def _match_region(value: str, regions: Mapping[str, Sequence[Path]]) -> str | None:
    """The region named by ``value``: its id, else the region of an image file with that name or stem, else `None`.

    ``value`` is also matched after :func:`spatialdata.sanitize_name`, as the region ids are, so that e.g. ``A_[1,2]``
    names the region ``A__1_2_``. A file name may be followed by a non-alphanumeric character and more, e.g. QuPath's
    ``A_Scan1.qptiff - resolution #1``; a name that merely starts with another region's name (``A_2_Scan1.qptiff``)
    does not name it.
    """
    candidates = list(dict.fromkeys((value, sanitize_name(value))))
    if exact := next((candidate for candidate in candidates if candidate in regions), None):
        return exact
    stem = _IMAGE_SUFFIX.sub("", value)
    for region, files in regions.items():
        for name in (file.name for file in files):
            if stem == _IMAGE_SUFFIX.sub("", name) or (value.startswith(name) and not value[len(name)].isalnum()):
                return region
    return None


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
        The mask to read from a file with several: its index or channel name, the OME channel name (e.g.
        ``"cells"``) or, in an inForm ``*_binary_seg_maps.tif``, ``"Tissue"`` (the tissue class map) or the
        compartment (``"Nucleus"``, ``"Cytoplasm"``, ``"Membrane"``). If `None`, the first one; for an inForm
        segmentation map ``"Nucleus"``, whose labels are the ``Cell ID`` of every cell.

    Returns
    -------
    The labels element.
    """
    labels_models_kwargs, chunks = _labels_models_kwargs(labels_models_kwargs)
    image, _, _ = read_tiff(path, chunks)
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"{path} has dtype {image.dtype}; a segmentation mask must have an integer dtype.")

    n_masks = image.shape[0]
    names, inform = _mask_channel_names(path, n_masks)
    if isinstance(mask_channel, str):
        if mask_channel not in names:
            raise ValueError(f"{path} has no mask channel {mask_channel!r}; its channel names are {names}.")
        index = names.index(mask_channel)
    elif mask_channel is None and inform and _INFORM_DEFAULT_MASK in names:
        index = names.index(_INFORM_DEFAULT_MASK)
        logger.info(
            "%s is an inForm segmentation map %s; reading %r. Set `mask_channel` to choose another.",
            path,
            names,
            _INFORM_DEFAULT_MASK,
        )
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


def _mask_channel_names(path: Path, n_masks: int) -> tuple[list[str], bool]:
    """The channel names of a mask file and whether it is an inForm segmentation map.

    The names are the OME channel names, else for an inForm segmentation map (a file with a ``<SegmentationImage>``
    page) the page names, see :func:`_inform_page_name`, else ``[]``.
    """
    with tifffile.TiffFile(path) as tf:
        if tf.ome_metadata:
            return _ome_channel_names(tf.ome_metadata), False
        if len(tf.pages) != n_masks:
            return [], False
        descriptions = [str(tf.pages[i].description or "") for i in range(n_masks)]  # type: ignore[union-attr]
    roots = [match.group(1) if (match := _INFORM_XML_ROOT.search(d)) else None for d in descriptions]
    if _INFORM_SEGMENTATION not in roots:
        return [], False
    return [_inform_page_name(d, root, i) for i, (d, root) in enumerate(zip(descriptions, roots, strict=True))], True


def _inform_page_name(description: str, root: str | None, index: int) -> str:
    """The compartment of a ``<SegmentationImage>`` page, else its XML root tag (``Tissue`` for the tissue map)."""
    if (compartment := _INFORM_COMPARTMENT.search(description)) is not None:
        return compartment.group(1)
    if root is not None:
        return _INFORM_PAGE_NAMES.get(root, root)
    return f"page {index}"


def _ome_channel_names(ome_metadata: str) -> list[str]:
    images = tifffile.xml2dict(ome_metadata)["OME"]["Image"]
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
    """``tables`` with mask-pixel centroids and radii scaled to image pixels by their region's mask factor.

    A mask pixel covers ``f x f`` image pixels, so a centroid ``c`` moves to the centre of those, ``f * c + (f - 1) / 2``.
    """
    scaled = []
    for table in tables:
        if table.mask_pixels and factors:
            row_factors = table.region.map(factors).fillna(1).to_numpy(dtype=np.float64)
            table = replace(
                table,
                centroids_px=None
                if table.centroids_px is None
                else table.centroids_px * row_factors[:, None] + (row_factors[:, None] - 1) / 2,
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
    still kept as shapes, see :func:`polygon_shapes`. Regions none of whose cells has a polygon get no labels.

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
    The labels elements by region id, and the GeoJSON file of each region with polygons.
    """
    labels_models_kwargs, chunks = _labels_models_kwargs(labels_models_kwargs)
    elements, sources = {}, {}
    for table in tables:
        if table.polygons_px is None:
            continue
        for region, rows in _region_rows(table).items():
            rows = rows[pd.notna(table.polygons_px[rows])]
            # regions without polygons get circles, like tables without a GeoJSON
            if len(rows) == 0:
                continue
            sources[region] = str(table.geojson)
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
            data = rasterize_polygons_lazy(
                table.polygons_px[rows], table.instance_id.to_numpy()[rows], image_shapes[region], chunks
            )
            elements[region] = Labels2DModel.parse(
                data,
                dims=("y", "x"),
                transformations=_transformations_of(transformations, region),
                **labels_models_kwargs,
            )
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
    The table elements, ``{"table": ...}`` if all tables have the same ``var``, else ``{"<region>_table": ...}``
    per region, and the circle shapes by element name. The circles have the cells' area (radius 1 px without a
    positive one).
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
    # tables with other markers or other marker metadata (e.g. QuPath compartments) are not concatenated
    pieces = _split_by_region(tables) if any(not table.var.equals(tables[0].var) for table in tables) else None

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
    :class:`CellTable`) are scaled by the same factor (see :func:`_to_image_pixels`), so that the table and circles
    are in image pixels. A
    `UserWarning` is raised for a region if more than 5% of its cells lie outside its image: their coordinates are
    in another unit or relative to another origin (e.g. inForm positions in microns relative to the slide).

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
    _check_cells_in_images(cell_tables, image_shapes)
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


def _check_cells_in_images(tables: Sequence[CellTable], image_shapes: Mapping[str, tuple[int, int]]) -> None:
    """Warn for each region with an image if more than 5% of its cells' centroids lie outside the image."""
    for table in tables:
        if table.centroids_px is None:
            continue
        for region, rows in _region_rows(table).items():
            if region not in image_shapes:
                continue
            centroids = table.centroids_px[rows]
            centroids = centroids[np.isfinite(centroids).all(axis=1)]
            height, width = image_shapes[region]
            outside = int(np.count_nonzero(((centroids < 0) | (centroids > [width, height])).any(axis=1)))
            if outside > _MAX_OUTSIDE_FRACTION * len(centroids):
                warnings.warn(
                    f"{outside} of the {len(centroids)} cells of region '{region}' lie outside its image ({width} x "
                    f"{height} px): the table's coordinates are in another unit (see "
                    "`cells_table_kwargs['coordinates_unit']`) or relative to another origin (e.g. inForm positions "
                    "in microns, which may be relative to the slide).",
                    UserWarning,
                    stacklevel=4,
                )


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
