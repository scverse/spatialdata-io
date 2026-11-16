from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from dask import delayed
from geopandas import GeoDataFrame
from shapely import GeometryType, from_ragged_array
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    PointsModel,
    ShapesModel,
)
from spatialdata.transformations import set_transformation
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataTree

from spatialdata_io._constants._constants import AteraKeys
from spatialdata_io.readers._utils._utils import parse_physical_size

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from xarray import DataArray
    from zarr.abc.buffer import Buffer
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype


# `zarr.Group.get_array` / `zarr.Group.get_group` (typed lookups that raise on a missing node or on the
# wrong node kind) only exist from zarr 3.3.0. These two helpers provide the same behavior on older zarr
# 3.x releases. We deliberately do not pin `zarr>=3.3.0` for now: it is a recent release, spatialdata itself
# only requires `zarr>=3.0.0`, and a higher floor here could conflict with users' environments and other
# scverse packages. Once spatialdata-io requires `zarr>=3.3.0`, remove the helpers and call the methods directly,
# e.g. `_get_array(tile, key)` -> `tile.get_array(key)` and `_get_group(group, key)` -> `group.get_group(key)`.
def _get_array(group: zarr.Group, path: str) -> zarr.Array:
    """Return the array at ``path`` (relative to ``group``), raising if it is missing or a group."""
    node = group[path]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"Expected a zarr array at {path!r}, found {type(node).__name__}.")
    return node


def _get_group(group: zarr.Group, path: str) -> zarr.Group:
    """Return the group at ``path`` (relative to ``group``), raising if it is missing or an array."""
    node = group[path]
    if not isinstance(node, zarr.Group):
        raise TypeError(f"Expected a zarr group at {path!r}, found {type(node).__name__}.")
    return node


# `var` in cell_feature_matrix.zarr.zip also has ~135 per-cluster differential-expression columns
# (`de_leiden_res_1.0_c{N}_{logfc,pval,pval_adj,rank,score}`); the readers keep only this small
# default subset to avoid holding that wide table in memory. Use `read_var` to load the full table.
DEFAULT_VAR_COLUMNS = [
    str(AteraKeys.FEATURE_NAME),
    str(AteraKeys.VAR_FILTERED),
    str(AteraKeys.VAR_HIGHLY_VARIABLE),
    str(AteraKeys.FEATURE_ID),
    str(AteraKeys.VAR_FEATURE_TYPE),
    str(AteraKeys.VAR_GENOME),
]

# Number of `obs` rows read per `X` chunk in the readers' lazy `dask` array for the table.
DEFAULT_TABLE_ROW_CHUNK_SIZE = 50_000
# Max transcripts per `dask` partition / `atera_to_parquet` batch (~130 MB of decoded columns).
DEFAULT_TRANSCRIPT_CHUNK_ROWS = 5_000_000


class _PadRaggedVlenStore(zarr.storage.WrapperStore[zarr.storage.ZipStore]):
    """Pad the short trailing chunk of a 1-D ``vlen-utf8`` zarr v2 array to the nominal chunk size.

    10x writes the last chunk of the `obs`/`var` string columns (e.g. `barcode`, chunks of 1,000,000) with only the
    items it holds. zarr decodes every v2 chunk at the nominal size and raises "cannot reshape array of size 376951
    into shape (1000000,)"; zarr's own writer pads that chunk. Only a trailing chunk that holds exactly the remaining
    items is padded; any other chunk is passed on unchanged, so zarr still raises on a corrupt one.
    """

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        buf = await super().get(key, prototype, byte_range)
        parent, _, name = key.rpartition("/")
        if buf is None or byte_range is not None or not name.isdigit():
            return buf
        meta_buf = await super().get(f"{parent}/.zarray" if parent else ".zarray", prototype)
        meta = json.loads(meta_buf.to_bytes()) if meta_buf is not None else {}
        if len(meta.get("shape", ())) != 1 or [f["id"] for f in meta.get("filters") or []] != ["vlen-utf8"]:
            return buf
        (size,), (chunk,) = meta["shape"], meta["chunks"]
        n_items = size - int(name) * chunk
        if not 0 < n_items < chunk:
            return buf
        from numcodecs import VLenUTF8, get_codec

        compressor = get_codec(meta["compressor"]) if meta.get("compressor") else None
        raw = buf.to_bytes()
        items = VLenUTF8().decode(compressor.decode(raw) if compressor else raw)
        if len(items) != n_items:
            return buf
        padded = np.full(chunk, meta.get("fill_value") or "", dtype=object)
        padded[:n_items] = items
        encoded = VLenUTF8().encode(padded)
        return prototype.buffer.from_bytes(compressor.encode(encoded) if compressor else encoded)


@contextlib.contextmanager
def _open_zip(path: Path) -> Iterator[zarr.Group]:
    """Open a zipped zarr store read-only (see `_PadRaggedVlenStore`)."""
    with zarr.storage.ZipStore(path, read_only=True) as store:
        yield zarr.open_group(_PadRaggedVlenStore(store), mode="r")


def _decode_packed_cell_id(raw: np.ndarray) -> np.ndarray:
    """Decode a (N, 2) uint32 packed cell id array into the int64 ids used in the cell-feature-matrix obs.

    The two uint32 columns pack into an int64 as ``low + (high << 32)``, which is the little-endian int64 view of
    the pair. The sentinel ``(uint32 max, uint32 max)`` marks "no cell assigned" and so views as ``-1``.
    """
    return np.ascontiguousarray(raw, dtype="<u4").view("<i8").reshape(-1)


def _polygons_to_shapes(
    coords: np.ndarray,
    num_vertices: np.ndarray,
    cell_index: np.ndarray,
    cell_ids: np.ndarray | None,
    pixel_size: float,
    is_nucleus: bool,
) -> GeoDataFrame:
    """Build a boundary-polygon `ShapesModel` from ragged ``(coords, num_vertices, cell_index)``.

    ``coords`` is the flat, ragged (N total vertices, 2) array of (x, y) pairs; ``num_vertices``
    gives each polygon's vertex count (so ``coords`` splits into contiguous runs of this length).
    ``cell_index`` maps each polygon (0-based) to the owning cell's row position in the
    table/``cells.zarr`` (which share the same row order). For cells this mapping is the identity;
    for nuclei it may not be, since multinucleate cells have multiple nucleus polygons. Shared
    between `_get_polygons` (eager, `polygon_sets`) and `read_cell_boundaries` (tiled/pyramided,
    `gridded_polygon_sets`), which differ only in how they produce these ragged arrays.
    """
    n_polygons = len(num_vertices)
    ring_offsets = np.concatenate([[0], np.cumsum(num_vertices)])
    geom_offsets = np.arange(n_polygons + 1)
    geoms = from_ragged_array(GeometryType.POLYGON, coords, offsets=(ring_offsets, geom_offsets))

    owning_cell_id = cell_ids[cell_index] if cell_ids is not None else cell_index
    if is_nucleus:
        # multiple polygons can map to the same cell (multinucleate cells): use the 1-based
        # polygon position as the index, and keep the owning cell id as a column.
        geo_df = GeoDataFrame(
            {"geometry": geoms, str(AteraKeys.CELL_ID): owning_cell_id},
            index=pd.RangeIndex(1, n_polygons + 1),
        )
    else:
        geo_df = GeoDataFrame({"geometry": geoms}, index=owning_cell_id)
        geo_df.index.name = str(AteraKeys.CELL_ID)

    scale = Scale([1.0 / pixel_size, 1.0 / pixel_size], axes=("x", "y"))
    return ShapesModel.parse(geo_df, transformations={"global": scale})


def _get_polygons(
    cells_group: zarr.Group,
    mask_index: int,
    cell_ids: np.ndarray | None,
    pixel_size: float,
    is_nucleus: bool,
) -> GeoDataFrame:
    """Build boundary polygons from cells.zarr.zip polygon_sets/{mask_index}.

    Each row of ``vertices`` is a fixed-width (x, y) pair buffer, padded to a maximum number of
    vertices; ``num_vertices`` gives the number of valid pairs to use.
    """
    group = _get_group(cells_group, f"{AteraKeys.POLYGON_SETS_GROUP}/{mask_index}")
    vertices = np.asarray(_get_array(group, str(AteraKeys.POLYGON_VERTICES))[...])
    num_vertices = np.asarray(_get_array(group, str(AteraKeys.POLYGON_NUM_VERTICES))[...])
    cell_index = np.asarray(_get_array(group, str(AteraKeys.POLYGON_CELL_INDEX))[...])

    n_polygons, max_coords = vertices.shape
    max_vertices = max_coords // 2
    vertices = vertices.reshape(n_polygons, max_vertices, 2)
    valid = np.arange(max_vertices)[None, :] < num_vertices[:, None]
    coords = vertices[valid]

    return _polygons_to_shapes(coords, num_vertices, cell_index, cell_ids, pixel_size, is_nucleus)


def _tile_row_ranges(n_rows: int, chunk_rows: int = DEFAULT_TRANSCRIPT_CHUNK_ROWS) -> list[tuple[int, int]]:
    """Split a tile of ``n_rows`` transcripts into contiguous ``(start, stop)`` row ranges of at most `chunk_rows`.

    A single tile can hold ~100M+ transcripts, so reading it whole takes gigabytes (per concurrently running `dask`
    task, or per tile in `atera_to_parquet`). An empty tile still yields one empty range, so it keeps a partition.
    """
    return [(start, min(start + chunk_rows, n_rows)) for start in range(0, n_rows, chunk_rows)] or [(0, 0)]


def _transcript_rows(tile: zarr.Group, start: int, stop: int, feature_name: pd.Categorical) -> pd.DataFrame:
    """Decode rows ``[start, stop)`` of one tile of transcripts.zarr.zip, labelled with `feature_name`."""

    def rows(key: AteraKeys) -> np.ndarray:
        return np.asarray(_get_array(tile, str(key))[start:stop])

    location = rows(AteraKeys.TRANSCRIPTS_LOCATION)
    return pd.DataFrame(
        {
            str(AteraKeys.TRANSCRIPTS_X): location[:, 0].astype(np.float32),
            str(AteraKeys.TRANSCRIPTS_Y): location[:, 1].astype(np.float32),
            str(AteraKeys.TRANSCRIPTS_Z): location[:, 2].astype(np.float32),
            str(AteraKeys.FEATURE_NAME): feature_name,
            str(AteraKeys.CELL_ID): _decode_packed_cell_id(rows(AteraKeys.CELL_ID)),
            "overlaps_nucleus": rows(AteraKeys.TRANSCRIPTS_OVERLAPS_NUCLEUS).reshape(-1).astype(bool),
            "quality_score": rows(AteraKeys.TRANSCRIPTS_QUALITY_SCORE).reshape(-1).astype(np.float32),
        }
    )


def _read_transcript_tile(
    grid: zarr.Group, tile_key: str, feature_names: list[str], start: int = 0, stop: int | None = None
) -> pd.DataFrame:
    """Read and decode rows ``[start, stop)`` of a single spatial tile of transcripts.zarr.zip (default: the whole tile).

    `grid` is the store's ``grid`` group. All `dask` tasks share it, so the zip is opened once per process, not once
    per task.
    """
    tile = _get_group(grid, tile_key)
    n_rows = _get_array(tile, str(AteraKeys.TRANSCRIPTS_LOCATION)).shape[0]
    stop = n_rows if stop is None else min(stop, n_rows)
    gene_offset = np.asarray(_get_array(tile, str(AteraKeys.TRANSCRIPTS_GENE_OFFSET))[...])
    # Rows within a tile are sorted by gene; gene_offset[g] = [start, end) gives the row range for
    # gene g, so the per-row gene index must be reconstructed rather than read directly: row r belongs to
    # the first gene whose range ends after r.
    gene_index = np.searchsorted(gene_offset[:, 1], np.arange(start, stop), side="right")
    feature_name = pd.Categorical.from_codes(gene_index, categories=pd.Index(feature_names))
    return _transcript_rows(tile, start, stop, feature_name)


def _transcripts_meta(categories: list[str]) -> pd.DataFrame:
    """An empty transcripts table with the column dtypes of `_transcript_rows`."""
    return pd.DataFrame(
        {
            str(AteraKeys.TRANSCRIPTS_X): pd.array([], dtype="float32"),
            str(AteraKeys.TRANSCRIPTS_Y): pd.array([], dtype="float32"),
            str(AteraKeys.TRANSCRIPTS_Z): pd.array([], dtype="float32"),
            str(AteraKeys.FEATURE_NAME): pd.Categorical([], categories=categories),
            str(AteraKeys.CELL_ID): pd.array([], dtype="int64"),
            "overlaps_nucleus": pd.array([], dtype="bool"),
            "quality_score": pd.array([], dtype="float32"),
        }
    )


def _get_points(path: Path, pixel_size: float, feature_names: list[str]) -> dd.DataFrame:
    """Lazily read transcripts.zarr.zip, one `dask` task per spatial tile."""
    # the store stays referenced by the graph; it reopens on first use, also after pickling
    grid = zarr.open_group(zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, read_only=True), mode="r")
    grid = _get_group(grid, str(AteraKeys.GRID_GROUP))
    tile_keys = sorted(grid.group_keys())
    tile_sizes = {
        key: _get_array(_get_group(grid, key), str(AteraKeys.TRANSCRIPTS_LOCATION)).shape[0] for key in tile_keys
    }

    # One partition per `chunk_rows`-row slice of a tile (not per whole tile), so peak memory is bounded by
    # `dask`'s number of concurrent tasks times the chunk size, however large a tile is.
    delayed_tiles = [
        delayed(_read_transcript_tile)(grid, key, feature_names, start, stop)
        for key in tile_keys
        for start, stop in _tile_row_ranges(tile_sizes[key])
    ]
    table = dd.from_delayed(delayed_tiles, meta=_transcripts_meta(feature_names))

    transform = Scale([1.0 / pixel_size, 1.0 / pixel_size], axes=("x", "y"))
    return PointsModel.parse(
        table,
        coordinates={
            "x": str(AteraKeys.TRANSCRIPTS_X),
            "y": str(AteraKeys.TRANSCRIPTS_Y),
            "z": str(AteraKeys.TRANSCRIPTS_Z),
        },
        feature_key=str(AteraKeys.FEATURE_NAME),
        instance_key=str(AteraKeys.CELL_ID),
        transformations={"global": transform},
        sort=False,
    )


def _get_gene_codewords(path: Path, genes: list[str]) -> dict[str, list[int]]:
    """Map each of ``genes`` to its codeword id(s) via ``panel_config.json``.

    Each gene target can have multiple codewords (e.g. for redundant probe coverage); a transcript's
    ``codeword_identity`` (an index into the shared, dataset-wide codebook) is only mapped to a gene
    through this file, not through anything in `transcripts.zarr.zip` itself.
    """
    with open(path / AteraKeys.PANEL_CONFIG_FILE) as f:
        panel_config = json.load(f)

    codewords_by_gene: dict[str, list[int]] = {}
    for target in panel_config["payloads"][0]["targets"]:
        if target["type"]["descriptor"] != "gene":
            continue
        name = target["type"]["data"]["name"]
        codewords_by_gene[name] = target["codewords"]

    missing = [gene for gene in genes if gene not in codewords_by_gene]
    if missing:
        raise ValueError(f"No codewords found for gene(s) {missing} in {AteraKeys.PANEL_CONFIG_FILE}.")
    return {gene: codewords_by_gene[gene] for gene in genes}


def read_transcripts_for_genes(
    sdata: SpatialData,
    path: str | Path,
    genes: str | list[str],
    points_key: str = "transcripts_subset",
    by_codeword: bool = False,
) -> SpatialData:
    """Add a `points` element to ``sdata`` with the transcripts of only the given gene(s).

    Unlike reading with ``transcripts=True``, which lazily decodes every tile of
    ``transcripts.zarr.zip`` in full once computed, this reads only the requested genes'
    contiguous row ranges out of each spatial tile: rows within a tile are sorted by gene, with
    ``gene_offset`` giving each gene's ``[start, end)`` range, and the tile arrays are chunked
    finely enough (e.g. ~11k-row chunks for tiles with tens of millions of rows) that slicing
    ``array[start:end]`` only reads the overlapping chunks rather than the whole tile. This makes
    plotting a handful of genes across the whole dataset much cheaper than loading and filtering
    the full transcripts table.

    Requires ``sdata`` to have been read with ``cells_table=True``, to look up gene indices (via
    the table's `var`) and `pixel_size` (via ``sdata.attrs``).

    Parameters
    ----------
    sdata
        A `SpatialData` object previously returned by `atera`.
    path
        Path to the dataset (the same path originally passed to the reader).
    genes
        One gene name, or a list of gene names, to read.
    points_key
        Key under which the resulting `points` element is stored in ``sdata.points``.
    by_codeword
        If `True`, label each transcript by its specific codeword (e.g. ``"ACTB_cw1006"``) instead
        of just its gene name, via the `feature_name` column. Since a gene is decoded from multiple
        redundant codewords, this lets you plot each codeword's spatial distribution separately
        (e.g. with ``groups=``/``palette=`` on `render_points`) to spot a codeword with an
        inconsistent pattern, e.g. due to a non-specific probe.

    Returns
    -------
    ``sdata``, with ``sdata.points[points_key]`` added (mutated in place, and also returned).
    """
    if isinstance(genes, str):
        genes = [genes]
    path = Path(path)
    feature_names = sdata.tables["table"].var[str(AteraKeys.FEATURE_NAME)].astype(str).tolist()
    gene_indices = {gene: feature_names.index(gene) for gene in genes}
    pixel_size = cast(float, sdata.attrs[str(AteraKeys.PIXEL_SIZE)])

    if by_codeword:
        gene_codewords = _get_gene_codewords(path, genes)
        categories = [f"{gene}_cw{cw}" for gene in genes for cw in gene_codewords[gene]]
    else:
        categories = genes

    frames = []
    with _open_zip(path / AteraKeys.TRANSCRIPTS_FILE) as group:
        grid = _get_group(group, str(AteraKeys.GRID_GROUP))
        for tile_key in sorted(grid.group_keys()):
            tile = _get_group(grid, tile_key)
            gene_offset = _get_array(tile, str(AteraKeys.TRANSCRIPTS_GENE_OFFSET))
            for gene, gene_idx in gene_indices.items():
                start, end = (int(v) for v in np.asarray(gene_offset[gene_idx]))
                if end <= start:
                    continue
                if by_codeword:
                    codeword = np.asarray(_get_array(tile, str(AteraKeys.CODEWORD_IDENTITY))[start:end]).reshape(-1)
                    labels = [f"{gene}_cw{cw}" for cw in codeword]
                else:
                    labels = [gene] * (end - start)
                frames.append(_transcript_rows(tile, start, end, pd.Categorical(labels, categories=categories)))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_transcripts_meta([]).columns))

    transform = Scale([1.0 / pixel_size, 1.0 / pixel_size], axes=("x", "y"))
    sdata.points[points_key] = PointsModel.parse(
        df,
        coordinates={
            "x": str(AteraKeys.TRANSCRIPTS_X),
            "y": str(AteraKeys.TRANSCRIPTS_Y),
            "z": str(AteraKeys.TRANSCRIPTS_Z),
        },
        feature_key=str(AteraKeys.FEATURE_NAME),
        instance_key=str(AteraKeys.CELL_ID),
        transformations={"global": transform},
    )
    return sdata


def _tile_bounds(tile_key: str, tile_size: float) -> tuple[float, float, float, float]:
    """Return the (xmin, ymin, xmax, ymax) raw-unit spatial extent of a ``"tx,ty"`` grid tile key."""
    tile_x, tile_y = (int(v) for v in tile_key.split(","))
    return tile_x * tile_size, tile_y * tile_size, (tile_x + 1) * tile_size, (tile_y + 1) * tile_size


def _bbox_overlaps(boxes: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Which of the ``(..., 4)`` (xmin, ymin, xmax, ymax) `boxes` overlap `bbox`."""
    boxes = np.asarray(boxes)
    return (boxes[..., 0] < bbox[2]) & (bbox[0] < boxes[..., 2]) & (boxes[..., 1] < bbox[3]) & (bbox[1] < boxes[..., 3])


def _cell_row_positions_in_bbox(path: Path, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Row positions (0-based, matching table/`obs` row order) of cells overlapping `bbox`.

    `bbox` is in the same "global" (pixel) coordinate system as the rest of a reader-produced
    `SpatialData`; it is converted to raw/micron units via `pixel_size` before comparing against
    ``cells.zarr.zip``'s top-level `bboxes`, which shares row order with the table/`obs` (the same
    fact `read_cell_boundaries` relies on for its own `bbox` filtering).
    """
    pixel_size = _get_pixel_size(path)
    with _open_zip(path / AteraKeys.CELLS_FILE) as group:
        cell_bboxes = np.asarray(_get_array(group, str(AteraKeys.BBOXES))[...]).reshape(-1, 4)
    return np.nonzero(
        _bbox_overlaps(
            cell_bboxes, (bbox[0] * pixel_size, bbox[1] * pixel_size, bbox[2] * pixel_size, bbox[3] * pixel_size)
        )
    )[0]


def read_cell_boundaries(
    sdata: SpatialData,
    path: str | Path,
    shapes_key: str = "cell_boundaries",
    mask_index: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
    pyramid_level: int = 0,
) -> SpatialData:
    """Add a `shapes` element to ``sdata`` with boundary polygons read from ``gridded_polygon_sets``.

    Unlike reading with ``cells_boundaries=True``/``nucleus_boundaries=True``, which decode every
    cell's full-detail polygon eagerly from ``polygon_sets``, ``cells.zarr.zip`` also has a
    ``gridded_polygon_sets``: the same boundary polygons, but spatially tiled and available at
    multiple levels of vertex detail (a pyramid), analogous to an image pyramid. This lets you
    trade detail for speed/memory in two independent ways:

    - ``pyramid_level`` (`0` = finest, matching `polygon_sets`, up to `4` = coarsest): higher
      levels store the same cells with far fewer vertices per polygon (e.g. ~24 vertices/cell at
      level 0 vs. ~5 at level 4, in a typical bundle). Every level has every cell -- this does not
      subset cells, only simplifies their polygons, so it mainly helps rendering speed/memory, not
      table size.
    - ``bbox``: an ``(xmin, ymin, xmax, ymax)`` region, in the same "global" coordinate system as
      the rest of ``sdata`` (i.e. the same units as the images/other shapes), to load. Only the
      spatial tiles overlapping this region are read (mirroring the tile-based approach already
      used for transcripts in `read_transcripts_for_genes`), and cells are then filtered to those
      whose bounding box overlaps it. If `None`, every tile at ``pyramid_level`` is read (all
      cells).

    Vertices are stored quantized to a `uint8` range, relative to each cell's own bounding box (in
    the top-level ``bboxes`` array); this un-quantizes them via
    ``absolute = cell_min + relative / 255 * (cell_max - cell_min)``.

    Requires ``sdata`` to have been read with ``cells_table=True``, to map row positions in
    ``gridded_polygon_sets`` to cell ids (via the table's `obs`), and `pixel_size` (via
    ``sdata.attrs``).

    Parameters
    ----------
    sdata
        A `SpatialData` object previously returned by `atera`.
    path
        Path to the dataset (the same path originally passed to the reader).
    shapes_key
        Key under which the resulting `shapes` element is stored in ``sdata.shapes``.
    mask_index
        `0` for nucleus boundaries, `1` for cell boundaries (matching the reader's
        ``nucleus_boundaries``/``cells_boundaries``).
    bbox
        ``(xmin, ymin, xmax, ymax)``, in the same coordinate system as the rest of ``sdata``. If
        `None`, all cells at ``pyramid_level`` are read.
    pyramid_level
        Which polygon-detail pyramid level to read, from `0` (finest, full detail) to `4`
        (coarsest, most simplified).

    Returns
    -------
    ``sdata``, with ``sdata.shapes[shapes_key]`` added (mutated in place, and also returned).
    """
    path = Path(path)
    pixel_size = cast(float, sdata.attrs[str(AteraKeys.PIXEL_SIZE)])
    cell_ids = sdata.tables["table"].obs[str(AteraKeys.CELL_ID)].to_numpy()
    is_nucleus = mask_index == 0

    raw_bbox = None
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        raw_bbox = (xmin * pixel_size, ymin * pixel_size, xmax * pixel_size, ymax * pixel_size)

    with _open_zip(path / AteraKeys.CELLS_FILE) as group:
        cell_bboxes = np.asarray(_get_array(group, str(AteraKeys.BBOXES))[...]).reshape(-1, 4)
        mask_group = _get_group(group, f"{AteraKeys.GRIDDED_POLYGON_SETS_GROUP}/{mask_index}")
        # Each pyramid level merges 2x2 blocks of tiles from the level below, so a tile's spatial
        # extent doubles per level; `grid_size` (attrs, shared across levels) gives level 0's tile
        # size.
        grid_size = cast("list[float]", mask_group.attrs[str(AteraKeys.GRID_SIZE)])
        tile_size = float(grid_size[0]) * (2**pyramid_level)
        level_group = _get_group(mask_group, str(pyramid_level))

        tile_keys = list(level_group.group_keys())
        if raw_bbox is not None:
            tile_keys = [key for key in tile_keys if _bbox_overlaps(_tile_bounds(key, tile_size), raw_bbox)]

        coords_parts = []
        num_vertices_parts = []
        cell_index_parts = []
        for tile_key in tile_keys:
            tile = _get_group(level_group, tile_key)
            num_vertices = np.asarray(_get_array(tile, str(AteraKeys.POLYGON_NUM_VERTICES))[...])
            cell_index = np.asarray(_get_array(tile, str(AteraKeys.POLYGON_CELL_INDEX))[...])
            relative_vertices = np.asarray(_get_array(tile, str(AteraKeys.RELATIVE_VERTICES))[...]).reshape(-1, 2)

            if raw_bbox is not None:
                keep = _bbox_overlaps(cell_bboxes[cell_index], raw_bbox)
                if not keep.any():
                    continue
                relative_vertices = relative_vertices[np.repeat(keep, num_vertices)]
                num_vertices = num_vertices[keep]
                cell_index = cell_index[keep]

            vertex_cell_index = np.repeat(cell_index, num_vertices)
            vertex_bboxes = cell_bboxes[vertex_cell_index]
            vertex_mins = vertex_bboxes[:, [0, 1]]
            vertex_spans = vertex_bboxes[:, [2, 3]] - vertex_mins
            coords_parts.append((vertex_mins + relative_vertices / 255 * vertex_spans).astype(np.float32))
            num_vertices_parts.append(num_vertices)
            cell_index_parts.append(cell_index)

    coords = np.concatenate(coords_parts) if coords_parts else np.empty((0, 2), dtype=np.float32)
    num_vertices = np.concatenate(num_vertices_parts) if num_vertices_parts else np.empty((0,), dtype=np.int32)
    cell_index = np.concatenate(cell_index_parts) if cell_index_parts else np.empty((0,), dtype=np.uint32)

    sdata.shapes[shapes_key] = _polygons_to_shapes(coords, num_vertices, cell_index, cell_ids, pixel_size, is_nucleus)
    return sdata


def _parse_morphology_channel_name(filename: str) -> str:
    """Parse the channel name out of a ``chNNNN_<name>.ome.tif`` morphology image filename."""
    match = re.match(r"ch\d+_(.+)\.ome\.tif+$", filename)
    if match is None:
        raise ValueError(f"Expected a morphology image filename of the form 'chNNNN_<name>.ome.tif', found {filename}")
    return match.group(1)


def _parse_morphology_channel_index(filename: str) -> int:
    """Parse the channel index out of a ``chNNNN_<name>.ome.tif`` morphology image filename."""
    match = re.match(r"ch(\d+)_.+\.ome\.tif+$", filename)
    if match is None:
        raise ValueError(f"Expected a morphology image filename of the form 'chNNNN_<name>.ome.tif', found {filename}")
    return int(match.group(1))


def _get_pixel_size(path: Path) -> float:
    """Return the dataset's pixel size (microns/pixel).

    Prefers ``experiment.spatial`` when present, but some bundles omit it; in that case, fall back
    to the ``PhysicalSizeX`` embedded in the OME-XML metadata of the first ``morphology_2d``
    OME-TIFF (every morphology image in a bundle shares the same pixel size).
    """
    specs_file = path / AteraKeys.SPECS_FILE
    if specs_file.exists():
        with open(specs_file) as f:
            specs = json.load(f)
        return specs[str(AteraKeys.PIXEL_SIZE)]

    morphology_dir = path / AteraKeys.MORPHOLOGY_2D_DIR
    files = (
        sorted(f for f in morphology_dir.iterdir() if f.name.endswith(".ome.tif") and not f.name.startswith("._"))
        if morphology_dir.is_dir()
        else []
    )
    if not files:
        raise FileNotFoundError(
            f"Cannot determine the pixel size: {path!s} contains neither {AteraKeys.SPECS_FILE!s} nor any "
            f"morphology OME-TIFF files in {morphology_dir.name!s}/."
        )
    return parse_physical_size(files[0])


def _tiled_imread(
    path: Path, imread_kwargs: Mapping[str, Any] = MappingProxyType({}), n_levels: int = 1
) -> list[da.Array]:
    """Read an OME-TIFF as `dask` arrays (one per pyramid level) chunked by the file's own on-disk tile grid.

    `dask_image.imread.imread` gives each page (e.g. each channel or z-plane) exactly one `dask`
    chunk -- the whole plane, which for these whole-slide-image morphology files is tens of GB.
    That single giant chunk then has to be fully decoded before `Image2DModel.parse`/`Image3DModel.parse`
    can rechunk it into tiles or build a downsampled pyramid level, so even plotting one channel at
    the lowest pyramid resolution requires materializing the entire full-resolution plane in memory.

    This instead reads through `tifffile.TiffPageSeries.aszarr(level=i)`, which exposes the file's
    native tile grid (e.g. 1024x1024, matching `p.is_tiled`/`p.tilewidth`/`p.tilelength`) as `zarr`
    chunks, so only the tiles actually touched downstream get decoded.

    These OME-TIFFs embed their own multi-level pyramid (each level half the size of the previous one).
    Reading up to `n_levels` of those stored levels, rather than downsampling `level=0` lazily, means
    a low-resolution level only costs reading that (small) level from disk, not every full-resolution
    tile. If the file has fewer levels than requested, only the available ones are returned.
    """
    import tifffile

    tf = tifffile.TiffFile(path, **imread_kwargs)
    series = tf.series[0]
    n_levels = min(n_levels, len(series.levels))
    return [da.from_zarr(zarr.open(series.aszarr(level=i), mode="r")) for i in range(n_levels)]


def _parse_morphology_pyramid(
    levels: list[da.Array],
    dims: tuple[str, ...],
    c_coords: list[str],
    image_models_kwargs: Mapping[str, Any],
    model: type[Image2DModel] | type[Image3DModel],
) -> DataArray | DataTree:
    """Build a (multiscale) image model from pre-computed pyramid levels instead of downsampling `level=0`."""
    from xarray import Dataset

    kwargs = {k: v for k, v in image_models_kwargs.items() if k != "scale_factors"}
    parsed = [model.parse(level, dims=dims, c_coords=c_coords, **kwargs) for level in levels]
    if len(parsed) == 1:
        tree: DataArray | DataTree = parsed[0]
    else:
        tree = DataTree.from_dict({f"scale{i}": Dataset({"image": p}) for i, p in enumerate(parsed)})
    set_transformation(tree, Identity(), "global")
    model.validate(tree)
    return tree


def _get_morphology_images(
    dir_path: Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    three_d: bool = False,
) -> DataArray | DataTree:
    """Read the morphology OME-TIFF files in a directory as a single multi-channel image."""
    files = sorted(f for f in dir_path.iterdir() if f.name.endswith(".ome.tif") and not f.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No morphology OME-TIFF files found in {dir_path}.")
    # `scale_factors=[2, 2, 2, 2]` (the default) means 5 levels in total; use the file's stored levels for them.
    n_levels = len(image_models_kwargs.get("scale_factors") or []) + 1

    if three_d:
        # one file per channel, each file has shape (z, y, x); stack along a new channel axis
        channel_names = [_parse_morphology_channel_name(f.name) for f in files]
        per_file_levels = [_tiled_imread(f, imread_kwargs, n_levels) for f in files]
        n = min(len(lv) for lv in per_file_levels)
        levels = [da.stack([lv[i] for lv in per_file_levels], axis=0) for i in range(n)]
        return _parse_morphology_pyramid(levels, ("c", "z", "y", "x"), channel_names, image_models_kwargs, Image3DModel)

    # Each 2D file is itself a multi-page OME-TIFF containing every channel of the panel (shape
    # (n_channels, y, x)), but only the plane at the index given by the file's own "chNNNN" prefix
    # holds that channel's real image data; take just that plane from each file.
    from ome_types import from_tiff

    channel_indices = [_parse_morphology_channel_index(f.name) for f in files]
    channel_names = []
    for f, i in zip(files, channel_indices, strict=True):
        name = from_tiff(f).images[0].pixels.channels[i].name
        if name is None:
            raise ValueError(f"Channel {i} in {f!s} has no name in its OME-XML metadata.")
        channel_names.append(name)
    per_file_levels = [
        [level[i] for level in _tiled_imread(f, imread_kwargs, n_levels)]
        for f, i in zip(files, channel_indices, strict=True)
    ]
    n = min(len(lv) for lv in per_file_levels)
    levels = [da.stack([lv[j] for lv in per_file_levels], axis=0) for j in range(n)]
    return _parse_morphology_pyramid(levels, ("c", "y", "x"), channel_names, image_models_kwargs, Image2DModel)
