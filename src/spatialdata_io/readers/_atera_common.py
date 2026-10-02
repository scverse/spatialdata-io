from __future__ import annotations

import asyncio
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
    Labels2DModel,
    PointsModel,
    ShapesModel,
)
from spatialdata.transformations.transformations import Identity, Scale

from spatialdata_io._constants._constants import AteraKeys

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from xarray import DataArray, DataTree

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

_UINT32_SENTINEL = np.iinfo(np.uint32).max


@contextlib.contextmanager
def _patched_ragged_vlen_chunk_decode() -> Iterator[None]:
    """Work around ``zarr``'s `V2Codec` failing to decode a vlen (string) array's ragged trailing chunk.

    `obs`/`var`'s string columns (e.g. `barcode`) are stored as 1-D zarr v2 arrays with a fixed
    chunk size chosen independently of the row count (e.g. `1_000_000`), so the trailing chunk is
    usually smaller than that nominal size. For fixed-width dtypes this is harmless: `zarr` pads an
    incomplete edge chunk to the full chunk shape (with the fill value) before compression, so it
    always decodes back to the full nominal size. Variable-length (`vlen-utf8`/`StringDType`)
    chunks aren't padded this way -- only the elements actually written are stored -- but
    `V2Codec._decode_single` unconditionally reshapes the decoded chunk to the nominal chunk shape
    regardless of dtype, so reading across that trailing chunk raises e.g. ``ValueError: cannot
    reshape array of size 376951 into shape (1000000,)``. Reproduced against `zarr` 3.1.6 and
    3.4.0 (the latest release as of writing); no matching issue found upstream yet.

    This patches `V2Codec._decode_single` (the abstract-method entry point present on every
    `zarr` 3.x release, unlike the private `_decode_sync`/`_decode_single` split introduced partway
    through the 3.x series) for the duration of the context, falling back to the chunk's actual
    (smaller) size instead of raising, only for the 1-D case this affects; any other reshape failure
    (e.g. a genuinely corrupt chunk, or a ragged chunk in more than one dimension) is still raised.
    """
    try:
        from numcodecs.compat import ensure_ndarray_like
        from zarr.codecs._v2 import V2Codec
        from zarr.registry import get_ndbuffer_class
    except ImportError:
        # `zarr`'s internal layout changed enough that this patch no longer applies; read without
        # it rather than fail outright (bundles without a ragged vlen chunk are unaffected either way).
        yield
        return

    original_decode_single = V2Codec._decode_single

    async def patched_decode_single(self: V2Codec, chunk_bytes: Any, chunk_spec: Any) -> Any:
        def decode() -> Any:
            cdata = chunk_bytes.as_array_like()
            chunk = self.compressor.decode(cdata) if self.compressor else cdata
            if self.filters:
                for f in reversed(self.filters):
                    chunk = f.decode(chunk)
            chunk = ensure_ndarray_like(chunk)
            if chunk_spec.dtype.dtype_cls is not np.dtypes.ObjectDType:
                try:
                    chunk = chunk.view(chunk_spec.dtype.to_native_dtype())
                except TypeError:
                    chunk = np.array(chunk).astype(chunk_spec.dtype.to_native_dtype())
            elif chunk.dtype != object:
                raise RuntimeError("cannot read object array without object codec")

            chunk = chunk.reshape(-1, order="A")
            nominal_size = int(np.prod(chunk_spec.shape))
            if chunk.size == nominal_size:
                chunk = chunk.reshape(chunk_spec.shape, order=chunk_spec.order)
            elif len(chunk_spec.shape) != 1:
                raise ValueError(f"cannot reshape array of size {chunk.size} into shape {chunk_spec.shape}")
            # else: ragged trailing chunk of a 1-D vlen array -- keep its natural, smaller size.
            return get_ndbuffer_class().from_ndarray_like(chunk)

        return await asyncio.to_thread(decode)

    V2Codec._decode_single = patched_decode_single  # type: ignore[method-assign]
    try:
        yield
    finally:
        V2Codec._decode_single = original_decode_single  # type: ignore[method-assign]


def _decode_packed_cell_id(raw: np.ndarray) -> np.ndarray:
    """Decode a (N, 2) uint32 packed cell id array into the int64 ids used in the cell-feature-matrix obs.

    The two uint32 columns pack into an int64 as ``low + (high << 32)``. The sentinel
    ``(uint32 max, uint32 max)`` marks "no cell assigned" and is decoded to ``-1``.
    """
    low = raw[:, 0].astype(np.int64)
    high = raw[:, 1].astype(np.int64)
    packed = low + (high << 32)
    unassigned = (raw[:, 0] == _UINT32_SENTINEL) & (raw[:, 1] == _UINT32_SENTINEL)
    packed[unassigned] = -1
    return packed


def _get_labels(
    cells_group: zarr.Group,
    mask_index: int,
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> DataArray | DataTree:
    """Read the labels raster from cells.zarr.zip masks/{mask_index} (0 = nucleus, 1 = cell)."""
    masks = da.from_array(cells_group.get_array(f"{AteraKeys.MASKS_GROUP}/{mask_index}"))
    return Labels2DModel.parse(masks, dims=("y", "x"), transformations={"global": Identity()}, **labels_models_kwargs)


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
    group = cells_group.get_group(f"{AteraKeys.POLYGON_SETS_GROUP}/{mask_index}")
    vertices = np.asarray(group.get_array(str(AteraKeys.POLYGON_VERTICES))[...])
    num_vertices = np.asarray(group.get_array(str(AteraKeys.POLYGON_NUM_VERTICES))[...])
    cell_index = np.asarray(group.get_array(str(AteraKeys.POLYGON_CELL_INDEX))[...])

    n_polygons, max_coords = vertices.shape
    max_vertices = max_coords // 2
    vertices = vertices.reshape(n_polygons, max_vertices, 2)
    valid = np.arange(max_vertices)[None, :] < num_vertices[:, None]
    coords = vertices[valid]

    return _polygons_to_shapes(coords, num_vertices, cell_index, cell_ids, pixel_size, is_nucleus)


def _read_transcript_tile(path: Path, tile_key: str, feature_names: list[str]) -> pd.DataFrame:
    """Read and decode a single spatial tile of transcripts.zarr.zip.

    Opens its own zip store (rather than sharing one across tiles) so this function can safely
    be called from independent `dask` tasks.
    """
    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        tile = group.get_group(f"{AteraKeys.GRID_GROUP}/{tile_key}")
        location = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_LOCATION))[...])
        gene_offset = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_GENE_OFFSET))[...])
        cell_id_raw = np.asarray(tile.get_array(str(AteraKeys.CELL_ID))[...])
        overlaps_nucleus = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_OVERLAPS_NUCLEUS))[...]).reshape(-1)
        quality_score = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_QUALITY_SCORE))[...]).reshape(-1)
    finally:
        store.close()

    # Rows within a tile are sorted by gene; gene_offset[g] = [start, end) gives the row range for
    # gene g, so the per-row gene index must be reconstructed rather than read directly.
    gene_index = np.repeat(np.arange(len(gene_offset)), gene_offset[:, 1] - gene_offset[:, 0])
    feature_name = pd.Categorical.from_codes(gene_index, categories=pd.Index(feature_names))

    return pd.DataFrame(
        {
            str(AteraKeys.TRANSCRIPTS_X): location[:, 0].astype(np.float32),
            str(AteraKeys.TRANSCRIPTS_Y): location[:, 1].astype(np.float32),
            str(AteraKeys.TRANSCRIPTS_Z): location[:, 2].astype(np.float32),
            str(AteraKeys.FEATURE_NAME): feature_name,
            str(AteraKeys.CELL_ID): _decode_packed_cell_id(cell_id_raw),
            "overlaps_nucleus": overlaps_nucleus.astype(bool),
            "quality_score": quality_score.astype(np.float32),
        }
    )


def _get_points(path: Path, pixel_size: float, feature_names: list[str]) -> dd.DataFrame:
    """Lazily read transcripts.zarr.zip, one `dask` task per spatial tile."""
    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        tile_keys = sorted(group.get_group(str(AteraKeys.GRID_GROUP)).group_keys())
    finally:
        store.close()

    cat_dtype = pd.CategoricalDtype(categories=feature_names)
    meta = pd.DataFrame(
        {
            str(AteraKeys.TRANSCRIPTS_X): pd.array([], dtype="float32"),
            str(AteraKeys.TRANSCRIPTS_Y): pd.array([], dtype="float32"),
            str(AteraKeys.TRANSCRIPTS_Z): pd.array([], dtype="float32"),
            str(AteraKeys.FEATURE_NAME): pd.array([], dtype=cat_dtype),
            str(AteraKeys.CELL_ID): pd.array([], dtype="int64"),
            "overlaps_nucleus": pd.array([], dtype="bool"),
            "quality_score": pd.array([], dtype="float32"),
        }
    )
    delayed_tiles = [delayed(_read_transcript_tile)(path, key, feature_names) for key in tile_keys]
    table = dd.from_delayed(delayed_tiles, meta=meta)

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
    pixel_size = sdata.attrs[str(AteraKeys.PIXEL_SIZE)]

    if by_codeword:
        gene_codewords = _get_gene_codewords(path, genes)
        categories = [f"{gene}_cw{cw}" for gene in genes for cw in gene_codewords[gene]]
    else:
        categories = genes

    columns = [
        str(AteraKeys.TRANSCRIPTS_X),
        str(AteraKeys.TRANSCRIPTS_Y),
        str(AteraKeys.TRANSCRIPTS_Z),
        str(AteraKeys.FEATURE_NAME),
        str(AteraKeys.CELL_ID),
        "overlaps_nucleus",
        "quality_score",
    ]
    store = zarr.storage.ZipStore(path / AteraKeys.TRANSCRIPTS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        tile_keys = sorted(group.get_group(str(AteraKeys.GRID_GROUP)).group_keys())

        frames = []
        for tile_key in tile_keys:
            tile = group.get_group(f"{AteraKeys.GRID_GROUP}/{tile_key}")
            gene_offset = tile.get_array(str(AteraKeys.TRANSCRIPTS_GENE_OFFSET))
            for gene, gene_idx in gene_indices.items():
                start, end = np.asarray(gene_offset[gene_idx])
                if end <= start:
                    continue
                location = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_LOCATION))[start:end])
                cell_id_raw = np.asarray(tile.get_array(str(AteraKeys.CELL_ID))[start:end])
                overlaps_nucleus = np.asarray(
                    tile.get_array(str(AteraKeys.TRANSCRIPTS_OVERLAPS_NUCLEUS))[start:end]
                ).reshape(-1)
                quality_score = np.asarray(tile.get_array(str(AteraKeys.TRANSCRIPTS_QUALITY_SCORE))[start:end]).reshape(
                    -1
                )
                if by_codeword:
                    codeword_identity = np.asarray(tile.get_array(str(AteraKeys.CODEWORD_IDENTITY))[start:end]).reshape(
                        -1
                    )
                    feature_name = pd.Categorical([f"{gene}_cw{cw}" for cw in codeword_identity], categories=categories)
                else:
                    feature_name = pd.Categorical([gene] * (end - start), categories=categories)
                frames.append(
                    pd.DataFrame(
                        {
                            str(AteraKeys.TRANSCRIPTS_X): location[:, 0].astype(np.float32),
                            str(AteraKeys.TRANSCRIPTS_Y): location[:, 1].astype(np.float32),
                            str(AteraKeys.TRANSCRIPTS_Z): location[:, 2].astype(np.float32),
                            str(AteraKeys.FEATURE_NAME): feature_name,
                            str(AteraKeys.CELL_ID): _decode_packed_cell_id(cell_id_raw),
                            "overlaps_nucleus": overlaps_nucleus.astype(bool),
                            "quality_score": quality_score.astype(np.float32),
                        }
                    )
                )
    finally:
        store.close()

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

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


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _cell_row_positions_in_bbox(path: Path, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Row positions (0-based, matching table/`obs` row order) of cells overlapping `bbox`.

    `bbox` is in the same "global" (pixel) coordinate system as the rest of a reader-produced
    `SpatialData`; it is converted to raw/micron units via `pixel_size` before comparing against
    ``cells.zarr.zip``'s top-level `bboxes`, which shares row order with the table/`obs` (the same
    fact `read_cell_boundaries` relies on for its own `bbox` filtering).
    """
    pixel_size = _get_pixel_size(path)
    xmin, ymin, xmax, ymax = (v * pixel_size for v in bbox)

    store = zarr.storage.ZipStore(path / AteraKeys.CELLS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        cell_bboxes = np.asarray(group.get_array(str(AteraKeys.BBOXES))[...]).reshape(-1, 4)
    finally:
        store.close()

    keep = (
        (cell_bboxes[:, 0] < xmax)
        & (xmin < cell_bboxes[:, 2])
        & (cell_bboxes[:, 1] < ymax)
        & (ymin < cell_bboxes[:, 3])
    )
    return np.nonzero(keep)[0]


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
    pixel_size = sdata.attrs[str(AteraKeys.PIXEL_SIZE)]
    cell_ids = sdata.tables["table"].obs[str(AteraKeys.CELL_ID)].to_numpy()
    is_nucleus = mask_index == 0

    raw_bbox = None
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        raw_bbox = (xmin * pixel_size, ymin * pixel_size, xmax * pixel_size, ymax * pixel_size)

    store = zarr.storage.ZipStore(path / AteraKeys.CELLS_FILE, read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        cell_bboxes = np.asarray(group.get_array(str(AteraKeys.BBOXES))[...]).reshape(-1, 4)
        mask_group = group.get_group(f"{AteraKeys.GRIDDED_POLYGON_SETS_GROUP}/{mask_index}")
        # Each pyramid level merges 2x2 blocks of tiles from the level below, so a tile's spatial
        # extent doubles per level; `grid_size` (attrs, shared across levels) gives level 0's tile
        # size.
        grid_size = cast("list[float]", mask_group.attrs[str(AteraKeys.GRID_SIZE)])
        tile_size = float(grid_size[0]) * (2**pyramid_level)
        level_group = mask_group.get_group(str(pyramid_level))

        tile_keys = list(level_group.group_keys())
        if raw_bbox is not None:
            tile_keys = [key for key in tile_keys if _bbox_overlaps(_tile_bounds(key, tile_size), raw_bbox)]

        coords_parts = []
        num_vertices_parts = []
        cell_index_parts = []
        for tile_key in tile_keys:
            tile = level_group.get_group(tile_key)
            num_vertices = np.asarray(tile.get_array(str(AteraKeys.POLYGON_NUM_VERTICES))[...])
            cell_index = np.asarray(tile.get_array(str(AteraKeys.POLYGON_CELL_INDEX))[...])
            relative_vertices = np.asarray(tile.get_array(str(AteraKeys.RELATIVE_VERTICES))[...]).reshape(-1, 2)

            if raw_bbox is not None:
                tile_cell_bboxes = cell_bboxes[cell_index]
                keep = (
                    (tile_cell_bboxes[:, 0] < raw_bbox[2])
                    & (raw_bbox[0] < tile_cell_bboxes[:, 2])
                    & (tile_cell_bboxes[:, 1] < raw_bbox[3])
                    & (raw_bbox[1] < tile_cell_bboxes[:, 3])
                )
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
    finally:
        store.close()

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

    from ome_types import from_tiff

    morphology_dir = path / AteraKeys.MORPHOLOGY_2D_DIR
    files = (
        sorted(f for f in morphology_dir.iterdir() if f.name.endswith(".ome.tif") and not f.name.startswith("._"))
        if morphology_dir.is_dir()
        else []
    )
    if not files:
        raise FileNotFoundError(
            f"Found neither {AteraKeys.SPECS_FILE!s} nor any morphology OME-TIFF files in {morphology_dir!s} "
            "to read the pixel size from."
        )
    physical_size_x = from_tiff(files[0]).images[0].pixels.physical_size_x
    if physical_size_x is None:
        raise ValueError(f"{files[0]!s} does not specify `PhysicalSizeX` in its OME-XML metadata.")
    return physical_size_x


def _tiled_imread(path: Path, imread_kwargs: Mapping[str, Any] = MappingProxyType({})) -> da.Array:
    """Read an OME-TIFF as a `dask` array chunked by the file's own on-disk tile grid.

    `dask_image.imread.imread` gives each page (e.g. each channel or z-plane) exactly one `dask`
    chunk -- the whole plane, which for these whole-slide-image morphology files is tens of GB.
    That single giant chunk then has to be fully decoded before `Image2DModel.parse`/`Image3DModel.parse`
    can rechunk it into tiles or build a downsampled pyramid level, so even plotting one channel at
    the lowest pyramid resolution requires materializing the entire full-resolution plane in memory.

    This instead reads through `tifffile.TiffPageSeries.aszarr(level=0)`, which exposes the file's
    native tile grid (e.g. 1024x1024, matching `p.is_tiled`/`p.tilewidth`/`p.tilelength`) as `zarr`
    chunks, so only the tiles actually touched downstream get decoded.

    `level=0` always reads the file's full-resolution series: these OME-TIFFs happen to embed their
    own multi-level pyramid already, but `atera`'s morphology images build their own multiscale
    pyramid via `Image2DModel.parse`'s `scale_factors`, independent of it.
    """
    import tifffile

    tf = tifffile.TiffFile(path, **imread_kwargs)
    store = tf.series[0].aszarr(level=0)
    return da.from_zarr(zarr.open(store, mode="r"))


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

    if three_d:
        # one file per channel, each file has shape (z, y, x); stack along a new channel axis
        channel_names = [_parse_morphology_channel_name(f.name) for f in files]
        channels = [_tiled_imread(f, imread_kwargs) for f in files]
        image = da.stack(channels, axis=0)
        return Image3DModel.parse(
            image,
            dims=("c", "z", "y", "x"),
            c_coords=channel_names,
            transformations={"global": Identity()},
            **image_models_kwargs,
        )

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
    channels = [_tiled_imread(f, imread_kwargs)[i] for f, i in zip(files, channel_indices, strict=True)]
    image = da.stack(channels, axis=0)
    return Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        c_coords=channel_names,
        transformations={"global": Identity()},
        **image_models_kwargs,
    )
