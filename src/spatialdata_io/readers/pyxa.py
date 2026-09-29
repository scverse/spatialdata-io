from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
import zarr
from joblib.externals.loky import get_reusable_executor
from scipy import sparse
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image3DModel, Labels3DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Scale, Sequence, Translation, set_transformation
from xarray import DataArray, Dataset, DataTree

from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io._docs import inject_docs

__all__ = ["pyxa"]


def _validate_columns(df: pd.DataFrame | dd.DataFrame, required: set[str], file_name: str) -> None:
    """Raise a clear ``ValueError`` naming the file and any missing required column(s)."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {sorted(missing)}")


def _get_points(path: Path) -> dd.DataFrame:
    ddf = dd.read_csv(path, dtype={PyxaKeys.CELL_ID.value: str})
    _validate_columns(
        ddf,
        {
            PyxaKeys.CELL_ID.value,
            PyxaKeys.GENE.value,
            PyxaKeys.X_UM.value,
            PyxaKeys.Y_UM.value,
            PyxaKeys.Z_UM.value,
        },
        path.name,
    )
    ddf[PyxaKeys.ASSIGNED.value] = ~ddf[PyxaKeys.CELL_ID.value].str.endswith(PyxaKeys.UNASSIGNED_SUFFIX.value)
    # PointsModel needs known feature categories; computing them here is one pass over the gene column
    ddf[PyxaKeys.GENE.value] = ddf[PyxaKeys.GENE.value].astype("category").cat.as_known()
    return ddf


def _read_cells(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=PyxaKeys.CELL_ID.value, dtype={PyxaKeys.CELL_ID.value: str})


def _cluster_categorical(values: pd.Series) -> pd.Categorical[str]:
    """Cluster labels as string categories, in numeric order when the labels are integers."""
    present = values.dropna()
    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().all():
        labels = numeric.astype(int).astype(str)
        categories = [str(c) for c in sorted(numeric.astype(int).unique())]
    else:
        labels = present.astype(str)
        categories = sorted(labels.unique())
    return pd.Categorical(labels.reindex(values.index), categories=categories)


def _get_table(
    cell_by_gene_path: Path,
    cell_metadata_path: Path,
    pyxa_studio_path: Path | None = None,
) -> ad.AnnData:
    """Build the cell table from the counts and metadata, optionally with Pyxa Studio's clusters.

    ``pyxa_studio`` lists only the cells that passed Pyxa's filters, so its ``Cluster``
    (categorical) and ``obsm["X_umap"]`` are missing (NaN) for the other cells.
    """
    by_gene = _read_cells(cell_by_gene_path)
    metadata = _read_cells(cell_metadata_path)

    spatial_cols = [PyxaKeys.X_UM.value, PyxaKeys.Y_UM.value, PyxaKeys.Z_UM.value]
    _validate_columns(metadata, set(spatial_cols), cell_metadata_path.name)

    metadata = metadata.loc[by_gene.index]
    # Pyxa counts are mostly zeros: a full Region's dense float64 table is several GB
    adata = ad.AnnData(
        sparse.csr_matrix(by_gene.to_numpy()),
        obs=metadata.drop(columns=spatial_cols),
        var=pd.DataFrame(index=by_gene.columns),
    )
    adata.obsm["spatial"] = metadata[spatial_cols].values

    if pyxa_studio_path is not None:
        studio = _read_cells(pyxa_studio_path)
        _validate_columns(studio, {PyxaKeys.CLUSTER.value}, pyxa_studio_path.name)
        n_missing = int((~adata.obs_names.isin(studio.index)).sum())
        if n_missing:
            logger.info(
                f"{pyxa_studio_path.name}: {n_missing} of {adata.n_obs} cells were filtered out by Pyxa Studio; "
                f"their {PyxaKeys.CLUSTER.value!r} and {PyxaKeys.UMAP_KEY.value!r} are missing"
            )
        studio = studio.reindex(adata.obs_names)
        adata.obs[PyxaKeys.CLUSTER.value] = _cluster_categorical(studio[PyxaKeys.CLUSTER.value])
        umap_cols = [c for c in (PyxaKeys.X_UMAP.value, PyxaKeys.Y_UMAP.value, PyxaKeys.Z_UMAP.value) if c in studio]
        if umap_cols:
            adata.obsm[PyxaKeys.UMAP_KEY.value] = studio[umap_cols].to_numpy(dtype=np.float64)

    adata.obs[PyxaKeys.REGION_KEY.value] = pd.Series(PyxaKeys.REGION.value, index=adata.obs_names, dtype="category")
    adata.obs[PyxaKeys.CELL_ID.value] = adata.obs_names
    # the cell_id column carries the instance key; an index with the same name breaks table joins
    adata.obs.index.name = None
    return adata


def _get_voxel_size(cell_metadata_path: Path, n_rows: int = 10_000) -> tuple[float, float]:
    """Infer the (xy, z) voxel size in um of the segmentation polygons from the per-cell metadata.

    Polygons are stored in pixel coordinates (xy) with a z-plane index (``ZIndex``), while points
    and the table use micrometers. The metadata carries each cell centroid in both units, related
    by a pure scale per axis (no offset), so each size is a least-squares fit through the origin.
    A pure scale is fully determined by a few cells, so only the first ``n_rows`` cells are read;
    the fit is then checked to reproduce every sampled cell, and an error is raised if it does not.
    """
    columns = [
        PyxaKeys.X_UM.value,
        PyxaKeys.Y_UM.value,
        PyxaKeys.Z_UM.value,
        PyxaKeys.X_PIXELS.value,
        PyxaKeys.Y_PIXELS.value,
        PyxaKeys.Z_PIXELS.value,
    ]
    metadata = pd.read_csv(cell_metadata_path, usecols=lambda c: c in columns, nrows=n_rows)
    _validate_columns(metadata, set(columns), cell_metadata_path.name)

    def fit(um_cols: list[str], px_cols: list[str]) -> float:
        um = metadata[um_cols].to_numpy().ravel()
        px = metadata[px_cols].to_numpy().ravel()
        size = float(np.dot(um, px) / np.dot(px, px))
        residual = np.abs(um - size * px).max()
        if residual > 1e-6 * max(np.abs(um).max(), 1.0):
            raise ValueError(
                f"{cell_metadata_path.name}: {um_cols} and {px_cols} are not related by a pure scale "
                f"(max residual {residual:.3g} um with a fitted size of {size:.6g} um/pixel)"
            )
        return size

    xy = fit([PyxaKeys.X_UM.value, PyxaKeys.Y_UM.value], [PyxaKeys.X_PIXELS.value, PyxaKeys.Y_PIXELS.value])
    z = fit([PyxaKeys.Z_UM.value], [PyxaKeys.Z_PIXELS.value])
    return xy, z


def _polygonal_part(geometry: shapely.Geometry) -> shapely.Geometry:
    """Keep only the (multi)polygonal part of a geometry, dropping any lines or points."""
    if isinstance(geometry, shapely.Polygon | shapely.MultiPolygon):
        return geometry
    polygons = [p for p in shapely.get_parts(geometry) if isinstance(p, shapely.Polygon)]
    return shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0] if polygons else shapely.Polygon()


def _make_polygonal_valid(geometries: np.ndarray) -> np.ndarray:
    """Repair invalid geometries in place of dropping them, keeping each one a (Multi)Polygon.

    Valid geometries are returned untouched. Invalid ones are repaired with
    :func:`shapely.make_valid` (``"structure"`` method), which rebuilds the polygon from its
    rings, and any zero-area parts produced by the repair (lines, points) are discarded.
    """
    geometries = geometries.copy()
    invalid = ~shapely.is_valid(geometries)
    if invalid.any():
        repaired = shapely.make_valid(geometries[invalid], method="structure", keep_collapsed=False)
        geometries[invalid] = [_polygonal_part(g) for g in repaired]
    return geometries


def _get_shapes(path: Path, xy_size: float, z_size: float) -> gpd.GeoDataFrame:
    """Read the per-cell, per-z-plane segmentation polygons and convert them to micrometers.

    xy coordinates are scaled from pixels by ``xy_size``. Since shapes are 2D in spatialdata, z is
    stored as a ``Z_um`` column: plane ``k`` spans ``[k, k + 1)`` in ``Z_pixels`` units, so its
    centre sits at ``(k + 0.5) * z_size``. Scaling can turn polygons that touch themselves at a
    single vertex into self-intersecting ones through floating point rounding, so the scaled
    geometries are repaired and then validated.
    """
    parquet_file = pq.ParquetFile(path)
    chunks = []
    for batch in parquet_file.iter_batches():
        chunk = batch.to_pandas()
        chunk["geometry"] = shapely.from_wkb(chunk["geometry"])
        chunks.append(gpd.GeoDataFrame(chunk, geometry="geometry"))
    gdf = pd.concat(chunks, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    _validate_columns(gdf, {PyxaKeys.CELL_ID.value, PyxaKeys.Z_INDEX.value}, path.name)

    gdf[PyxaKeys.CELL_ID.value] = gdf[PyxaKeys.CELL_ID.value].astype(str)
    gdf[PyxaKeys.Z_UM.value] = (gdf[PyxaKeys.Z_INDEX.value] + 0.5) * z_size

    scaled = shapely.transform(gdf.geometry.to_numpy(), lambda coords: coords * xy_size)
    n_invalid = int((~shapely.is_valid(scaled)).sum())
    fixed = _make_polygonal_valid(scaled)
    if n_invalid:
        area_change = np.abs(shapely.area(fixed) - shapely.area(scaled)) / np.maximum(shapely.area(scaled), 1e-12)
        logger.info(
            f"{path.name}: repaired {n_invalid} invalid polygon(s) after scaling to micrometers "
            f"(max relative area change {area_change.max():.2g})"
        )
    gdf = gdf.set_geometry(fixed)

    empty = gdf.geometry.is_empty.to_numpy()
    if empty.any():
        logger.warning(f"{path.name}: dropping {int(empty.sum())} polygon(s) with no area left after repair")
        gdf = gdf[~empty]
    if not gdf.geometry.is_valid.all() or not set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{path.name}: segmentation polygons are still invalid or non-polygonal after repair")

    return gdf.reset_index(drop=True)


def _get_footprints(planes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge each cell's z-plane polygons into one 2D footprint, indexed by cell id.

    A table can only annotate an element with one row per instance, so the per-plane polygons are
    kept as a separate element and the table annotates these footprints instead. The union is the
    expensive step and is independent per cell; shapely releases the GIL, so cells are merged in a
    thread pool.
    """
    cell_ids = planes[PyxaKeys.CELL_ID.value].to_numpy()
    order = np.argsort(cell_ids, kind="stable")
    cell_ids, geometries = cell_ids[order], planes.geometry.to_numpy()[order]
    unique_ids, starts = np.unique(cell_ids, return_index=True)
    stops = np.append(starts[1:], len(cell_ids))

    with ThreadPoolExecutor() as executor:
        unions = list(
            executor.map(lambda bounds: shapely.union_all(geometries[slice(*bounds)]), zip(starts, stops, strict=True))
        )

    footprints = gpd.GeoDataFrame(
        geometry=_make_polygonal_valid(np.array(unions, dtype=object)),
        index=pd.Index(unique_ids, name=PyxaKeys.CELL_ID.value),
    )
    if not footprints.geometry.is_valid.all():
        raise ValueError("cell footprints are invalid after merging the z-plane polygons")
    return footprints


def _open_mosaic(path: Path) -> zarr.Group:
    """Open a mosaic OME-Zarr read-only, from its directory or from a zip of it (read in place).

    A zip holds either the group at its root or a single top-level ``<name>.ome.zarr/`` directory,
    as in the Stellaromics/demo dataset. Top-level entries starting with ``__`` (e.g. the
    ``__MACOSX/`` tree of a zip made on macOS) are not the mosaic and are ignored.
    """
    if path.suffix != ".zip":
        return zarr.open_group(store=str(path), mode="r")
    with zipfile.ZipFile(path) as zf:
        tops = {top for name in zf.namelist() if not (top := name.split("/", 1)[0]).startswith("__")}
    group_path = "" if "zarr.json" in tops or len(tops) != 1 else tops.pop()
    return zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r", path=group_path)


@dataclass(frozen=True)
class _MosaicGrid:
    """The mosaic's voxel grid: every level's (z, y, x) shape and level 0's frame in micrometers."""

    shapes: tuple[tuple[int, int, int], ...]
    scale: tuple[float, float, float]
    translation: tuple[float, float, float]

    @property
    def transformation(self) -> Sequence:
        axes = ("z", "y", "x")
        return Sequence([Scale(list(self.scale), axes=axes), Translation(list(self.translation), axes=axes)])

    def step(self, level: int) -> tuple[int, int, int]:
        """Level-0 voxels per voxel of ``level``, per axis (nearest-neighbour stride)."""
        n0, n = self.shapes[0], self.shapes[level]
        return tuple(max(1, round(a / b)) for a, b in zip(n0, n, strict=True))  # type: ignore[return-value]


def _mosaic_grid(path: Path) -> _MosaicGrid:
    """The mosaic's level shapes (z, y, x) and level-0 scale and translation, from its OME-NGFF metadata."""
    group = _open_mosaic(path)
    multiscale = cast("dict[str, Any]", group.attrs.asdict()["ome"])["multiscales"][0]
    axes = [a["name"] for a in multiscale["axes"]]
    zyx = [axes.index(a) for a in ("z", "y", "x")]
    datasets = multiscale["datasets"]
    shapes = tuple(tuple(int(cast("Any", group[d["path"]]).shape[i]) for i in zyx) for d in datasets)
    transforms = {t["type"]: t for t in datasets[0]["coordinateTransformations"]}
    return _MosaicGrid(
        shapes=shapes,  # type: ignore[arg-type]
        scale=tuple(float(transforms["scale"]["scale"][i]) for i in zyx),  # type: ignore[arg-type]
        translation=tuple(float(transforms["translation"]["translation"][i]) for i in zyx),  # type: ignore[arg-type]
    )


def _get_image(path: Path) -> DataTree:
    """Load every level of an OME-Zarr (OME-NGFF v0.5) mosaic, from a directory or a zip, as a multiscale image.

    The pyramid levels already in the store are opened lazily, not recomputed. As in spatialdata's
    own OME-Zarr reader, the ``global`` transformation comes from the full-resolution level and
    each coarser level is related to it by the ratio of the array shapes.
    """
    group = _open_mosaic(path)
    ome = cast("dict[str, Any]", group.attrs.asdict()["ome"])
    multiscale = ome["multiscales"][0]
    datasets = multiscale["datasets"]

    # drop the singleton "t" axis, which spatialdata's image models don't model
    all_axes = [a["name"] for a in multiscale["axes"]]
    t_index = all_axes.index("t")
    axes = tuple(a for a in all_axes if a != "t")
    arrays = [da.squeeze(da.from_zarr(group[d["path"]]), axis=t_index) for d in datasets]

    transformation = _mosaic_grid(path).transformation
    spatial_axes = tuple(a for a in axes if a != "c")

    n_channels = arrays[0].shape[axes.index("c")]
    channel_labels = [c.get("label") for c in ome.get("omero", {}).get("channels", [])]
    c_coords = channel_labels if len(channel_labels) == n_channels else list(range(n_channels))

    levels = {}
    for i, array in enumerate(arrays):
        # coordinates of every level are pixel centres in scale0 pixel units, as spatialdata assigns them
        coords: dict[str, Any] = {"c": c_coords}
        for ax in spatial_axes:
            n0, n = arrays[0].shape[axes.index(ax)], array.shape[axes.index(ax)]
            coords[ax] = np.linspace(0, n0, n + 1)[:-1] + n0 / n / 2
        levels[f"scale{i}"] = Dataset({"image": DataArray(array, dims=axes, coords=coords)})
    image = DataTree.from_dict(levels)
    set_transformation(image, {"global": transformation}, set_all=True)
    Image3DModel.validate(image)
    return image


# --- 3D cell labels, rasterized lazily from the segmentation polygons ------------------------------
# The polygons (one per cell per z-plane, in pixel units) are drawn onto the mosaic image's level-0
# voxel grid, so labels and image overlay voxel for voxel at level 0. Ring decoding (``_read_rings``)
# is eager; drawing (``_draw`` / ``_rasterize_tile``) is one ``dask.delayed`` task per tile and runs
# only when the labels are computed or written. See ``_get_labels`` for how the coarser pyramid
# levels are obtained without redrawing.

# PIL draws labels into signed 32-bit ("I" mode) images
_MAX_LABEL = 2**31 - 1


def _label_ids(cell_ids: pd.Index) -> tuple[np.ndarray, str]:
    """Integer label per cell, and the rule used.

    The trailing integer of each ``cell_id`` (after any prefix, e.g. ``Region_17`` -> 17) when every
    cell has one and they are unique, > 0 (0 is background) and < 2^31; otherwise 1..n in table order.
    """
    n = len(cell_ids)
    digits = pd.Series(np.asarray(cell_ids, dtype=str)).str.extract(r"(\d+)$", expand=False)
    if digits.isna().any():
        why = "no trailing integer in some cell_id"
    elif (digits.str.len() > 10).any() or (numbers := digits.astype(np.int64)).max() > _MAX_LABEL:
        why = "a trailing integer is not below 2^31"
    elif numbers.min() < 1:
        why = "a trailing integer is 0 (the background)"
    elif not numbers.is_unique:
        why = "trailing integers are not unique"
    else:
        return numbers.to_numpy(dtype=np.uint32), "trailing integer of cell_id"
    return np.arange(1, n + 1, dtype=np.uint32), f"1..n in table order ({why})"


@dataclass(frozen=True)
class _Rings:
    """Polygon exterior rings in level-0 voxel index space (voxel centres at integer coordinates)."""

    label: np.ndarray
    plane: np.ndarray
    length: np.ndarray
    coords: np.ndarray
    bounds: np.ndarray

    def __len__(self) -> int:
        return len(self.label)


# Rows decoded to shapely at once, per row group: bounds a worker's peak memory to one batch's
# geometries and their transformed/simplified copies, instead of a whole row group's (up to ~1.36M
# polygons for the largest Region measured), at a small cost in per-batch overhead.
_DECODE_BATCH_ROWS = 65_536


def _rings_from_row_group(
    path: Path,
    row_group: int,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    simplify: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """One parquet row group's rings on the grid, and how many polygon parts were dropped and why.

    Streamed in ``_DECODE_BATCH_ROWS``-row batches, so only one batch's shapely geometries (and their
    transformed/simplified copies) are held in memory at a time, not the whole row group's.
    """
    sz, sy, sx = grid.scale
    tz, ty, tx = grid.translation
    nz = grid.shapes[0][0]
    batches = pq.ParquetFile(path).iter_batches(
        batch_size=_DECODE_BATCH_ROWS,
        row_groups=[row_group],
        columns=[PyxaKeys.CELL_ID.value, PyxaKeys.Z_INDEX.value, "geometry"],
    )
    batch_arrays: list[dict[str, np.ndarray]] = []
    dropped = {"not in the table": 0, "off the mosaic's z range": 0, "empty": 0}
    for batch in batches:
        cell_id = pc.cast(batch.column(PyxaKeys.CELL_ID.value), "string").to_numpy(zero_copy_only=False)
        row = labels.index.get_indexer(cell_id)
        zindex = batch.column(PyxaKeys.Z_INDEX.value).to_numpy()
        geoms = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
        poly_parts, part_of = shapely.get_parts(geoms, return_index=True)
        rings = shapely.get_exterior_ring(poly_parts)
        rings = shapely.transform(
            rings,
            lambda c: np.column_stack(((c[:, 0] * xy_size - tx) / sx, (c[:, 1] * xy_size - ty) / sy)),
        )
        rings = shapely.simplify(rings, simplify)
        # plane k holds Z_um = (ZIndex + 0.5) * z_size, the reader's plane centre
        plane = np.rint(((zindex[part_of] + 0.5) * z_size - tz) / sz).astype(np.int32)
        in_table = row[part_of] >= 0
        on_grid = (plane >= 0) & (plane < nz)
        empty = shapely.is_empty(rings) | (shapely.get_num_coordinates(rings) < 3)
        keep = in_table & on_grid & ~empty
        rings = rings[keep]
        coords, ring_of = shapely.get_coordinates(rings, return_index=True)
        batch_arrays.append(
            {
                "label": labels.to_numpy(dtype=np.uint32)[row[part_of][keep]],
                "plane": plane[keep],
                "length": np.bincount(ring_of, minlength=len(rings)).astype(np.int64),
                "coords": coords.astype(np.float32),
                "bounds": shapely.bounds(rings).astype(np.float32).reshape(-1, 4),
            }
        )
        dropped["not in the table"] += int((~in_table).sum())
        dropped["off the mosaic's z range"] += int((in_table & ~on_grid).sum())
        dropped["empty"] += int((in_table & on_grid & empty).sum())
        # drop this batch's shapely/numpy arrays before decoding the next one
        del geoms, poly_parts, part_of, rings, plane, in_table, on_grid, empty, keep, coords, ring_of
    if batch_arrays:
        out = {k: np.concatenate([b[k] for b in batch_arrays]) for k in batch_arrays[0]}
    else:
        # the row group's batches kept nothing (every polygon filtered out, or no batches at all)
        out = {
            "label": np.empty(0, dtype=np.uint32),
            "plane": np.empty(0, dtype=np.int32),
            "length": np.empty(0, dtype=np.int64),
            "coords": np.empty((0, 2), dtype=np.float32),
            "bounds": np.empty((0, 4), dtype=np.float32),
        }
    return out, dropped


def _read_rings(
    path: Path,
    labels: pd.Series,
    grid: _MosaicGrid,
    xy_size: float,
    z_size: float,
    *,
    simplify: float = 0.25,
) -> _Rings:
    """Every segmentation polygon's exterior ring on the mosaic's level-0 grid, with its label and plane.

    ``labels`` maps ``cell_id`` to the cell's label. Row groups are decoded in worker processes, one
    task per row group across up to the CPU count of workers (``joblib``, preferring processes, so
    ``joblib.parallel_config(backend=...)`` can choose another backend); the default ``loky`` workers
    are shut down once decoding ends. Although pyarrow and shapely release the GIL, decoding a full
    Region's row groups concurrently on threads was measured to contend badly on the allocator (one
    process, many threads each doing large shapely/numpy malloc/free traffic) rather than the GIL,
    taking ~24x longer wall time and ~3x the peak memory of the same work split across processes.
    A single row group runs directly, with no pool. Rings are simplified to ``simplify`` voxels: the
    polygons are traced on a finer pixel grid than the mosaic's, and the staircase vertices add
    nothing at its resolution. Holes are ignored (exteriors are filled).
    """
    n_groups = pq.ParquetFile(path).metadata.num_row_groups
    if n_groups <= 1:
        results = [_rings_from_row_group(path, i, labels, grid, xy_size, z_size, simplify) for i in range(n_groups)]
    else:
        n_jobs = min(n_groups, joblib.cpu_count())
        backend, _ = joblib.parallel.get_active_backend(prefer="processes")
        results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
            joblib.delayed(_rings_from_row_group)(path, i, labels, grid, xy_size, z_size, simplify)
            for i in range(n_groups)
        )
        if n_jobs > 1 and isinstance(backend, joblib.parallel.BACKENDS["loky"]):
            # loky keeps its workers (hundreds of MB each after a decode) alive for reuse; nothing else
            # here needs them, so release them now rather than holding that memory through the write
            get_reusable_executor(reuse=True).shutdown(wait=True)
    parts = [r[0] for r in results]
    dropped: dict[str, int] = {}
    for _, d in results:
        for why, count in d.items():
            dropped[why] = dropped.get(why, 0) + count
    if parts:
        rings = _Rings(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    else:
        # no row groups at all (e.g. a crop with no cells): nothing to concatenate over
        rings = _Rings(
            label=np.empty(0, dtype=np.uint32),
            plane=np.empty(0, dtype=np.int32),
            length=np.empty(0, dtype=np.int64),
            coords=np.empty((0, 2), dtype=np.float32),
            bounds=np.empty((0, 4), dtype=np.float32),
        )
    summary = ", ".join(f"{count} {why}" for why, count in dropped.items() if count)
    logger.info(
        f"{path.name}: {len(rings)} polygon rings on the mosaic grid" + (f"; dropped {summary}" if summary else "")
    )
    return rings


# one drawing task: a whole number of storage chunks, so tiles never share a chunk
TILE = (32, 1024, 1024)
# storage chunks, as the mosaic's: an inspect window reads only the chunks it covers
CHUNKS = (32, 256, 256)


@dataclass(frozen=True)
class _Tile:
    """The rings to draw into one tile of one pyramid level."""

    origin: tuple[int, int, int]
    shape: tuple[int, int, int]
    step: tuple[int, int]
    label: np.ndarray
    plane: np.ndarray
    offsets: np.ndarray
    coords: np.ndarray


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Indices of the concatenated slices ``[s, s + n)`` for each (s, n), in order."""
    new_starts = np.cumsum(lengths) - lengths
    return np.arange(int(lengths.sum())) - np.repeat(new_starts, lengths) + np.repeat(starts, lengths)


def _plan_tiles(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    tile: tuple[int, int, int] = TILE,
) -> list[_Tile]:
    """Group the rings of one level (``shape``, ``step`` from level 0) by tile.

    The level keeps the planes a nearest-neighbour stride of level 0 keeps; a ring crossing tiles goes
    to each. Within a tile, rings are ordered by label, so where they overlap the higher label wins.
    """
    nz, ny, nx = shape
    dz, dy, dx = step
    tz, ty, tx = tile
    keep = np.flatnonzero(rings.plane % dz == 0)
    plane = rings.plane[keep] // dz
    on = plane < nz
    keep, plane = keep[on], plane[on]
    # bounds in level-voxel index space. Drawing snaps each vertex to its nearest voxel, so a ring fills
    # voxels floor(lo + 0.5)..floor(hi + 0.5): its high side can reach the next tile's first voxel.
    # Tiles are assigned from bounds padded by one voxel on the high side (the low side needs none); a
    # tile that the ring does not actually reach draws nothing for it.
    bounds = rings.bounds[keep] / np.array([dx, dy, dx, dy], dtype=np.float32)
    n_ty, n_tx = -(-ny // ty), -(-nx // tx)
    y_lo = np.clip(np.floor(bounds[:, 1] / ty), 0, n_ty - 1).astype(np.int64)
    y_hi = np.clip(np.floor((bounds[:, 3] + 1) / ty), 0, n_ty - 1).astype(np.int64)
    x_lo = np.clip(np.floor(bounds[:, 0] / tx), 0, n_tx - 1).astype(np.int64)
    x_hi = np.clip(np.floor((bounds[:, 2] + 1) / tx), 0, n_tx - 1).astype(np.int64)
    idx, keys = [], []
    for oy in range(int((y_hi - y_lo).max(initial=0)) + 1):
        for ox in range(int((x_hi - x_lo).max(initial=0)) + 1):
            hit = np.flatnonzero((y_lo + oy <= y_hi) & (x_lo + ox <= x_hi))
            idx.append(hit)
            keys.append(((plane[hit] // tz) * n_ty + y_lo[hit] + oy) * n_tx + x_lo[hit] + ox)
    idx_arr, keys_arr = np.concatenate(idx), np.concatenate(keys)
    if keys_arr.size == 0:
        # no ring falls in this level at all (e.g. an empty _Rings): nothing to draw, no tiles
        return []
    order = np.lexsort((rings.label[keep][idx_arr], keys_arr))
    idx_arr, keys_arr = idx_arr[order], keys_arr[order]

    ring = keep[idx_arr]
    starts = np.cumsum(rings.length) - rings.length
    lengths = rings.length[ring]
    coords = rings.coords[_ragged_gather(starts[ring], lengths)]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    tiles = []
    edges = np.flatnonzero(np.diff(keys_arr)) + 1
    for lo, hi in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(keys_arr)]]), strict=True):
        kz, rest = divmod(int(keys_arr[lo]), n_ty * n_tx)
        ky, kx = divmod(rest, n_tx)
        origin = (kz * tz, ky * ty, kx * tx)
        c0, c1 = int(offsets[lo]), int(offsets[hi])
        tiles.append(
            _Tile(
                origin=origin,
                shape=(min(tz, nz - origin[0]), min(ty, ny - origin[1]), min(tx, nx - origin[2])),
                step=(dy, dx),
                label=rings.label[ring[lo:hi]],
                plane=plane[idx_arr[lo:hi]] - origin[0],
                offsets=offsets[lo : hi + 1] - c0,
                coords=coords[c0:c1],
            )
        )
    return tiles


def _rasterize_tile(tile: _Tile) -> np.ndarray:
    """Fill the tile's rings plane by plane into a ``uint32`` block (0 = background)."""
    from PIL import Image, ImageDraw

    depth, height, width = tile.shape
    block = np.zeros(tile.shape, dtype=np.uint32)
    dy, dx = tile.step
    _, y0, x0 = tile.origin
    # Snap each vertex to the voxel whose centre is nearest (PIL fills the pixels of an integer polygon,
    # vertices included), in level voxel index space, so the snap does not depend on the tile. PIL draws
    # a polygon differently when some vertices are negative (it truncates toward zero and clips), so the
    # canvas starts at the lowest vertex, not the tile's origin, and is cropped to the tile after drawing:
    # a tile then draws exactly what one whole-level draw would, and neighbouring tiles join seamlessly.
    xs = np.floor(tile.coords[:, 0] / dx + 0.5).astype(np.int64)
    ys = np.floor(tile.coords[:, 1] / dy + 0.5).astype(np.int64)
    cx, cy = min(x0, int(xs.min(initial=x0))), min(y0, int(ys.min(initial=y0)))
    xy = np.column_stack((xs - cx, ys - cy))
    for plane in np.unique(tile.plane):
        image = Image.new("I", (x0 + width - cx, y0 + height - cy))
        draw = ImageDraw.Draw(image)
        for r in np.flatnonzero(tile.plane == plane):
            draw.polygon(xy[tile.offsets[r] : tile.offsets[r + 1]].ravel().tolist(), fill=int(tile.label[r]))
        block[plane] = np.asarray(image, dtype=np.int32)[y0 - cy :, x0 - cx :]
    return block


def _labels_level(
    rings: _Rings,
    shape: tuple[int, int, int],
    step: tuple[int, int, int],
    *,
    tile: tuple[int, int, int] = TILE,
    chunks: tuple[int, int, int] = CHUNKS,
) -> da.Array:
    """One pyramid level as a lazy array: a ``dask.delayed`` drawing task per tile, zeros where no ring falls."""
    return _tiles_array(_plan_tiles(rings, shape, step, tile), shape, tile=tile, chunks=chunks)


def _tiles_array(
    tiles: list[_Tile],
    shape: tuple[int, int, int],
    *,
    tile: tuple[int, int, int] = TILE,
    chunks: tuple[int, int, int] = CHUNKS,
) -> da.Array:
    """A level's planned tiles as a lazy array, zeros where no tile is planned."""
    by_origin = {t.origin: t for t in tiles}
    nz, ny, nx = shape
    tz, ty, tx = tile
    blocks = []
    for z in range(0, nz, tz):
        rows = []
        for y in range(0, ny, ty):
            row = []
            for x in range(0, nx, tx):
                size = (min(tz, nz - z), min(ty, ny - y), min(tx, nx - x))
                t = by_origin.get((z, y, x))
                if t is None:
                    row.append(da.zeros(size, dtype=np.uint32, chunks=size))
                else:
                    # looked up at call time, so the drawing function can be patched in tests
                    row.append(da.from_delayed(dask.delayed(_draw)(t), shape=size, dtype=np.uint32))
            rows.append(row)
        blocks.append(rows)
    return da.block(blocks).rechunk(tuple(min(c, s) for c, s in zip(chunks, shape, strict=True)))


def _draw(tile: _Tile) -> np.ndarray:
    return _rasterize_tile(tile)


def _get_labels(rings: _Rings, grid: _MosaicGrid) -> DataTree:
    """The cells as a lazy multiscale ``Labels3DModel`` on the mosaic's grid, one level per mosaic level.

    Level 0 is drawn from the rings, one ``dask.delayed`` tile at a time. Every coarser level is a
    nearest-neighbour strided *view* of the level-0 array (matching ``grid.step``), not redrawn
    independently, so a level-0 tile's drawing task is shared by every level that needs it: computing or
    writing the whole tree draws each level-0 tile at most once. The stride starts at each block's centre
    (offset ``step // 2`` per axis, less where that would leave the level short of the mosaic's shape):
    the mosaic's own pyramid is a smoothed block average, so centre samples track it best.
    """
    n0 = grid.shapes[0]
    tiles = _plan_tiles(rings, n0, (1, 1, 1))
    level0 = _tiles_array(tiles, n0)
    levels = {}
    for i, shape in enumerate(grid.shapes):
        if i == 0:
            array = level0
        else:
            step = grid.step(i)
            # each coarse voxel takes the level-0 voxel at its block's centre, as the mosaic's pyramid
            # (a smoothed block average) centres it there; the offset shrinks where it would run off the end
            oz, oy, ox = (max(0, min(d // 2, a - 1 - (b - 1) * d)) for a, b, d in zip(n0, shape, step, strict=True))
            dz, dy, dx = step
            strided = level0[oz::dz, oy::dy, ox::dx]
            if any(a < b for a, b in zip(strided.shape, shape, strict=True)):
                raise ValueError(
                    f"scale{i}: level-0 stride {step} gives shape {strided.shape}, "
                    f"shorter than the mosaic's {shape} on some axis"
                )
            sz, sy, sx = shape
            array = strided[:sz, :sy, :sx].rechunk(tuple(min(c, s) for c, s in zip(CHUNKS, shape, strict=True)))
        # coordinates of every level are pixel centres in scale0 pixel units, as spatialdata assigns them
        coords = {ax: np.linspace(0, a, b + 1)[:-1] + a / b / 2 for ax, a, b in zip("zyx", n0, shape, strict=True)}
        levels[f"scale{i}"] = Dataset({"image": DataArray(array, dims=("z", "y", "x"), coords=coords)})
    tree = DataTree.from_dict(levels)
    set_transformation(tree, {"global": grid.transformation}, set_all=True)
    Labels3DModel.validate(tree)
    logger.info(
        f"{PyxaKeys.CELL_LABELS.value}: {len(rings)} rings in {len(tiles)} level-0 tiles, "
        f"{len(grid.shapes)} levels planned; drawn when computed or written"
    )
    return tree


InputPath = str | Path | bool | None

RequiredPath = str | Path | None


def _required(value: RequiredPath) -> str | Path | bool:
    """A required file is never skipped; unset means it must be in ``path``."""
    if isinstance(value, bool):
        raise TypeError("cell_by_gene and cell_metadata are required; pass a path or leave them unset")
    return True if value is None else value


def _resolve_input(value: InputPath, path: Path | None, file_name: str) -> Path | None:
    """Where to read one Pyxa file from, or ``None`` to skip it.

    ``False`` skips the file. ``None`` reads ``path / file_name`` when that exists and skips it
    otherwise; ``True`` requires it. A path reads that file, which must exist.
    """
    if value is False:
        return None
    if value is None or value is True:
        candidate = path / file_name if path is not None else None
        if candidate is not None and candidate.exists():
            return candidate
        if value is True:
            raise FileNotFoundError(f"Expected Pyxa output file not found: {candidate or file_name}")
        return None
    explicit = Path(value)
    if not explicit.exists():
        raise FileNotFoundError(f"Expected Pyxa output file not found: {explicit}")
    return explicit


def _resolve_image(value: InputPath, path: Path | None) -> Path | None:
    """Where to read the mosaic from, or ``None`` to skip it.

    Like :func:`_resolve_input`, looking in ``path`` for the unzipped ``mosaic_3d.ome.zarr`` first
    and then for ``mosaic_3d.ome.zarr.zip``. An explicit path may be either.
    """
    if value is False:
        return None
    if value is None or value is True:
        for name in (PyxaKeys.MOSAIC_FILE.value, PyxaKeys.MOSAIC_ZIP_FILE.value):
            if path is not None and (path / name).exists():
                return path / name
        if value is True:
            raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {PyxaKeys.MOSAIC_FILE.value}(.zip)")
        return None
    explicit = Path(value)
    if not explicit.exists():
        raise FileNotFoundError(f"Expected Pyxa mosaic image not found: {explicit}")
    return explicit


@inject_docs(px=PyxaKeys)
def pyxa(
    path: str | Path | None = None,
    dataset_id: str = "pyxa",
    *,
    cell_by_gene: RequiredPath = None,
    cell_metadata: RequiredPath = None,
    cell_assigned_gene: InputPath = None,
    segmentation_geometries: InputPath = None,
    pyxa_studio: InputPath = None,
    image: InputPath = None,
    shapes: bool | None = None,
    labels: bool = False,
) -> SpatialData:
    """
    Read *Pyxa* (Stellaromics) output.

    The ``rna`` table is always read, from two required files:

        - ``{px.CELL_BY_GENE_FILE!r}``: Per-cell gene expression counts, the table's ``X``.
        - ``{px.CELL_METADATA_FILE!r}``: Per-cell metadata (volume, spatial coordinates), the
          table's ``obs`` and ``obsm["spatial"]``.

    Everything else is optional, each read when present:

        - ``{px.CELL_ASSIGNED_GENE_FILE!r}``: Transcript-level gene assignments, as the
          ``transcripts`` points.
        - ``{px.SEGMENTATION_GEOMETRIES_FILE!r}``: Per-cell segmentation polygons, as shapes, or
          with ``labels=True`` as 3D cell labels. With ``labels=False`` the table annotates the
          cell footprints only when these are read.
        - ``{px.PYXA_STUDIO_FILE!r}``: Pyxa Studio's export of the cells that passed its filters,
          adding ``{px.CLUSTER!r}`` (categorical) to the table's ``obs`` and the 3D UMAP as
          ``obsm[{px.UMAP_KEY!r}]``. Cells it filtered out keep missing values there.
        - ``{px.MOSAIC_FILE!r}`` (or ``{px.MOSAIC_ZIP_FILE!r}``, read in place): the mosaic
          OME-Zarr (OME-NGFF v0.5) image, all pyramid levels, as ``{px.MOSAIC_IMAGE!r}``.

    Files are looked up in ``path`` by default. Pass a path for a file to read it from
    elsewhere, and for an optional file ``False`` to skip it even if present (e.g. the
    transcripts of a large Region) or ``True`` to require it.

    No public specification exists for this format at the time of writing; this
    reader is validated against the public demo dataset at
    https://huggingface.co/datasets/Stellaromics/demo.

    All elements are returned in micrometers in the ``global`` coordinate
    system. Segmentation polygons are stored on disk in pixel units, one polygon
    per cell per z-plane (``ZIndex``); the reader converts them to micrometers
    (repairing any polygon that the conversion makes invalid) and adds their z
    as a ``Z_um`` column (the centre of the z-plane), since shapes are 2D in
    spatialdata. Both voxel sizes are inferred from ``{px.CELL_METADATA_FILE!r}``,
    whose per-cell centroids are the area-weighted centroids of each cell's
    polygons in both units.

    As shapes (by default only with ``labels=False``; see ``shapes``), the polygons are two elements:

        - ``{px.REGION!r}``: one 2D footprint per cell (the union of its z-plane
          polygons), indexed by ``cell_id`` and, with ``labels=False``, annotated by the ``rna`` table.
          Cells stacked in z have overlapping footprints, so use these for 2D
          display and table annotation, not for 2D spatial aggregation (the
          transcripts' ``cell_id`` already gives each transcript's cell).
        - ``{px.CELL_BOUNDARIES_Z!r}``: the per-cell, per-z-plane polygons, with
          ``cell_id``, ``ZIndex`` and ``Z_um`` columns.

    With ``labels=True`` the polygons are instead rasterized into ``{px.CELL_LABELS!r}``, a 3D
    labels element on the mosaic's voxel grid (same pyramid levels and transformation as
    ``{px.MOSAIC_IMAGE!r}``), and the table annotates it through the integer
    ``{px.LABEL_ID!r}``: the trailing integer of each ``cell_id`` (``Region_17`` -> 17) when
    those are unique, positive and below 2^31, otherwise 1..n in table order. Holes are
    filled, and where two cells overlap on a plane the higher label wins. The labels are
    lazy: level 0 is drawn, one task per 32 x 1024 x 1024 tile, when computed or written, and
    the coarser levels are strided views of it (nearest neighbour, sampled at each coarse
    voxel's block centre), so writing draws each tile once, with dask's default threaded
    scheduler (a process scheduler is much slower here, since every drawn tile is pickled
    back). The lazy labels' dask graph holds every ring's coordinates (GBs for a full Region)
    for as long as the element is alive, and would ship them all to a distributed scheduler.
    Decoding the polygons is eager, in worker processes: one task per parquet row group,
    across up to the CPU count of workers (threads contend badly on the allocator with this
    many large shapely/numpy arrays; ``joblib.parallel_config(backend=...)`` can choose another
    backend), each row group streamed in small batches so a worker never holds more than one
    batch's geometries at once. That takes about a minute and tens of GB for a full Region of
    ~20M polygons.

    Unassigned transcripts (``cell_id`` ending in ``"_-1"``) are kept in the
    points element, flagged via an ``assigned`` column, rather than dropped.

    Parameters
    ----------
    path
        Directory holding Pyxa's output files. ``None`` reads only the files given
        explicitly, in which case ``cell_by_gene`` and ``cell_metadata`` must be.
    dataset_id
        Dataset identifier, currently unused for element naming (reserved for
        future multi-sample support).
    cell_by_gene, cell_metadata
        Required files: ``None`` (default) reads them from ``path``, a path reads that file.
    cell_assigned_gene, segmentation_geometries, pyxa_studio, image
        Optional files: ``None`` (default) reads one from ``path`` if present, a path reads
        that file, ``False`` skips it and ``True`` requires it in ``path``.

        ``image`` may be the mosaic's directory or a zip of it; a zipped mosaic is read in place.
        Compute a zipped mosaic with dask's threaded scheduler (the default): a process scheduler
        pickles its ``ZipStore``, which reopens the zip in every worker.
    shapes
        Return the polygons as shapes. ``None`` (default): when the segmentation geometries are
        read and ``labels`` is ``False``.
    labels
        Rasterize the polygons into 3D cell labels on the mosaic's grid (needs the segmentation
        geometries and the mosaic); the table then annotates the labels.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    directory = Path(path) if path is not None else None
    if directory is not None and not directory.is_dir():
        raise FileNotFoundError(f"Pyxa output directory not found: {directory}")
    by_gene_path = _resolve_input(_required(cell_by_gene), directory, PyxaKeys.CELL_BY_GENE_FILE.value)
    metadata_path = _resolve_input(_required(cell_metadata), directory, PyxaKeys.CELL_METADATA_FILE.value)
    assigned_gene_path = _resolve_input(cell_assigned_gene, directory, PyxaKeys.CELL_ASSIGNED_GENE_FILE.value)
    geometries_path = _resolve_input(segmentation_geometries, directory, PyxaKeys.SEGMENTATION_GEOMETRIES_FILE.value)
    studio_path = _resolve_input(pyxa_studio, directory, PyxaKeys.PYXA_STUDIO_FILE.value)
    if by_gene_path is None or metadata_path is None:  # unreachable: required inputs resolve or raise
        raise FileNotFoundError("cell_by_gene and cell_metadata are required")

    image_source = _resolve_image(image, directory)
    inputs = [p for p in (by_gene_path, metadata_path, assigned_gene_path, geometries_path, studio_path) if p]
    logger.info(f"Reading Pyxa {', '.join(p.name for p in inputs)}")

    if labels:
        missing = [
            name
            for name, found in (("segmentation_geometries", geometries_path), ("a mosaic image", image_source))
            if found is None
        ]
        if missing:
            raise ValueError(
                f"labels=True needs segmentation_geometries and a mosaic image; missing: {', '.join(missing)}"
            )
    if shapes and geometries_path is None:
        raise FileNotFoundError(
            f"Expected Pyxa output file not found: {PyxaKeys.SEGMENTATION_GEOMETRIES_FILE.value} (shapes=True)"
        )
    read_shapes = geometries_path is not None and (shapes if shapes is not None else not labels)

    points = {}
    if assigned_gene_path is not None:
        points["transcripts"] = PointsModel.parse(
            _get_points(assigned_gene_path),
            coordinates={"x": PyxaKeys.X_UM.value, "y": PyxaKeys.Y_UM.value, "z": PyxaKeys.Z_UM.value},
            feature_key=PyxaKeys.GENE.value,
            instance_key=PyxaKeys.CELL_ID.value,
        )

    xy_size, z_size = _get_voxel_size(metadata_path) if (read_shapes or labels) else (1.0, 1.0)
    shapes_elements = {}
    if read_shapes:
        planes = _get_shapes(geometries_path, xy_size, z_size)  # type: ignore[arg-type]
        shapes_elements[PyxaKeys.REGION.value] = ShapesModel.parse(_get_footprints(planes))
        shapes_elements[PyxaKeys.CELL_BOUNDARIES_Z.value] = ShapesModel.parse(planes)

    adata = _get_table(by_gene_path, metadata_path, studio_path)
    labels_elements = {}
    if labels:
        ids, rule = _label_ids(adata.obs_names)
        logger.info(f"{PyxaKeys.LABEL_ID.value}: {rule}")
        adata.obs[PyxaKeys.LABEL_ID.value] = ids
        adata.obs[PyxaKeys.REGION_KEY.value] = pd.Series(
            PyxaKeys.CELL_LABELS.value, index=adata.obs_names, dtype="category"
        )
        grid = _mosaic_grid(image_source)  # type: ignore[arg-type]
        rings = _read_rings(
            geometries_path,  # type: ignore[arg-type]
            pd.Series(ids, index=adata.obs_names),
            grid,
            xy_size,
            z_size,
        )
        labels_elements[PyxaKeys.CELL_LABELS.value] = _get_labels(rings, grid)
        table = TableModel.parse(
            adata,
            region=PyxaKeys.CELL_LABELS.value,
            region_key=PyxaKeys.REGION_KEY.value,
            instance_key=PyxaKeys.LABEL_ID.value,
        )
    elif shapes_elements:
        table = TableModel.parse(
            adata,
            region=PyxaKeys.REGION.value,
            region_key=PyxaKeys.REGION_KEY.value,
            instance_key=PyxaKeys.INSTANCE_KEY.value,
        )
    else:
        table = TableModel.parse(adata)

    images = {}
    if image_source is not None:
        images[PyxaKeys.MOSAIC_IMAGE.value] = _get_image(image_source)

    return SpatialData(
        points=points, shapes=shapes_elements, labels=labels_elements, tables={"rna": table}, images=images
    )
