from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import anndata as ad
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import zarr
from scipy import sparse
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image3DModel, PointsModel, ShapesModel, TableModel
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
    as in the Stellaromics/demo dataset.
    """
    if path.suffix != ".zip":
        return zarr.open_group(store=str(path), mode="r")
    with zipfile.ZipFile(path) as zf:
        tops = {name.split("/", 1)[0] for name in zf.namelist()}
    group_path = "" if "zarr.json" in tops or len(tops) != 1 else tops.pop()
    return zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r", path=group_path)


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

    coordinate_transformations = {ct["type"]: ct for ct in datasets[0]["coordinateTransformations"]}
    spatial_axes = tuple(a for a in axes if a != "c")
    scale_values = [
        v for v, a in zip(coordinate_transformations["scale"]["scale"], all_axes, strict=True) if a in spatial_axes
    ]
    translation_values = [
        v
        for v, a in zip(coordinate_transformations["translation"]["translation"], all_axes, strict=True)
        if a in spatial_axes
    ]
    transformation = Sequence(
        [Scale(scale_values, axes=spatial_axes), Translation(translation_values, axes=spatial_axes)]
    )

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
        - ``{px.SEGMENTATION_GEOMETRIES_FILE!r}``: Per-cell segmentation polygons, as shapes. The
          table annotates the cell footprints only when these are read.
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

    The polygons are returned as two shapes elements:

        - ``{px.REGION!r}``: one 2D footprint per cell (the union of its z-plane
          polygons), indexed by ``cell_id`` and annotated by the ``rna`` table.
          Cells stacked in z have overlapping footprints, so use these for 2D
          display and table annotation, not for 2D spatial aggregation (the
          transcripts' ``cell_id`` already gives each transcript's cell).
        - ``{px.CELL_BOUNDARIES_Z!r}``: the per-cell, per-z-plane polygons, with
          ``cell_id``, ``ZIndex`` and ``Z_um`` columns.

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

        ``image`` may be the mosaic's directory or a zip of it; a zipped mosaic is read in place
        with the threaded dask scheduler (a ``ZipStore`` is not shared across processes).

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

    points = {}
    if assigned_gene_path is not None:
        points["transcripts"] = PointsModel.parse(
            _get_points(assigned_gene_path),
            coordinates={"x": PyxaKeys.X_UM.value, "y": PyxaKeys.Y_UM.value, "z": PyxaKeys.Z_UM.value},
            feature_key=PyxaKeys.GENE.value,
            instance_key=PyxaKeys.CELL_ID.value,
        )

    shapes = {}
    if geometries_path is not None:
        xy_size, z_size = _get_voxel_size(metadata_path)
        planes = _get_shapes(geometries_path, xy_size, z_size)
        shapes[PyxaKeys.REGION.value] = ShapesModel.parse(_get_footprints(planes))
        shapes[PyxaKeys.CELL_BOUNDARIES_Z.value] = ShapesModel.parse(planes)

    adata = _get_table(by_gene_path, metadata_path, studio_path)
    if shapes:
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

    return SpatialData(points=points, shapes=shapes, tables={"rna": table}, images=images)
