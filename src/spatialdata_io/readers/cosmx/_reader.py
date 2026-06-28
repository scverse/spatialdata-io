"""CosMx reader for spatialdata.

Reads NanoString CosMx spatial transcriptomics data into a
:class:`spatialdata.SpatialData` object.  Supports all known CosMx export
formats (AtomX flat files, nested CellStatsDir layouts, multimodal
RNA+Protein runs) and transparently handles stitching, polygon
rasterization, and coordinate normalization.

Public API
----------
cosmx
    Top-level reader function (single or multimodal datasets).
CosMxDataset
    Frozen dataclass describing the on-disk layout of a CosMx dataset.
CosMxDatasetReader
    Stateful reader that orchestrates per-element I/O.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import tifffile
from shapely.affinity import translate as _translate
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations import Translation

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

from ._discovery import (
    _infer_dataset_id,
    _set_up_cosmx_dataset_for_conversion,
)
from ._io import (
    COSMX_FOV_SIZE_PX,
    _default_image_kwargs,
    _find_matching_fov_file,
    _get_cosmx_morphology_coords,
    _maybe_warn_big_file,
    _parquet_cache_for_tx,
    _read_expr_mat_polars,
    _read_fov_image,
    _read_fov_locs,
    _read_metadata_polars,
    _read_polygons_csv,
    _read_transcripts_csv,
    place_local_in_fov_grid,
)
from ._stitching import (
    _canvas_from_fov_locs_for_polygons,
    _plot_fov_preview,
    _polygons_to_label_raster,
    _read_stitched_cell_labels_from_dir,
    _read_stitched_image,
    stitch_segmentation_label_image,
)
from ._utils import (
    _dask_categoricals_to_string,
    _normalize_image_channels,
    _pandas_categoricals_to_string,
    detect_fovs_with_data,
)

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["cosmx"]

# ---------------------------------------------------------------------------
# Dataset descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CosMxDataset:
    """Frozen descriptor of a CosMx dataset directory layout.

    Automatically infers ``dataset_id`` from marker-file prefixes on disk.
    For multimodal runs (e.g. S3RNA + S3Protein), the ``modalities`` field
    maps each modality name to its own ``CosMxDataset``.
    """

    path: Path = field(default=Path("."), metadata={"help": "Root directory."})
    dataset_id: str | None = field(default=None)
    exprMat_file: Path | None = field(default=None)
    fov_positions_file: Path | None = field(default=None)
    metadata_file: Path | None = field(default=None)
    tx_file: Path | None = field(default=None)
    polygons_file: Path | None = field(default=None)
    analysis_results_dir: Path | None = field(default=None)
    cell_stats_dir: Path | None = field(default=None)
    run_summary_dir: Path | None = field(default=None)
    cell_composite_dir: Path | None = field(default=None)
    cell_labels_dir: Path | None = field(default=None)
    cell_overlay_dir: Path | None = field(default=None)
    celltype_accessory_data: Path | None = field(default=None)
    compartment_labels_dir: Path | None = field(default=None)
    protein_dir: Path | None = field(default=None)
    morphology_2d_dir: Path | None = field(default=None)
    morphology_3d_dir: Path | None = field(default=None)
    modalities: dict[str, CosMxDataset] | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        if self.modalities:
            if not isinstance(self.modalities, dict) or not self.modalities:
                raise ValueError("modalities must be a non-empty dict.")
            return
        # At least one data file must be present.
        has_data = any(
            val is not None for name, val in self.__dict__.items() if name not in {"dataset_id", "modalities", "path"}
        )
        if not has_data:
            raise ValueError("At least one data file or directory must be present.")
        inferred = _infer_dataset_id(self.path, self.dataset_id)
        object.__setattr__(self, "dataset_id", inferred)

    # -- pretty-printing --

    def __str__(self) -> str:
        root = self.path
        lines = [self.__class__.__name__, f"  path: {root}", f"  dataset_id: {self.dataset_id or '<not set>'}"]
        for name, val in self.__dict__.items():
            if name in {"path", "dataset_id", "modalities"} or val is None:
                continue
            if isinstance(val, Path):
                if not val.exists():
                    continue
                try:
                    val = f"<path>/{val.resolve().relative_to(root).as_posix()}"
                except ValueError:
                    val = str(val.resolve())
                lines.append(f"  {name}: {val}")
            else:
                lines.append(f"  {name}: {val!r}")
        for mod_name, mod_ds in (self.modalities or {}).items():
            lines.append(f"  {mod_name}:")
            lines.append(textwrap.indent(str(mod_ds), "    "))
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle) -> None:
        p.text(self.__class__.__name__ + "(…)" if cycle else str(self))


# ---------------------------------------------------------------------------
# Transformation helper
# ---------------------------------------------------------------------------


def _translation_transform(ox: float, oy: float) -> Translation:
    """Translation by ``(ox, oy)`` in the global xy coordinate system."""
    return Translation([ox, oy], axes=("x", "y"))


# ---------------------------------------------------------------------------
# Element-name helpers
# ---------------------------------------------------------------------------


def _element_name(base: str, fovs: set[int] | None, *, max_listed: int = 8) -> str:
    """Build a context-dependent element name from *base* and the FOV set."""
    if not fovs:
        return base
    if len(fovs) == 1:
        return f"F{next(iter(fovs)):05d}_{base}"
    sorted_fovs = sorted(fovs)
    if len(sorted_fovs) <= max_listed:
        return "subset_" + "_".join(f"F{f:05d}" for f in sorted_fovs) + f"_{base}"
    return base


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _collect_global_cell_ids_from_tables(tables: dict[str, AnnData]) -> set[int]:
    """Gather all non-zero global_cell_ids across all tables."""
    ids: set[int] = set()
    for table in tables.values():
        adata = table.table if hasattr(table, "table") else table
        if "global_cell_id" not in adata.obs:
            continue
        col = adata.obs["global_cell_id"]
        if isinstance(col.dtype, pd.CategoricalDtype):
            col = col.astype("Int64")
        for v in col:
            if pd.notna(v) and int(v) != 0:
                ids.add(int(v))
    return ids


def _filter_tables_to_ids(
    tables: dict[str, AnnData],
    valid_ids: set[int],
    *,
    source_label: str = "label IDs",
) -> dict[str, AnnData]:
    """Keep only table rows whose global_cell_id is in *valid_ids*."""
    if not valid_ids:
        return tables
    filtered: dict[str, AnnData] = {}
    for name, table in tables.items():
        adata = table.table if hasattr(table, "table") else table
        if "global_cell_id" not in adata.obs:
            filtered[name] = table
            continue
        col = adata.obs["global_cell_id"]
        if isinstance(col.dtype, pd.CategoricalDtype):
            col = col.astype("Int64")
        mask = col.isin(valid_ids)
        if not mask.any():
            logger.warning("Table %s: all rows removed after matching to %s.", name, source_label)
            continue
        if mask.all():
            filtered[name] = table
            continue
        logger.info("Table %s: dropping %d row(s) not in %s.", name, int((~mask).sum()), source_label)
        ad2 = adata[mask].copy()
        region_val = ad2.obs["region_key"].iloc[0] if "region_key" in ad2.obs else None
        filtered[name] = TableModel.parse(
            ad2,
            region=region_val,
            region_key="region_key" if region_val else None,
            instance_key="global_cell_id",
            overwrite_metadata=True,
        )
    return filtered


# ---------------------------------------------------------------------------
# CosMxDatasetReader
# ---------------------------------------------------------------------------


class CosMxDatasetReader:
    """Stateful reader that orchestrates reading of all CosMx elements.

    Manages FOV subsetting, coordinate origin normalization,
    global_cell_id computation, and lazy polygon caching.

    Parameters
    ----------
    dataset
        A :class:`CosMxDataset` describing the files on disk.
    fovs
        Optional set of FOV IDs to read (``None`` = all).
    n_workers
        Number of parallel workers for dask/zarr operations.
    flip_image
        Whether to flip images vertically.
    polygons_as_labels
        If ``True``, rasterize polygons into a label image.
    fov_locs
        Pre-parsed FOV positions (skips re-reading the CSV).
    align_rasters_to_polygons
        If ``True``, anchor images/labels to the polygon coordinate origin.
    keep_polygons_after_rasterize
        If ``True``, keep vector polygons even when rasterizing to labels.
    """

    def __init__(
        self,
        dataset: CosMxDataset,
        *,
        fovs: set[int] | None = None,
        n_workers: int | None = None,
        flip_image: bool | None = None,
        polygons_as_labels: bool = False,
        fov_locs: pd.DataFrame | None = None,
        align_rasters_to_polygons: bool | None = None,
        keep_polygons_after_rasterize: bool = False,
        image_normalization_percentile: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.fovs = fovs
        self.n_workers = n_workers or 1
        self.flip_image = bool(flip_image) if flip_image is not None else False
        self.polygons_as_labels = polygons_as_labels
        self.keep_polygons_after_rasterize = keep_polygons_after_rasterize
        if image_normalization_percentile is not None:
            if not 0.0 <= image_normalization_percentile <= 100.0:
                raise ValueError(
                    f"image_normalization_percentile must be in [0, 100] or None, got "
                    f"{image_normalization_percentile!r}."
                )
            if image_normalization_percentile < 1.0:
                logger.warning(
                    "image_normalization_percentile=%s is < 1; percentiles are 0-100 — did you mean e.g. 99.9?",
                    image_normalization_percentile,
                )
        self.image_normalization_percentile = image_normalization_percentile
        self._image_norm_logged = False

        # Read or reuse FOV positions
        if fov_locs is not None:
            self.fov_locs = fov_locs
        elif dataset.fov_positions_file is not None:
            self.fov_locs = _read_fov_locs(
                dataset.fov_positions_file,
                fovs=sorted(fovs) if fovs else None,
            )
        else:
            raise ValueError("FOV positions file is required to read CosMx data.")

        self._base_origin_x = float(self.fov_locs["xmin"].min())
        self._base_origin_y = float(self.fov_locs["ymin"].min())

        # Polygon cache and origin (set lazily by _get_polygons)
        self._poly_df: pd.DataFrame | None = None
        self._origin_x: float | None = None
        self._origin_y: float | None = None

        if align_rasters_to_polygons is None:
            align_rasters_to_polygons = polygons_as_labels
        self.align_rasters_to_polygons = bool(align_rasters_to_polygons)

        # Frozen after first use to ensure consistent IDs across elements
        self.max_cell_id: int | None = None

    # -- Polygon loading and origin computation --

    def _get_polygons(self) -> pd.DataFrame:
        """Load polygons (cached). Sets ``_origin_x`` / ``_origin_y``."""
        if self._poly_df is not None:
            return self._poly_df
        if self.dataset.polygons_file is None:
            raise FileNotFoundError("Dataset has no polygons_file.")

        poly_df = _read_polygons_csv(
            self.dataset.polygons_file,
            fov_locs=self.fov_locs,
            n_workers=self.n_workers,
            fov_set=self.fovs,
            use_polars=True,
        )
        poly_df["global_cell_id"] = self.global_cell_id(poly_df)

        # Compute origin from actual polygon bounds
        xs = [float(g.bounds[0]) for g in poly_df["geometry"] if g is not None and not g.is_empty]
        ys = [float(g.bounds[1]) for g in poly_df["geometry"] if g is not None and not g.is_empty]
        ox = min(xs) if xs else self._base_origin_x
        oy = min(ys) if ys else self._base_origin_y

        # Shift polygons to origin
        if ox != 0.0 or oy != 0.0:
            poly_df["geometry"] = [
                _translate(g, xoff=-ox, yoff=-oy) if g is not None else None for g in poly_df["geometry"]
            ]
            poly_df = poly_df.dropna(subset=["geometry"]).reset_index(drop=True)

        self._origin_x = ox
        self._origin_y = oy
        self._poly_df = poly_df
        return self._poly_df

    def _ensure_origin(self) -> tuple[float, float]:
        """Return the coordinate origin, loading polygons if needed."""
        if self._origin_x is not None and self._origin_y is not None:
            return self._origin_x, self._origin_y
        if self.dataset.polygons_file is not None and (self.align_rasters_to_polygons or self.polygons_as_labels):
            self._get_polygons()
            return self._origin_x, self._origin_y  # type: ignore[return-value]
        self._origin_x = self._base_origin_x
        self._origin_y = self._base_origin_y
        return self._origin_x, self._origin_y

    # -- Global cell ID --

    def _lock_max_cell_id(self, current_max: int) -> int:
        """Lock or validate ``max_cell_id`` and return the base multiplier.

        On first call, locks ``max_cell_id`` to *current_max*.  On subsequent
        calls, validates that *current_max* does not exceed the locked value.

        Returns ``max_cell_id + 1`` (the base used in the ID formula).
        """
        if self.max_cell_id is None:
            self.max_cell_id = current_max
        elif current_max > self.max_cell_id:
            raise ValueError(
                f"cell_ID up to {current_max} but reader locked to "
                f"max_cell_id={self.max_cell_id}. Load the file with the "
                "largest per-FOV cell_ID first (usually the polygon CSV)."
            )
        return self.max_cell_id + 1

    def global_cell_id(self, df: pd.DataFrame) -> pd.Series:
        """Compute ``global_cell_id = fov * (max_cell_id + 1) * (cell_ID > 0) + cell_ID``.

        Locks ``max_cell_id`` on first call for consistency across elements.
        """
        if "fov" not in df.columns or "cell_ID" not in df.columns:
            raise ValueError("Expected columns 'fov' and 'cell_ID'.")
        fov_ser = pd.to_numeric(df["fov"], errors="coerce").fillna(0).astype(int)
        cell_ser = pd.to_numeric(df["cell_ID"], errors="coerce").fillna(0).astype(int)
        current_max = int(cell_ser.max()) if len(cell_ser) else 0
        base = self._lock_max_cell_id(current_max)
        return fov_ser * base * (cell_ser > 0).astype(int) + cell_ser

    # -- Read images --

    def read_images(
        self,
        *,
        read_proteins: bool,
        image_models_kwargs: dict[str, Any],
        imread_kwargs: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, Image2DModel]:
        """Read morphology (and optionally protein) images."""
        if self.dataset.morphology_2d_dir is None:
            return {}

        images_dir = self.dataset.morphology_2d_dir
        morphology_coords = _get_cosmx_morphology_coords(images_dir)

        if self.align_rasters_to_polygons and self.dataset.polygons_file is not None:
            self._get_polygons()
        ox, oy = self._ensure_origin()

        # Morphology TIFFs are stored y-inverted relative to the FOV-grid
        # placement, so they need a vertical flip to co-register with
        # transcripts/labels.  This must NOT depend on whether polygons were
        # read — gating it on ``align_rasters_to_polygons`` left px-only images
        # mirrored when read without polygons (issue #42).  ``flip_image``
        # defaults to True (a user may pass ``flip_image=False`` for a dataset
        # stored the other way).
        per_fov_flip = self.flip_image

        protein_dirs: dict[int, Path] = {}
        if read_proteins and self.dataset.analysis_results_dir is not None:
            protein_dirs = {
                int(p.parent.name[3:]): p for p in self.dataset.analysis_results_dir.rglob("**/FOV*/ProteinImages")
            }

        transform = _translation_transform(ox, oy)

        # Single FOV — no stitching
        if self.fovs and len(self.fovs) == 1:
            fov = next(iter(self.fovs))
            try:
                fov_path = _find_matching_fov_file(images_dir, fov)
            except FileNotFoundError:
                logger.warning("No image file found for FOV %d — skipping images.", fov)
                return {}
            image, c_coords = _read_fov_image(
                fov_path,
                protein_dirs.get(fov),
                morphology_coords,
                selected_channels=channels,
                **imread_kwargs,
            )
            if per_fov_flip:
                image = image[:, ::-1, :]
            return {
                _element_name("image", self.fovs): self._normalize_and_parse_image(
                    image, c_coords, transform, image_models_kwargs
                )
            }

        # Multi-FOV — stitch
        fovs_filter = self.fovs if self.fovs else None
        sel_locs = self.fov_locs.loc[sorted(self.fovs)] if self.fovs else self.fov_locs
        prot_subset = {k: v for k, v in protein_dirs.items() if k in self.fovs} if self.fovs else protein_dirs

        # Tighten the image canvas to image-bearing FOVs (issue #37), but only
        # when the base FOV origin governs; when polygons drive the origin keep
        # it so the image stays co-registered with transcripts/labels.
        tighten = not (
            self.dataset.polygons_file is not None and (self.align_rasters_to_polygons or self.polygons_as_labels)
        )
        image, c_coords, used_ox, used_oy = _read_stitched_image(
            images_dir,
            sel_locs,
            prot_subset,
            morphology_coords,
            flip_image=per_fov_flip,
            fovs_filter=fovs_filter,
            selected_channels=channels,
            n_workers=self.n_workers,
            tighten_to_seen=tighten,
            **imread_kwargs,
        )
        if image is None:
            return {}
        img_transform = _translation_transform(used_ox, used_oy) if tighten else transform
        name = "stitched_image" if not self.fovs else _element_name("image", self.fovs)
        return {name: self._normalize_and_parse_image(image, c_coords, img_transform, image_models_kwargs)}

    def _normalize_and_parse_image(
        self,
        image: da.Array,
        c_coords: list[str],
        transform: Any,
        image_models_kwargs: dict[str, Any],
    ) -> Image2DModel:
        """Parse the assembled image, optionally applying a per-channel percentile stretch.

        Post-stitch; the reversible per-channel divisors are recorded in the element's
        ``attrs`` for promotion into ``sdata.attrs`` at assembly.
        """
        pct = self.image_normalization_percentile
        image, scales = _normalize_image_channels(image, c_coords, percentile=pct, n_workers=self.n_workers)
        stretched = any(div != 1.0 for div in scales.values())  # all-1.0 == nothing applied
        if stretched and not self._image_norm_logged:
            logger.info(
                "Applying per-channel %.4g-th-percentile image normalization (scale-only, reversible via attrs).",
                pct,
            )
            self._image_norm_logged = True
        parsed = Image2DModel.parse(
            image,
            dims=("c", "y", "x"),
            c_coords=c_coords,
            transformations={"global": transform},
            **image_models_kwargs,
        )
        if stretched:
            parsed.attrs["cosmx_image_normalization"] = {"percentile": pct, "channel_scales": scales}
        return parsed

    # -- Read transcripts --

    def read_transcripts(self) -> dict[str, PointsModel]:
        """Read transcript coordinates into a PointsModel."""
        if self.dataset.tx_file is None:
            return {}
        src = self.dataset.tx_file
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Transcript file not found: {src}")

        is_big = _maybe_warn_big_file(src)

        if is_big:
            return self._read_transcripts_parquet(src)
        return self._read_transcripts_csv(src)

    def _place_transcripts(self, df):
        """Map transcripts into the stitched FOV grid.

        Transcripts and polygons share the same per-FOV local coordinate
        system, so we place transcript points with the exact mapping used
        for polygon vertices (``fov_locs`` origin + ``flip_y``).  This
        co-registers transcripts with polygons/images/labels by construction
        on every dataset, regardless of flip or any global-frame offset.

        Falls back to the raw ``x_global_px`` / ``y_global_px`` columns when
        the per-FOV local columns are not available.
        """
        if {"x_local_px", "y_local_px", "fov"}.issubset(df.columns):
            return place_local_in_fov_grid(df, self.fov_locs)
        logger.warning(
            "[transcripts] No local px columns found — using raw global "
            "coordinates, which may not align with images/polygons."
        )
        return df

    def _read_transcripts_parquet(self, src: Path) -> dict[str, PointsModel]:
        """Read large transcripts via Parquet cache + Dask."""
        pq_file = _parquet_cache_for_tx(src, self.dataset.dataset_id, n_workers=self.n_workers)
        logger.info("[transcripts] Reading from Parquet cache...")
        df = dd.read_parquet(pq_file, engine="pyarrow")

        if self.fovs:
            df = df[df["fov"].isin(list(self.fovs))]

        df = self._place_transcripts(df)

        if {"fov", "cell_ID"}.issubset(set(df.columns)):
            raw_max = df["cell_ID"].max().compute()
            if pd.isna(raw_max):
                logger.warning("[transcripts] No transcripts found for the requested FOVs — skipping.")
                return {}
            current_max = int(raw_max)
            base = self._lock_max_cell_id(current_max)
            df = df.assign(global_cell_id=df["fov"] * base * (df["cell_ID"] > 0) + df["cell_ID"])

        df = _dask_categoricals_to_string(df)
        name, coord_map = self._transcript_naming(df)

        if isinstance(df, dd.DataFrame):
            # Apply coord transforms for Dask
            if "x" not in df.columns:
                ox, oy = self._ensure_origin()
                df = df.assign(x=df["x_global_px"] - ox, y=df["y_global_px"] - oy)

        ox, oy = self._ensure_origin()
        return {
            name: PointsModel.parse(
                df,
                coordinates=coord_map,
                feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    def _read_transcripts_csv(self, src: Path) -> dict[str, PointsModel]:
        """Read small transcripts directly from CSV."""
        logger.info("[transcripts] Reading %s into memory...", src.name)
        df = _read_transcripts_csv(src, self.dataset.dataset_id)
        if self.fovs:
            df = df[df["fov"].isin(list(self.fovs))]
        if df.empty:
            logger.warning("[transcripts] No transcripts found for the requested FOVs — skipping.")
            return {}

        df = self._place_transcripts(df)

        df["global_cell_id"] = self.global_cell_id(df)

        name, coord_map = self._transcript_naming(df)
        _pandas_categoricals_to_string(df)

        # Convert to Dask before PointsModel.parse to avoid a partition
        # alignment error in spatialdata when it independently converts the
        # feature column and the coordinate columns to separate Dask objects.
        ox, oy = self._ensure_origin()
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // 2_000_000))
        return {
            name: PointsModel.parse(
                ddf,
                coordinates=coord_map,
                feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    def _transcript_naming(self, df) -> tuple[str, dict[str, str]]:
        """Determine element name and coordinate mapping for transcripts.

        Coordinates are shifted to the shared 0-based origin used by
        images/labels/shapes so all elements align in the ``global``
        coordinate system.  (For Dask frames the shift is applied by the
        caller, since ``x`` may not yet exist here.)
        """
        ox, oy = self._ensure_origin()

        if not isinstance(df, dd.DataFrame):
            df["x"] = (df["x_global_px"] - ox).astype(float)
            df["y"] = (df["y_global_px"] - oy).astype(float)

        return _element_name("points", self.fovs), {"x": "x", "y": "y"}

    # -- Read shapes (polygons) --

    def read_shapes(self) -> dict[str, ShapesModel]:
        """Read cell polygons as vector shapes."""
        if self.dataset.polygons_file is None:
            return {}
        if self.polygons_as_labels and not self.keep_polygons_after_rasterize:
            return {}

        poly_df = self._get_polygons()
        ox, oy = self._origin_x, self._origin_y
        gdf = gpd.GeoDataFrame(poly_df.copy(), geometry="geometry").set_index("global_cell_id", drop=False)

        return {
            _element_name("cells_polygons", self.fovs): ShapesModel.parse(
                gdf,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    # -- Build FOV box shapes --

    def build_fov_shapes(self) -> dict[str, ShapesModel]:
        """Build one square per FOV as shape elements."""
        if self.fov_locs is None or self.fov_locs.empty:
            return {}
        ox, oy = self._ensure_origin()

        locs = self.fov_locs.loc[sorted(self.fovs)].copy() if self.fovs else self.fov_locs.copy()

        geoms, fov_vals, skipped = [], [], []
        for fov, row in locs.iterrows():
            xmin = float(row.get("xmin", math.nan))
            ymin = float(row.get("ymin", math.nan))
            if not math.isfinite(xmin) or not math.isfinite(ymin):
                skipped.append(int(fov))
                continue
            x0, y0 = xmin - ox, ymin - oy
            geoms.append(sgeom.box(x0, y0, x0 + COSMX_FOV_SIZE_PX, y0 + COSMX_FOV_SIZE_PX))
            fov_vals.append(int(fov))

        if skipped:
            logger.warning("Skipped %d FOV(s) with NaN coords: %s", len(skipped), skipped)
        if not geoms:
            return {}

        gdf = gpd.GeoDataFrame({"fov": fov_vals, "geometry": geoms}, geometry="geometry")
        return {
            _element_name("fov_boxes", self.fovs): ShapesModel.parse(
                gdf,
                transformations={"global": _translation_transform(ox, oy)},
            )
        }

    # -- Read labels --

    def read_labels(
        self,
        *,
        allowed_global_ids: set[int] | None = None,
    ) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Read or rasterize cell labels.

        Tries in order:
        1. Polygon rasterization (if ``polygons_as_labels`` and polygons exist)
        2. CellLabels directory (pre-computed TIFFs)
        3. Legacy CellStatsDir layout

        Returns ``(labels_dict, set_of_label_ids_present)``.
        """
        # --- Path 1: polygon → raster ---
        if self.polygons_as_labels and self.dataset.polygons_file is not None:
            return self._labels_from_polygons(allowed_global_ids)

        # --- Path 2: flat CellLabels directory ---
        if self.dataset.cell_labels_dir is not None:
            return self._labels_from_cell_labels_dir()

        # --- Path 3: legacy CellStatsDir ---
        if self.dataset.cell_stats_dir is not None:
            return self._labels_from_cell_stats_dir()

        return {}, set()

    def _labels_from_polygons(
        self,
        allowed_global_ids: set[int] | None,
    ) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Rasterize polygons into a label image."""
        poly_df = self._get_polygons()
        ox, oy = self._origin_x, self._origin_y

        if allowed_global_ids is not None:
            before = len(poly_df)
            poly_df = poly_df[poly_df["global_cell_id"].isin(allowed_global_ids)].reset_index(drop=True)
            dropped = before - len(poly_df)
            if dropped:
                logger.info("Labels: removed %d polygon(s) without a matching table row.", dropped)
            self._poly_df = poly_df

        if poly_df.empty:
            logger.warning("No polygons left after alignment. Skipping labels.")
            return {}, set()

        canvas_min_x, canvas_min_y, canvas_w, canvas_h = _canvas_from_fov_locs_for_polygons(
            self.fov_locs,
            origin_x=ox,
            origin_y=oy,
            fovs=self.fovs,
        )

        raster = _polygons_to_label_raster(
            poly_df,
            chunks=(2048, 2048),
            n_jobs=self.n_workers,
            canvas_min_x=canvas_min_x,
            canvas_min_y=canvas_min_y,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )
        raster.attrs.pop("transform", None)

        present_ids = {int(x) for x in da.unique(raster.data).compute() if x != 0}
        name = _element_name("polygons_labels", self.fovs)
        lbl = Labels2DModel.parse(raster, dims=("y", "x"), transformations={"global": _translation_transform(ox, oy)})
        return {name: lbl}, present_ids

    def _labels_from_cell_labels_dir(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Stitch pre-computed CellLabels TIFFs."""
        # The per-FOV tile flip is derived from the ``flip_y`` column inside the
        # stitcher (it is the inverse of the transcript-placement flip). We pass
        # only ``self.flip_image`` here as the fallback used when no ``flip_y``
        # column is present (labels read without transcripts). Folding ``flip_y``
        # into ``flip_image`` here would double-apply it and mirror the labels.
        fov_local_to_global = self._build_cell_label_luts()

        # When no cell_info.csv is available, build LUTs from the TIFFs
        # themselves using the same global_cell_id formula as read_tables().
        # Without this, labels keep local cell IDs that collide across FOVs
        # and don't match the global IDs in the expression table.
        if fov_local_to_global is None and self.dataset.cell_labels_dir is not None:
            fov_local_to_global = self._build_cell_label_luts_from_tiffs()

        stitched, present_ids, used_fov_locs = _read_stitched_cell_labels_from_dir(
            self.dataset.cell_labels_dir,
            self.fov_locs,
            fovs=self.fovs,
            flip_image=self.flip_image,
            n_workers=self.n_workers,
            fov_local_to_global=fov_local_to_global,
        )

        if stitched is None:
            return {}, set()

        # Update reader state to match the FOVs we actually stitched
        ox, oy = float(used_fov_locs["xmin"].min()), float(used_fov_locs["ymin"].min())
        self.fov_locs = used_fov_locs
        self._base_origin_x = ox
        self._base_origin_y = oy
        self._origin_x = ox
        self._origin_y = oy

        name = _element_name("cell_labels", self.fovs)
        lbl = Labels2DModel.parse(stitched, dims=("y", "x"), transformations={"global": _translation_transform(ox, oy)})
        return {name: lbl}, present_ids

    def _labels_from_cell_stats_dir(self) -> tuple[dict[str, Labels2DModel], set[int]]:
        """Legacy label stitching from CellStatsDir."""
        ox, oy = self._ensure_origin()
        stitched, df = stitch_segmentation_label_image(
            path=self.dataset.path,
            fov_position_file=str(self.dataset.fov_positions_file),
            cell_info_file=str(self.dataset.cell_stats_dir / f"{self.dataset.dataset_id}_cell_info.csv"),
            dataset_id=self.dataset.dataset_id,
            flip_image=self.flip_image,
            n_workers=self.n_workers,
            fovs=self.fovs if self.fovs else None,
        )
        present_ids = {int(x) for x in df["global_cell_id"].to_numpy() if x != 0}
        lbl = Labels2DModel.parse(stitched, dims=("y", "x"), transformations={"global": _translation_transform(ox, oy)})
        return {"cell_labels_from_cellstats": lbl}, present_ids

    def _build_cell_label_luts(self) -> dict[int, np.ndarray] | None:
        """Build local→global cell ID LUTs for CellLabels stitching."""
        if self.dataset.cell_stats_dir is None or self.dataset.dataset_id is None:
            return None
        info_path = self.dataset.cell_stats_dir / f"{self.dataset.dataset_id}_cell_info.csv"
        if not info_path.exists():
            return None
        cell_info = pd.read_csv(info_path)
        luts: dict[int, np.ndarray] = {}
        for fov in sorted(cell_info["fov"].unique()):
            local_ids = np.unique(cell_info.loc[cell_info["fov"] == fov, "cellID"].astype(int))
            if local_ids.size == 0:
                continue
            lut = np.zeros(int(local_ids.max()) + 1, dtype=int)
            df_f = cell_info.loc[cell_info["fov"] == fov]
            for _, row in df_f.iterrows():
                lut[int(row["cellID"])] = int(row["global_cell_id"])
            luts[fov] = lut
        return luts

    def _build_cell_label_luts_from_tiffs(self) -> dict[int, np.ndarray] | None:
        """Build local→global cell ID LUTs by scanning CellLabels TIFFs.

        Used when no ``cell_info.csv`` is available. Scans each TIFF for
        its maximum local cell ID, then builds per-FOV look-up tables
        using the same ``global_cell_id`` formula as :meth:`global_cell_id`
        so that label IDs match the table's global_cell_id values.
        """
        from ._utils import find_cell_label_tifs

        cell_labels_dir = self.dataset.cell_labels_dir
        if cell_labels_dir is None:
            return None

        fov_tif_map = find_cell_label_tifs(cell_labels_dir)

        # Pass 1: find the max local cell ID across all relevant FOVs.
        fov_max_ids: dict[int, int] = {}
        for fov, img_path in fov_tif_map.items():
            if self.fovs is not None and fov not in self.fovs:
                continue
            if fov not in self.fov_locs.index:
                continue
            arr = tifffile.imread(img_path)
            fov_max_ids[fov] = int(arr.max())

        if not fov_max_ids:
            return None

        global_max_local = max(fov_max_ids.values())

        # Lock max_cell_id if not yet locked; otherwise use the existing
        # locked value so that the LUTs stay consistent with read_tables().
        if self.max_cell_id is None:
            self.max_cell_id = global_max_local
        # If already locked to a smaller value, warn but keep the locked
        # value — labels with IDs beyond the locked max will map to IDs
        # that have no matching table row (acceptable: those cells simply
        # aren't in the expression matrix).
        if global_max_local > self.max_cell_id:
            logger.warning(
                "CellLabels contain local cell IDs up to %d but "
                "max_cell_id is locked to %d. Some label IDs may not "
                "match table rows.",
                global_max_local,
                self.max_cell_id,
            )

        base = self.max_cell_id + 1

        # Pass 2: build per-FOV LUTs.
        luts: dict[int, np.ndarray] = {}
        for fov, max_local in fov_max_ids.items():
            lut = np.zeros(max_local + 1, dtype=np.int64)
            for local_id in range(1, max_local + 1):
                lut[local_id] = fov * base + local_id
            # cell_ID == 0 stays 0 (background)
            luts[fov] = lut

        return luts

    # -- Read tables (expression + metadata) --

    def read_tables(self, regions: list[str], *, modality: str | None = None) -> dict[str, AnnData]:
        """Read expression matrix and metadata into an AnnData table."""
        if self.dataset.exprMat_file is None:
            return {}

        adata, bg_df = _read_expr_mat_polars(
            self.dataset.exprMat_file,
            n_rows=None,
            fovs=self.fovs if self.fovs else None,
        )

        if adata.n_obs == 0:
            label = modality or "expression"
            logger.warning("[tables] No %s data for the requested FOVs — skipping.", label)
            return {}

        if self.dataset.metadata_file is not None:
            meta = _read_metadata_polars(self.dataset.metadata_file, n_rows=None)
            common = adata.obs.index.intersection(meta.index)
            adata = adata[common].copy()
            adata.obs = meta.loc[common]

        # Compute global cell IDs
        if "fov" in adata.obs.columns and "cell_ID" in adata.obs.columns:
            tmp = adata.obs.reset_index(drop=False)
            tmp["global_cell_id"] = self.global_cell_id(tmp)
            adata.obs["global_cell_id"] = tmp.set_index("fov_cellID").loc[adata.obs.index, "global_cell_id"]
        else:
            adata.obs["global_cell_id"] = np.arange(1, adata.n_obs + 1, dtype=int)

        # Determine region
        region = regions[0] if regions else "cells"
        if (
            region == "cells"
            and self.polygons_as_labels
            and self.dataset.polygons_file is not None
            and not self.keep_polygons_after_rasterize
        ):
            region = _element_name("polygons_labels", self.fovs)

        adata.obs["region_key"] = pd.Series(region, index=adata.obs.index, dtype="category")

        # Sanitize column names for zarr compatibility
        obs = adata.obs.copy()
        seen: dict[str, str] = {}
        to_drop = []
        for col in obs.columns:
            key = col.lower()
            if key in seen:
                to_drop.append(col)
            else:
                seen[key] = col
        if to_drop:
            obs = obs.drop(columns=to_drop)

        preserve = {"global_cell_id", "region_key"}
        rename_map: dict[str, str] = {}
        taken: set[str] = set(obs.columns)
        for col in list(obs.columns):
            if col in preserve:
                continue
            new = re.sub(r"[^0-9A-Za-z_]", "_", col.strip())
            if not re.match(r"[A-Za-z_]", new):
                new = f"col_{new}"
            base, i = new, 1
            while new in taken and new != col:
                i += 1
                new = f"{base}_{i}"
            if new != col:
                rename_map[col] = new
                taken.add(new)
        if rename_map:
            obs = obs.rename(columns=rename_map)

        _pandas_categoricals_to_string(obs)
        adata.obs = obs

        if bg_df is not None:
            adata.uns["fov_bg_signal"] = bg_df

        table = TableModel.parse(
            adata,
            region=region,
            region_key="region_key",
            instance_key="global_cell_id",
            overwrite_metadata=True,
        )
        return {"cell_data": table}


# ---------------------------------------------------------------------------
# Shared read orchestration (single- and multimodal paths)
# ---------------------------------------------------------------------------


def _read_images_and_shapes(
    reader: CosMxDatasetReader,
    *,
    read_images: bool,
    read_proteins: bool,
    read_polygons: bool,
    image_models_kwargs: dict[str, Any] | None,
    imread_kwargs: dict[str, Any] | None,
    channels: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read morphology/protein images and polygon shapes (gate shared by both paths)."""
    images = (
        reader.read_images(
            read_proteins=read_proteins,
            image_models_kwargs=image_models_kwargs,
            imread_kwargs=imread_kwargs,
            channels=channels,
        )
        if read_images and reader.dataset.morphology_2d_dir is not None
        else {}
    )
    shapes = reader.read_shapes() if read_polygons else {}
    return images, shapes


# ---------------------------------------------------------------------------
# Multi-modal orchestrator
# ---------------------------------------------------------------------------


def _cosmx_multi(
    dataset: CosMxDataset,
    *,
    read_images: bool,
    read_labels: bool,
    read_proteins: bool,
    read_transcripts: bool,
    read_polygons: bool,
    read_gexp: bool,
    fovs: list[int] | int | None,
    channels: list[str] | None,
    n_workers: int,
    flip_image: bool | None,
    image_normalization_percentile: float | None,
    image_models_kwargs: dict[str, Any] | None,
    imread_kwargs: dict[str, Any] | None,
    polygons_as_labels: bool,
    keep_polygons_after_rasterize: bool,
    align_rasters_to_polygons: bool | None,
    add_fovs_as_shapes: bool,
    skip_empty_fovs: bool,
    preview_fovs: bool,
) -> SpatialData | None:
    """Read a multimodal CosMx dataset (e.g. RNA + Protein)."""
    modalities = dataset.modalities
    if not modalities:
        raise ValueError("Expected modalities for multi-modal dataset.")

    fov_set = _normalize_fovs(fovs)
    image_models_kwargs, imread_kwargs = _default_image_kwargs(image_models_kwargs, imread_kwargs)

    # Pick modality for shared spatial elements (prefer one with polygons/labels)
    label_ds = next(
        (ds for ds in modalities.values() if ds.polygons_file is not None or ds.cell_labels_dir is not None),
        next(iter(modalities.values())),
    )
    if label_ds.fov_positions_file is None:
        raise ValueError("FOV positions file required for multimodal runs.")

    fov_locs = _read_fov_locs(label_ds.fov_positions_file, fovs=sorted(fov_set) if fov_set else None)

    if skip_empty_fovs and (read_images or read_labels):
        fov_locs = _prune_empty_fovs(dataset, fov_locs, fov_set)

    if preview_fovs:
        _plot_fov_preview(fov_locs, fov_set)
        return None

    n_fovs = len(fov_set) if fov_set else len(fov_locs)
    n_mods = len(modalities)
    logger.info(
        "Reading multimodal CosMx dataset (%d modalities, %d FOV(s)). "
        "Source files are gzip-compressed CSVs that must be fully decompressed "
        "even for a FOV subset — this may take a while on first load.",
        n_mods,
        n_fovs,
    )

    # Shared reader for spatial elements
    logger.debug("[1/5] Setting up reader and loading polygons...")
    reader = CosMxDatasetReader(
        label_ds,
        fovs=fov_set,
        n_workers=n_workers,
        flip_image=flip_image,
        polygons_as_labels=polygons_as_labels,
        fov_locs=fov_locs,
        align_rasters_to_polygons=align_rasters_to_polygons,
        keep_polygons_after_rasterize=keep_polygons_after_rasterize,
        image_normalization_percentile=image_normalization_percentile,
    )

    # Pre-load polygons to freeze max_cell_id
    base_max_cell_id: int | None = None
    try:
        reader._get_polygons()
        base_max_cell_id = reader.max_cell_id
    except Exception as e:
        logger.debug("Could not pre-load polygons to freeze max_cell_id: %s", e)

    def _modality_reader(ds: CosMxDataset) -> CosMxDatasetReader:
        """Secondary reader (transcripts/tables) sharing the frozen max_cell_id."""
        r = CosMxDatasetReader(
            ds,
            fovs=fov_set,
            n_workers=n_workers,
            flip_image=flip_image,
            polygons_as_labels=False,
            fov_locs=fov_locs,
            align_rasters_to_polygons=align_rasters_to_polygons,
        )
        if base_max_cell_id is not None:
            r.max_cell_id = base_max_cell_id
        return r

    logger.debug("[2/5] Reading images and labels...")
    images, shapes = _read_images_and_shapes(
        reader,
        read_images=read_images,
        read_proteins=read_proteins,
        read_polygons=read_polygons,
        image_models_kwargs=image_models_kwargs,
        imread_kwargs=imread_kwargs,
        channels=channels,
    )
    labels, label_ids = reader.read_labels() if read_labels else ({}, set())

    if add_fovs_as_shapes:
        shapes = {**shapes, **reader.build_fov_shapes()}

    # Transcripts (from whichever modality has tx_file)
    logger.debug("[3/5] Reading transcripts...")
    points: dict[str, PointsModel] = {}
    tx_mod = next((name for name, ds in modalities.items() if ds.tx_file is not None), None)
    if read_transcripts and tx_mod is not None:
        tx_points = _modality_reader(modalities[tx_mod]).read_transcripts()
        if tx_points:
            points = {"transcripts": next(iter(tx_points.values()))}

    # Table/label alignment — the multimodal path is LABEL-ANCHORED: labels are read
    # first (above), then each modality's table is filtered to the label IDs. The
    # single-modality path in cosmx() is table-anchored instead; keep them distinct.
    tables: dict[str, AnnData] = {}
    region_refs = list(labels.keys()) or list(shapes.keys()) or ["cells"]
    if read_gexp:
        for mod_name, mod_ds in modalities.items():
            logger.debug("[4/5] Reading %s data...", mod_name)
            mod_tables = _modality_reader(mod_ds).read_tables(region_refs, modality=mod_name)
            if label_ids:
                mod_tables = _filter_tables_to_ids(mod_tables, label_ids, source_label="rasterised labels")
            if mod_tables:
                tables[mod_name] = next(iter(mod_tables.values()))

    logger.debug("[5/5] Assembling SpatialData object...")
    return _assemble_sdata(images, points, labels, shapes, tables)


# ---------------------------------------------------------------------------
# SpatialData assembly
# ---------------------------------------------------------------------------


def _assemble_sdata(
    images: dict[str, Any],
    points: dict[str, Any],
    labels: dict[str, Any],
    shapes: dict[str, Any],
    tables: dict[str, Any],
) -> SpatialData:
    """Build and annotate the final SpatialData object."""
    # Move each image's per-channel normalization divisors from the (non-persisted) element
    # attrs into the persisted SpatialData attrs, so the reversible stretch survives a
    # write/read round-trip. Popped from the element so the metadata has a single home.
    attrs: dict[str, Any] = {}
    image_norm = {
        nm: im.attrs.pop("cosmx_image_normalization")
        for nm, im in images.items()
        if "cosmx_image_normalization" in getattr(im, "attrs", {})
    }
    if image_norm:
        attrs["cosmx_image_normalization"] = image_norm

    sdata = SpatialData(
        images=images,
        points=points,
        labels=labels,
        shapes=shapes,
        tables=tables,
        attrs=attrs,
    )

    # Wire tables to label elements
    if labels and sdata.tables:
        main_label = next(iter(labels))
        for table_name, table in list(sdata.tables.items()):
            raw = table.table if hasattr(table, "table") else table
            if "region_key" not in raw.obs:
                continue
            reg_vals = pd.Series(raw.obs["region_key"]).astype("string").unique()
            if len(reg_vals) == 1 and reg_vals[0] == main_label:
                sdata.set_table_annotates_spatialelement(table_name, region=main_label)
                continue
            raw.uns.pop("spatialdata_attrs", None)
            raw.obs["region_key"] = pd.Series(main_label, index=raw.obs.index, dtype="category")
            sdata.tables[table_name] = TableModel.parse(
                raw,
                region=main_label,
                region_key="region_key",
                instance_key="global_cell_id",
                overwrite_metadata=True,
            )
            sdata.set_table_annotates_spatialelement(table_name, region=main_label)

    return sdata


# ---------------------------------------------------------------------------
# FOV normalization
# ---------------------------------------------------------------------------


def _normalize_fovs(fovs: list[int | str] | int | str | None) -> set[int] | None:
    """Convert the user-facing ``fovs`` argument to an optional set of ints."""
    if fovs is None:
        return None
    if isinstance(fovs, int | str):
        return {int(fovs)}
    return {int(f) for f in fovs}


def _present_fovs(dataset: CosMxDataset) -> set[int]:
    """FOV ids with per-FOV data on disk, unioned over the dataset + modalities (#37)."""
    sources = [dataset, *(dataset.modalities.values() if dataset.modalities else [])]
    present: set[int] = set()
    for s in sources:
        present |= detect_fovs_with_data(
            morphology_2d_dir=s.morphology_2d_dir,
            cell_labels_dir=s.cell_labels_dir,
            cell_stats_dir=s.cell_stats_dir,
            analysis_results_dir=s.analysis_results_dir,
        )
    return present


def _prune_empty_fovs(
    dataset: CosMxDataset,
    fov_locs: pd.DataFrame | None,
    fov_set: set[int] | None,
) -> pd.DataFrame | None:
    """Drop phantom FOVs (listed in positions, no data files) from *fov_locs* (#37).

    Pruned before any element is read so origin, canvases and FOV boxes stay
    consistent. Never drops an explicitly requested FOV; keeps every FOV when
    detection finds nothing (transcript-/gexp-only datasets).
    """
    if fov_locs is None or fov_locs.empty:
        return fov_locs
    present = _present_fovs(dataset)
    if not present:
        return fov_locs
    listed = {int(f) for f in fov_locs.index}
    phantom = listed - present
    if fov_set is not None:
        phantom -= fov_set
    if not phantom:
        return fov_locs
    logger.info("skip_empty_fovs: dropping %d FOV(s) with no data files: %s", len(phantom), sorted(phantom))
    return fov_locs.loc[sorted(listed - phantom)]


def _prescan_max_cell_id(
    reader: CosMxDatasetReader,
    dataset: CosMxDataset,
    fov_set: set[int] | None,
) -> None:
    """Pre-scan cell_ID column from all available files to lock max_cell_id.

    This removes the load-order dependency: without pre-scanning,
    whichever element calls ``global_cell_id()`` first locks the max,
    and subsequent elements with a higher max will crash.
    """
    import polars as pl

    max_ids: list[int] = []
    for path in [dataset.polygons_file, dataset.exprMat_file, dataset.metadata_file]:
        if path is None or not path.exists():
            continue
        try:
            lf = pl.scan_csv(path, n_rows=None)
            if fov_set is not None and "fov" in lf.collect_schema().names():
                lf = lf.filter(pl.col("fov").is_in(list(fov_set)))
            val = lf.select(pl.col("cell_ID").max()).collect().item()
            if val is not None:
                max_ids.append(int(val))
        except Exception:
            continue

    if max_ids:
        reader.max_cell_id = max(max_ids)
        logger.debug("Pre-scanned max_cell_id=%d from %d file(s).", reader.max_cell_id, len(max_ids))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: str | None = None,
    *,
    read_images: bool = True,
    read_labels: bool = True,
    read_proteins: bool = True,
    read_transcripts: bool = True,
    read_polygons: bool = True,
    read_gexp: bool = True,
    fovs: list[int | str] | int | str | None = None,
    channels: list[str] | None = None,
    n_workers: int | None = None,
    flip_image: bool | None = None,
    image_normalization_percentile: float | None = None,
    image_models_kwargs: dict[str, Any] | None = None,
    imread_kwargs: dict[str, Any] | None = None,
    polygons_as_labels: bool = True,
    keep_polygons_after_rasterize: bool = False,
    align_rasters_to_polygons: bool | None = None,
    add_fovs_as_shapes: bool = True,
    skip_empty_fovs: bool = True,
    preview_fovs: bool = False,
) -> SpatialData | None:
    """Read *CosMx Nanostring* data into a :class:`spatialdata.SpatialData` object.

    Supports all known CosMx export formats: flat CSV files, nested
    CellStatsDir layouts, multimodal RNA+Protein runs, and both old-style
    (px-only / mm-only) and new-style (px+mm) FOV positions files.

    Files are recognized by their standard CosMx suffixes — counts
    ``{cx.COUNTS_SUFFIX!r}``, metadata ``{cx.METADATA_SUFFIX!r}``, transcripts
    ``{cx.TRANSCRIPTS_SUFFIX!r}``, and FOV positions ``{cx.FOV_SUFFIX!r}`` —
    typically prefixed with the dataset id.

    Parameters
    ----------
    path
        Path to the root directory containing CosMx files.
    dataset_id
        Optional dataset identifier.  Inferred from file prefixes if not given.
    read_images
        Whether to read morphology images.
    read_labels
        Whether to read or rasterize cell labels.
    read_proteins
        Whether to include protein image channels.
    read_transcripts
        Whether to read transcript coordinates.
    read_polygons
        Whether to read cell boundary polygons as shapes.
    read_gexp
        Whether to read the gene expression matrix.
    fovs
        Specific FOV(s) to read.  ``None`` reads all FOVs.
    channels
        Specific image channel names to include.
    n_workers
        Number of parallel workers for stitching operations.
    flip_image
        Flip morphology images vertically to co-register with transcripts and
        labels.  Defaults to ``True`` when ``None`` (CosMx morphology TIFFs are
        stored y-inverted relative to the FOV-grid placement); pass ``False`` for
        a dataset stored the other way.
    image_normalization_percentile
        Optional per-channel contrast normalization for morphology/protein images, applied
        post-stitching on top of the dtype-max scaling.  ``None`` (default) keeps the plain
        dtype-max ``[0, 1]`` image.  A float in ``[0, 100]`` (e.g. ``99.9``) divides each
        channel by that percentile of its non-zero pixels, recovering channels whose real
        signal sits far below the dtype ceiling and would otherwise render near-black
        (issue #38).  The stretch is scale-only (no clipping), so the brightest pixels may
        exceed ``1.0``; it is reversible via the divisors recorded in
        ``sdata.attrs['cosmx_image_normalization']`` — these are in dtype-max ``[0, 1]`` units
        (they reverse to the dtype-max image, not raw counts).  The percentile is approximate
        for large multi-chunk images, so the exact divisor may vary slightly with chunking.
    image_models_kwargs
        Extra kwargs for :class:`spatialdata.models.Image2DModel`.
    imread_kwargs
        Extra kwargs for :func:`dask_image.imread.imread`.
    polygons_as_labels
        Rasterize polygons into a label image.
    keep_polygons_after_rasterize
        Keep vector polygons even when ``polygons_as_labels=True``.
    align_rasters_to_polygons
        Anchor images and labels to the polygon coordinate origin.
    add_fovs_as_shapes
        Add FOV bounding boxes as shape elements.
    skip_empty_fovs
        Drop FOVs that appear in the positions file but have no data files on
        disk (issue #37).  CosMx positions files often list more FOVs than ship
        data; those phantom FOVs otherwise inflate the image canvas, desync the
        image and label canvases, and add empty FOV boxes.  Defaults to ``True``;
        FOVs are pruned only when detection finds a strict, non-empty subset with
        data (transcript-/gexp-only datasets are left untouched).  Pass ``False``
        to keep every listed FOV.
    preview_fovs
        Show a preview plot of FOV positions and return ``None``.

    Returns
    -------
    :class:`spatialdata.SpatialData` or ``None`` (if ``preview_fovs=True``).
    """
    n_workers = n_workers or 1
    path = Path(path).resolve()

    dataset = _set_up_cosmx_dataset_for_conversion(path=path, dataset_id=dataset_id)

    # Morphology TIFFs are stored y-inverted relative to the FOV-grid placement,
    # so they need a vertical flip to co-register with transcripts/labels.  When
    # the user does not specify, default to flipping (issue #42).  This replaces
    # an earlier heuristic that inferred the flip from the transcript coordinate
    # convention (``not flip_y``) — the wrong signal, which left px-only images
    # mirrored when read without polygons.  Resolve here, BEFORE the multimodal
    # dispatch, so both single- and multi-modal paths get the same default.
    if flip_image is None:
        flip_image = True
    flip_image = bool(flip_image)

    # --- Multimodal dispatch ---
    if dataset.modalities:
        return _cosmx_multi(
            dataset=dataset,
            read_images=read_images,
            read_labels=read_labels,
            read_proteins=read_proteins,
            read_transcripts=read_transcripts,
            read_polygons=read_polygons,
            read_gexp=read_gexp,
            fovs=fovs,
            channels=channels,
            n_workers=n_workers,
            flip_image=flip_image,
            image_normalization_percentile=image_normalization_percentile,
            image_models_kwargs=image_models_kwargs,
            imread_kwargs=imread_kwargs,
            polygons_as_labels=polygons_as_labels,
            keep_polygons_after_rasterize=keep_polygons_after_rasterize,
            align_rasters_to_polygons=align_rasters_to_polygons,
            add_fovs_as_shapes=add_fovs_as_shapes,
            skip_empty_fovs=skip_empty_fovs,
            preview_fovs=preview_fovs,
        )

    # --- Single-modality path ---
    logger.info(
        "Reading single-modality CosMx dataset (ID: %s). "
        "Gzip-compressed CSV sources must be fully decompressed even for a "
        "FOV subset — this may take a while on first load.",
        dataset.dataset_id,
    )

    fov_set = _normalize_fovs(fovs)
    image_models_kwargs, imread_kwargs = _default_image_kwargs(image_models_kwargs, imread_kwargs)

    fov_locs = _read_fov_locs(dataset.fov_positions_file) if dataset.fov_positions_file else None

    if skip_empty_fovs and (read_images or read_labels):
        fov_locs = _prune_empty_fovs(dataset, fov_locs, fov_set)

    if preview_fovs:
        if fov_locs is None:
            raise ValueError("preview_fovs=True but no FOV positions file found.")
        _plot_fov_preview(fov_locs, fov_set)
        return None

    if align_rasters_to_polygons is None:
        align_rasters_to_polygons = bool(read_polygons or (read_labels and polygons_as_labels))

    reader = CosMxDatasetReader(
        dataset,
        fovs=fov_set,
        n_workers=n_workers,
        flip_image=flip_image,
        polygons_as_labels=polygons_as_labels,
        fov_locs=fov_locs,
        align_rasters_to_polygons=align_rasters_to_polygons,
        keep_polygons_after_rasterize=keep_polygons_after_rasterize,
        image_normalization_percentile=image_normalization_percentile,
    )

    # Pre-scan cell_ID max across all available files to avoid
    # load-order dependency crashes in global_cell_id().
    _prescan_max_cell_id(reader, dataset, fov_set)

    # --- Read elements ---
    images, shapes = _read_images_and_shapes(
        reader,
        read_images=read_images,
        read_proteins=read_proteins,
        read_polygons=read_polygons,
        image_models_kwargs=image_models_kwargs,
        imread_kwargs=imread_kwargs,
        channels=channels,
    )
    points = reader.read_transcripts() if read_transcripts and dataset.tx_file is not None else {}

    # Table/label alignment — the single-modality path is TABLE-ANCHORED: read tables
    # first, then constrain labels to the table/polygon IDs. The multimodal path in
    # _cosmx_multi is label-anchored instead; keep the two strategies distinct.
    region_refs = list(shapes.keys()) + list(images.keys()) or ["cells"]
    tables = reader.read_tables(region_refs) if read_gexp else {}

    # Align tables ↔ polygons
    polygon_id_set: set[int] | None = None
    if reader.polygons_as_labels and reader.dataset.polygons_file is not None:
        poly_df = reader._get_polygons()
        polygon_ids = {int(v) for v in poly_df["global_cell_id"].to_numpy() if pd.notna(v) and int(v) != 0}
        tables = _filter_tables_to_ids(tables, polygon_ids, source_label="polygons")
        polygon_id_set = polygon_ids

    # Determine allowed label IDs
    table_ids = _collect_global_cell_ids_from_tables(tables)
    allowed_label_ids = (table_ids & polygon_id_set) if polygon_id_set else table_ids
    if not allowed_label_ids:
        allowed_label_ids = None

    labels, label_ids = reader.read_labels(allowed_global_ids=allowed_label_ids) if read_labels else ({}, set())

    if add_fovs_as_shapes:
        shapes = {**shapes, **reader.build_fov_shapes()}

    sdata = _assemble_sdata(images, points, labels, shapes, tables)

    # Final pass: drop table rows not in rasterised labels
    if label_ids:
        sdata.tables = _filter_tables_to_ids(dict(sdata.tables), label_ids, source_label="rasterised labels")

    return sdata
