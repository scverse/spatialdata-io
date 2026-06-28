"""Pure utilities for the CosMx reader — no domain logic."""

from __future__ import annotations

import re
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from spatialdata._logging import logger

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Dask parallelism helpers
# ---------------------------------------------------------------------------


@contextmanager
def dask_thread_pool(n_workers: int | None):
    """Temporarily cap dask parallelism to *n_workers* threads."""
    if n_workers is None or n_workers <= 0:
        yield
        return
    with ThreadPool(n_workers) as pool:
        with dask.config.set(scheduler="threads", pool=pool):
            yield


def compute_with_limit(*tasks: Any, n_workers: int | None = None) -> tuple[Any, ...]:
    """``dask.compute`` with an optional thread-pool cap."""
    if not tasks:
        return ()
    with dask_thread_pool(n_workers):
        return dask.compute(*tasks)


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------


def _to_float01_dtype_max(arr: da.Array) -> da.Array:
    """Normalize an array to float32 in [0, 1] by its dtype range (legacy CosMx behavior).

    Unsigned/signed integers are scaled by their dtype min/max; floating arrays are assumed
    already normalized and passed through unchanged (dtype/precision preserved).
    """
    dt = arr.dtype
    if np.issubdtype(dt, np.floating):
        return arr
    if np.issubdtype(dt, np.unsignedinteger):
        return (arr.astype("float32") / float(np.iinfo(dt).max)).astype("float32")
    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        return ((arr.astype("float32") - float(info.min)) / float(info.max - info.min)).astype("float32")
    return arr.astype("float32")


def _normalize_image_channels(
    arr: da.Array,
    channel_names: list[str] | None = None,
    *,
    percentile: float | None = None,
    n_workers: int | None = None,
) -> tuple[da.Array, dict[str, float]]:
    """Optionally apply a per-channel percentile contrast stretch to a ``(c, y, x)`` image.

    ``percentile=None`` returns *arr* unchanged (the IO layer already dtype-max normalized
    it to ``[0, 1]``).  A float such as ``99.9`` divides each channel by that percentile of
    its **non-zero, finite** pixels — non-zero so the zero padding between non-contiguous
    FOVs does not bias the estimate — recovering channels whose real signal sits far below
    the dtype ceiling and otherwise render near-black (issue #38).

    The stretch is scale-only (no clipping), hence exactly reversible via the returned
    *scales* (channel name -> divisor).  ``da.percentile`` is approximate for multi-chunk
    images, so a divisor is a close estimate.  A channel with no positive signal is left
    unscaled (divisor ``1.0``) with a warning.
    """
    if percentile is None:
        return arr.astype("float32"), {}

    is_2d = arr.ndim == 2
    channels = [arr] if is_2d else [arr[i] for i in range(arr.shape[0])]
    if channel_names is not None and len(channel_names) == len(channels):
        names = list(channel_names)
    else:
        names = [str(i) for i in range(len(channels))]

    scales: dict[str, float] = {}
    out: list[da.Array] = []
    for nm, ch in zip(names, channels, strict=False):
        flat = ch.ravel()
        sample = flat[(flat != 0) & da.isfinite(flat)]
        try:
            (p,) = compute_with_limit(da.percentile(sample, percentile), n_workers=n_workers)
            p = float(np.asarray(p).ravel()[0])
        except ValueError:  # no non-zero finite pixels
            p = float("nan")
        if np.isfinite(p) and p > 0.0:
            div = p
        else:
            logger.warning(
                "Image channel %r: no positive signal at the %.4g-th percentile; left unscaled.", nm, percentile
            )
            div = 1.0
        scales[nm] = div
        out.append(ch.astype("float32") / div)

    return (out[0] if is_2d else da.stack(out, axis=0)), scales


# ---------------------------------------------------------------------------
# Categorical / string helpers (zarr compatibility)
# ---------------------------------------------------------------------------


def _pandas_categoricals_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to string dtype (zarr cannot serialize categoricals)."""
    if df.empty:
        return df
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("")
    return df


def _dask_categoricals_to_string(df: dd.DataFrame) -> dd.DataFrame:
    """Partition-wise categorical → string conversion for dask DataFrames."""
    cat_cols = [
        col for col, dtype in df.dtypes.items() if isinstance(dtype, pd.CategoricalDtype) or str(dtype) == "category"
    ]
    if not cat_cols:
        return df

    def _to_str(pdf: pd.DataFrame) -> pd.DataFrame:
        for col in cat_cols:
            if col in pdf.columns:
                pdf[col] = pdf[col].astype("string").fillna("")
        return pdf

    return df.map_partitions(_to_str)


# ---------------------------------------------------------------------------
# CSV header matching for polygon files
# ---------------------------------------------------------------------------

# Canonical column names → list of regex patterns to try (case-insensitive).
_CANONICAL: dict[str, list[str]] = {
    "fov": [r"^fov$", r"^roi$", r"^field_?of_?view$", r"^fov_id$"],
    "cell_ID": [r"^cell[_ ]?id$", r"^cellid$", r"^cell$", r"^object_id$", r"^cell_identifier$"],
    "polygon_index": [
        r"^polygon_index$",
        r"^(poly|shape|segm)[-_ ]?index$",
        r"^(poly|shape|segm)[-_ ]?idx$",
        r"^object_index$",
        r"^segmentation_index$",
    ],
    "x": [r"^x_global_px$", r"^x_local_px$", r"^x[_ ]?px$", r"^x$"],
    "y": [r"^y_global_px$", r"^y_local_px$", r"^y[_ ]?px$", r"^y$"],
}


def _match_canonical(hdr: list[str], canon: str) -> str | None:
    """Return the raw column in *hdr* matching canonical name *canon*, else ``None``.

    Uses the same case-insensitive alias patterns as :func:`_match_header`, but for a
    single column and without requiring the full polygon schema. This is the shared
    entry point so every caller honours the *same* alias set (e.g. ``cell_ID`` also
    matching ``cellID``/``cell_id``/``object_id``) — a narrower ad-hoc match risks
    missing columns one path accepts and another silently drops.
    """
    hdr_stripped = [c.strip() for c in hdr]
    for pat in _CANONICAL.get(canon, []):
        hit = next(
            (
                orig
                for orig, stripped in zip(hdr, hdr_stripped, strict=False)
                if re.fullmatch(pat, stripped, flags=re.IGNORECASE)
            ),
            None,
        )
        if hit is not None:
            return hit
    return None


def _match_header(hdr: list[str]) -> dict[str, str]:
    """Map raw CSV column names to canonical names via regex matching.

    Returns ``{original_name: canonical_name}`` for each matched column.
    ``polygon_index`` is optional; ``fov``, ``cell_ID``, ``x``, ``y`` are required.

    Raises
    ------
    ValueError
        If required columns cannot be identified.
    """
    rename: dict[str, str] = {}

    for canon in _CANONICAL:
        hit = _match_canonical(hdr, canon)
        if hit is not None:
            rename[hit] = canon

    # Heuristic fallback for x/y — these vary the most across exports.
    def _first_like(coord: str) -> str | None:
        patt = re.compile(rf"^{coord}.*px$", flags=re.IGNORECASE)
        for orig in hdr:
            if patt.match(orig.strip()):
                return orig
        for orig in hdr:
            if orig.strip().lower() == coord:
                return orig
        return None

    if "x" not in rename.values():
        maybe_x = _first_like("x")
        if maybe_x is not None:
            rename[maybe_x] = "x"
    if "y" not in rename.values():
        maybe_y = _first_like("y")
        if maybe_y is not None:
            rename[maybe_y] = "y"

    required = {"fov", "cell_ID", "x", "y"}
    missing = required - set(rename.values())
    if missing:
        raise ValueError(f"Failed to identify required columns {sorted(missing)} in polygons header {hdr}")
    return rename


# ---------------------------------------------------------------------------
# CellLabels TIF discovery
# ---------------------------------------------------------------------------

_CELL_LABELS_RE = re.compile(r"CellLabels_F(\d+)\.tif$", re.IGNORECASE)


def find_cell_label_tifs(directory: Path) -> dict[int, Path]:
    """Find ``CellLabels_F*.tif`` files under *directory* (recursive).

    Returns a ``{fov_id: path}`` mapping.
    """
    result: dict[int, Path] = {}
    for p in directory.rglob("CellLabels_F*.[tT][iI][fF]"):
        m = _CELL_LABELS_RE.search(p.name)
        if m:
            result[int(m.group(1))] = p
    return result


# Canonical FOV-id patterns, shared with the stitchers so detection matches
# what the reader actually loads: morphology TIFFs (``*_F<digits>``) and protein
# FOV directories (``FOV<digits>``).
MORPH_FOV_RE = re.compile(r".*_F(\d+)", re.IGNORECASE)
FOV_DIR_RE = re.compile(r"FOV0*(\d+)$", re.IGNORECASE)


def detect_fovs_with_data(
    *,
    morphology_2d_dir: Path | None = None,
    cell_labels_dir: Path | None = None,
    cell_stats_dir: Path | None = None,
    analysis_results_dir: Path | None = None,
) -> set[int]:
    """FOV ids with at least one per-FOV data file on disk, by union (issue #37).

    Sources: morphology TIFFs, ``CellLabels`` TIFFs (incl. nested under a
    CellStatsDir), and protein ``FOV<id>/ProteinImages`` dirs. One listing per
    source. Transcripts/exprMat are not per-FOV, so an empty result means
    "cannot tell" — callers must not prune on it.
    """
    present: set[int] = set()
    if morphology_2d_dir and morphology_2d_dir.exists():
        present |= {int(m.group(1)) for p in morphology_2d_dir.glob("*.TIF") if (m := MORPH_FOV_RE.match(p.name))}
    if cell_labels_dir and cell_labels_dir.exists():
        present |= find_cell_label_tifs(cell_labels_dir).keys()
    if cell_stats_dir and cell_stats_dir.exists():
        present |= find_cell_label_tifs(cell_stats_dir).keys()
    if analysis_results_dir and analysis_results_dir.exists():
        present |= {
            int(m.group(1))
            for p in analysis_results_dir.rglob("FOV*/ProteinImages")
            if (m := FOV_DIR_RE.match(p.parent.name))
        }
    return present
