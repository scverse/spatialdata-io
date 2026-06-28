"""CSV / TIFF / Parquet I/O helpers for the CosMx reader."""

from __future__ import annotations

import csv
import gzip
import math
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import dask.array as da
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import scipy.sparse
import shapely.geometry as sgeom
import tifffile
from anndata import AnnData
from anndata.utils import make_index_unique
from dask_image.imread import imread
from spatialdata._logging import logger
from tqdm import tqdm

from ._utils import (
    _match_header,
    _pandas_categoricals_to_string,
    _to_float01_dtype_max,
)

if TYPE_CHECKING:
    import io
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COSMX_PIXEL_SIZE = 0.120280945
MM_TO_PX = 1000.0 / COSMX_PIXEL_SIZE
COSMX_FOV_SIZE_PX = 4256.0


# ---------------------------------------------------------------------------
# default image kwargs
# ---------------------------------------------------------------------------


def _default_image_kwargs(
    image_models_kwargs: dict[str, Any] | None = None,
    imread_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_models_kwargs = {} if image_models_kwargs is None else image_models_kwargs
    imread_kwargs = {} if imread_kwargs is None else imread_kwargs

    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (1, 1024, 1024)
    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    return image_models_kwargs, imread_kwargs


# ---------------------------------------------------------------------------
# FOV positions
# ---------------------------------------------------------------------------


def _read_fov_locs(
    csv_path: Path,
    *,
    fovs: list[int] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1) find FOV column
    cols_lower = {c.lower(): c for c in df.columns}
    fov_col: str | None = None
    for cand in ("fov", "fov_id", "fov_idx", "field_of_view", "roi", "order"):
        if cand in cols_lower:
            fov_col = cols_lower[cand]
            break
    if fov_col is None:
        raise ValueError(f"{csv_path.name}: cannot identify FOV column in {list(df.columns)!r}")

    # 2) detect orientation source BEFORE we synthesize px from mm
    raw_cols = [c.lower() for c in df.columns]
    file_has_px = any("px" in c for c in raw_cols)
    file_has_mm = any("mm" in c for c in raw_cols)
    needs_extra_flip = not (file_has_px and file_has_mm)

    # 3) try pixel columns first
    def _find_px(colnames: list[str]) -> tuple[str | None, str | None]:
        x_px_col = None
        y_px_col = None
        for c in colnames:
            lc = c.lower()
            if "px" in lc:
                if lc.startswith("x") or "x_" in lc:
                    x_px_col = c
                elif lc.startswith("y") or "y_" in lc:
                    y_px_col = c
        return x_px_col, y_px_col

    x_px_col, y_px_col = _find_px(list(df.columns))

    # 4) if no px -> look for mm and convert
    if x_px_col is None or y_px_col is None:
        x_mm_col = None
        y_mm_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in {"x_mm", "x_global_mm"} or lc.endswith("_x_mm"):
                x_mm_col = c
            elif lc in {"y_mm", "y_global_mm"} or lc.endswith("_y_mm"):
                y_mm_col = c
        if x_mm_col is None and "X_mm" in df.columns:
            x_mm_col = "X_mm"
        if y_mm_col is None and "Y_mm" in df.columns:
            y_mm_col = "Y_mm"

        if x_mm_col is None or y_mm_col is None:
            raise ValueError(
                f"{csv_path.name}: neither pixel nor mm coordinate columns found; got columns {list(df.columns)!r}"
            )

        df["x_global_px"] = df[x_mm_col].astype(float) * MM_TO_PX
        df["y_global_px"] = df[y_mm_col].astype(float) * MM_TO_PX
        x_px_col = "x_global_px"
        y_px_col = "y_global_px"

    # 5) normalize names
    fov_ser = pd.to_numeric(df[fov_col], errors="coerce").astype("Int64")
    if fov_ser.isna().any():
        raise ValueError(f"{csv_path.name}: FOV column {fov_col!r} contains NaNs / non-numeric values.")
    df["fov"] = fov_ser.astype(int)

    df = df.rename(
        columns={
            x_px_col: "xmin",
            y_px_col: "ymin",
        }
    )

    # 6) add width/height and the per-FOV flip flag
    df["xmax"] = df["xmin"].astype(float) + COSMX_FOV_SIZE_PX
    df["ymax"] = df["ymin"].astype(float) + COSMX_FOV_SIZE_PX
    df["flip_y"] = bool(needs_extra_flip)

    # 7) index + optional subset
    df = df.set_index("fov", verify_integrity=True).sort_index()

    if fovs is not None:
        fovs_sorted = sorted(int(f) for f in fovs)
        missing = set(fovs_sorted) - set(df.index)
        if missing:
            raise KeyError(f"Requested FOVs not found in positions file {csv_path.name}: {sorted(missing)}")
        df = df.loc[fovs_sorted]

    return df


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------


def _get_cosmx_morphology_coords(images_dir: Path) -> list[str]:
    images_paths = list(images_dir.glob("*.TIF"))
    if len(images_paths) == 0:
        raise FileNotFoundError(f"Expected to find images inside {images_dir}")

    with tifffile.TiffFile(images_paths[0]) as tif:
        description = tif.pages[0].description
        substrings = re.findall(r'"BiologicalTarget": "(.*?)",', description)
        channels = re.findall(r'"ChannelId": "(.*?)",', description)
        channel_order = list(re.findall(r'"ChannelOrder": "(.*?)",', description)[0])
        return [substrings[channels.index(x)] if x in channels else x for x in channel_order]


def _get_cosmx_protein_name(image_path: Path) -> str:
    with tifffile.TiffFile(image_path) as tif:
        description = tif.pages[0].description
        substrings = re.findall(r'"DisplayName": "(.*?)",', description)
        return substrings[0].replace("/", ".")


def _read_protein_fov(protein_dir: Path) -> tuple[da.Array, list[str]]:
    images_paths = list(protein_dir.rglob("*.TIF"))
    protein_imgs = [imread(image_path) for image_path in images_paths]
    protein_imgs = [_to_float01_dtype_max(img) for img in protein_imgs]
    protein_image = da.concatenate(protein_imgs, axis=0)
    channel_names = [_get_cosmx_protein_name(image_path) for image_path in images_paths]
    return protein_image, channel_names


def _find_matching_fov_file(images_dir: Path, fov: int) -> Path:
    pattern = re.compile(rf".*_F0*{fov}\.TIF")
    fov_files = [file for file in images_dir.rglob("*") if pattern.match(file.name)]
    if len(fov_files) == 0:
        raise FileNotFoundError(f"No file matches the pattern {pattern} inside {images_dir}")
    if len(fov_files) != 1:
        raise ValueError(f"Multiple files match the pattern {pattern}: {', '.join(map(str, fov_files))}")
    return fov_files[0]


def _read_fov_image(
    morphology_path: Path,
    protein_path: Path | None,
    morphology_coords: list[str],
    *,
    selected_channels: list[str] | None = None,
    **imread_kwargs: Any,
) -> tuple[da.Array, list[str]]:
    image = imread(morphology_path, **imread_kwargs)
    image = _to_float01_dtype_max(image)

    protein_names: list[str] = []
    if protein_path is not None:
        protein_image, protein_names = _read_protein_fov(protein_path)
        image = da.concatenate([image, protein_image], axis=0)

    all_names = make_index_unique(pd.Index(morphology_coords + protein_names)).tolist()

    if selected_channels is not None:
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        present = [c for c in selected_channels if c in name_to_idx]
        missing_here = [c for c in selected_channels if c not in name_to_idx]
        if missing_here:
            logger.warning(
                "FOV %s: skipping %d missing requested channel(s): %s. Present: %s",
                morphology_path.name,
                len(missing_here),
                missing_here,
                all_names,
            )
        if not present:
            raise ValueError(
                f"No requested channels present in file {morphology_path.name}. "
                f"Requested {selected_channels}, available {all_names}."
            )
        idx = [name_to_idx[c] for c in present]
        image = image[idx, :, :]
        all_names = present

    return image, all_names


# ---------------------------------------------------------------------------
# expr / metadata readers (polars)
# ---------------------------------------------------------------------------


def _read_expr_mat_polars(
    expr_path: Path,
    n_rows: int | None = None,
    fovs: set[int] | None = None,
) -> tuple[AnnData, pd.DataFrame | None]:
    """Read expression matrix, filtering out background (cell_ID=0).

    Returns
    -------
    tuple of (AnnData, bg_df or None)
        The cell expression table and, if any cell_ID=0 rows were present,
        a DataFrame of per-FOV background signal indexed by ``fov``.
    """
    sample = pl.read_csv(expr_path, n_rows=1)
    expr_cols = [c for c in sample.columns if c not in ("fov", "cell_ID")]

    lf = pl.scan_csv(expr_path, n_rows=n_rows)
    if fovs is not None:
        lf = lf.filter(pl.col("fov").is_in(fovs))

    lf = lf.with_columns(
        (pl.col("fov").cast(pl.Utf8) + "_" + pl.col("cell_ID").cast(pl.Utf8)).alias("fov_cellID")
    ).select(["fov", "cell_ID", "fov_cellID"] + expr_cols)
    pdf = lf.collect().to_pandas()

    # Separate background rows (cell_ID=0) before building the AnnData.
    bg_mask = pdf["cell_ID"] == 0
    bg_df: pd.DataFrame | None = None
    if bg_mask.any():
        bg_rows = pdf.loc[bg_mask]
        bg_df = bg_rows.set_index("fov")[expr_cols]
        bg_df.index = bg_df.index.astype(int)
        bg_df.index.name = "fov"
        logger.info(
            "Filtered %d background row(s) (cell_ID=0) from expression matrix across %d FOV(s).",
            len(bg_df),
            bg_df.index.nunique(),
        )
        pdf = pdf.loc[~bg_mask]

    obs = pdf[["fov", "cell_ID", "fov_cellID"]].set_index("fov_cellID", verify_integrity=True)
    X = scipy.sparse.csr_matrix(pdf[expr_cols].values)
    var = pd.DataFrame({"gene": expr_cols}).set_index("gene")

    return AnnData(X=X, obs=obs, var=var), bg_df


def _read_metadata_polars(
    meta_path: Path,
    n_rows: int | None = None,
) -> pd.DataFrame:
    df = (
        pl.read_csv(
            meta_path,
            n_rows=n_rows,
            infer_schema_length=n_rows,
        )
        .with_columns((pl.col("fov").cast(pl.Utf8) + "_" + pl.col("cell_ID").cast(pl.Utf8)).alias("fov_cellID"))
        .filter(pl.col("cell_ID") != 0)
    )

    pdf = df.to_pandas()
    return pdf.set_index("fov_cellID", verify_integrity=True)


# ---------------------------------------------------------------------------
# per-FOV local -> stitched-grid placement (shared by polygons & transcripts)
# ---------------------------------------------------------------------------


def place_local_in_fov_grid(
    df: Any,
    fov_locs: pd.DataFrame,
    *,
    fov_col: str = "fov",
    x_local_col: str = "x_local_px",
    y_local_col: str = "y_local_px",
    out_x: str = "x_global_px",
    out_y: str = "y_global_px",
) -> Any:
    """Overwrite *out_x* / *out_y* with stitched-grid coordinates.

    Derived from per-FOV local px columns, matching polygon placement::

        x_global = x0 + x_local
        y_global = y0 + y_local                  if the FOV is flipped
                 = y0 + (FOV_SIZE - y_local)      otherwise

    where ``x0``/``y0`` are the FOV's ``xmin``/``ymin``.  This mirrors the
    placement :func:`_read_polygons_csv` applies to polygon vertices, so
    transcripts co-register with polygons by construction (both share the
    same per-FOV local coordinate system).

    The flip height is the fixed CosMx FOV size, not ``fov_locs['ymax']`` —
    image stitching may overwrite ``ymax`` with the actual raster height,
    which must not perturb coordinate placement.

    Looked up per row via ``Series.map`` (no join/shuffle), so it works for
    both pandas and Dask DataFrames.  *df* must contain *fov_col*,
    *x_local_col* and *y_local_col*.
    """
    x0 = fov_locs["xmin"].astype(float).to_dict()
    y0 = fov_locs["ymin"].astype(float)
    flip = fov_locs["flip_y"].astype(bool) if "flip_y" in fov_locs.columns else pd.Series(False, index=fov_locs.index)
    # Express the y flip as a per-FOV affine (offset + sign * y_local) so the
    # placement is plain arithmetic — no boolean ``Series.where``, whose
    # condition is fragile on Dask (``Series.map`` of a dict yields object
    # dtype there, which ``where`` rejects):
    #   flip:   y = y0 + y_local            -> offset = y0,        sign = +1
    #   noflip: y = y0 + (FOV - y_local)    -> offset = y0 + FOV,  sign = -1
    y_off = (y0 + (~flip) * COSMX_FOV_SIZE_PX).to_dict()
    y_sgn = flip.map({True: 1.0, False: -1.0}).to_dict()

    # ``.astype(float)`` is required: on Dask, ``Series.map(dict)`` yields object
    # dtype, which would propagate through the arithmetic below.
    fov = df[fov_col]
    x_origin = fov.map(x0).astype(float)
    y_origin = fov.map(y_off).astype(float)
    y_sign = fov.map(y_sgn).astype(float)
    return df.assign(
        **{
            out_x: x_origin + df[x_local_col],
            out_y: y_origin + y_sign * df[y_local_col],
        }
    )


# ---------------------------------------------------------------------------
# polygons CSV -> DataFrame (global px)
# ---------------------------------------------------------------------------


def _read_polygons_csv(
    csv_path: Path,
    *,
    fov_locs: pd.DataFrame,
    use_polars: bool = True,
    n_workers: int | None = None,
    fov_set: set[int] | None = None,
) -> pd.DataFrame:
    if fov_locs is None:
        raise ValueError("fov_locs is required to globalize polygon coordinates.")

    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt") as fh:
        raw_hdr = next(csv.reader(fh))
    rename = _match_header(raw_hdr)

    xl = next((c for c in raw_hdr if re.fullmatch(r"x_local_px", c, flags=re.I)), None)
    yl = next((c for c in raw_hdr if re.fullmatch(r"y_local_px", c, flags=re.I)), None)
    if xl is None or yl is None:
        xl = next((c for c in raw_hdr if c.strip().lower() in {"x", "x_px"}), None)
        yl = next((c for c in raw_hdr if c.strip().lower() in {"y", "y_px"}), None)
    if xl is None or yl is None:
        raise ValueError(f"{csv_path.name}: missing x_local_px / y_local_px columns (or acceptable fallback).")

    core_cols = {orig: canon for orig, canon in rename.items() if canon in {"fov", "cell_ID", "polygon_index"}}

    if use_polars:
        select_cols = [pl.col(orig).alias(core_cols[orig]) for orig in core_cols] + [
            pl.col(xl).alias("x_local"),
            pl.col(yl).alias("y_local"),
        ]
        lf = pl.scan_csv(csv_path).select(select_cols)
        if "polygon_index" not in core_cols.values():
            lf = lf.with_columns(pl.int_ranges(0, pl.len()).over(["fov", "cell_ID"]).alias("polygon_index"))
        pdf = (
            lf.group_by(["fov", "cell_ID", "polygon_index"])
            .agg([pl.col("x_local").alias("vx_local"), pl.col("y_local").alias("vy_local")])
            .collect()
            .to_pandas()
        )
    else:
        usecols = list(core_cols.keys()) + [xl, yl]
        raw = pd.read_csv(
            csv_path,
            usecols=usecols,
            compression="gzip" if csv_path.suffix == ".gz" else None,
        ).rename(columns={**core_cols, xl: "x_local", yl: "y_local"})
        if "polygon_index" not in raw.columns:
            raw["polygon_index"] = raw.groupby(["fov", "cell_ID"]).cumcount()
        pdf = (
            raw.groupby(["fov", "cell_ID", "polygon_index"])
            .agg({"x_local": list, "y_local": list})
            .reset_index()
            .rename(columns={"x_local": "vx_local", "y_local": "vy_local"})
        )

    if fov_set is not None:
        pdf = pdf[pdf["fov"].isin(fov_set)].reset_index(drop=True)

    x_off = fov_locs["xmin"].astype(float).to_dict()
    y_off = fov_locs["ymin"].astype(float).to_dict()
    y_max = fov_locs["ymax"].astype(float).to_dict() if "ymax" in fov_locs.columns else None
    flip_map = fov_locs["flip_y"].astype(bool).to_dict() if "flip_y" in fov_locs.columns else None

    local_max_y: dict[int, float] = {}
    for fov, vy_local in zip(pdf["fov"], pdf["vy_local"], strict=False):
        m = max(float(v) for v in vy_local)
        if fov not in local_max_y or m > local_max_y[fov]:
            local_max_y[fov] = m

    geoms: list[sgeom.Polygon | None] = []
    bad = 0

    for fov, vx_local, vy_local in zip(pdf["fov"], pdf["vx_local"], pdf["vy_local"], strict=False):
        if fov not in x_off:
            raise KeyError(f"FOV {fov} in polygons has no entry in fov_locs")

        x0 = x_off[fov]
        y0 = y_off[fov]

        if y_max is not None and not math.isnan(y_max[fov]) and y_max[fov] > y0:
            h = y_max[fov] - y0
        else:
            h = max(COSMX_FOV_SIZE_PX, local_max_y.get(fov, 0.0))

        extra_flip = bool(flip_map.get(fov, False)) if flip_map is not None else False

        pts: list[tuple[float, float]] = []
        for xl_, yl_ in zip(vx_local, vy_local, strict=False):
            try:
                x = x0 + float(xl_)
                y_local = float(yl_)
                if extra_flip:
                    y = y0 + y_local
                else:
                    y = y0 + (h - y_local)
            except Exception:
                continue
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))

        if len({(round(x, 6), round(y, 6)) for x, y in pts}) < 3:
            geoms.append(None)
            bad += 1
            continue

        if pts[0] != pts[-1]:
            pts.append(pts[0])

        poly = sgeom.Polygon(pts)
        if not poly.is_valid or poly.is_empty:
            geoms.append(None)
            bad += 1
        else:
            geoms.append(poly)

    pdf["geometry"] = geoms
    if bad:
        logger.warning("%s: skipped %d malformed polygon(s).", csv_path.name, bad)
    pdf = pdf.dropna(subset=["geometry"]).reset_index(drop=True)

    # Merge multi-polygon cells: a cell with multiple polygon_index values
    # must become a single MultiPolygon row to avoid duplicate global_cell_id.
    if (pdf.groupby(["fov", "cell_ID"]).size() > 1).any():

        def _merge_geoms(geoms):
            geoms = [g for g in geoms if g is not None]
            if len(geoms) == 1:
                return geoms[0]
            return sgeom.MultiPolygon(geoms)

        pdf = pdf.groupby(["fov", "cell_ID"], sort=False).agg({"geometry": _merge_geoms}).reset_index()

    pdf["bounds_max_x"] = pdf.geometry.map(lambda g: g.bounds[2])
    pdf["bounds_max_y"] = pdf.geometry.map(lambda g: g.bounds[3])

    return pdf


# ---------------------------------------------------------------------------
# parquet cache for transcripts
# ---------------------------------------------------------------------------


def _is_valid_parquet(path: Path) -> bool:
    try:
        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def _stream_csvgz_to_parquet(
    src: Path,
    parquet_path: Path,
    *,
    row_group_rows: int = 5_000_000,
    n_workers: int | None = None,
) -> None:
    total_rows = _count_csv_rows(src, n_workers=n_workers)
    if total_rows == 0:
        raise ValueError(f"{src} appears empty or header-only")

    if src.suffix != ".gz":
        fin: io.BufferedReader | io.RawIOBase = open(src, "rb")
    elif n_workers and n_workers > 1 and shutil.which("pigz"):
        proc = subprocess.Popen(
            ["pigz", f"-p{n_workers}", "-dc", str(src)],
            stdout=subprocess.PIPE,
            bufsize=2**20,
        )
        fin = proc.stdout  # type: ignore[assignment]
    else:
        fin = gzip.open(src, "rb")

    # Columns like y_global_px can mix plain ints and scientific notation
    # (e.g. "1e+05"), which Arrow's int64 auto-inference can't handle.
    # Force all plausible numeric coordinate columns to float64.
    _float_cols = {
        "x_local_px",
        "y_local_px",
        "x_global_px",
        "y_global_px",
        "x_global_mm",
        "y_global_mm",
        "z",
    }
    read_opts = pacsv.ReadOptions(block_size=32 << 20)
    convert_opts = pacsv.ConvertOptions(
        auto_dict_encode=True,
        column_types={c: pa.float64() for c in _float_cols},
    )
    reader = pacsv.open_csv(fin, read_options=read_opts, convert_options=convert_opts)

    with (
        pq.ParquetWriter(parquet_path, reader.schema, compression="zstd") as writer,
        tqdm(total=total_rows, unit="rows", desc="CSV -> Parquet", unit_scale=True, dynamic_ncols=True) as bar,
    ):
        for batch in reader:
            writer.write_table(
                pa.Table.from_batches([batch]),
                row_group_size=row_group_rows,
            )
            bar.update(batch.num_rows)

    if hasattr(fin, "close"):
        fin.close()


def _count_csv_rows(src: Path, *, n_workers: int | None = None) -> int:
    if src.suffix == ".gz" and shutil.which("pigz") and shutil.which("wc"):
        cmd = f"pigz -dc -p{n_workers or 1} {src} | wc -l"
        rows = int(subprocess.check_output(cmd, shell=True).strip())
        return max(rows - 1, 0)
    if src.suffix != ".gz" and shutil.which("wc"):
        rows = int(subprocess.check_output(["wc", "-l", str(src)]).split()[0])
        return max(rows - 1, 0)

    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rb") as fh:
        buf_size = 2**20
        rows = 0
        while True:
            chunk = fh.read(buf_size)
            if not chunk:
                break
            rows += chunk.count(b"\n")
    return max(rows - 1, 0)


def _resolve_tx_path(path: Path, dataset_id: str | None) -> Path:
    """Resolve a transcript CSV path.

    If *path* is a file, return it.  Otherwise, expect a directory
    containing ``<dataset_id>_tx_file.csv[.gz]``.
    """
    if path.is_file():
        return path

    if dataset_id is None:
        raise FileNotFoundError("Transcript path is a directory but no dataset_id was provided.")

    gz_path = path / f"{dataset_id}_tx_file.csv.gz"
    csv_path = path / f"{dataset_id}_tx_file.csv"
    src = gz_path if gz_path.exists() else csv_path
    if not src.exists():
        raise FileNotFoundError(f"No *_tx_file.* for dataset {dataset_id!r} under {path}")
    return src


def _parquet_cache_for_tx(
    path: Path,
    dataset_id: str | None,
    *,
    n_workers: int | None = None,
    row_group_rows: int = 5_000_000,
) -> Path:
    src = _resolve_tx_path(path, dataset_id)
    parquet_path = src.with_suffix(".parquet")

    if parquet_path.exists() and _is_valid_parquet(parquet_path):
        logger.info("[transcripts] Found Parquet cache — skipping conversion.")
        return parquet_path

    if parquet_path.exists():
        logger.warning("[transcripts] Found corrupted Parquet cache — rebuilding.")
        parquet_path.unlink()

    size_gb = src.stat().st_size / 1_073_741_824
    logger.warning(
        "[transcripts] Converting %s (%.1f GB) to Parquet for faster reads. "
        "This is a one-time operation but may take several minutes...",
        src.name,
        size_gb,
    )

    _stream_csvgz_to_parquet(
        src,
        parquet_path,
        row_group_rows=row_group_rows,
        n_workers=n_workers,
    )

    if not _is_valid_parquet(parquet_path):
        parquet_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create a valid Parquet file at {parquet_path}")

    logger.info("[transcripts] Parquet cache ready at %s.", parquet_path.name)
    return parquet_path


# ---------------------------------------------------------------------------
# transcript CSV (small) -> pandas
# ---------------------------------------------------------------------------


def _maybe_warn_big_file(src: Path, *, threshold_gb: float = 1.0) -> bool:
    size_gb = src.stat().st_size / 1_073_741_824
    if size_gb > threshold_gb:
        return True
    return False


def _read_transcripts_csv(path: Path, dataset_id: str | None, nrows: int | None = None) -> pd.DataFrame:
    src = _resolve_tx_path(path, dataset_id)
    if src.suffix == ".gz":
        df = pd.read_csv(src, compression="gzip", nrows=nrows)
    else:
        df = pd.read_csv(src, nrows=nrows)

    needed = ["x_global_px", "y_global_px", "target"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"The file {src} must contain the following columns: {', '.join(needed)}. Missing: {missing}")

    _pandas_categoricals_to_string(df)

    fov_ser = pd.to_numeric(df["fov"], errors="coerce").fillna(0).astype(int)
    cell_ser = pd.to_numeric(df["cell_ID"], errors="coerce").fillna(0).astype(int)
    df["fov"] = fov_ser
    df["cell_ID"] = cell_ser

    max_cell = int(cell_ser.max()) if len(cell_ser) else 0
    df["unique_cell_id"] = fov_ser * (max_cell + 1) * (cell_ser > 0).astype(int) + cell_ser

    return df
