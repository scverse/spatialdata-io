"""CosmX reader regression tests with synthetic fixtures.

Fixture matrix (derived from real data):

  Fixture A — "new-style" (lymph/tonsil-like)
    px+mm FOV positions, flip_y=False, .csv.gz files, uppercase FOV column,
    Morphology2D images, single modality, TWO ADJACENT FOVs

  Fixture B — "old-style px-only" (hippocampus-like)
    px-only FOV positions, flip_y=True, .csv (uncompressed), lowercase fov
    column, Morphology2D images, single modality, TWO NON-ADJACENT FOVs

  Fixture C — "old-style mm-only" (pancreas-like)
    mm-only positions, flip_y=True, extra columns (Slide, ROI, Order),
    CellLabels TIFFs instead of Morphology2D, single modality, SINGLE FOV

After standardization, ALL fixtures should produce the same normalized
SpatialData structure.

Run:  pixi run -e dev-py313 python -m pytest tests/test_cosmx_regression.py -v
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
COSMX_PIXEL_SIZE = 0.120280945
MM_TO_PX = 1000.0 / COSMX_PIXEL_SIZE
# The reader forces FOV_SIZE_PX=4256 in _snap_fov_grid.
# Our polygon local coords and FOV positions must be consistent with this.
# Real CosmX = 4256. We use a small size for fast test TIFFs.
# _snap_fov_grid will warn but still force 4256 for coordinates.
TIFF_FOV_SIZE = 256
# The reader's internal spec size (used for polygon coords, FOV positions, grid snapping)
SPEC_FOV_SIZE = 4256
N_CHANNELS = 3
N_CELLS_PER_FOV = 5
N_GENES = 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiff_description() -> str:
    """JSON that _get_cosmx_morphology_coords can parse via regex."""
    return json.dumps(
        {
            "ChannelOrder": "BGU",
            "MorphologyKit": {
                "MorphologyReagents": [
                    {"BiologicalTarget": "Histone", "Fluorophore": {"ChannelId": "B"}},
                    {"BiologicalTarget": "DNA", "Fluorophore": {"ChannelId": "G"}},
                    {"BiologicalTarget": "rRNA", "Fluorophore": {"ChannelId": "U"}},
                ]
            },
        }
    )


def _write_morphology_tiff(path: Path, n_channels: int = N_CHANNELS, fov_size: int = SPEC_FOV_SIZE):
    """Write a multi-page TIFF that dask_image.imread reads as (C, Y, X)."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    desc = _tiff_description()
    data = np.random.default_rng(0).integers(0, 65535, (n_channels, fov_size, fov_size), dtype=np.uint16)
    tifffile.imwrite(str(path), data, description=desc, photometric="minisblack")


def _write_asymmetric_morphology_tiff(path: Path, n_channels: int = N_CHANNELS, fov_size: int = TIFF_FOV_SIZE):
    """Morphology TIFF whose top half (100) differs from its bottom half (200),
    so a vertical flip is detectable by comparing top- vs bottom-row means.
    """
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n_channels, fov_size, fov_size), dtype=np.uint16)
    data[:, : fov_size // 2, :] = 100
    data[:, fov_size // 2 :, :] = 200
    tifffile.imwrite(str(path), data, description=_tiff_description(), photometric="minisblack")


def _write_cell_label_tiff(path: Path, n_cells: int = N_CELLS_PER_FOV, fov_size: int = TIFF_FOV_SIZE):
    """Write a CellLabels TIFF with small blocks of unique cell IDs."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((fov_size, fov_size), dtype=np.uint16)
    block = fov_size // (n_cells + 2)
    for cid in range(1, n_cells + 1):
        y0, y1 = cid * block, cid * block + block
        x0, x1 = cid * block, cid * block + block
        data[y0:y1, x0:x1] = cid
    tifffile.imwrite(str(path), data)


def _write_csv(path: Path, df: pd.DataFrame, compress: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if compress else None)


def _make_polygon_rows(fov: int, cell_id: int, cx: float, cy: float, r: float = 50.0, n_pts: int = 6):
    """Hexagon vertices for one cell in LOCAL coordinates."""
    rows = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        rows.append(
            {
                "fov": fov,
                "cellID": cell_id,
                "x_local_px": cx + r * math.cos(angle),
                "y_local_px": cy + r * math.sin(angle),
            }
        )
    return rows


def _make_polygon_df(fov_positions: dict[int, tuple[float, float]], n_cells: int = N_CELLS_PER_FOV):
    """Build polygon CSV with both local and global coords.

    fov_positions: {fov_id: (x_global_px, y_global_px)}
    """
    rows = []
    spacing = SPEC_FOV_SIZE / (n_cells + 1)
    for fov, (x_off, y_off) in fov_positions.items():
        for cid in range(1, n_cells + 1):
            cx_local = spacing * cid
            cy_local = spacing * cid
            for pt in _make_polygon_rows(fov, cid, cx_local, cy_local):
                pt["x_global_px"] = pt["x_local_px"] + x_off
                pt["y_global_px"] = pt["y_local_px"] + y_off
                rows.append(pt)
    return pd.DataFrame(rows)


def _make_expr_mat(fovs: list[int], n_cells: int = N_CELLS_PER_FOV, n_genes: int = N_GENES):
    rows = []
    rng = np.random.default_rng(42)
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    for fov in fovs:
        for cid in range(1, n_cells + 1):
            row = {"fov": fov, "cell_ID": cid}
            for g in gene_names:
                row[g] = int(rng.integers(0, 100))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_metadata(fovs: list[int], n_cells: int = N_CELLS_PER_FOV):
    rows = []
    rng = np.random.default_rng(42)
    for fov in fovs:
        for cid in range(1, n_cells + 1):
            rows.append(
                {
                    "fov": fov,
                    "cell_ID": cid,
                    "Area": float(rng.uniform(50, 500)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixture A: new-style, TWO ADJACENT FOVs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_a(tmp_path_factory) -> Path:
    """px+mm, Morphology2D, .csv.gz. FOVs 1 & 2 are side-by-side."""
    root = tmp_path_factory.mktemp("fixture_a")
    prefix = "S0"
    fovs = [1, 2]

    # Adjacent: FOV 2 is directly right of FOV 1
    fov_pos = {1: (0.0, 0.0), 2: (float(SPEC_FOV_SIZE), 0.0)}
    x_mm = {f: x / MM_TO_PX for f, (x, _) in fov_pos.items()}
    y_mm = {f: y / MM_TO_PX for f, (_, y) in fov_pos.items()}

    fov_df = pd.DataFrame(
        {
            "FOV": fovs,
            "x_global_px": [fov_pos[f][0] for f in fovs],
            "y_global_px": [fov_pos[f][1] for f in fovs],
            "x_global_mm": [x_mm[f] for f in fovs],
            "y_global_mm": [y_mm[f] for f in fovs],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv.gz", fov_df, compress=True)
    _write_csv(root / f"{prefix}_exprMat_file.csv.gz", _make_expr_mat(fovs), compress=True)
    _write_csv(root / f"{prefix}_metadata_file.csv.gz", _make_metadata(fovs), compress=True)
    _write_csv(root / f"{prefix}-polygons.csv.gz", _make_polygon_df(fov_pos), compress=True)

    morph = root / "Morphology2D"
    for fov in fovs:
        _write_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF", fov_size=TIFF_FOV_SIZE)

    return root


# ---------------------------------------------------------------------------
# Fixture B: old-style px-only, TWO NON-ADJACENT FOVs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_b(tmp_path_factory) -> Path:
    """px-only → flip_y=True, .csv, lowercase fov. FOVs 1 & 5 are spaced apart."""
    root = tmp_path_factory.mktemp("fixture_b")
    prefix = "Run5642_S3_Quarter"
    fovs = [1, 5]

    # Non-adjacent: FOV 5 is 3 FOV-widths away
    fov_pos = {1: (0.0, 0.0), 5: (3.0 * SPEC_FOV_SIZE, 2.0 * SPEC_FOV_SIZE)}

    fov_df = pd.DataFrame(
        {
            "fov": fovs,
            "x_global_px": [fov_pos[f][0] for f in fovs],
            "y_global_px": [fov_pos[f][1] for f in fovs],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv", fov_df)
    _write_csv(root / f"{prefix}_exprMat_file.csv", _make_expr_mat(fovs))
    _write_csv(root / f"{prefix}_metadata_file.csv", _make_metadata(fovs))
    _write_csv(root / f"{prefix}-polygons.csv", _make_polygon_df(fov_pos))

    morph = root / "Morphology2D"
    for fov in fovs:
        _write_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF", fov_size=TIFF_FOV_SIZE)

    return root


# ---------------------------------------------------------------------------
# Fixture C: mm-only, SINGLE FOV, CellLabels instead of Morphology2D
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_c(tmp_path_factory) -> Path:
    """mm-only → flip_y=True, extra columns, CellLabels dir, single FOV."""
    root = tmp_path_factory.mktemp("fixture_c")
    prefix = "Pancreas"
    fovs = [1]

    x_mm_val = 1.0
    y_mm_val = 2.0
    fov_pos = {1: (x_mm_val * MM_TO_PX, y_mm_val * MM_TO_PX)}

    fov_df = pd.DataFrame(
        {
            "Slide": ["S1"],
            "X_mm": [x_mm_val],
            "Y_mm": [y_mm_val],
            "Z_mm": [0.0],
            "ZOffset_mm": [0.0],
            "ROI": [1],
            "FOV": fovs,
            "Order": fovs,
            "Run_Tissue_name": ["tissue"],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv", fov_df)
    _write_csv(root / f"{prefix}_exprMat_file.csv", _make_expr_mat(fovs))
    _write_csv(root / f"{prefix}_metadata_file.csv", _make_metadata(fovs))
    _write_csv(root / f"{prefix}-polygons.csv", _make_polygon_df(fov_pos))

    labels_dir = root / "CellLabels"
    for fov in fovs:
        _write_cell_label_tiff(labels_dir / f"CellLabels_F{fov:03d}.tif")

    return root


# ---------------------------------------------------------------------------
# Fixture MM: multimodal (RNA + Protein), prefix-based (V2-style), SINGLE FOV
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_multimodal(tmp_path_factory) -> Path:
    """Two modality prefixes (``S0RNA`` / ``S0Protein``) → base id ``S0`` →
    triggers the multimodal (``_cosmx_multi``) dispatch. px-only (flip_y=True),
    single FOV. Morphology is asymmetric (top 100 / bottom 200) so the image
    flip is detectable. The RNA modality carries the shared spatial files
    (it's chosen as the label modality because it has polygons).
    """
    root = tmp_path_factory.mktemp("fixture_mm")
    fovs = [1]
    fov_pos = {1: (0.0, 0.0)}

    for prefix in ("S0RNA", "S0Protein"):
        _write_csv(root / f"{prefix}_exprMat_file.csv", _make_expr_mat(fovs))
        _write_csv(root / f"{prefix}_metadata_file.csv", _make_metadata(fovs))

    # Shared spatial files live with the RNA (label) modality.
    fov_df = pd.DataFrame(
        {
            "fov": fovs,
            "x_global_px": [fov_pos[f][0] for f in fovs],  # px-only → flip_y=True
            "y_global_px": [fov_pos[f][1] for f in fovs],
        }
    )
    _write_csv(root / "S0RNA_fov_positions_file.csv", fov_df)
    _write_csv(root / "S0RNA-polygons.csv", _make_polygon_df(fov_pos))

    morph = root / "Morphology2D"
    for fov in fovs:
        _write_asymmetric_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF")

    return root


# ---------------------------------------------------------------------------
# shared sdio fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sdio():
    import spatialdata_io as _sdio

    return _sdio


def _read(sdio, path: Path, **kw):
    """Read a fixture with sensible defaults for fast testing."""
    defaults = {"n_workers": 1, "read_transcripts": False}
    defaults.update(kw)
    return sdio.cosmx(path, **defaults)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════


class TestDatasetDiscovery:
    def test_fixture_a(self, fixture_a):
        from spatialdata_io.readers.cosmx._discovery import _set_up_cosmx_dataset_for_conversion

        ds = _set_up_cosmx_dataset_for_conversion(fixture_a)
        assert ds.dataset_id == "S0"
        assert ds.fov_positions_file is not None
        assert ds.polygons_file is not None
        assert ds.exprMat_file is not None
        assert ds.metadata_file is not None
        assert ds.morphology_2d_dir is not None
        assert ds.modalities is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. FOV POSITIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestFovPositions:
    def test_mm_only_flip_and_conversion(self, fixture_c):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs

        fl = _read_fov_locs(fixture_c / "Pancreas_fov_positions_file.csv")
        assert fl["flip_y"].all()
        assert fl["xmin"].iloc[0] > 1000  # mm→px conversion

    def test_all_produce_standard_columns(self, fixture_a, fixture_b, fixture_c):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs

        paths = [
            fixture_a / "S0_fov_positions_file.csv.gz",
            fixture_b / "Run5642_S3_Quarter_fov_positions_file.csv",
            fixture_c / "Pancreas_fov_positions_file.csv",
        ]
        for p in paths:
            fl = _read_fov_locs(p)
            for col in ("xmin", "ymin", "xmax", "ymax", "flip_y"):
                assert col in fl.columns, f"{p.name} missing {col}"
            assert (fl["xmax"] > fl["xmin"]).all()
            assert (fl["ymax"] > fl["ymin"]).all()

    def test_missing_fov_raises(self, fixture_a):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs

        with pytest.raises(KeyError):
            _read_fov_locs(fixture_a / "S0_fov_positions_file.csv.gz", fovs=[99])


# ═══════════════════════════════════════════════════════════════════════════
# 3. HEADER MATCHING (no I/O)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 4. GLOBAL CELL ID (no I/O)
# ═══════════════════════════════════════════════════════════════════════════


class TestGlobalCellId:
    def test_formula(self):
        base = 6
        df = pd.DataFrame({"fov": [1, 1, 2, 2], "cell_ID": [0, 5, 0, 3]})
        gids = df["fov"] * base * (df["cell_ID"] > 0).astype(int) + df["cell_ID"]
        assert gids.tolist() == [0, 11, 0, 15]

    def test_prescan_prevents_crash(self, tmp_path):
        """max_cell_id pre-scan should prevent ValueError when elements
        have different max cell_IDs.
        """
        from spatialdata_io.readers.cosmx._reader import CosMxDatasetReader, _prescan_max_cell_id

        # Create two CSVs: polygons with max cell_ID=10, expr with max=20
        poly_csv = tmp_path / "poly.csv"
        expr_csv = tmp_path / "expr.csv"
        poly_csv.write_text("fov,cell_ID,x,y\n1,10,0,0\n")
        expr_csv.write_text("fov,cell_ID,Gene_0\n1,20,5\n")

        class FakeDataset:
            polygons_file = poly_csv
            exprMat_file = expr_csv
            metadata_file = None

        class FakeReader:
            max_cell_id = None

        reader = FakeReader()
        _prescan_max_cell_id(reader, FakeDataset(), fov_set=None)
        assert reader.max_cell_id == 20, "Should lock to the global max across all files"

        # Now calling global_cell_id with cell_ID=20 should NOT crash
        real_reader = CosMxDatasetReader.__new__(CosMxDatasetReader)
        real_reader.max_cell_id = reader.max_cell_id
        gid = real_reader.global_cell_id(pd.DataFrame({"fov": [1], "cell_ID": [20]}))
        assert gid.iloc[0] == 1 * 21 + 20

    def test_prescan_matches_aliased_id_and_fov_columns(self, tmp_path):
        """The pre-scan must honour the same column aliases as the header
        canonicaliser. A polygon CSV using ``object_id``/``roi`` (instead of
        ``cell_ID``/``fov``) must still contribute its max — otherwise an
        orphan (segmented-only) cell with the highest per-FOV id is missed and
        max_cell_id is mis-locked. The old hardcoded ("cell_ID", "cellID")
        match would skip this file entirely.
        """
        from spatialdata_io.readers.cosmx._reader import _prescan_max_cell_id

        poly_csv = tmp_path / "poly.csv"
        expr_csv = tmp_path / "expr.csv"
        # Orphan id 30 lives only in the polygons, under aliased headers. The
        # fov-2 id=99 row must be excluded once we restrict to FOV 1 — which
        # only works if the aliased ``roi`` column is resolved too.
        poly_csv.write_text("roi,object_id,x,y\n1,30,0,0\n2,99,0,0\n")
        expr_csv.write_text("fov,cell_ID,Gene_0\n1,20,5\n")

        class FakeDataset:
            polygons_file = poly_csv
            exprMat_file = expr_csv
            metadata_file = None

        class FakeReader:
            max_cell_id = None

        reader = FakeReader()
        _prescan_max_cell_id(reader, FakeDataset(), fov_set={1})
        assert reader.max_cell_id == 30

    def test_prescan_warns_instead_of_silently_dropping_a_file(self, tmp_path, caplog):
        """A file that exists but has no recognizable cell-ID column must be
        surfaced (warning), not silently swallowed — a silent drop in a pre-scan
        that establishes the max_cell_id invariant hides the exact schema
        mismatch that mis-locks the max.
        """
        import logging

        from spatialdata._logging import logger as sd_logger

        from spatialdata_io.readers.cosmx._reader import _prescan_max_cell_id

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("fov,mystery,x,y\n1,7,0,0\n")

        class FakeDataset:
            polygons_file = bad_csv
            exprMat_file = None
            metadata_file = None

        class FakeReader:
            max_cell_id = None

        reader = FakeReader()
        # spatialdata's logger does not propagate to the root logger caplog
        # attaches to, so enable it for the duration of this assertion.
        prev_propagate = sd_logger.propagate
        sd_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING):
                _prescan_max_cell_id(reader, FakeDataset(), fov_set=None)
        finally:
            sd_logger.propagate = prev_propagate

        assert reader.max_cell_id is None
        assert any("cell_ID" in rec.getMessage() for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════
# 4b. TILE CLIPPING HELPER
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 4c. TRANSCRIPT PLACEMENT (local-coord; replaces the old fov_shift / +4256)
# ═══════════════════════════════════════════════════════════════════════════


class TestTranscriptPlacement:
    """Transcripts must be placed with the SAME per-FOV local->grid mapping as
    polygons, so they co-register by construction (issue #39 / #18).
    """

    _fov_locs = pd.DataFrame(
        {
            "xmin": [0.0, 4256.0],
            "ymin": [0.0, 0.0],
            "xmax": [4256.0, 8512.0],
            "ymax": [4256.0, 4256.0],
            "flip_y": [False, False],
        },
        index=pd.Index([1, 2], name="fov"),
    )

    def _df(self):
        return pd.DataFrame(
            {
                "fov": [1, 2],
                "x_local_px": [10.0, 20.0],
                "y_local_px": [100.0, 200.0],
                # raw global deliberately bogus to prove it is NOT used
                "x_global_px": [-9e9, -9e9],
                "y_global_px": [-9e9, -9e9],
            }
        )

    def test_flip_matches_polygon_formula(self):
        from spatialdata_io.readers.cosmx._io import place_local_in_fov_grid

        fov_locs = self._fov_locs.copy()
        fov_locs["flip_y"] = [True, True]
        out = place_local_in_fov_grid(self._df(), fov_locs)
        # flip: y = y0 + y_local
        assert out["x_global_px"].tolist() == [10.0, 4256.0 + 20.0]
        assert out["y_global_px"].tolist() == [100.0, 200.0]

    @pytest.mark.parametrize("flip", [True, False])
    def test_dask_path_matches_pandas(self, flip):
        """Regression: the real transcript path runs on a Dask DataFrame (CSV is
        converted to Parquet, then read lazily). A prior rewrite expressed the
        flip as ``y_local.where(fov.map(flip_dict), ...)`` — on Dask,
        ``Series.map(dict)`` is object dtype and ``where`` rejects a non-bool
        condition, crashing every ``read_labels=True`` read. The synthetic tests
        only covered the pandas path, so it slipped through. Assert the Dask path
        runs AND yields the same numbers as pandas.
        """
        import dask.dataframe as dd

        from spatialdata_io.readers.cosmx._io import place_local_in_fov_grid

        fov_locs = self._fov_locs.copy()
        fov_locs["flip_y"] = [flip, flip]

        pdf = self._df()
        expect = place_local_in_fov_grid(pdf, fov_locs)

        ddf = dd.from_pandas(pdf, npartitions=2)
        out = place_local_in_fov_grid(ddf, fov_locs)
        # must be lazy (not eagerly computed to pandas) and must not have raised
        assert hasattr(out, "compute"), "expected a Dask DataFrame back"
        got = out.compute()
        assert got["x_global_px"].tolist() == expect["x_global_px"].tolist()
        assert got["y_global_px"].tolist() == expect["y_global_px"].tolist()


def _lookup_centres(raster, fov_locs, fov, n_cells, ox, oy):
    """Place one transcript at each cell's local block-centre via
    ``place_local_in_fov_grid`` and read the label *raster* there; the returned
    list holds the raster value under each cell's centre (``-1`` if off-canvas).
    """
    from spatialdata_io.readers.cosmx._io import place_local_in_fov_grid

    block = SPEC_FOV_SIZE // (n_cells + 2)
    centres = [c * block + block // 2 for c in range(1, n_cells + 1)]
    placed = place_local_in_fov_grid(
        pd.DataFrame({"fov": [fov] * n_cells, "x_local_px": centres, "y_local_px": centres}),
        fov_locs,
    )
    col = np.round(placed["x_global_px"].to_numpy() - ox).astype(int)
    row = np.round(placed["y_global_px"].to_numpy() - oy).astype(int)
    H, W = raster.shape[-2:]
    ib = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    return np.where(ib, raster[np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)], -1).tolist()


class TestLabelTranscriptCoRegistration:
    """CellLabels-TIFF rasters must co-register with transcripts placed via
    ``place_local_in_fov_grid``.

    Regression for the stitcher tile-flip bug (#39): the CellLabels stitcher
    flipped each tile by ``flip_y`` (``do_flip = flip_image or flip_y``), but
    transcripts with ``flip_y=True`` are placed DIRECT (``y = y0 + y_local``) —
    so the raster ended up vertically mirrored relative to the points. Confirmed
    on pancreas (mm-only, the only real dataset with CellLabels TIFFs): the
    stitched-label lookup ``label[row, col] == global_cell_id`` was 0.8% direct /
    100% within-FOV-mirrored, and 100% / 0.8% after the fix.

    Earlier regression tests only exercised the pandas transcript path and the
    polygon-rasterized label path, so neither this flip nor the Dask ``.where``
    crash was covered. This test drives the CellLabels-TIFF stitcher and the
    transcript placement together and asserts they index the same cell.
    """

    @staticmethod
    def _stitch_and_lookup(tmp_path: Path, flip_y: bool):
        import tifffile

        from spatialdata_io.readers.cosmx._stitching import _read_stitched_cell_labels_from_dir

        # Use the real FOV size so the flip arithmetic in placement (which uses
        # COSMX_FOV_SIZE_PX) matches the label tile height.
        fov_size = SPEC_FOV_SIZE
        n_cells = 5
        labels_dir = tmp_path / "CellLabels"
        _write_cell_label_tiff(labels_dir / "CellLabels_F001.tif", n_cells=n_cells, fov_size=fov_size)

        xmin, ymin = 1000.0, 2000.0  # non-zero origin to catch frame mistakes
        fov_locs = pd.DataFrame(
            {
                "xmin": [xmin],
                "ymin": [ymin],
                "xmax": [xmin + fov_size],
                "ymax": [ymin + fov_size],
                "flip_y": [flip_y],
            },
            index=pd.Index([1], name="fov"),
        )

        stitched, _, used = _read_stitched_cell_labels_from_dir(
            labels_dir,
            fov_locs.copy(),
            fovs={1},
            flip_image=False,
            n_workers=1,
            fov_local_to_global=None,
        )
        arr = np.asarray(stitched.compute() if hasattr(stitched, "compute") else stitched)
        ox, oy = float(used["xmin"].min()), float(used["ymin"].min())
        looked_up = _lookup_centres(arr, fov_locs, 1, n_cells, ox, oy)
        return looked_up, list(range(1, n_cells + 1))

    @pytest.mark.parametrize("flip_y", [True, False])
    def test_transcripts_hit_their_cell(self, tmp_path, flip_y):
        looked_up, expected = self._stitch_and_lookup(tmp_path, flip_y)
        assert looked_up == expected, (
            f"flip_y={flip_y}: each transcript placed from local coords must "
            f"index its own cell in the stitched CellLabels raster. Got "
            f"{looked_up}, expected {expected}. A mismatch means the label tile "
            f"flip and the transcript placement disagree (the #39 mirror bug)."
        )


class TestLabelTileFlip:
    """Unit tests for the per-FOV label-tile flip rule (``_label_tile_flip``)."""

    @staticmethod
    def _locs(flip_y: bool):
        return pd.DataFrame(
            {"xmin": [0.0, 4256.0], "ymin": [0.0, 0.0], "flip_y": [flip_y, flip_y]},
            index=pd.Index([1, 2], name="fov"),
        )

    def test_flip_y_true_means_no_tile_flip(self):
        from spatialdata_io.readers.cosmx._stitching import _label_tile_flip

        locs = self._locs(True)
        assert _label_tile_flip(locs, 1, flip_image=False) is False
        # flip_y is authoritative — it overrides the flip_image fallback
        assert _label_tile_flip(locs, 2, flip_image=True) is False

    def test_fallback_when_fov_absent_from_locs(self):
        from spatialdata_io.readers.cosmx._stitching import _label_tile_flip

        locs = self._locs(True)  # only FOVs 1, 2
        assert _label_tile_flip(locs, 99, flip_image=True) is True
        assert _label_tile_flip(locs, 99, flip_image=False) is False

    def test_cellstats_stitcher_coregisters_px_only(self, tmp_path):
        """End-to-end #41: the legacy CellStatsDir stitcher must co-register
        labels with transcripts on a px-only (flip_y=True) dataset.

        We pass ``flip_image=True`` to prove ``flip_y`` overrides it: pre-fix the
        tile flipped by ``flip_image`` (mirrored vs the direct-placed
        transcripts); post-fix it flips ``not flip_y`` (= no flip).
        """
        from spatialdata_io.readers.cosmx._io import _read_fov_locs
        from spatialdata_io.readers.cosmx._stitching import stitch_segmentation_label_image

        prefix = "S0"
        fov = 1
        n_cells = N_CELLS_PER_FOV

        # px-only positions -> flip_y=True (the convention the bug affects)
        pos_file = tmp_path / f"{prefix}_fov_positions_file.csv"
        _write_csv(
            pos_file,
            pd.DataFrame(
                {
                    "fov": [fov],
                    "x_global_px": [1000.0],
                    "y_global_px": [2000.0],
                }
            ),
        )
        cs = tmp_path / "CellStatsDir"
        ci_file = cs / f"{prefix}_cell_info.csv"
        _write_csv(
            ci_file,
            pd.DataFrame(
                {
                    "fov": [fov] * n_cells,
                    "cellID": list(range(1, n_cells + 1)),
                }
            ),
        )
        _write_cell_label_tiff(
            cs / f"FOV{fov:03d}" / f"CellLabels_F{fov:03d}.tif", n_cells=n_cells, fov_size=SPEC_FOV_SIZE
        )

        stitched, df = stitch_segmentation_label_image(
            path=tmp_path,
            fov_position_file=str(pos_file),
            cell_info_file=str(ci_file),
            dataset_id=prefix,
            flip_image=True,
            n_workers=1,
            fovs={fov},  # inert here: overridden by flip_y
        )
        lab = np.asarray(stitched.compute() if hasattr(stitched, "compute") else stitched)

        # The stitcher returns only (labels, cell_info); re-read positions for the
        # global origin (xmin/ymin) — _snap_fov_grid zeroes only the raster origin.
        fov_locs = _read_fov_locs(pos_file)
        ox, oy = float(fov_locs.loc[fov, "xmin"]), float(fov_locs.loc[fov, "ymin"])
        gid = {int(r.cellID): int(r.global_cell_id) for r in df.itertuples()}
        hit = _lookup_centres(lab, fov_locs, fov, n_cells, ox, oy)
        expected = [gid[c] for c in range(1, n_cells + 1)]
        assert hit == expected, (
            f"cell_stats labels not co-registered with transcripts (#41): got {hit}, expected {expected}"
        )


class TestTranscriptAlignmentE2E:
    """End-to-end: transcript points must land on their cells' polygons in the
    output 'global' coordinate system, regardless of the global-frame offset
    (the scenario that used to trigger fov_shift).
    """

    @staticmethod
    def _build(root: Path, prefix: str, poly_y_offset: float) -> Path:
        """Dataset where polygon GLOBAL y sits *poly_y_offset* from the FOV
        position (e.g. -SPEC_FOV_SIZE reproduces the old fov_shift trigger).
        Transcripts are written at each cell's local centroid.
        """
        fovs = [1, 2]
        fov_pos = {1: (0.0, 0.0), 2: (float(SPEC_FOV_SIZE), 0.0)}
        x_mm = {f: x / MM_TO_PX for f, (x, _) in fov_pos.items()}
        y_mm = {f: y / MM_TO_PX for f, (_, y) in fov_pos.items()}
        fov_df = pd.DataFrame(
            {
                "FOV": fovs,
                "x_global_px": [fov_pos[f][0] for f in fovs],
                "y_global_px": [fov_pos[f][1] for f in fovs],
                "x_global_mm": [x_mm[f] for f in fovs],
                "y_global_mm": [y_mm[f] for f in fovs],
            }
        )
        _write_csv(root / f"{prefix}_fov_positions_file.csv.gz", fov_df, compress=True)
        _write_csv(root / f"{prefix}_exprMat_file.csv.gz", _make_expr_mat(fovs), compress=True)
        _write_csv(root / f"{prefix}_metadata_file.csv.gz", _make_metadata(fovs), compress=True)

        # polygons globalised with an arbitrary y offset
        poly_off = {f: (x, y + poly_y_offset) for f, (x, y) in fov_pos.items()}
        _write_csv(root / f"{prefix}-polygons.csv.gz", _make_polygon_df(poly_off), compress=True)

        # transcripts: one per cell at the cell's LOCAL centroid
        spacing = SPEC_FOV_SIZE / (N_CELLS_PER_FOV + 1)
        rows = []
        for fov in fovs:
            ox, oy = poly_off[fov]
            for cid in range(1, N_CELLS_PER_FOV + 1):
                cl = spacing * cid
                rows.append(
                    {
                        "fov": fov,
                        "cell_ID": cid,
                        "x_local_px": cl,
                        "y_local_px": cl,
                        "x_global_px": cl + ox,
                        "y_global_px": cl + oy,
                        "target": f"Gene_{cid % 3}",
                    }
                )
        _write_csv(root / f"{prefix}_tx_file.csv.gz", pd.DataFrame(rows), compress=True)

        morph = root / "Morphology2D"
        for fov in fovs:
            _write_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF", fov_size=TIFF_FOV_SIZE)
        return root

    @staticmethod
    def _gy_extent(elem):
        import numpy as np
        from spatialdata.transformations import get_transformation

        t = get_transformation(elem, "global")
        m = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        a, b = float(m[1, 1]), float(m[1, 2])
        try:  # points
            d = elem.compute()
            ys = np.asarray(d["y"], float)
            return a * ys.min() + b, a * ys.max() + b
        except AttributeError:  # shapes
            lo = float(elem.geometry.bounds["miny"].min())
            hi = float(elem.geometry.bounds["maxy"].max())
            return a * lo + b, a * hi + b

    @pytest.mark.parametrize("offset", [0.0, -SPEC_FOV_SIZE, 3.0 * SPEC_FOV_SIZE])
    def test_transcripts_overlap_polygons(self, sdio, tmp_path, offset):
        root = self._build(tmp_path, "TxAlign", offset)
        # keep cell polygons as shapes (don't rasterise) so we can compare
        sdata = sdio.cosmx(
            root,
            n_workers=1,
            read_transcripts=True,
            read_proteins=False,
            polygons_as_labels=False,
        )
        pts = next(iter(sdata.points.values()))
        shp = next(v for k, v in sdata.shapes.items() if "box" not in k.lower())
        p_lo, p_hi = self._gy_extent(pts)
        s_lo, s_hi = self._gy_extent(shp)
        # transcript y-extent must sit within the polygon y-extent (+margin),
        # i.e. no one-FOV-height drift regardless of the global offset
        margin = SPEC_FOV_SIZE * 0.25
        assert p_lo >= s_lo - margin and p_hi <= s_hi + margin, (
            f"offset={offset}: points y[{p_lo:.0f},{p_hi:.0f}] not within polygons y[{s_lo:.0f},{s_hi:.0f}]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. POLYGON READING
# ═══════════════════════════════════════════════════════════════════════════


class TestPolygonReading:
    def test_valid_geometries(self, fixture_a):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs, _read_polygons_csv

        fl = _read_fov_locs(fixture_a / "S0_fov_positions_file.csv.gz")
        pdf = _read_polygons_csv(fixture_a / "S0-polygons.csv.gz", fov_locs=fl, fov_set={1}, use_polars=True)
        assert len(pdf) > 0
        assert pdf["geometry"].apply(lambda g: g.is_valid).all()

    def test_flip_y_still_valid(self, fixture_b):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs, _read_polygons_csv

        fl = _read_fov_locs(fixture_b / "Run5642_S3_Quarter_fov_positions_file.csv")
        pdf = _read_polygons_csv(
            fixture_b / "Run5642_S3_Quarter-polygons.csv", fov_locs=fl, fov_set={1}, use_polars=True
        )
        assert len(pdf) > 0
        assert pdf["geometry"].apply(lambda g: g.is_valid).all()

    def test_coords_finite(self, fixture_a):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs, _read_polygons_csv

        fl = _read_fov_locs(fixture_a / "S0_fov_positions_file.csv.gz")
        pdf = _read_polygons_csv(fixture_a / "S0-polygons.csv.gz", fov_locs=fl, fov_set={1, 2}, use_polars=True)
        for g in pdf["geometry"]:
            assert all(math.isfinite(v) for v in g.bounds)

    def test_cell_count_per_fov(self, fixture_a):
        from spatialdata_io.readers.cosmx._io import _read_fov_locs, _read_polygons_csv

        fl = _read_fov_locs(fixture_a / "S0_fov_positions_file.csv.gz")
        pdf = _read_polygons_csv(fixture_a / "S0-polygons.csv.gz", fov_locs=fl, fov_set={1}, use_polars=True)
        assert pdf["cell_ID"].nunique() == N_CELLS_PER_FOV

    def test_multi_polygon_cells_merged(self, tmp_path):
        """Cells with multiple polygon_index values are merged into MultiPolygon."""
        import shapely.geometry as sgeom

        from spatialdata_io.readers.cosmx._io import _read_polygons_csv

        # Build a polygon CSV where cell_ID=1 has TWO polygon parts
        spacing = SPEC_FOV_SIZE / (N_CELLS_PER_FOV + 1)
        rows = []
        # Part 1 of cell 1
        for pt in _make_polygon_rows(1, 1, spacing, spacing, r=50.0):
            pt["polygon_index"] = 0
            rows.append(pt)
        # Part 2 of cell 1 (offset)
        for pt in _make_polygon_rows(1, 1, spacing + 200, spacing + 200, r=50.0):
            pt["polygon_index"] = 1
            rows.append(pt)
        # Cell 2 — single polygon
        for pt in _make_polygon_rows(1, 2, spacing * 2, spacing * 2, r=50.0):
            pt["polygon_index"] = 0
            rows.append(pt)

        poly_csv = tmp_path / "multi_poly.csv"
        _write_csv(poly_csv, pd.DataFrame(rows))

        fl = pd.DataFrame(
            {
                "xmin": [0.0],
                "ymin": [0.0],
                "xmax": [float(SPEC_FOV_SIZE)],
                "ymax": [float(SPEC_FOV_SIZE)],
            },
            index=[1],
        )
        fl.index.name = "fov"

        pdf = _read_polygons_csv(poly_csv, fov_locs=fl, fov_set={1}, use_polars=True)

        # Should have exactly 2 rows: one per cell
        assert len(pdf) == 2, f"Expected 2 rows (one per cell), got {len(pdf)}"
        assert set(pdf["cell_ID"]) == {1, 2}

        # Cell 1 should be a MultiPolygon, cell 2 a Polygon
        cell1 = pdf[pdf["cell_ID"] == 1].iloc[0]["geometry"]
        cell2 = pdf[pdf["cell_ID"] == 2].iloc[0]["geometry"]
        assert isinstance(cell1, sgeom.MultiPolygon), f"Expected MultiPolygon, got {type(cell1)}"
        assert len(cell1.geoms) == 2
        assert isinstance(cell2, sgeom.Polygon)


# ═══════════════════════════════════════════════════════════════════════════
# 6. END-TO-END NORMALIZED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════


def _assert_normalized(sdata):
    """Invariants that must hold for ALL fixture outputs."""
    from spatialdata import SpatialData

    assert isinstance(sdata, SpatialData)

    # tables
    assert len(sdata.tables) > 0
    for _name, tbl in sdata.tables.items():
        assert tbl.n_obs > 0
        assert tbl.n_vars > 0
        assert "global_cell_id" in tbl.obs.columns
        assert (tbl.obs["global_cell_id"] != 0).all(), "background rows should be filtered"
        assert "region_key" in tbl.obs.columns
        # region_key is allowed to be categorical (spatialdata enforces it)
        for col in tbl.obs.columns:
            if col == "region_key":
                continue
            assert not isinstance(tbl.obs[col].dtype, pd.CategoricalDtype), f"categorical {col} will break zarr"

    # shapes
    assert len(sdata.shapes) > 0
    fov_box_keys = [k for k in sdata.shapes if "fov_box" in k]
    assert len(fov_box_keys) > 0


class TestNormalizedOutput:
    """All fixtures produce a standardized SpatialData after cosmx()."""

    def test_fixture_a_adjacent(self, sdio, fixture_a):
        """Two adjacent FOVs: images should stitch, labels/tables should align."""
        sdata = _read(sdio, fixture_a, fovs=[1, 2])
        _assert_normalized(sdata)
        assert len(sdata.images) > 0
        assert len(sdata.labels) > 0

    def test_fixture_b_non_adjacent(self, sdio, fixture_b):
        """Two non-adjacent FOVs: gap between FOVs should be handled."""
        sdata = _read(sdio, fixture_b, fovs=[1, 5])
        _assert_normalized(sdata)
        assert len(sdata.images) > 0
        assert len(sdata.labels) > 0

    def test_fixture_c_single_fov(self, sdio, fixture_c):
        """Single FOV with CellLabels (no Morphology2D)."""
        sdata = _read(sdio, fixture_c, fovs=[1], read_images=False)
        _assert_normalized(sdata)
        assert len(sdata.labels) > 0

    def test_single_fov_from_multi(self, sdio, fixture_a):
        """Selecting 1 FOV from a multi-FOV dataset should use F00001_ prefix."""
        sdata = _read(sdio, fixture_a, fovs=[1])
        _assert_normalized(sdata)
        all_names = list(sdata.images.keys()) + list(sdata.labels.keys()) + list(sdata.shapes.keys())
        assert any("F00001" in n for n in all_names), f"Expected F00001 prefix, got {all_names}"

    def test_polygons_as_shapes(self, sdio, fixture_a):
        """polygons_as_labels=False → shape elements contain cell polygons."""
        sdata = _read(sdio, fixture_a, fovs=[1, 2], polygons_as_labels=False)
        poly_keys = [k for k in sdata.shapes if "cells_polygons" in k]
        assert len(poly_keys) > 0
        gdf = sdata.shapes[poly_keys[0]]
        assert len(gdf) > 0
        assert gdf.geometry.is_valid.all()


# ═══════════════════════════════════════════════════════════════════════════
# 7. FOV PLACEMENT GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════


def _img_shape(sdata):
    """Get (C, Y, X) shape from the first image (may be DataTree or DataArray)."""
    img = next(iter(sdata.images.values()))
    # Image2DModel with scale_factors produces a DataTree
    if hasattr(img, "ds"):
        # xarray DataTree: base level is scale0
        arr = img["scale0"].ds["image"]
    elif hasattr(img, "shape"):
        arr = img
    else:
        # try to get the base resolution from the tree
        arr = img[list(img.keys())[0]].ds[list(img[list(img.keys())[0]].ds.keys())[0]]
    return tuple(arr.shape)


class TestLabelAnchoring:
    """Co-registration is label-anchored: the segmentation is the ground truth."""

    def test_segmented_but_unquantified_cell_kept(self, sdio, tmp_path):
        # A polygon with no expression row (cell 6, the highest cell_ID) is an
        # orphan. It must be KEPT in the labels (segmentation = ground truth) while
        # the table holds only the 5 quantified cells. Its high cell_ID also guards
        # the max_cell_id prescan, which must read the polygon 'cellID' column.
        import dask.array as da

        root = tmp_path / "orphan"
        root.mkdir()
        fov_pos = {1: (0.0, 0.0)}
        fov_df = pd.DataFrame(
            {"FOV": [1], "x_global_px": [0.0], "y_global_px": [0.0], "x_global_mm": [0.0], "y_global_mm": [0.0]}
        )
        _write_csv(root / "S0_fov_positions_file.csv.gz", fov_df, compress=True)
        _write_csv(root / "S0-polygons.csv.gz", _make_polygon_df(fov_pos, n_cells=6), compress=True)
        _write_csv(root / "S0_exprMat_file.csv.gz", _make_expr_mat([1], n_cells=5), compress=True)
        _write_csv(root / "S0_metadata_file.csv.gz", _make_metadata([1], n_cells=5), compress=True)

        sd = sdio.cosmx(root, fovs=[1], read_images=False, read_transcripts=False, read_proteins=False, n_workers=1)
        lab = next(iter(sd.labels.values()))
        arr = lab.data if hasattr(lab, "data") else lab["scale0"]["image"].data
        n_label = int((np.asarray(da.unique(arr).compute()) != 0).sum())
        n_table = next(iter(sd.tables.values())).n_obs
        assert n_label == 6, f"labels must keep all 6 segmented cells (incl. the orphan), got {n_label}"
        assert n_table == 5, f"table must hold only the 5 quantified cells, got {n_table}"


class TestFovPlacement:
    """Verify stitching geometry for adjacent, non-adjacent, and single FOV."""

    def test_adjacent_stitched_width(self, sdio, fixture_a):
        """Two side-by-side FOVs → stitched canvas ≈ 2 * SPEC_FOV_SIZE wide."""
        import dask.array as da

        sdata = _read(sdio, fixture_a, fovs=[1, 2])
        if not sdata.images:
            pytest.skip("No images")
        c, h, w = _img_shape(sdata)
        # _snap_fov_grid forces 4256, so canvas = 2*4256
        assert w > SPEC_FOV_SIZE, f"Expected width > {SPEC_FOV_SIZE}, got {w}"
        assert w <= 2 * SPEC_FOV_SIZE + 10

        # Verify the image has actual data, not all zeros
        img = next(iter(sdata.images.values()))
        if hasattr(img, "ds"):
            arr = img["scale0"].ds["image"]
        else:
            arr = img
        data = arr.data if isinstance(arr.data, da.Array) else arr
        # Check a region that should have FOV data
        sample = data[0, :TIFF_FOV_SIZE, :TIFF_FOV_SIZE].compute()
        assert sample.max() > 0, "Stitched image is all zeros in FOV 1 region"

    def test_non_adjacent_has_gap(self, sdio, fixture_b):
        """Non-adjacent FOVs → canvas larger than 2 * SPEC_FOV_SIZE."""
        sdata = _read(sdio, fixture_b, fovs=[1, 5])
        if not sdata.images:
            pytest.skip("No images")
        c, h, w = _img_shape(sdata)
        assert w > 2 * SPEC_FOV_SIZE, f"Non-adjacent gap not reflected: w={w}"

    def test_single_fov_size(self, sdio, fixture_a):
        """Single FOV → image matches the actual TIFF size (no snapping)."""
        sdata = _read(sdio, fixture_a, fovs=[1])
        if not sdata.images:
            pytest.skip("No images")
        c, h, w = _img_shape(sdata)
        assert h == TIFF_FOV_SIZE
        assert w == TIFF_FOV_SIZE

    def test_fov_boxes_geometry(self, sdio, fixture_a):
        """FOV boxes should be SPEC_FOV_SIZE x SPEC_FOV_SIZE squares."""
        sdata = _read(sdio, fixture_a, fovs=[1, 2])
        fov_keys = [k for k in sdata.shapes if "fov_box" in k]
        assert len(fov_keys) == 1
        gdf = sdata.shapes[fov_keys[0]]
        assert len(gdf) == 2
        for geom in gdf.geometry:
            minx, miny, maxx, maxy = geom.bounds
            w = maxx - minx
            h = maxy - miny
            assert abs(w - SPEC_FOV_SIZE) < 1
            assert abs(h - SPEC_FOV_SIZE) < 1

    def test_non_adjacent_fov_boxes_separated(self, sdio, fixture_b):
        """Non-adjacent FOV boxes should not overlap."""
        sdata = _read(sdio, fixture_b, fovs=[1, 5])
        fov_keys = [k for k in sdata.shapes if "fov_box" in k]
        assert len(fov_keys) == 1
        gdf = sdata.shapes[fov_keys[0]]
        assert len(gdf) == 2
        box1, box2 = gdf.geometry.iloc[0], gdf.geometry.iloc[1]
        assert not box1.intersects(box2), "Non-adjacent FOV boxes should not overlap"


def _img_array(sdata):
    """Materialize the first image element as a numpy array (C, Y, X)."""
    img = next(iter(sdata.images.values()))
    if hasattr(img, "ds"):
        arr = img["scale0"].ds["image"]
    elif hasattr(img, "shape"):
        arr = img
    else:
        arr = img[list(img.keys())[0]].ds[list(img[list(img.keys())[0]].ds.keys())[0]]
    data = arr.data
    return np.asarray(data.compute() if hasattr(data, "compute") else data)


class TestImageFlipOrientation:
    """Morphology image orientation must not depend on whether polygons are read.

    Regression for #42: the morphology vertical flip was gated on
    ``align_rasters_to_polygons`` (which is True only when polygons/labels are
    read), so px-only (``flip_y=True``) images were flipped when polygons were
    read but left mirrored otherwise. The flip is now applied consistently
    (``flip_image`` defaults to True), independent of polygon reading.

    fixture_b is px-only (``flip_y=True``) with random-valued Morphology2D TIFFs,
    so a vertical flip is detectable. Pre-fix: read_polygons=True flips the image
    while read_polygons=False does not, so the two arrays differ. Post-fix: both
    are flipped, so the arrays are identical.
    """

    def test_orientation_independent_of_polygons(self, sdio, fixture_b):
        common = {
            "n_workers": 1,
            "read_transcripts": False,
            "read_proteins": False,
            "read_gexp": False,
            "read_labels": False,
        }
        sd_with = sdio.cosmx(fixture_b, read_polygons=True, **common)
        sd_without = sdio.cosmx(fixture_b, read_polygons=False, **common)
        if not sd_with.images or not sd_without.images:
            pytest.skip("No images")
        a_with = _img_array(sd_with)
        a_without = _img_array(sd_without)
        assert a_with.shape == a_without.shape, (
            f"image shape changed with read_polygons: {a_with.shape} vs {a_without.shape}"
        )
        assert np.array_equal(a_with, a_without), (
            "morphology image orientation changed depending on read_polygons (#42)"
        )

    def test_explicit_flip_image_false_respected(self, sdio, fixture_b):
        """An explicit flip_image=False must override the default flip.

        Single FOV so a per-tile flip equals a whole-canvas flip (the reader
        flips each FOV tile, not the stitched canvas).
        """
        common = {
            "n_workers": 1,
            "fovs": [1],
            "read_transcripts": False,
            "read_proteins": False,
            "read_gexp": False,
            "read_labels": False,
            "read_polygons": False,
        }
        a_default = _img_array(sdio.cosmx(fixture_b, **common))  # flip_image=None -> True
        a_noflip = _img_array(sdio.cosmx(fixture_b, flip_image=False, **common))
        assert np.array_equal(a_default, a_noflip[:, ::-1, :]), (
            "flip_image=False should yield the un-flipped image (vertical mirror of the default flipped one)"
        )


class TestMultimodalImageFlip:
    """The morphology flip must also be applied on the multimodal (RNA+Protein)
    read path.

    Regression guard for #42: the `flip_image` default (True) is resolved at the
    top of `cosmx()`, BEFORE the multimodal dispatch — resolving it afterwards
    left multimodal `flip_image=None -> False`, so multimodal morphology images
    came out un-flipped (mirrored). The single-modality flip tests above do NOT
    exercise `_cosmx_multi`, so this dedicated multimodal fixture is needed.

    fixture_multimodal's morphology is asymmetric (top 100 / bottom 200); a
    correct vertical flip makes the stored top half hold the larger (bottom)
    value, which we detect via top- vs bottom-region means (robust to the
    reader's float intensity normalization).
    """

    def test_dispatches_to_multimodal(self, fixture_multimodal):
        from spatialdata_io.readers.cosmx._discovery import (
            _set_up_cosmx_dataset_for_conversion,
        )

        ds = _set_up_cosmx_dataset_for_conversion(fixture_multimodal)
        assert ds.modalities is not None, "fixture did not register as multimodal"
        assert set(ds.modalities) == {"RNA", "Protein"}

    def test_multimodal_morphology_is_flipped(self, sdio, fixture_multimodal):
        sdata = sdio.cosmx(
            fixture_multimodal,
            fovs=[1],
            n_workers=1,
            read_transcripts=False,
            read_proteins=False,
            read_gexp=False,
            read_labels=False,
            read_polygons=True,
        )
        if not sdata.images:
            pytest.skip("No images")
        arr = _img_array(sdata)  # (C, Y, X), float-normalized
        h = arr.shape[1]
        top = float(arr[0, : h // 2, :].mean())
        bottom = float(arr[0, h // 2 :, :].mean())
        # raw top=100, bottom=200; after the vertical flip top must hold the
        # larger (originally-bottom) value.
        assert top > bottom, (
            "multimodal morphology image was not vertically flipped (#42): the "
            f"flip default did not reach the multimodal path (top={top:.5f} "
            f"!> bottom={bottom:.5f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. PREVIEW
# ═══════════════════════════════════════════════════════════════════════════


class TestPreview:
    def test_returns_none(self, sdio, fixture_a):
        import matplotlib

        matplotlib.use("Agg")
        assert sdio.cosmx(fixture_a, preview_fovs=True) is None

    def test_with_subset(self, sdio, fixture_a):
        import matplotlib

        matplotlib.use("Agg")
        assert sdio.cosmx(fixture_a, fovs=[1], preview_fovs=True) is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. ZARR ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════


class TestZarrRoundTrip:
    def test_write_read(self, sdio, fixture_a, tmp_path):
        import dask.array as da
        import spatialdata as sd

        sdata = _read(sdio, fixture_a, fovs=[1], read_images=False)

        # Verify labels have non-zero values BEFORE writing
        if sdata.labels:
            lbl = next(iter(sdata.labels.values()))
            lbl_np = lbl.data.compute() if isinstance(lbl.data, da.Array) else np.asarray(lbl)
            assert lbl_np.max() > 0, "Labels are all zeros before writing"

        zarr_path = tmp_path / "test.zarr"
        sdata.write(zarr_path)
        sdata2 = sd.read_zarr(zarr_path)
        assert set(sdata.labels.keys()) == set(sdata2.labels.keys())
        assert set(sdata.shapes.keys()) == set(sdata2.shapes.keys())
        assert set(sdata.tables.keys()) == set(sdata2.tables.keys())
        for name in sdata.tables:
            assert sdata.tables[name].n_obs == sdata2.tables[name].n_obs

        # Verify labels survived round-trip with non-zero values
        if sdata2.labels:
            lbl2 = next(iter(sdata2.labels.values()))
            lbl2_np = lbl2.data.compute() if isinstance(lbl2.data, da.Array) else np.asarray(lbl2)
            assert lbl2_np.max() > 0, "Labels are all zeros after round-trip"
            assert set(np.unique(lbl2_np)) == set(np.unique(lbl_np)), "Label IDs changed after round-trip"


# ═══════════════════════════════════════════════════════════════════════════
# 10. SNAPSHOT VALUES
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# skip_empty_fovs / phantom FOVs (issue #37)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def fixture_phantom(tmp_path_factory) -> Path:
    """Positions list FOV 1 (phantom: no data files) + FOV 2 (real). px-only.

    Only FOV 2 ships a Morphology2D TIFF and a CellLabels TIFF, so a correct
    reader should size both rasters to one FOV. Phantom FOV 1 sits at the origin
    (smaller coords) so leaving it in inflates the image canvas (#37).
    """
    root = tmp_path_factory.mktemp("fixture_phantom")
    prefix = "Run_Ph"
    pos_fovs = [1, 2]
    real = [2]
    fov_pos = {1: (0.0, 0.0), 2: (float(SPEC_FOV_SIZE), 0.0)}

    fov_df = pd.DataFrame(
        {
            "fov": pos_fovs,
            "x_global_px": [fov_pos[f][0] for f in pos_fovs],
            "y_global_px": [fov_pos[f][1] for f in pos_fovs],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv", fov_df)
    _write_csv(root / f"{prefix}_exprMat_file.csv", _make_expr_mat(real))
    _write_csv(root / f"{prefix}_metadata_file.csv", _make_metadata(real))

    morph = root / "Morphology2D"
    labels = root / "CellLabels"
    for fov in real:
        _write_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF", fov_size=TIFF_FOV_SIZE)
        _write_cell_label_tiff(labels / f"CellLabels_F{fov:03d}.tif")
    return root


@pytest.fixture(scope="session")
def fixture_no_perfov(tmp_path_factory) -> Path:
    """Positions for 2 FOVs but NO per-FOV image/label files (transcript/gexp
    style). ``detect_fovs_with_data`` returns empty → ``skip_empty_fovs`` must
    keep all FOVs rather than dropping the whole dataset (#37 safe rule).
    """
    root = tmp_path_factory.mktemp("fixture_noperfov")
    prefix = "Run_NP"
    fovs = [1, 2]
    fov_pos = {1: (0.0, 0.0), 2: (float(SPEC_FOV_SIZE), 0.0)}
    fov_df = pd.DataFrame(
        {
            "fov": fovs,
            "x_global_px": [fov_pos[f][0] for f in fovs],
            "y_global_px": [fov_pos[f][1] for f in fovs],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv", fov_df)
    _write_csv(root / f"{prefix}_exprMat_file.csv", _make_expr_mat(fovs))
    _write_csv(root / f"{prefix}_metadata_file.csv", _make_metadata(fovs))
    return root


def _label_shape(sdata):
    """(Y, X) shape of the first label element."""
    lbl = next(iter(sdata.labels.values()))
    if hasattr(lbl, "ds"):
        arr = lbl["scale0"].ds["image"]
    elif hasattr(lbl, "shape"):
        arr = lbl
    else:
        arr = lbl[list(lbl.keys())[0]].ds[list(lbl[list(lbl.keys())[0]].ds.keys())[0]]
    return tuple(arr.shape)


def _global_x(elem):
    """Global-transform x translation of a raster element."""
    from spatialdata.transformations import get_transformation

    t = get_transformation(elem, "global")
    return float(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))[0, 2])


class TestSkipEmptyFovs:
    """Regression for #37: phantom FOVs (positions but no data files) must not
    inflate the image canvas, desync the image/label canvases, or add empty
    FOV boxes.
    """

    _COMMON = {
        "n_workers": 1,
        "read_transcripts": False,
        "read_proteins": False,
        "read_gexp": False,
        "read_polygons": False,
        "polygons_as_labels": False,
        "read_images": True,
        "read_labels": True,
    }

    def test_prune_tightens_and_matches_labels(self, sdio, fixture_phantom):
        sd = sdio.cosmx(fixture_phantom, skip_empty_fovs=True, **self._COMMON)
        _, ih, iw = _img_shape(sd)
        lh, lw = _label_shape(sd)
        assert iw <= SPEC_FOV_SIZE + 10, f"image canvas not tightened: {iw}"
        assert (ih, iw) == (lh, lw), f"image {ih}x{iw} != label {lh}x{lw}"
        # both rasters land on the real FOV 2 (global x = SPEC_FOV_SIZE)
        assert abs(_global_x(next(iter(sd.images.values()))) - SPEC_FOV_SIZE) <= 1
        assert abs(_global_x(next(iter(sd.labels.values()))) - SPEC_FOV_SIZE) <= 1

    def test_prune_drops_phantom_fov_box(self, sdio, fixture_phantom):
        sd = sdio.cosmx(fixture_phantom, skip_empty_fovs=True, add_fovs_as_shapes=True, **self._COMMON)
        keys = [k for k in sd.shapes if "fov_box" in k]
        assert keys and len(sd.shapes[keys[0]]) == 1, "phantom FOV box not pruned"

    def test_no_skip_still_tightens_image_via_seen_filter(self, sdio, fixture_phantom):
        # Part 2: even with skip_empty_fovs=False, the image canvas is tightened
        # to image-bearing FOVs and stays co-registered with the labels.
        sd = sdio.cosmx(fixture_phantom, skip_empty_fovs=False, **self._COMMON)
        _, ih, iw = _img_shape(sd)
        lh, lw = _label_shape(sd)
        assert iw <= SPEC_FOV_SIZE + 10, f"image canvas not tightened w/o skip: {iw}"
        assert (ih, iw) == (lh, lw)
        assert abs(_global_x(next(iter(sd.images.values()))) - SPEC_FOV_SIZE) <= 1

    def test_no_skip_keeps_phantom_fov_box(self, sdio, fixture_phantom):
        # Read without labels: the label path independently prunes fov_locs to
        # seen labels, so this isolates the skip_empty_fovs=False behaviour
        # (phantom FOV kept in fov_locs → still gets a box).
        sd = sdio.cosmx(
            fixture_phantom,
            skip_empty_fovs=False,
            add_fovs_as_shapes=True,
            n_workers=1,
            read_transcripts=False,
            read_proteins=False,
            read_gexp=False,
            read_polygons=False,
            polygons_as_labels=False,
            read_images=True,
            read_labels=False,
        )
        keys = [k for k in sd.shapes if "fov_box" in k]
        assert keys and len(sd.shapes[keys[0]]) == 2, "skip_empty_fovs=False must keep every listed FOV box"

    def test_explicit_phantom_request_not_read_as_all(self, sdio, fixture_phantom):
        # Requesting only the phantom FOV must NOT collapse to "all FOVs": the
        # reader should produce no image (FOV 1 has none), not silently load FOV 2.
        sd = sdio.cosmx(
            fixture_phantom,
            skip_empty_fovs=True,
            fovs=[1],
            n_workers=1,
            read_transcripts=False,
            read_proteins=False,
            read_gexp=False,
            read_polygons=False,
            polygons_as_labels=False,
            read_images=True,
            read_labels=False,
        )
        assert not sd.images, "requesting a phantom FOV silently read other FOVs' images"

    def test_no_perfov_files_keeps_all_fovs(self, sdio, fixture_no_perfov):
        # detection finds no per-FOV files → must NOT prune (would nuke the set)
        sd = sdio.cosmx(
            fixture_no_perfov,
            skip_empty_fovs=True,
            n_workers=1,
            read_images=False,
            read_labels=False,
            read_transcripts=False,
            read_gexp=False,
            read_polygons=False,
            polygons_as_labels=False,
            add_fovs_as_shapes=True,
        )
        keys = [k for k in sd.shapes if "fov_box" in k]
        assert keys and len(sd.shapes[keys[0]]) == 2


class TestDetectFovsWithData:
    """Unit tests for the per-FOV data detector (#37)."""

    def test_detects_across_sources(self, tmp_path):
        from spatialdata_io.readers.cosmx._utils import detect_fovs_with_data

        morph = tmp_path / "Morphology2D"
        _write_morphology_tiff(morph / "20240101_S0_F00003.TIF", fov_size=TIFF_FOV_SIZE)
        labels = tmp_path / "CellLabels"
        _write_cell_label_tiff(labels / "CellLabels_F007.tif")
        # CellStatsDir: a FOV with a label TIF counts; an empty FOV dir does NOT.
        _write_cell_label_tiff(tmp_path / "CellStatsDir" / "FOV010" / "CellLabels_F010.tif")
        (tmp_path / "CellStatsDir" / "FOV011").mkdir(parents=True)
        got = detect_fovs_with_data(
            morphology_2d_dir=morph,
            cell_labels_dir=labels,
            cell_stats_dir=tmp_path / "CellStatsDir",
        )
        assert got == {3, 7, 10}  # 11 excluded: empty dir, no TIF

    def test_empty_when_no_files(self, tmp_path):
        from spatialdata_io.readers.cosmx._utils import detect_fovs_with_data

        assert detect_fovs_with_data(morphology_2d_dir=tmp_path / "missing") == set()


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE NORMALIZATION (issue #38) — opt-in, scale-only
# ═══════════════════════════════════════════════════════════════════════════


def _write_low_signal_morphology_tiff(
    path: Path, n_channels: int = N_CHANNELS, fov_size: int = TIFF_FOV_SIZE, bg: int = 100, bright: int = 5000
):
    """Morphology TIFF whose real signal (``bright``) sits FAR below the uint16
    ceiling (65535) — so dtype-max scaling renders it near-black (#38) while a
    per-channel percentile stretch recovers it.  A quarter-FOV block is bright
    (well above the 99.9th-percentile floor); the rest is low background.
    """
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((n_channels, fov_size, fov_size), bg, dtype=np.uint16)
    q = fov_size // 4
    data[:, :q, :q] = bright
    tifffile.imwrite(str(path), data, description=_tiff_description(), photometric="minisblack")


@pytest.fixture(scope="session")
def fixture_lowsig(tmp_path_factory) -> Path:
    """px+mm, single modality, TWO ADJACENT FOVs with LOW-signal morphology
    (bright=5000 ≪ 65535) — exercises the #38 stretch end-to-end.
    """
    root = tmp_path_factory.mktemp("fixture_lowsig")
    prefix = "S0"
    fovs = [1, 2]
    fov_pos = {1: (0.0, 0.0), 2: (float(SPEC_FOV_SIZE), 0.0)}
    x_mm = {f: x / MM_TO_PX for f, (x, _) in fov_pos.items()}
    y_mm = {f: y / MM_TO_PX for f, (_, y) in fov_pos.items()}
    fov_df = pd.DataFrame(
        {
            "FOV": fovs,
            "x_global_px": [fov_pos[f][0] for f in fovs],
            "y_global_px": [fov_pos[f][1] for f in fovs],
            "x_global_mm": [x_mm[f] for f in fovs],
            "y_global_mm": [y_mm[f] for f in fovs],
        }
    )
    _write_csv(root / f"{prefix}_fov_positions_file.csv.gz", fov_df, compress=True)
    _write_csv(root / f"{prefix}_exprMat_file.csv.gz", _make_expr_mat(fovs), compress=True)
    _write_csv(root / f"{prefix}_metadata_file.csv.gz", _make_metadata(fovs), compress=True)
    _write_csv(root / f"{prefix}-polygons.csv.gz", _make_polygon_df(fov_pos), compress=True)
    morph = root / "Morphology2D"
    for fov in fovs:
        _write_low_signal_morphology_tiff(morph / f"20240101_S0_F{fov:05d}.TIF", fov_size=TIFF_FOV_SIZE)
    return root


class TestImageNormalizationPrimitive:
    """Unit tests for ``_normalize_image_channels`` (issue #38)."""

    @staticmethod
    def _norm(arr, names, **kw):
        import dask.array as da

        from spatialdata_io.readers.cosmx._utils import _normalize_image_channels

        return _normalize_image_channels(da.from_array(arr, chunks=(1,) + arr.shape[1:]), names, **kw)

    def test_none_is_passthrough(self):
        # Opt-in: percentile=None leaves the (already dtype-max normalized) image untouched.
        ch = np.full((1, 64, 64), 0.0015, dtype="float32")
        ch[0, :8, :8] = 0.05
        out, scales = self._norm(ch, ["DNA"], percentile=None)
        out = np.asarray(out.compute())
        assert out.dtype == np.float32 and scales == {}
        assert np.array_equal(out, ch)

    def test_percentile_recovers_low_signal(self):
        # Regression for #38: signal at 3000 (≪ ceiling) reaches ~1.0 via its own percentile.
        ch = np.full((1, 128, 128), 50, dtype="uint16")
        ch[0, :16, :16] = 3000
        out, scales = self._norm(ch, ["DNA"], percentile=99.9)
        out = np.asarray(out.compute())
        assert out.dtype == np.float32
        assert out.max() > 0.9, f"low-signal channel not recovered: max={out.max()}"
        assert scales["DNA"] > 0

    def test_scale_only_is_reversible(self):
        # No clipping: the brightest pixels (above the percentile) exceed 1.0, and
        # multiplying back by the divisor recovers the input exactly.
        ch = np.full((1, 64, 64), 40, dtype="uint16")
        ch[0, :10, :10] = 5000  # bulk signal -> ~99.9th percentile
        ch[0, 0, :2] = 50000  # a few pixels brighter than the percentile
        out, scales = self._norm(ch, ["DNA"], percentile=99.9)
        out = np.asarray(out.compute())
        assert out.max() > 1.0, "scale-only must not clip the brightest pixels to 1.0"
        np.testing.assert_allclose(out * scales["DNA"], ch.astype("float32"), rtol=1e-4)

    def test_multi_chunk_percentile(self):
        # Production stitched images are multi-chunk, where da.percentile is approximate.
        # Exercise that path: it must run, recover signal, and stay exactly reversible.
        import dask.array as da

        from spatialdata_io.readers.cosmx._utils import _normalize_image_channels

        arr = np.full((1, 300, 300), 40, dtype="uint16")
        arr[0, :30, :30] = 8000
        out, scales = _normalize_image_channels(da.from_array(arr, chunks=(1, 128, 128)), ["DNA"], percentile=99.9)
        out = np.asarray(out.compute())
        assert out.dtype == np.float32 and scales["DNA"] > 0
        assert out.max() > 0.5, "signal not recovered on the multi-chunk path"
        np.testing.assert_allclose(out * scales["DNA"], arr.astype("float32"), rtol=1e-4)

    def test_zeros_ignored_in_percentile(self):
        # Zero padding (inter-FOV gaps) must not bias the percentile toward 0.
        ch = np.zeros((1, 64, 64), dtype="uint16")  # mostly zero "canvas"
        ch[0, :8, :8] = 100  # a single covered FOV: background
        ch[0, :2, :2] = 5000  # signal within it
        _, scales = self._norm(ch, ["DNA"], percentile=99.9)
        assert scales["DNA"] > 100, "percentile collapsed onto the zero padding"

    def test_empty_channel_left_unscaled_without_crash(self):
        zeros = np.zeros((1, 32, 32), dtype="uint16")
        out, scales = self._norm(zeros, ["empty"], percentile=99.9)
        out = np.asarray(out.compute())
        assert out.max() == 0.0 and scales["empty"] == 1.0  # unscaled fallback, no crash

    def test_float_channel_with_nan_does_not_propagate(self):
        fl = np.full((1, 16, 16), 5.0, dtype="float32")
        fl[0, 0, 0] = np.nan
        out, _ = self._norm(fl, ["f"], percentile=99.9)
        out = np.asarray(out.compute())
        assert np.isnan(out).sum() == 1, "a single NaN must not spread to the whole channel"


class TestImageNormalizationEndToEnd:
    """End-to-end through ``read_images`` — single-, multi-FOV, and multimodal paths (#38)."""

    def test_default_is_legacy_dim(self, sdio, fixture_lowsig):
        # Opt-in: the default (None) keeps dtype-max scaling, so low signal stays dim.
        arr = _img_array(_read(sdio, fixture_lowsig, fovs=[1]))
        assert arr.dtype == np.float32
        assert arr.max() < 0.2, "default must keep legacy dtype-max (dim) scaling"

    def test_percentile_recovers_single_fov(self, sdio, fixture_lowsig):
        arr = _img_array(_read(sdio, fixture_lowsig, fovs=[1], image_normalization_percentile=99.9))
        assert arr.max() > 0.9, "opt-in percentile did not recover the low-signal single-FOV image (#38)"

    def test_percentile_recovers_stitched_multi_fov(self, sdio, fixture_lowsig):
        arr = _img_array(_read(sdio, fixture_lowsig, fovs=[1, 2], image_normalization_percentile=99.9))
        assert arr.dtype == np.float32
        assert arr.max() > 0.9, "opt-in percentile did not reach the stitched multi-FOV image (#38)"

    def test_multimodal_percentile_reaches_morphology(self, sdio, fixture_multimodal):
        # The param must thread through _cosmx_multi; fixture morphology is low-signal.
        kw = {
            "fovs": [1],
            "n_workers": 1,
            "read_transcripts": False,
            "read_proteins": False,
            "read_gexp": False,
            "read_labels": False,
            "read_polygons": True,
        }
        arr = _img_array(sdio.cosmx(fixture_multimodal, image_normalization_percentile=99.9, **kw))
        assert arr.max() > 0.9, "opt-in normalization did not reach the multimodal morphology image"
        arr0 = _img_array(sdio.cosmx(fixture_multimodal, **kw))
        assert arr0.max() < 0.05, "default should keep legacy (dim) scaling on the multimodal path"

    def test_scales_recorded_reversible_and_survive_round_trip(self, sdio, fixture_lowsig, tmp_path):
        import spatialdata as sd_mod

        sd = _read(sdio, fixture_lowsig, fovs=[1], image_normalization_percentile=99.9)
        meta = sd.attrs.get("cosmx_image_normalization")
        assert meta, "per-channel normalization divisors not recorded in sdata.attrs"
        (img_meta,) = list(meta.values())
        assert img_meta["percentile"] == 99.9 and img_meta["channel_scales"]
        # Reversible: normalized * divisor recovers the dtype-max image read with percentile=None.
        a_norm = np.asarray(_img_array(sd))
        a_raw = np.asarray(_img_array(_read(sdio, fixture_lowsig, fovs=[1])))
        divs = np.array(list(img_meta["channel_scales"].values()), dtype="float32")[:, None, None]
        np.testing.assert_allclose(a_norm * divs, a_raw, rtol=1e-3, atol=1e-4)
        zp = tmp_path / "norm.zarr"
        sd.write(zp)
        assert sd_mod.read_zarr(zp).attrs.get("cosmx_image_normalization"), "normalization attrs lost on round-trip"

    def test_out_of_range_percentile_raises(self, sdio, fixture_lowsig):
        with pytest.raises(ValueError, match="image_normalization_percentile"):
            _read(sdio, fixture_lowsig, fovs=[1], image_normalization_percentile=150.0)
