"""Tests for the mcmicro reader across the labsyspharm and nf-core pipeline layouts.

The reader accepts two different mcmicro output directory layouts (the original
``labsyspharm/mcmicro`` Nextflow pipeline and the newer ``nf-core/mcmicro`` pipeline), in
either whole-slide-image (WSI) or tissue-microarray (TMA) mode. These tests build minimal
synthetic output trees on disk (following the ``test_macsima.py`` pattern) and assert that the
reader auto-detects the pipeline and produces the expected image/label/table keys.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from click.testing import CliRunner
from spatialdata import get_extent, read_zarr

from spatialdata_io.__main__ import mcmicro_wrapper
from spatialdata_io._constants._constants import McmicroPipeline
from spatialdata_io.readers.mcmicro import _detect_pipeline, mcmicro

N_CHANNELS = 2
MARKER_NAMES = ["DAPI", "CD3"]


def _write_image(path: Path, n_channels: int = N_CHANNELS, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(n_channels, size, size), dtype=np.uint16)
    tifffile.imwrite(path, data)


def _write_mask(path: Path, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((size, size), dtype=np.uint32)
    data[:4, :4] = 1
    data[4:, 4:] = 2
    tifffile.imwrite(path, data)


def _write_markers(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"channel_number": range(1, N_CHANNELS + 1), "marker_name": MARKER_NAMES}).to_csv(path, index=False)


def _write_quant(path: Path, n_cells: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "CellID": range(1, n_cells + 1),
            **{name: np.arange(n_cells, dtype=float) for name in MARKER_NAMES},
            "X_centroid": np.arange(n_cells, dtype=float),
            "Y_centroid": np.arange(n_cells, dtype=float),
            "Area": np.arange(n_cells, dtype=float),
        }
    )
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------------------------- #
# tree builders
# --------------------------------------------------------------------------------------------- #
def _build_labsyspharm_wsi(root: Path, sample: str = "test") -> None:
    (root / "qc").mkdir(parents=True, exist_ok=True)
    with open(root / "qc" / "params.yml", "w") as fp:
        yaml.safe_dump({"workflow": {"tma": False}}, fp)
    _write_markers(root / "markers.csv")
    _write_image(root / "registration" / f"{sample}.ome.tif")
    _write_mask(root / "segmentation" / f"unmicst-{sample}" / "cell.ome.tif")
    _write_quant(root / "quantification" / f"{sample}--unmicst_cell.csv")


def _build_nfcore_wsi(root: Path, samples: list[str], markers: bool = True) -> None:
    (root / "pipeline_info").mkdir(parents=True, exist_ok=True)
    if markers:
        _write_markers(root / "markers.csv")
    for sample in samples:
        _write_image(root / "registration" / "ashlar" / f"{sample}.ome.tif")
        _write_mask(root / "segmentation" / "cellpose" / f"{sample}_mask.ome.tif")
        _write_quant(root / "quantification" / "mcquant" / "cellpose" / f"{sample}.csv")


def _write_centroids(path: Path, n_cores: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{10 * (i + 1)} {20 * (i + 1)}" for i in range(n_cores)]
    path.write_text("\n".join(lines) + "\n")


def _build_labsyspharm_tma(root: Path, cores: list[int]) -> None:
    (root / "qc").mkdir(parents=True, exist_ok=True)
    with open(root / "qc" / "params.yml", "w") as fp:
        yaml.safe_dump({"workflow": {"tma": True}}, fp)
    _write_markers(root / "markers.csv")
    _write_image(root / "registration" / "tma.ome.tif")
    _write_centroids(root / "qc" / "coreograph" / "centroidsY-X.txt", len(cores))
    for core in cores:
        _write_image(root / "dearray" / f"{core}.ome.tif")
        _write_mask(root / "dearray" / "masks" / f"{core}_mask.tif")
        _write_mask(root / "segmentation" / f"unmicst-{core}" / "cell.ome.tif")
        _write_quant(root / "quantification" / f"{core}--unmicst_cell.csv")


def _build_nfcore_tma(root: Path, cores: list[int], sample: str = "exemplar-002") -> None:
    """Mirror the real nf-core TMA layout.

    Numbered cores + a TMA_MAP, two segmenters with tool-specific mask names (cellpose
    ``{core}_cp_masks.tif``, mesmer ``{sample}_{core}_mask.tif``), and per-core-per-segmenter
    quantification with a prelude markersheet.
    """
    (root / "pipeline_info").mkdir(parents=True, exist_ok=True)
    (root / "prelude").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"channel_number": range(1, N_CHANNELS + 1), "marker_name": MARKER_NAMES}).to_csv(
        root / "prelude" / "markers_markersheet_mqc.tsv", sep="\t", index=False
    )
    _write_image(root / "registration" / "ashlar" / f"{sample}.ome.tif")
    _write_centroids(root / "tma_dearray" / "centroidsY-X.txt", len(cores))
    _write_mask(root / "tma_dearray" / "TMA_MAP.tif")  # must be ignored as a core image
    for core in cores:
        _write_image(root / "tma_dearray" / f"{core}.tif")
        _write_mask(root / "tma_dearray" / "masks" / f"{core}_mask.tif")
        _write_mask(root / "segmentation" / "cellpose" / f"{core}_cp_masks.tif")
        _write_mask(root / "segmentation" / "deepcell_mesmer" / f"{sample}_{core}_mask.tif")
        _write_quant(root / "quantification" / "mcquant" / "cellpose" / f"{core}_{core}_cp_masks.csv")
        _write_quant(root / "quantification" / "mcquant" / "mesmer" / f"{core}_{sample}_{core}_mask.csv")


# --------------------------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------------------------- #
def test_detect_labsyspharm(tmp_path: Path) -> None:
    _build_labsyspharm_wsi(tmp_path)
    assert _detect_pipeline(tmp_path) == McmicroPipeline.LABSYSPHARM


def test_detect_nfcore(tmp_path: Path) -> None:
    _build_nfcore_wsi(tmp_path, ["s1"])
    assert _detect_pipeline(tmp_path) == McmicroPipeline.NFCORE


def test_detect_raises_when_ambiguous(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not detect"):
        _detect_pipeline(tmp_path)


# --------------------------------------------------------------------------------------------- #
# WSI
# --------------------------------------------------------------------------------------------- #
def test_labsyspharm_wsi(tmp_path: Path) -> None:
    _build_labsyspharm_wsi(tmp_path, "test")
    sdata = mcmicro(tmp_path)
    assert set(sdata.images) == {"test_image"}
    assert "test_unmicst_cell" in sdata.labels
    assert "test--unmicst_cell" in sdata.tables
    assert sdata.tables["test--unmicst_cell"].obs["region"].cat.categories.tolist() == ["test_unmicst_cell"]
    assert list(sdata.images["test_image"].c.values) == MARKER_NAMES


def test_labsyspharm_wsi_multi_segmenter(tmp_path: Path) -> None:
    """Two segmenters writing the same `cell.ome.tif` must not collide on the label key."""
    (tmp_path / "qc").mkdir(parents=True)
    with open(tmp_path / "qc" / "params.yml", "w") as fp:
        yaml.safe_dump({"workflow": {"tma": False}}, fp)
    _write_markers(tmp_path / "markers.csv")
    _write_image(tmp_path / "registration" / "test.ome.tif")
    for seg in ["unmicst", "ilastik"]:
        _write_mask(tmp_path / "segmentation" / f"{seg}-test" / "cell.ome.tif")
        _write_quant(tmp_path / "quantification" / f"test--{seg}_cell.csv")
    sdata = mcmicro(tmp_path)
    assert {"test_unmicst_cell", "test_ilastik_cell"} <= set(sdata.labels)
    assert {"test--unmicst_cell", "test--ilastik_cell"} == set(sdata.tables)
    assert sdata.tables["test--ilastik_cell"].obs["region"].cat.categories.tolist() == ["test_ilastik_cell"]


def test_nfcore_wsi_single_sample(tmp_path: Path) -> None:
    _build_nfcore_wsi(tmp_path, ["s1"])
    sdata = mcmicro(tmp_path)
    assert set(sdata.images) == {"s1_image"}
    assert "s1_cellpose" in sdata.labels
    assert "s1_cellpose_table" in sdata.tables
    assert sdata.tables["s1_cellpose_table"].obs["region"].cat.categories.tolist() == ["s1_cellpose"]
    # WSI uses an identity transform: image and its label share the same global extent, anchored at 0
    img_ext = get_extent(sdata["s1_image"], coordinate_system="global")
    lab_ext = get_extent(sdata["s1_cellpose"], coordinate_system="global")
    assert img_ext["x"][0] == 0 and img_ext["y"][0] == 0
    assert img_ext == lab_ext


def test_nfcore_wsi_multi_sample(tmp_path: Path) -> None:
    _build_nfcore_wsi(tmp_path, ["s1", "s2", "s3"])
    sdata = mcmicro(tmp_path)
    assert set(sdata.images) == {"s1_image", "s2_image", "s3_image"}
    assert {"s1_cellpose", "s2_cellpose", "s3_cellpose"} <= set(sdata.labels)
    assert {"s1_cellpose_table", "s2_cellpose_table", "s3_cellpose_table"} == set(sdata.tables)


def test_nfcore_wsi_multi_segmenter(tmp_path: Path) -> None:
    """Mirror the real S3 multiseg layout: tool-specific mask names and mismatched dir names."""
    sample = "exemplar-001-10cycles"
    (tmp_path / "pipeline_info").mkdir(parents=True)
    _write_markers(tmp_path / "markers.csv")
    _write_image(tmp_path / "registration" / "ashlar" / f"{sample}.ome.tif")
    # segmentation dirs and mask filenames vary by tool
    _write_mask(tmp_path / "segmentation" / "cellpose" / f"{sample}_backsub.ome_cp_masks.tif")
    _write_mask(tmp_path / "segmentation" / "mccellpose" / f"{sample}_mask.ome.tif")
    _write_mask(tmp_path / "segmentation" / "deepcell_mesmer" / f"{sample}_mask.tif")
    # quantification dir names differ from segmentation dirs (mesmer vs deepcell_mesmer)
    _write_quant(tmp_path / "quantification" / "mcquant" / "cellpose" / f"{sample}_backsub_{sample}_backsub.csv")
    _write_quant(tmp_path / "quantification" / "mcquant" / "mccellpose" / f"{sample}.csv")
    _write_quant(tmp_path / "quantification" / "mcquant" / "mesmer" / f"{sample}.csv")

    sdata = mcmicro(tmp_path)
    assert set(sdata.images) == {f"{sample}_image"}
    assert {f"{sample}_cellpose", f"{sample}_mccellpose", f"{sample}_deepcell_mesmer"} <= set(sdata.labels)
    assert {f"{sample}_cellpose_table", f"{sample}_mccellpose_table", f"{sample}_mesmer_table"} == set(sdata.tables)
    # the mesmer table must link (by substring) to the deepcell_mesmer label element
    mesmer_region = sdata.tables[f"{sample}_mesmer_table"].obs["region"].cat.categories.tolist()
    assert mesmer_region == [f"{sample}_deepcell_mesmer"]


def test_nfcore_wsi_markers_fallback(tmp_path: Path) -> None:
    _build_nfcore_wsi(tmp_path, ["s1"], markers=False)
    # drop quantification so the derived channel names (which need not match the marker columns
    # written by mcquant) don't collide with the synthetic CSV columns
    import shutil

    shutil.rmtree(tmp_path / "quantification")
    with pytest.warns(UserWarning, match="No markers file found"):
        sdata = mcmicro(tmp_path)
    # OME metadata carries no channel names for our synthetic tiffs -> integer channel names
    assert len(sdata.images["s1_image"].c) == N_CHANNELS
    assert list(sdata.images["s1_image"].c.values) == ["channel_0", "channel_1"]


def test_explicit_pipeline_override(tmp_path: Path) -> None:
    # nf-core tree without pipeline_info/ still auto-detects via registration/ashlar,
    # and an explicit override is honored regardless.
    _build_nfcore_wsi(tmp_path, ["s1"])
    (tmp_path / "pipeline_info").rmdir()
    assert _detect_pipeline(tmp_path) == McmicroPipeline.NFCORE
    sdata = mcmicro(tmp_path, pipeline="nfcore")
    assert set(sdata.images) == {"s1_image"}


# --------------------------------------------------------------------------------------------- #
# TMA
# --------------------------------------------------------------------------------------------- #
def test_labsyspharm_tma(tmp_path: Path) -> None:
    _build_labsyspharm_tma(tmp_path, [1, 2])
    sdata = mcmicro(tmp_path)
    assert "tma_map" in sdata.images
    assert {"core_1_image", "core_2_image"} <= set(sdata.images)
    assert {"core_1_unmicst_cell", "core_2_unmicst_cell"} <= set(sdata.labels)
    assert "segmentation_table" in sdata.tables


def test_nfcore_tma(tmp_path: Path) -> None:
    sample = "exemplar-002"
    _build_nfcore_tma(tmp_path, [1, 2, 3, 4], sample=sample)
    sdata = mcmicro(tmp_path)
    # global mosaic + one image per core (TMA_MAP.tif must be excluded)
    assert "tma_map" in sdata.images
    assert {f"core_{c}_image" for c in (1, 2, 3, 4)} | {"tma_map"} == set(sdata.images)
    # per-core, per-segmenter labels (core id parsed correctly despite "exemplar-002" in the name)
    assert {f"core_{c}_cellpose" for c in (1, 2, 3, 4)} <= set(sdata.labels)
    assert {f"core_{c}_deepcell_mesmer" for c in (1, 2, 3, 4)} <= set(sdata.labels)
    # one table per segmenter, each concatenating its 4 cores
    assert {"cellpose_table", "mesmer_table"} == set(sdata.tables)
    mesmer_regions = sorted(sdata.tables["mesmer_table"].obs["region"].cat.categories.tolist())
    assert mesmer_regions == [f"core_{c}_deepcell_mesmer" for c in (1, 2, 3, 4)]
    # each core is translated to its centroid, so cores occupy distinct, non-origin global extents
    ext = {c: get_extent(sdata[f"core_{c}_image"], coordinate_system="global") for c in (1, 2, 3, 4)}
    origins = {(ext[c]["x"][0], ext[c]["y"][0]) for c in (1, 2, 3, 4)}
    assert len(origins) == 4  # no two cores overlap at the same origin
    assert all(ext[c]["x"][0] > 0 and ext[c]["y"][0] > 0 for c in (1, 2, 3, 4))
    # a core image and its label share the same translated extent (aligned)
    assert get_extent(sdata["core_1_cellpose"], coordinate_system="global") == ext[1]


# --------------------------------------------------------------------------------------------- #
# real public dataset (downloaded by .github/workflows/prepare_test_data.yaml)
# --------------------------------------------------------------------------------------------- #
# Real nf-core/mcmicro 2.0.0 `test` profile output. See prepare_test_data.yaml for provenance.
_REAL_NFCORE = Path("./data/mcmicro-nfcore-wsi-test")


@pytest.mark.skipif(not _REAL_NFCORE.is_dir(), reason="mcmicro-nfcore-wsi-test data not downloaded")
def test_nfcore_real_data() -> None:
    # the `test` profile output has no usable marker sheet, so channels fall back to integer names
    with pytest.warns(UserWarning, match="No markers file found"):
        sdata = mcmicro(_REAL_NFCORE)
    assert _detect_pipeline(_REAL_NFCORE) == McmicroPipeline.NFCORE
    assert "TEST1_image" in sdata.images
    assert "TEST1_mccellpose" in sdata.labels
    table = sdata.tables["TEST1_mccellpose_table"]
    assert table.obs["region"].cat.categories.tolist() == ["TEST1_mccellpose"]
    assert table.n_obs > 0


@pytest.mark.skipif(not _REAL_NFCORE.is_dir(), reason="mcmicro-nfcore-wsi-test data not downloaded")
def test_cli_nfcore_real_data(runner: CliRunner) -> None:
    with TemporaryDirectory() as outdir:
        output_zarr = Path(outdir) / "data.zarr"
        result = runner.invoke(mcmicro_wrapper, ["--input", str(_REAL_NFCORE), "--output", str(output_zarr)])
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)


# --------------------------------------------------------------------------------------------- #
# CLI round-trip
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", [_build_labsyspharm_wsi, lambda root: _build_nfcore_wsi(root, ["s1"])])
def test_cli_mcmicro(runner: CliRunner, tmp_path: Path, builder) -> None:  # type: ignore[no-untyped-def]
    builder(tmp_path)
    with TemporaryDirectory() as outdir:
        output_zarr = Path(outdir) / "data.zarr"
        result = runner.invoke(mcmicro_wrapper, ["--input", str(tmp_path), "--output", str(output_zarr)])
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)
