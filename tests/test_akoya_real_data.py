"""Tests of the Akoya readers on real public datasets.

They run only when the environment variable ``AKOYA_TEST_DATA`` points to a directory holding the files below (paths
relative to it). The files are public (Zenodo, Figshare, BioImage Archive, GitHub) but too large to
download in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
from spatialdata.models import get_channel_names
from spatialdata.transformations import get_transformation

from spatialdata_io import codex, phenocycler

_ROOT = os.environ.get("AKOYA_TEST_DATA")

pytestmark = [
    pytest.mark.skipif(_ROOT is None, reason="set AKOYA_TEST_DATA to the directory of the real Akoya datasets"),
    # upstream noise unrelated to the readers: ome-zarr's deprecated `Scaler`; unclosed files must not be left behind
    pytest.mark.filterwarnings(
        "ignore:Call to deprecated class Scaler",
        "error::ResourceWarning",
        "error::pytest.PytestUnraisableExceptionWarning",
    ),
]


@dataclass(frozen=True)
class Scan:
    """What a real dataset is expected to read as: ``shape`` starts ``(c, y, x)``, ``names`` the first channels."""

    path: str
    region: str
    generation: str
    names: tuple[str, ...]
    pixel_size_um: float | None
    shape: tuple[int, ...]
    dtype: str | None = None


# one per generation and file format
_SCANS = [
    Scan(
        "figshare_25123538/AML2_183_Scan1.qptiff", "AML2_183", "fusion1", ("DAPI", "NPM1C"), 0.5069, (54, 10800, 7680)
    ),
    # Fusion 2.x in PhenoImager (Opal) mode: the names come from `<Name>`, there is no `<Biomarker>`
    Scan("bia_S-BIAD3467/Tonsil-ctrl_Scan1.qptiff", "Tonsil-ctrl", "fusion2", ("DAPI", "Opal 690"), 0.4994, (6,)),
    # tifffile's QPI series builder finds only 3 of the 5 levels; the full-resolution shape must still be right
    Scan(
        "zenodo_22107401/Arlene_Tonsil_9x9.qptiff",
        "Arlene_Tonsil_9x9",
        "cim_qptiff",
        ("DAPI-01",),
        0.3774,
        (43, 9508, 12672),
    ),
    # CODEX Uploader montage, TCYX (21, 4, y, x); the names come from `channelNames.txt`
    Scan("bia_S-BIAD859/stitched/6371v1", "reg001", "legacy_processor", ("HOECHST", "CollIV"), None, (84, 3528, 3360)),
    # CODEX Processor per-marker stitched images `reg001_cycCCC_chHHH_<marker>.tiff`
    Scan(
        "zenodo_7742474/human_pancreas_codex_zip_subset",
        "reg001",
        "legacy_processor",
        ("Hoechst", "CGC"),
        None,
        (2, 18440, 25816),
    ),
    # Vectra / PhenoImager (generation `qpi_other`): unmixed uint16, brightfield RGB and inForm float32 component data
    Scan(
        "zenodo_22107401/Panel33C_Scan2.unmixed.qptiff",
        "Panel33C_Scan2.unmixed",
        "qpi_other",
        ("CD21",),
        0.4965,
        (8, 36000, 24960),
        "uint16",
    ),
    Scan(
        "ome_vectra_qptiff/HandEcompressed_Scan1.qptiff",
        "HandEcompressed",
        "qpi_other",
        ("r", "g", "b"),
        0.4989,
        (3, 26640, 30720),
        "uint8",
    ),
    Scan(
        "ome_vectra_qptiff/LuCa-7color_[13860,52919]_1x1component_data.tif",
        "LuCa-7color__13860_52919__1x1component_data",
        "qpi_other",
        ("PDL1 (Opal 520)",),
        0.498,
        (8, 1400, 1868),
        "float32",
    ),
]


@pytest.mark.parametrize("scan", _SCANS, ids=[scan.path.split("/")[-1] for scan in _SCANS])
def test_real_scan(scan: Scan) -> None:
    path = Path(_ROOT) / scan.path  # type: ignore[arg-type]
    if not path.exists():
        pytest.skip(f"{path} is missing")
    reader = codex if scan.generation in ("cim_qptiff", "legacy_processor") else phenocycler

    sdata = reader(path)

    image = sdata.images[f"{scan.region}_image"]
    data = image["scale0"]["image"].data
    assert isinstance(data, da.Array)
    assert data.shape[: len(scan.shape)] == scan.shape
    assert scan.dtype is None or data.dtype == scan.dtype
    names = get_channel_names(image)
    assert tuple(names[: len(scan.names)]) == scan.names
    assert len(set(names)) == len(names)
    metadata = sdata.attrs["akoya"]["regions"][scan.region]
    assert metadata["generation"] == scan.generation
    # a brightfield scan has one RGB page, which is read as three channels
    assert len(metadata["channels"]) == (1 if names == ["r", "g", "b"] else scan.shape[0])
    if scan.pixel_size_um is None:
        assert metadata["pixel_size_um"] is None
    else:
        assert metadata["pixel_size_um"] == pytest.approx(scan.pixel_size_um, abs=1e-3)
    assert sdata.attrs["spatialdata_io_reader"] == reader.__name__
    if path.is_file():
        _assert_tile_matches_tifffile(path, data, channel=scan.shape[0] - 1)


def _assert_tile_matches_tifffile(path: Path, data: da.Array, channel: int) -> None:
    """A full-resolution tile of ``channel`` equals tifffile's decoding of the whole page (not through zarr)."""
    with tifffile.TiffFile(path) as tf:
        full = [page for page in tf.pages if page.shape[:2] == data.shape[1:]]
        # brightfield: one RGB page holds the three channels
        page = full[0] if full[0].ndim == 3 else full[channel]
        y, x = data.shape[1] // 2, data.shape[2] // 2
        expected = page.asarray()[y : y + 256, x : x + 256]
    if expected.ndim == 3:
        expected = expected[..., channel]
    np.testing.assert_array_equal(data[channel, y : y + 256, x : x + 256].compute(), expected)


def test_real_same_qptiff_through_both_readers() -> None:
    path = Path(_ROOT) / "figshare_25123538/AML2_183_Scan1.qptiff"  # type: ignore[arg-type]
    if not path.exists():
        pytest.skip(f"{path} is missing")

    a, b = phenocycler(path), codex(path)

    image_a, image_b = a.images["AML2_183_image"], b.images["AML2_183_image"]
    assert get_channel_names(image_a) == get_channel_names(image_b)
    assert list(image_a) == list(image_b)
    coarsest = list(image_a)[-1]
    np.testing.assert_array_equal(image_a[coarsest]["image"].values, image_b[coarsest]["image"].values)
    assert get_transformation(image_a, get_all=True) == get_transformation(image_b, get_all=True)
    assert a.attrs["akoya"]["regions"] == b.attrs["akoya"]["regions"]


def test_real_codex_processor_csv() -> None:
    # a 500-cell subset of a CODEX Processor CSV with the `cyc001_ch001:DAPI1` header (github.com/igordot/phenomenalist)
    path = Path(_ROOT) / "github_codex_metadata/phenomenalist/tonsil-akoya-2018-500.csv"  # type: ignore[arg-type]
    if not path.exists():
        pytest.skip(f"{path} is missing")

    with pytest.warns(UserWarning, match="cells are represented as circles"):
        sdata = codex(path)

    table = sdata.tables["table"]
    assert table.shape == (500, 32)
    assert list(table.var_names[:5]) == ["DAPI1", "Blank", "Blank (2)", "Blank (3)", "DAPI2"]
    assert list(table.obs["region"].cat.categories) == ["reg001_cells"]
    assert table.obsm["spatial"].shape == (500, 2)
    assert len(sdata.shapes["reg001_cells"]) == 500
