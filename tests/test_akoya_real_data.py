"""Tests of the Akoya readers on real public datasets.

They run only when the environment variable ``AKOYA_TEST_DATA`` points to a directory holding the files below (paths
relative to it). The files are public (Zenodo, Figshare, BioImage Archive, Dryad, GitHub, the MCMICRO bucket) but too large to download
in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from spatialdata.models import get_channel_names

from spatialdata_io import codex, phenocycler
from spatialdata_io.readers._utils._cell_tables import detect_table_format, read_cell_table

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
    """What a real dataset is expected to read as."""

    path: str
    reader: str
    region: str
    generation: str
    n_channels: int
    first_names: tuple[str, ...]
    pixel_size_um: float | None
    shape: tuple[int, int, int] | None = None
    dtype: str | None = None


_SCANS = [
    Scan(
        "figshare_25123538/AML2_183_Scan1.qptiff",
        "phenocycler",
        "AML2_183",
        "fusion1",
        54,
        ("DAPI", "NPM1C", "ASMA", "CD4"),
        0.5069,
        (54, 10800, 7680),
    ),
    Scan(
        "zenodo_7916784/DON126-B3A.qptiff",
        "phenocycler",
        "DON126-B3A",
        "fusion1",
        28,
        ("DAPI", "GCG", "CD45", "CD19"),
        0.5079,
    ),
    Scan(
        "zenodo_16732565/VI_CODEX_12_Scan1.qptiff",
        "phenocycler",
        "VI_CODEX_12",
        "fusion2",
        15,
        ("DAPI", "GATA6", "S100a2", "LamininG"),
        0.5062,
    ),
    Scan(
        "zenodo_14741083/EWS_Akoya_Slide6.qptiff",
        "phenocycler",
        "EWS_Akoya_Slide6",
        "fusion2",
        42,
        ("DAPI", "WGA", "Na/K ATPase", "CD11c"),
        0.5075,
    ),
    # Fusion 2.x in PhenoImager (Opal) mode: the names come from `<Name>`, there is no `<Biomarker>`
    Scan(
        "bia_S-BIAD3467/Tonsil-ctrl_Scan1.qptiff",
        "phenocycler",
        "Tonsil-ctrl",
        "fusion2",
        6,
        ("DAPI", "Opal 690", "Opal 620", "Opal 780", "Opal 520", "Sample AF"),
        0.4994,
    ),
    Scan(
        "figshare_31956057/20210825-ipLN-9-hpi-VSV.qptiff",
        "codex",
        "20210825-ipLN-9-hpi-VSV",
        "cim_qptiff",
        37,
        ("DAPI", "Runx2", "CXCR4", "VSV"),
        0.3774,
    ),
    # tifffile's QPI series builder finds only 3 of the 5 levels; the full-resolution shape must still be right
    Scan(
        "zenodo_22107401/Arlene_Tonsil_9x9.qptiff",
        "codex",
        "Arlene_Tonsil_9x9",
        "cim_qptiff",
        43,
        ("DAPI-01", "HLA-A", "CD20", "Vimentin"),
        0.3774,
        (43, 9508, 12672),
    ),
    # CODEX Uploader montage, TCYX (21, 4, y, x); the names come from `channelNames.txt`
    Scan(
        "bia_S-BIAD859/stitched/6371v1",
        "codex",
        "reg001",
        "legacy_processor",
        84,
        ("HOECHST", "CollIV", "NaKATPase", "HLA-ABC"),
        None,
        (84, 3528, 3360),
    ),
    # CODEX Processor per-marker stitched images `reg001_cycCCC_chHHH_<marker>.tiff`
    Scan(
        "zenodo_7742474/human_pancreas_codex_zip_subset",
        "codex",
        "reg001",
        "legacy_processor",
        2,
        ("Hoechst", "CGC"),
        None,
        (2, 18440, 25816),
    ),
    # Vectra / PhenoImager (generation `qpi_other`): Opal, unmixed uint16, JPEG, brightfield RGB and inForm float32
    Scan(
        "zenodo_15265351/P074_Scan1.qptiff",
        "phenocycler",
        "P074",
        "qpi_other",
        8,
        ("DAPI", "Opal 570", "Opal 690", "Opal 480"),
        0.4984,
        (8, 17280, 17280),
        "uint8",
    ),
    Scan(
        "ome_vectra_qptiff/LuCa-7color_Scan1.qptiff",
        "phenocycler",
        "LuCa-7color",
        "qpi_other",
        5,
        ("DAPI", "FITC", "CY3", "Texas Red", "CY5"),
        0.498,
        (5, 34560, 24960),
        "uint8",
    ),
    Scan(
        "zenodo_22107401/Panel33C_Scan2.unmixed.qptiff",
        "phenocycler",
        "Panel33C_Scan2.unmixed",
        "qpi_other",
        8,
        ("CD21", "CD23", "CD20", "CD4", "CK"),
        0.4965,
        (8, 36000, 24960),
        "uint16",
    ),
    # a third-party (KFBIO) scanner writing JPEG-compressed QPI
    Scan(
        "zenodo_16569043/2025-07-15_13_13_34.qptiff",
        "phenocycler",
        "2025-07-15_13_13_34",
        "qpi_other",
        3,
        ("DAPI", "520", "620"),
        0.25,
        (3, 71472, 61440),
        "uint8",
    ),
    # brightfield: one RGB page read as the channels r, g, b
    Scan(
        "ome_vectra_qptiff/HandEcompressed_Scan1.qptiff",
        "phenocycler",
        "HandEcompressed",
        "qpi_other",
        3,
        ("r", "g", "b"),
        0.4989,
        (3, 26640, 30720),
        "uint8",
    ),
    # inForm component data: a single `.tif` file with float32 pages
    Scan(
        "ome_vectra_qptiff/LuCa-7color_[13860,52919]_1x1component_data.tif",
        "phenocycler",
        "LuCa-7color__13860_52919__1x1component_data",
        "qpi_other",
        8,
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

    sdata = (phenocycler if scan.reader == "phenocycler" else codex)(path)

    image = sdata.images[f"{scan.region}_image"]
    data = image["scale0"]["image"].data
    assert isinstance(data, da.Array)
    assert data.shape[0] == scan.n_channels
    if scan.shape is not None:
        assert data.shape == scan.shape
    if scan.dtype is not None:
        assert data.dtype == scan.dtype
    names = get_channel_names(image)
    assert tuple(names[: len(scan.first_names)]) == scan.first_names
    assert len(set(names)) == len(names)
    metadata = sdata.attrs["akoya"]["regions"][scan.region]
    assert metadata["generation"] == scan.generation
    # a brightfield scan has one RGB page, which is read as three channels
    assert len(metadata["channels"]) == (1 if names == ["r", "g", "b"] else scan.n_channels)
    if scan.pixel_size_um is None:
        assert metadata["pixel_size_um"] is None
    else:
        assert metadata["pixel_size_um"] == pytest.approx(scan.pixel_size_um, abs=1e-3)
    assert sdata.attrs["spatialdata_io_reader"] == scan.reader


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


# public cell-table exports (see the docstrings of `read_cell_table` and `read_labels` for the variants)
_EXPORTS = "exports"
_PHENOPTR = f"{_EXPORTS}/github_phenoptr/repo/inst/extdata/sample/Set4_1-6plex_[16142,55840]"
_DRYAD = f"{_EXPORTS}/dryad_hqbzkh1v1"
# the OME `PhysicalSizeX` of the Dryad images, which no Akoya reader reads
_DRYAD_PIXEL_SIZE_UM = 0.5072844


def _export(relative: str) -> Path:
    path = Path(_ROOT) / relative  # type: ignore[arg-type]
    if not path.exists():
        pytest.skip(f"{path} is missing")
    return path


def test_real_inform_table_with_seg_maps_and_component_data() -> None:
    # inForm 2.3 (github.com/akoyabio/phenoptr): positions in pixels, the `Cell ID`s are the labels of page 1 (nuclei)
    # of the segmentation map; page 0 is the tissue class map
    image = _export(f"{_PHENOPTR}_component_data.tif")

    sdata = phenocycler(
        image,
        cells_table=_export(f"{_PHENOPTR}_cell_seg_data.txt"),
        labels=_export(f"{_PHENOPTR}_binary_seg_maps.tif"),
    )

    region = "Set4_1-6plex__16142_55840__component_data"
    table = sdata.tables["table"]
    assert table.shape == (6072, 8)
    assert list(table.var_names[:2]) == ["PDL1 (Opal 520)", "CD8 (Opal 540)"]
    assert list(table.obs["region"].cat.categories) == [f"{region}_labels"]
    labels = sdata.labels[f"{region}_labels"]["scale0"]["image"]
    assert labels.shape == (1400, 1868)
    assert set(table.obs["cell_id"]) == set(np.unique(labels.values)) - {0}
    np.testing.assert_allclose(table.obsm["spatial"][0], [515, 13])


def test_real_inform_micron_variant() -> None:
    # github.com/lilab-stanford/auto-cell: positions in microns without a unit in their name, trailing spaces
    path = _export(f"{_EXPORTS}/github_auto-cell/repo/cell_annotation/sample_cell_seg_data.txt")

    assert detect_table_format(path) == "inform"
    table = read_cell_table(path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={})

    assert len(table.obs) == 4
    assert list(table.var.index) == [
        "PD1 (PPD520)",
        "FOXP3 (PPD540)",
        "CD20 (PPD620)",
        "CD3 (PPD650)",
        "PANCK (PPD690)",
        "PPDDAPI",
        "Autofluorescence",
    ]
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px[0], [4230.4 / 0.5, 29684.5 / 0.5])
    assert table.radius_px is not None


def test_real_qupath_r_exported_coordinates_with_geojson(tmp_path: Path) -> None:
    # doi:10.5061/dryad.hqbzkh1v1: R-exported QuPath centroids (`Centroid.X.um`) and measurements in two files, and
    # the GeoJSON of all StarDist cells (UUID ids, one `annotation`); 116 cells were filtered out of the tables
    coords = _export(f"{_DRYAD}/C-7_coords.csv")
    geojson = _export(f"{_DRYAD}/C-7.geojson")
    expression = _export(f"{_DRYAD}/C-7_expression.csv")
    kwargs = {"geojson": geojson}

    assert detect_table_format(coords) == "qupath"
    with pytest.warns(UserWarning, match="116 of the 9531 cell features"):
        table = read_cell_table(
            coords, "auto", pixel_size_um=_DRYAD_PIXEL_SIZE_UM, default_region="C-7", cells_table_kwargs=kwargs
        )
    assert table.polygons_px is not None
    assert pd.notna(table.polygons_px).all()
    assert len(table.obs) == 9415

    # joined on the object id, the measurements are the markers, from the `Cell` compartment (`DAPI_Cell_Mean`); the
    # expression file alone has no centroids in QuPath's form (its `Y_coord` is negated)
    joined = tmp_path / "C-7.csv"
    pd.read_csv(coords).merge(pd.read_csv(expression), on="Object.ID", validate="1:1").to_csv(joined, index=False)
    with pytest.warns(UserWarning, match="116 of the 9531 cell features"):
        table = read_cell_table(
            joined, "auto", pixel_size_um=_DRYAD_PIXEL_SIZE_UM, default_region="C-7", cells_table_kwargs=kwargs
        )
    assert table.X.shape == (9415, 30)
    assert {"DAPI", "HLA.A", "Pan.Cytokeratin"} <= set(table.var.index)
    assert set(table.var["compartment"]) == {"Cell"}
    assert table.var.loc["DAPI", "source_column"] == "DAPI_Cell_Mean"


def test_real_qupath_detection_xy_and_bare_markers() -> None:
    # zenodo.org/records/15514605: `X`/`Y` centroids in µm, markers named after the marker only, `Parent` = TMA core
    path = _export(f"{_EXPORTS}/zenodo_15514605/measurement_TA4369.csv")

    assert detect_table_format(path) == "qupath"
    with pytest.warns(UserWarning, match="reading the 48 other numeric columns"):
        table = read_cell_table(
            path, "auto", pixel_size_um=0.5, default_region=None, cells_table_kwargs={"region_column": "Parent"}
        )

    assert table.X.shape == (348314, 48)
    assert list(table.var.index[:3]) == ["CD44", "CD68", "CD31"]
    assert table.region.nunique() == 59
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px[0], [2620.8 / 0.5, 661.3 / 0.5])


def test_real_mcquant_table_with_ome_tiff() -> None:
    # MCMICRO exemplar-001: mcquant table in pixels with the OME-TIFF of its first cycle, read as a hyperstack
    image = _export(f"{_EXPORTS}/s3_mcmicro_exemplar001/exemplar-001-cycle6.ome.tif")
    cells = _export(f"{_EXPORTS}/s3_mcmicro_exemplar001/unmicst-exemplar-001_cell.csv")

    with pytest.warns(UserWarning, match="cells are represented as circles"):
        sdata = codex(image, cells_table=cells)

    table = sdata.tables["table"]
    assert table.shape == (11170, 12)
    assert list(table.var_names[:4]) == ["DNA_6", "ELANE", "CD57", "CD45"]
    assert len(sdata.shapes["exemplar-001-cycle6.ome_cells"]) == 11170
