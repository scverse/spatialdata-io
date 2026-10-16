import json
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
from tests._akoya_data import cim_experiment_v4_xml, fusion_scan_profile_json, write_qptiff

from spatialdata_io.readers._utils._qptiff import read_qptiff_image, read_qptiff_metadata


def test_fusion2_channels_from_biomarker(tmp_path: Path) -> None:
    p = tmp_path / "scan.qptiff"
    write_qptiff(p, biomarkers=["DAPI", "CD3", "DAPI"], names=["DAPI", "ATTO 550", "DAPI"])
    scan = read_qptiff_metadata(p)
    assert [c.name for c in scan.channels] == ["DAPI", "CD3", "DAPI"]
    assert scan.channels[1].fluor == "ATTO 550"
    assert scan.generation == "fusion2"
    assert scan.region == "scan"
    assert scan.slide_id == "SLIDE1"
    assert scan.software == "Fusion 2.3.1"
    assert scan.channels[1].exposure_ms == pytest.approx(450.0)
    assert scan.channels[1].color == "255,255,0"


def test_opal_mode_uses_name(tmp_path: Path) -> None:
    p = tmp_path / "opal.qptiff"
    names = ["DAPI", "Opal 480", "Opal 520", "Opal 570", "Opal 620", "Opal 690"]
    write_qptiff(p, biomarkers=[None] * 6, names=names)
    scan = read_qptiff_metadata(p)
    assert [c.name for c in scan.channels] == names
    assert all(c.fluor is None for c in scan.channels)
    assert scan.generation == "fusion2"


def test_fusion1_generation_and_region_override(tmp_path: Path) -> None:
    p = tmp_path / "scan.er.qptiff"
    write_qptiff(p, biomarkers=["DAPI"], software="Fusion 1.0.5", dtype="uint16")
    assert read_qptiff_metadata(p).region == "scan"
    scan = read_qptiff_metadata(p, region="reg001")
    assert scan.generation == "fusion1"
    assert scan.region == "reg001"


def test_vectra_is_qpi_other(tmp_path: Path) -> None:
    p = tmp_path / "vectra.qptiff"
    write_qptiff(p, biomarkers=[None, None], names=["DAPI", "FITC"], software="VectraPolaris 1.0")
    assert read_qptiff_metadata(p).generation == "qpi_other"


def test_pixel_size_from_xresolution(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    write_qptiff(p, biomarkers=["DAPI"], resolution_px_per_cm=1e4 / 0.5069)
    scan = read_qptiff_metadata(p)
    assert scan.pixel_size_um == pytest.approx(0.5069, rel=1e-6)
    assert scan.nominal_pixel_size_um is None


def test_96dpi_resolution_is_unknown(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    write_qptiff(p, biomarkers=["DAPI"], resolution_px_per_cm=96, resolution_unit="INCH")
    with pytest.warns(UserWarning, match="Could not determine the pixel size"):
        scan = read_qptiff_metadata(p)
    assert scan.pixel_size_um is None


def test_stage_position(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    write_qptiff(p, biomarkers=["DAPI"], stage_position_cm=(1.5, 0.25))
    scan = read_qptiff_metadata(p)
    assert scan.stage_position_um == pytest.approx((15000.0, 2500.0))
    write_qptiff(p, biomarkers=["DAPI"])
    assert read_qptiff_metadata(p).stage_position_um is None


def test_cim_without_label_page_counts_pages_by_imagetype(tmp_path: Path) -> None:
    p = tmp_path / "cim.qptiff"
    biomarkers: list[str | None] = ["DAPI", "CD3", "CD8", "DAPI", "CD20"]
    profile = cim_experiment_v4_xml([["DAPI", "CD3", "CD8"], ["DAPI", "CD20"]])
    written = write_qptiff(
        p,
        biomarkers=biomarkers,
        software="Codex Instrument Manager 1.30.0.12",
        description_version=3,
        scan_profile=profile,
        levels=3,
        label_page=False,
        resolution_px_per_cm=1e4 / 0.37739,
    )
    image = read_qptiff_image(p)
    assert image.shape == (5, 256, 320)
    assert np.array_equal(image.compute(), written)
    scan = read_qptiff_metadata(p)
    assert scan.generation == "cim_qptiff"
    assert [c.name for c in scan.channels] == biomarkers
    assert scan.channels[1].cycle == 1
    assert scan.channels[1].channel == 2
    assert scan.channels[1].exposure_ms is not None
    assert scan.channels[1].extra["wavelength_nm"] == pytest.approx(558.0)
    assert [c.cycle for c in scan.channels] == [1, 1, 1, 2, 2]
    assert scan.nominal_pixel_size_um == pytest.approx(0.37739)


def test_fusion_scan_profile_enrichment(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    wells = [[("DAPI", "DAPI"), ("CD3", "ATTO 550")], [("DAPI", "DAPI"), ("CD8", "AF 750")]]
    write_qptiff(p, biomarkers=["DAPI", "CD3", "DAPI", "CD8"], scan_profile=fusion_scan_profile_json(wells))
    scan = read_qptiff_metadata(p)
    assert [c.cycle for c in scan.channels] == [1, 1, 2, 2]
    assert [c.channel for c in scan.channels] == [1, 2, 1, 2]
    assert isinstance(scan.raw["scan_profile"], dict)


def test_fusion_reagent_placeholders_are_dropped(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    profile = json.loads(fusion_scan_profile_json([[("DAPI", "DAPI"), ("CD3", "ATTO 550")]]))
    items = profile["experimentDescription"]["wells"][0]["items"]
    items[0].update(reporter="None", barcode="None", clone="None")
    items[1].update(reporter="GN65", barcode="GN65", clone="XXX")
    write_qptiff(p, biomarkers=["DAPI", "CD3"], scan_profile=json.dumps(profile))
    dapi, cd3 = read_qptiff_metadata(p).channels
    assert not {"reporter", "barcode", "clone"} & dapi.extra.keys()
    assert (cd3.extra["reporter"], cd3.extra["barcode"], cd3.extra["clone"]) == ("GN65", "GN65", "XXX")
    assert cd3.extra["dye"] == "ATTO 550"


def test_enrichment_leaves_unmatched_channels_empty(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    wells = [[("DAPI", "DAPI"), ("CD3", "ATTO 550")]]
    write_qptiff(p, biomarkers=["DAPI", "CD3", "CD4"], scan_profile=fusion_scan_profile_json(wells))
    scan = read_qptiff_metadata(p)
    assert [c.name for c in scan.channels] == ["DAPI", "CD3", "CD4"]
    assert [c.cycle for c in scan.channels] == [1, 1, None]


def test_enrichment_ambiguous_name_left_unset(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    wells = [[("DAPI", "DAPI"), ("CD3", "ATTO 550")], [("DAPI", "DAPI"), ("CD8", "AF 750")], [("DAPI", "DAPI")]]
    write_qptiff(p, biomarkers=["DAPI", "CD3", "CD8"], scan_profile=fusion_scan_profile_json(wells))
    scan = read_qptiff_metadata(p)
    assert [c.cycle for c in scan.channels] == [None, 1, 2]
    assert scan.channels[0].channel is None


def test_enrichment_positional_when_counts_match(tmp_path: Path) -> None:
    p = tmp_path / "cim.qptiff"
    write_qptiff(
        p,
        biomarkers=["DAPI", "F4-80", "NK1"],
        software="Codex Instrument Manager 1.30.0.12",
        scan_profile=cim_experiment_v4_xml([["DAPI", "F4/80"], ["NK1.1", "BLANK"]]),
    )
    scan = read_qptiff_metadata(p)
    assert [c.name for c in scan.channels] == ["DAPI", "F4-80", "NK1"]
    assert [(c.cycle, c.channel) for c in scan.channels] == [(1, 1), (1, 2), (2, 1)]


def test_enrichment_not_positional_when_orders_differ(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    wells = [[("DAPI", "DAPI"), ("CD3", "ATTO 550"), ("CD8", "CY5"), ("CD20", "AF 750")]]
    write_qptiff(p, biomarkers=["DAPI", "CD3", "CD20", "CD8"], scan_profile=fusion_scan_profile_json(wells))
    scan = read_qptiff_metadata(p)
    assert [c.channel for c in scan.channels] == [1, 2, 4, 3]


def test_profile_record_without_marker_skipped(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    profile = json.loads(fusion_scan_profile_json([[("DAPI", "DAPI"), ("CD3", "ATTO 550")]]))
    del profile["experimentDescription"]["wells"][0]["items"][1]["markerName"]
    write_qptiff(p, biomarkers=["DAPI", "CD3"], scan_profile=json.dumps(profile))
    dapi, cd3 = read_qptiff_metadata(p).channels
    assert dapi.cycle == 1
    assert cd3.cycle is None


@pytest.mark.parametrize(("dtype", "name"), [("uint8", "x.qptiff"), ("uint16", "x.er.qptiff")])
def test_image_is_lazy_and_correct(tmp_path: Path, dtype: str, name: str) -> None:
    p = tmp_path / name
    written = write_qptiff(p, biomarkers=["DAPI", "CD3"], dtype=dtype)
    arr = read_qptiff_image(p)
    assert isinstance(arr, da.Array)
    assert arr.chunksize != arr.shape
    assert arr.dtype == np.dtype(dtype)
    assert np.array_equal(arr.compute(), written)


def test_three_channel_not_rgb(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    written = write_qptiff(p, biomarkers=["DAPI", "CD3", "CD8"])
    scan = read_qptiff_metadata(p)
    assert scan.rgb is False
    image = read_qptiff_image(p)
    assert image.ndim == 3
    assert image.shape == (3, 256, 320)
    assert np.array_equal(image.compute(), written)


def test_brightfield_rgb(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    written = write_qptiff(p, biomarkers=[None], names=["BF"], rgb=True, software="VectraPolaris 1.0")
    scan = read_qptiff_metadata(p)
    assert scan.rgb is True
    image = read_qptiff_image(p)
    assert image.shape == (256, 320, 3)
    assert np.array_equal(image.compute(), written)


def test_not_qpi_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    tifffile.imwrite(p, np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="not a PerkinElmer"):
        read_qptiff_metadata(p)
    with pytest.raises(ValueError, match="not a PerkinElmer"):
        read_qptiff_image(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_qptiff_metadata(tmp_path / "missing.qptiff")


def test_classic_tiff_and_stripped_level(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    written = write_qptiff(p, biomarkers=["DAPI", "CD3"], bigtiff=False, tile=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scan = read_qptiff_metadata(p)
    assert len(scan.channels) == 2
    assert np.array_equal(read_qptiff_image(p).compute(), written)
