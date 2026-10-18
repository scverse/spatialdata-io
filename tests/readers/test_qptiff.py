import gc
import pickle
import warnings
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
from tests._akoya_data import cim_experiment_v4_xml, fusion_scan_profile_json, write_qptiff

from spatialdata_io.readers._utils import _qptiff
from spatialdata_io.readers._utils._qptiff import read_qptiff_image, read_qptiff_metadata

_CIM = "Codex Instrument Manager 1.30.0.12"


@pytest.mark.parametrize(
    ("kwargs", "names", "generation"),
    [
        ({"biomarkers": ["DAPI", "CD3"], "names": ["DAPI", "ATTO 550"]}, ["DAPI", "CD3"], "fusion2"),
        # PhenoImager (Opal) mode has no `<Biomarker>`
        ({"biomarkers": [None, None], "names": ["DAPI", "Opal 480"]}, ["DAPI", "Opal 480"], "fusion2"),
        ({"biomarkers": ["DAPI"], "software": "Fusion 1.0.5"}, ["DAPI"], "fusion1"),
        ({"biomarkers": [None], "names": ["FITC"], "software": "VectraPolaris 1.0"}, ["FITC"], "qpi_other"),
    ],
)
def test_channel_names_and_generation(tmp_path: Path, kwargs: dict[str, Any], names: list[str], generation: str):
    write_qptiff(tmp_path / "S_Scan1.qptiff", **kwargs)

    scan = read_qptiff_metadata(tmp_path / "S_Scan1.qptiff")

    assert [c.name for c in scan.channels] == names
    assert scan.generation == generation


def test_pixel_size_and_stage_position(tmp_path: Path) -> None:
    p = tmp_path / "x.qptiff"
    write_qptiff(p, biomarkers=["DAPI"], stage_position_cm=(1.5, 0.25))
    scan = read_qptiff_metadata(p)
    assert scan.pixel_size_um == pytest.approx(0.5069, rel=1e-6)
    assert scan.stage_position_um == pytest.approx((15000.0, 2500.0))

    # 96 dpi is a screen resolution, not a calibration
    write_qptiff(p, biomarkers=["DAPI"], resolution_px_per_cm=96, resolution_unit="INCH")
    with pytest.warns(UserWarning, match="Could not determine the pixel size"):
        assert read_qptiff_metadata(p).pixel_size_um is None


def test_cim_without_label_page_counts_pages_by_image_type(tmp_path: Path) -> None:
    p = tmp_path / "cim.qptiff"
    biomarkers: list[str | None] = ["DAPI", "CD3", "CD8", "DAPI", "CD20"]
    profile = cim_experiment_v4_xml([["DAPI", "CD3", "CD8"], ["DAPI", "CD20"]])
    written = write_qptiff(p, biomarkers=biomarkers, software=_CIM, scan_profile=profile, levels=3, label_page=False)

    scan = read_qptiff_metadata(p)

    assert scan.generation == "cim_qptiff"
    assert [(c.name, c.cycle, c.channel) for c in scan.channels][:2] == [("DAPI", 1, 1), ("CD3", 1, 2)]
    assert scan.nominal_pixel_size_um == pytest.approx(0.37739)
    np.testing.assert_array_equal(read_qptiff_image(scan).compute(), written)


@pytest.mark.parametrize(
    ("biomarkers", "profile", "expected"),
    [
        # by name; a page without a record is left unset
        (["DAPI", "CD3", "CD4"], [[("DAPI", "DAPI"), ("CD3", "ATTO 550")]], [(1, 1), (1, 2), (None, None)]),
        # DAPI is in several cycles, so it is not paired
        (
            ["DAPI", "CD3", "CD8"],
            [[("DAPI", "D"), ("CD3", "A")], [("DAPI", "D"), ("CD8", "C")]],
            [(None, None), (1, 2), (2, 2)],
        ),
        # names that differ only in non-ASCII letters, written as character references in the page XML
        (["TCR&#947;&#948;", "TCR&#945;&#946;"], [[("TCRαβ", "A"), ("TCRγδ", "C")]], [(1, 2), (1, 1)]),
    ],
)
def test_scan_profile_sets_cycle_and_channel(
    tmp_path: Path, biomarkers: list[str | None], profile: list[list[tuple[str, str]]], expected: list[tuple]
) -> None:
    p = tmp_path / "x.qptiff"
    write_qptiff(p, biomarkers=biomarkers, scan_profile=fusion_scan_profile_json(profile))

    assert [(c.cycle, c.channel) for c in read_qptiff_metadata(p).channels] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"biomarkers": ["DAPI", "CD3", "CD8"]},  # three channels are not RGB
        {"biomarkers": ["DAPI"], "dtype": "uint16"},
        {"biomarkers": [None], "names": ["BF"], "rgb": True, "software": "VectraPolaris 1.0"},
        {"biomarkers": ["DAPI", "CD3"], "bigtiff": False, "tile": None},
    ],
    ids=["multiplex", "uint16", "brightfield", "classic_stripped"],
)
def test_image_is_lazy_and_correct(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    p = tmp_path / "x.qptiff"
    written = write_qptiff(p, **kwargs)

    scan = read_qptiff_metadata(p)
    image = read_qptiff_image(scan)

    assert scan.rgb == kwargs.get("rgb", False)
    assert isinstance(image, da.Array)
    assert image.dtype == written.dtype
    np.testing.assert_array_equal(image.compute(), written)


@pytest.mark.parametrize("series_match", [True, False])
def test_image_is_picklable_and_closes_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, series_match: bool
) -> None:
    # dask.distributed pickles the graph, so it must not hold open file handles; also for the per-page fallback,
    # used when tifffile's series do not match the full-resolution pages of the XML
    p = tmp_path / "x.qptiff"
    written = write_qptiff(p, biomarkers=["DAPI", "CD3"])
    if not series_match:
        real = _qptiff.lazy_tiff_array

        def wrong_series(path: Path, *, series: int = 0, page: int | None = None) -> da.Array:
            return real(path, series=series, page=page) if page is not None else da.zeros((1, 1, 1))

        monkeypatch.setattr(_qptiff, "lazy_tiff_array", wrong_series)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ResourceWarning)
        image = read_qptiff_image(read_qptiff_metadata(p))
        restored = pickle.loads(pickle.dumps(image))
        np.testing.assert_array_equal(restored.compute(), written)
        del image, restored
        gc.collect()
    assert not [warning for warning in record if issubclass(warning.category, ResourceWarning)]
