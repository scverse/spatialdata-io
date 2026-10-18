"""Tests for the PhenoCycler-Fusion reader on synthetic qptiff files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation

from spatialdata_io import phenocycler
from spatialdata_io.__main__ import phenocycler_wrapper
from tests._akoya_data import forbid_pixel_reads, write_qptiff

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def test_read_single_file(tmp_path: Path) -> None:
    path = tmp_path / "Tonsil_Scan1.qptiff"
    data = write_qptiff(path, biomarkers=["DAPI", "CD3", "CD3"])

    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["Tonsil_image"]
    assert get_channel_names(sdata.images["Tonsil_image"]) == ["DAPI", "CD3", "CD3 (2)"]
    np.testing.assert_array_equal(_level0(sdata, "Tonsil_image").values, data)
    assert sdata.attrs["akoya"]["regions"]["Tonsil"]["generation"] == "fusion2"
    assert sdata.attrs["spatialdata_io_reader"] == "phenocycler"


def test_vectra_unmixed_float32_component_qptiff(tmp_path: Path) -> None:
    path = tmp_path / "Slide_Scan1.unmixed.qptiff"
    names = ["PDL1 (Opal 520)", "DAPI"]
    data = write_qptiff(
        path,
        biomarkers=[None, None],
        names=names,
        dtype="float32",
        software="VectraPolaris 1.0",
        resolution_px_per_cm=2e4,
    )

    sdata = phenocycler(path, image_models_kwargs=_FAST)

    image = _level0(sdata, "Slide_Scan1.unmixed_image")
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image.values, data)
    assert get_channel_names(sdata.images["Slide_Scan1.unmixed_image"]) == names
    metadata = sdata.attrs["akoya"]["regions"]["Slide_Scan1.unmixed"]
    assert metadata["generation"] == "qpi_other"
    assert metadata["pixel_size_um"] == pytest.approx(0.5)


def test_vectra_brightfield_rgb_qptiff(tmp_path: Path) -> None:
    path = tmp_path / "HandE_Scan1.qptiff"
    data = write_qptiff(path, biomarkers=[None], names=["BF"], rgb=True, software="VectraPolaris 1.0")

    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert get_channel_names(sdata.images["HandE_image"]) == ["r", "g", "b"]
    np.testing.assert_array_equal(_level0(sdata, "HandE_image").values, np.moveaxis(data, -1, 0))
    assert sdata.attrs["akoya"]["regions"]["HandE"]["generation"] == "qpi_other"


def test_inform_component_tif_with_special_characters_in_name(tmp_path: Path) -> None:
    # inForm names its component files `<sample>_[x,y]_component_data.tif`; the brackets and comma are not valid
    # in an element name
    path = tmp_path / "LuCa_[13860,52919]_1x1component_data.tif"
    data = write_qptiff(
        path, biomarkers=[None, None], names=["A (Opal 520)", "B"], dtype="float32", software="inForm", levels=1
    )

    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["LuCa__13860_52919__1x1component_data_image"]
    image = _level0(sdata, "LuCa__13860_52919__1x1component_data_image")
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image.values, data)


def test_non_qpi_tif_raises(tmp_path: Path) -> None:
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="not a PerkinElmer/Akoya QPTIFF"):
        phenocycler(path)


def test_read_directory_multiple_scans_and_subset(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI", "CD8"])

    sdata = phenocycler(str(tmp_path), image_models_kwargs=_FAST)
    assert sorted(sdata.images) == ["A_image", "B_image"]
    assert sorted(sdata.attrs["akoya"]["regions"]) == ["A", "B"]

    subset = phenocycler(tmp_path, regions=["B"], image_models_kwargs=_FAST)
    assert list(subset.images) == ["B_image"]
    assert list(subset.attrs["akoya"]["regions"]) == ["B"]

    with pytest.raises(ValueError, match="Z"):
        phenocycler(tmp_path, regions=["Z"])


def test_phenocycler_unrelated_scans_own_coordinate_systems(tmp_path: Path) -> None:
    for region, slide in (("A", "SLIDE1"), ("B", "SLIDE2")):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"], slide_id=slide)
    _write_mask(tmp_path / "A_mask.tif")

    sdata = phenocycler(tmp_path, labels={"A": tmp_path / "A_mask.tif"}, image_models_kwargs=_FAST)

    assert sorted(sdata.coordinate_systems) == ["A", "B"]
    for name in ("A_image", "A_labels"):
        assert get_transformation(sdata[name], get_all=True) == {"A": Identity()}
    assert get_transformation(sdata["B_image"], get_all=True) == {"B": Identity()}


def test_phenocycler_same_slide_scans_placed_in_global(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"], stage_position_cm=(1.0, 2.0))
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI"], stage_position_cm=(1.01, 2.0))

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert sdata.coordinate_systems == ["global"]
    assert get_transformation(sdata["A_image"], get_all=True) == {"global": Translation([0, 0], axes=("x", "y"))}
    offset = get_transformation(sdata["B_image"], "global")
    assert isinstance(offset, Translation)
    np.testing.assert_allclose(offset.translation, [100 / 0.5069, 0], rtol=1e-3)


def test_region_collision_keeps_full_stem(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "S_Scan2.qptiff", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["S_Scan1_image", "S_Scan2_image"]


def test_er_preferred(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint8")
    write_qptiff(tmp_path / "S_Scan1.er.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint16")

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]
    assert _level0(sdata, "S_image").dtype == np.uint16
    assert sdata.attrs["akoya"]["regions"]["S"]["path"].endswith("S_Scan1.er.qptiff")


def test_read_all_bit_depths(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint8")
    write_qptiff(tmp_path / "S_Scan1.er.qptiff", biomarkers=["DAPI", "CD3"], dtype="uint16")

    sdata = phenocycler(tmp_path, read_all_bit_depths=True, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["S_8bit_image", "S_image"]
    assert _level0(sdata, "S_image").dtype == np.uint16
    assert _level0(sdata, "S_8bit_image").dtype == np.uint8
    assert get_transformation(sdata["S_8bit_image"], get_all=True) == get_transformation(sdata["S_image"], get_all=True)
    assert list(get_transformation(sdata["S_image"], get_all=True)) == ["global"]
    assert sdata.attrs["akoya"]["regions"]["S"]["path"].endswith("S_Scan1.er.qptiff")
    assert sdata.attrs["akoya"]["regions"]["S_8bit"]["path"].endswith("S_Scan1.qptiff")


def test_read_all_bit_depths_of_an_extended_range_file(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.QPTIFF", biomarkers=["DAPI"], dtype="uint8")
    write_qptiff(tmp_path / "S_Scan1.ER.QPTIFF", biomarkers=["DAPI"], dtype="uint16")

    sdata = phenocycler(tmp_path / "S_Scan1.ER.QPTIFF", read_all_bit_depths=True, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["S_8bit_image", "S_image"]
    assert _level0(sdata, "S_8bit_image").dtype == np.uint8
    # the 8-bit file alone is read as it is
    assert list(phenocycler(tmp_path / "S_Scan1.QPTIFF", read_all_bit_depths=True).images) == ["S_image"]


def test_read_all_bit_depths_shares_stage_placement(tmp_path: Path) -> None:
    for name, pos in (("A", (0.0, 0.0)), ("B", (0.1, 0.0))):
        write_qptiff(tmp_path / f"{name}.qptiff", biomarkers=["DAPI"], dtype="uint8", stage_position_cm=pos)
        write_qptiff(tmp_path / f"{name}.er.qptiff", biomarkers=["DAPI"], dtype="uint16", stage_position_cm=pos)

    sdata = phenocycler(tmp_path, read_all_bit_depths=True, regions=["B"], image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["B_8bit_image", "B_image"]
    assert get_transformation(sdata["B_8bit_image"], get_all=True) == get_transformation(sdata["B_image"], get_all=True)

    # with both regions placed by their stage positions, the 8-bit image gets the translation of its region
    sdata = phenocycler(tmp_path, read_all_bit_depths=True, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["A_8bit_image", "A_image", "B_8bit_image", "B_image"]
    translation = get_transformation(sdata["B_image"], "global")
    assert isinstance(translation, Translation)
    assert translation.translation[0] > 0
    for region in ("A", "B"):
        expected = get_transformation(sdata[f"{region}_image"], get_all=True)
        assert get_transformation(sdata[f"{region}_8bit_image"], get_all=True) == expected


def test_read_all_bit_depths_with_labels(tmp_path: Path) -> None:
    for suffix, dtype in ((".qptiff", "uint8"), (".er.qptiff", "uint16")):
        write_qptiff(
            tmp_path / f"S{suffix}", biomarkers=["DAPI"], shape=(64, 80), dtype=dtype, stage_position_cm=(1, 2)
        )
    mask = np.zeros((32, 40), dtype=np.uint32)
    mask[2:4, 2:4], mask[5:7, 5:7] = 1, 2
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    sdata = phenocycler(
        tmp_path,
        labels=tmp_path / "mask.tif",
        read_all_bit_depths=True,
        image_models_kwargs=_FAST,
    )

    image_transformations = get_transformation(sdata["S_image"], get_all=True)
    assert get_transformation(sdata["S_8bit_image"], get_all=True) == image_transformations
    # the mask is scaled onto the extended-range region only
    assert list(sdata.labels) == ["S_labels"]
    labels_transformation = get_transformation(sdata["S_labels"], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations == [Scale([2, 2], axes=("y", "x")), image_transformations["global"]]

    with pytest.raises(ValueError, match=r"Unknown region\(s\) \['S_8bit'\]"):
        phenocycler(tmp_path, labels={"S_8bit": tmp_path / "mask.tif"}, read_all_bit_depths=True)


def test_read_all_bit_depths_shape_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S.qptiff", biomarkers=["DAPI"], shape=(64, 80), dtype="uint8")
    write_qptiff(tmp_path / "S.er.qptiff", biomarkers=["DAPI"], shape=(64, 96), dtype="uint16")

    with pytest.warns(UserWarning, match=r"8-bit scan .*\(64, 80\).*\(64, 96\)"):
        sdata = phenocycler(tmp_path, read_all_bit_depths=True, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["S_8bit_image", "S_image"]


def test_read_all_bit_depths_without_pair_reads_one(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S.qptiff", biomarkers=["DAPI"], dtype="uint8")

    sdata = phenocycler(tmp_path, read_all_bit_depths=True, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]


def test_read_all_bit_depths_region_name_clash_raises(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S.qptiff", biomarkers=["DAPI"], dtype="uint8")
    write_qptiff(tmp_path / "S.er.qptiff", biomarkers=["DAPI"], dtype="uint16")
    write_qptiff(tmp_path / "S_8bit.qptiff", biomarkers=["DAPI"], dtype="uint8")

    with pytest.raises(ValueError, match="S_8bit"):
        phenocycler(tmp_path, read_all_bit_depths=True, image_models_kwargs=_FAST)


def test_ignores_acquisition_files(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "images"
    acquisition = images / "phenocycler"
    acquisition.mkdir(parents=True)
    write_qptiff(images / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(acquisition / "S_Scan1_Cycle1.qptiff.raw", biomarkers=["DAPI"])
    write_qptiff(acquisition / "S.raw.qptiff", biomarkers=["DAPI"])
    write_qptiff(acquisition / "S_Scan1_Cycle1.qptiff", biomarkers=["DAPI"])
    write_qptiff(images / "S_Scan1.qptiff.intermediate", biomarkers=["DAPI"])
    write_qptiff(images / "S_Scan1.raw.qptiff", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]


def test_discovery_ignores_suffix_case(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.QPTIFF", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "B_Scan1.ER.QPTIFF", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["A_image", "B_image"]


def test_discovery_skips_hidden_files_and_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    for folder in (".Trash-1000/files", "__MACOSX"):
        (root / folder).mkdir(parents=True)
        write_qptiff(root / folder / "A_Scan1.qptiff", biomarkers=["DAPI"])
    write_qptiff(root / "A_Scan1.qptiff", biomarkers=["DAPI"])
    (root / "._A_Scan1.qptiff").write_bytes(b"AppleDouble")
    # a scan folder on another volume, and a link back up the tree
    (tmp_path / "scratch").mkdir()
    write_qptiff(tmp_path / "scratch" / "B_Scan1.qptiff", biomarkers=["DAPI"])
    (root / "scans").symlink_to(tmp_path / "scratch", target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)

    sdata = phenocycler(root, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["A_image", "B_image"]


def test_dataset_below_an_acquisition_like_folder_is_read(tmp_path: Path) -> None:
    # only the folders below `path` are checked for the HuBMAP acquisition folder `raw/images/phenocycler/`
    root = tmp_path / "raw" / "images" / "phenocycler" / "project"
    root.mkdir(parents=True)
    write_qptiff(root / "S_Scan1.qptiff", biomarkers=["DAPI"])

    sdata = phenocycler(root, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["S_image"]


@pytest.mark.parametrize("relative", [False, True])
def test_sidecars_hubmap_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool) -> None:
    raw_images = tmp_path / "raw" / "images"
    processed = tmp_path / "lab_processed" / "images"
    raw_images.mkdir(parents=True)
    processed.mkdir(parents=True)
    write_qptiff(raw_images / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    wells = [
        {"wellName": "A1", "items": [{"markerName": "DAPI"}]},
        {"wellName": "A2", "items": [{"markerName": "DAPI"}, {"markerName": "CD3"}, {"markerName": "--"}]},
    ]
    xpd = {"formatType": "Protein", "wells": wells}
    (raw_images / "S_Scan1.xpd").write_text(json.dumps(xpd))
    (processed / "S_Scan1.qptiff.channels.csv").write_text("Channel,Marker\n1,DAPI\n2,CD3\n")

    if relative:
        monkeypatch.chdir(raw_images)

    # DAPI repeats per cycle and blanks do not count as channels, so the sidecars match the two pages
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler("." if relative else tmp_path, image_models_kwargs=_FAST)

    raw = sdata.attrs["akoya"]["regions"]["S"]["raw"]
    assert raw["xpd"] == xpd
    assert raw["channels_csv"] == [{"Channel": "1", "Marker": "DAPI"}, {"Channel": "2", "Marker": "CD3"}]


def test_sidecar_channel_mismatch_warns(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3", "CD8"])
    xpd = {"wells": [{"wellName": "A1", "items": [{"markerName": "DAPI"}, {"markerName": "CD4"}]}]}
    (tmp_path / "S_Scan1.xpd").write_text(json.dumps(xpd))

    with pytest.warns(UserWarning, match="3") as record:
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert any("2" in str(warning.message) for warning in record)
    assert get_channel_names(sdata.images["S_image"]) == ["DAPI", "CD3", "CD8"]


def test_channels_csv_mismatch_warns(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])
    (tmp_path / "S_Scan1.qptiff.channels.csv").write_text("Marker\nDAPI\nCD3\nCD8\n")

    with pytest.warns(UserWarning, match="channels.csv"):
        phenocycler(path, image_models_kwargs=_FAST)


def test_invalid_xpd_warns_and_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    (tmp_path / "S_Scan1.xpd").write_text("not json")

    with pytest.warns(UserWarning, match="not valid JSON"):
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert "xpd" not in sdata.attrs["akoya"]["regions"]["S"]["raw"]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (json.dumps({"wells": [{"items": ["DAPI"]}]}).encode(), "no readable markers"),
        (json.dumps({"wells": "x"}).encode(), "no readable markers"),
        (json.dumps({"wells": [{"items": [{"markerName": "DAPI"}]}]}).encode("utf-16"), "not valid"),
    ],
    ids=["items-not-objects", "wells-not-a-list", "utf-16"],
)
def test_malformed_xpd_warns_and_image_is_read(tmp_path: Path, content: bytes, match: str) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    data = write_qptiff(path, biomarkers=["DAPI"])
    (tmp_path / "S_Scan1.xpd").write_bytes(content)

    with pytest.warns(UserWarning, match=match) as record:
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert not any("distinct markers" in str(warning.message) for warning in record)
    np.testing.assert_array_equal(_level0(sdata, "S_image").values, data)


def test_xpd_without_wells_is_kept_without_warning(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    # a UTF-8 byte order mark, as Windows tools write it
    (tmp_path / "S_Scan1.xpd").write_text(json.dumps({"formatType": "Protein"}), encoding="utf-8-sig")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert sdata.attrs["akoya"]["regions"]["S"]["raw"]["xpd"] == {"formatType": "Protein"}


def test_channels_csv_with_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])
    (tmp_path / "S_Scan1.qptiff.channels.csv").write_text("Marker\nDAPI\nCD3\nCD8\n", encoding="utf-8-sig")

    with pytest.warns(UserWarning, match="channels.csv lists 3 distinct markers"):
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert sdata.attrs["akoya"]["regions"]["S"]["raw"]["channels_csv"][0] == {"Marker": "DAPI"}


def test_unreadable_channels_csv_warns_and_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    (tmp_path / "S_Scan1.qptiff.channels.csv").write_bytes("Marker\nDAPI\n".encode("utf-16"))

    with pytest.warns(UserWarning, match="Ignoring"):
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert "channels_csv" not in sdata.attrs["akoya"]["regions"]["S"]["raw"]


def test_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="\\*.qptiff"):
        phenocycler(str(tmp_path))


def test_channel_names_override(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])

    sdata = phenocycler(path, channel_names=["nuclei", "T cells"], image_models_kwargs=_FAST)
    assert get_channel_names(sdata.images["S_image"]) == ["nuclei", "T cells"]
    channels = sdata.attrs["akoya"]["regions"]["S"]["channels"]
    assert [channel["name_deduped"] for channel in channels] == ["nuclei", "T cells"]

    with pytest.raises(ValueError, match="expected 2"):
        phenocycler(path, channel_names=["only one"])


def _write_mask(path: Path, n: int = 5, shape: tuple[int, int] = (256, 320)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint32)
    for i in range(n):
        mask[10 + 20 * i : 20 + 20 * i, 10:20] = i + 1
    tifffile.imwrite(path, mask)
    return mask


def test_labels_without_cells_table(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")

    sdata = phenocycler(path, labels=str(tmp_path / "mask.tif"), image_models_kwargs=_FAST)

    assert list(sdata.labels) == ["S_labels"]
    assert len(sdata.tables) == 0


def test_mapping_with_unknown_region_raises(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "A_mask.tif")

    with pytest.raises(ValueError, match="'Z'"):
        phenocycler(tmp_path, labels={"Z": tmp_path / "A_mask.tif"})


def test_mapping_keys_of_unselected_regions_are_ignored(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
        _write_mask(tmp_path / f"{region}_mask.tif")

    sdata = phenocycler(
        tmp_path,
        regions=["A"],
        labels={"A": tmp_path / "A_mask.tif", "B": tmp_path / "B_mask.tif"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert list(sdata.labels) == ["A_labels"]
    assert sdata.attrs["akoya"]["labels"] == {"A": str(tmp_path / "A_mask.tif")}


def test_raw_region_names_select_sanitised_regions(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Slide 1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(tmp_path / "Slide 2.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mask(tmp_path / "mask.tif")

    sdata = phenocycler(
        tmp_path, regions=["Slide 1"], labels={"Slide 1": tmp_path / "mask.tif"}, image_models_kwargs=_FAST
    )

    assert list(sdata.images) == ["Slide_1_image"]
    assert list(sdata.labels) == ["Slide_1_labels"]


def test_mapping_keys_colliding_after_sanitising_raise(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Slide_1.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mask(tmp_path / "mask.tif")

    with pytest.raises(ValueError, match="same region"):
        phenocycler(
            tmp_path,
            labels={"Slide 1": tmp_path / "mask.tif", "Slide_1": tmp_path / "mask.tif"},
            image_models_kwargs=_FAST,
        )


def test_mask_shape_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif", shape=(200, 300))

    with pytest.warns(UserWarning, match=r"\(200, 300\).*\(256, 320\)"):
        sdata = phenocycler(tmp_path, labels=tmp_path / "mask.tif", image_models_kwargs=_FAST)

    assert list(sdata.labels) == ["S_labels"]


def test_regions_colliding_after_sanitising_raise(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A [1].qptiff", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "A_[1].qptiff", biomarkers=["DAPI"])

    with pytest.raises(ValueError, match=r"same region\(s\) \['A__1_'\]"):
        phenocycler(tmp_path, image_models_kwargs=_FAST)


def test_single_mask_many_regions_raises(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")

    with pytest.raises(ValueError, match="mapping"):
        phenocycler(tmp_path, labels=tmp_path / "mask.tif")


def test_cli_labels_mapping(runner: CliRunner, tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mask(tmp_path / "mask.tif")
    output = tmp_path / "out.zarr"

    result = runner.invoke(
        phenocycler_wrapper,
        [
            "--input",
            str(tmp_path / "S_Scan1.qptiff"),
            "--output",
            str(output),
            "--labels",
            json.dumps({"S": str(tmp_path / "mask.tif")}),
            "--image-models-kwargs",
            json.dumps(_FAST),
            "--labels-models-kwargs",
            json.dumps(_FAST),
        ],
    )

    assert result.exit_code == 0, result.output
    sdata = read_zarr(output)
    assert list(sdata.labels) == ["S_labels"]


@pytest.mark.parametrize("value", ['{"S": 1}', "{not json"])
def test_cli_bad_mapping_is_bad_parameter(runner: CliRunner, tmp_path: Path, value: str) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])

    result = runner.invoke(
        phenocycler_wrapper, ["--input", str(tmp_path), "--output", str(tmp_path / "o.zarr"), "--labels", value]
    )

    assert result.exit_code == 2
    assert "labels" in result.output


def test_cli_regions_and_channel_names_may_hold_commas(runner: CliRunner, tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_[1,2].qptiff", biomarkers=["DAPI", "HLA-A"])
    write_qptiff(tmp_path / "B.qptiff", biomarkers=["DAPI", "CD3"])
    args = ["--input", str(tmp_path), "--image-models-kwargs", json.dumps(_FAST)]
    names = ["--channel-names", "DAPI", "--channel-names", "HLA-A,B,C"]

    result = runner.invoke(phenocycler_wrapper, [*args, "-o", str(tmp_path / "a.zarr"), "--regions", "A_[1,2]", *names])
    assert result.exit_code == 0, result.output
    sdata = read_zarr(tmp_path / "a.zarr")
    assert list(sdata.images) == ["A__1_2__image"]
    assert get_channel_names(sdata.images["A__1_2__image"]) == ["DAPI", "HLA-A,B,C"]

    result = runner.invoke(
        phenocycler_wrapper, [*args, "-o", str(tmp_path / "b.zarr"), "--regions", "A__1_2_", "--regions", "B"]
    )
    assert result.exit_code == 0, result.output
    assert sorted(read_zarr(tmp_path / "b.zarr").images) == ["A__1_2__image", "B_image"]


@pytest.mark.parametrize("option", ["--image-models-kwargs", "--labels-models-kwargs"])
def test_cli_kwargs_must_be_json_objects(runner: CliRunner, tmp_path: Path, option: str) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])

    result = runner.invoke(phenocycler_wrapper, ["-i", str(tmp_path), "-o", str(tmp_path / "o.zarr"), option, "null"])

    assert result.exit_code == 2
    assert "JSON object" in result.output


def test_str_regions_and_channel_names(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "AB_Scan1.qptiff", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "C_Scan1.qptiff", biomarkers=["DAPI"])

    sdata = phenocycler(tmp_path, regions="AB", channel_names="nuclei", image_models_kwargs=_FAST)

    assert list(sdata.images) == ["AB_image"]
    assert get_channel_names(sdata.images["AB_image"]) == ["nuclei"]


def test_write_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3", "CD3"])
    sdata = phenocycler(path, image_models_kwargs=_FAST)

    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")

    assert get_channel_names(read.images["S_image"]) == get_channel_names(sdata.images["S_image"])
    assert read.attrs["akoya"] == sdata.attrs["akoya"]
    assert read.attrs["spatialdata_io_reader"] == "phenocycler"


def test_is_lazy(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    written = write_qptiff(path, biomarkers=["DAPI", "CD3"])

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert isinstance(_level0(sdata, "S_image").data, da.Array)
    np.testing.assert_array_equal(_level0(sdata, "S_image").values, written)


def test_phenocycler_keyword_only(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"])
    with pytest.raises(TypeError, match="positional argument"):
        phenocycler(tmp_path, ["A"])  # type: ignore[misc]
