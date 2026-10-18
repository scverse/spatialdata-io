"""Tests for the PhenoCycler-Fusion reader on synthetic qptiff files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation

from spatialdata_io import codex, phenocycler
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
        _write_mcquant(tmp_path / f"{region}.csv")
    _write_mask(tmp_path / "A_mask.tif")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path,
            cells_table={"A": tmp_path / "A.csv", "B": tmp_path / "B.csv"},
            labels={"A": tmp_path / "A_mask.tif"},
            image_models_kwargs=_FAST,
        )

    assert sorted(sdata.coordinate_systems) == ["A", "B"]
    for name in ("A_image", "A_labels"):
        assert get_transformation(sdata[name], get_all=True) == {"A": Identity()}
    for name in ("B_image", "B_cells"):
        assert get_transformation(sdata[name], get_all=True) == {"B": Identity()}


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


def test_read_all_bit_depths_with_labels_and_cells_table(tmp_path: Path) -> None:
    for suffix, dtype in ((".qptiff", "uint8"), (".er.qptiff", "uint16")):
        write_qptiff(
            tmp_path / f"S{suffix}", biomarkers=["DAPI"], shape=(64, 80), dtype=dtype, stage_position_cm=(1, 2)
        )
    mask = np.zeros((32, 40), dtype=np.uint32)
    mask[2:4, 2:4], mask[5:7, 5:7] = 1, 2
    tifffile.imwrite(tmp_path / "mask.tif", mask)
    (tmp_path / "cells.csv").write_text("CellID,CD3,X_centroid,Y_centroid,Area\n1,1,3,3,4\n2,2,6,6,4\n")

    sdata = phenocycler(
        tmp_path,
        cells_table=tmp_path / "cells.csv",
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
    table = sdata.tables["table"]
    assert table.uns["spatialdata_attrs"]["region"] == ["S_labels"]
    np.testing.assert_allclose(table.obsm["spatial"], [[6, 6], [12, 12]])

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


def _write_mcquant(path: Path, n: int = 5, markers: tuple[str, ...] = ("DAPI", "CD3")) -> None:
    rng = np.random.default_rng(0)
    cells = pd.DataFrame(
        {
            "CellID": np.arange(1, n + 1),
            **{marker: rng.uniform(0, 100, n) for marker in markers},
            "X_centroid": rng.uniform(0, 300, n),
            "Y_centroid": rng.uniform(0, 250, n),
            "Area": rng.uniform(10, 50, n),
        }
    )
    cells.to_csv(path, index=False)


def _write_mask(path: Path, n: int = 5, shape: tuple[int, int] = (256, 320)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint32)
    for i in range(n):
        mask[10 + 20 * i : 20 + 20 * i, 10:20] = i + 1
    tifffile.imwrite(path, mask)
    return mask


def test_mcquant_with_mask_labels(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"], stage_position_cm=(1.0, 2.0))
    _write_mcquant(tmp_path / "cells.csv")
    mask = _write_mask(tmp_path / "mask.tif")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(
            path,
            cells_table=tmp_path / "cells.csv",
            labels=tmp_path / "mask.tif",
            image_models_kwargs=_FAST,
            labels_models_kwargs=_FAST,
        )

    assert list(sdata.labels) == ["S_labels"]
    assert len(sdata.shapes) == 0
    np.testing.assert_array_equal(sdata.labels["S_labels"]["scale0"]["image"].values, mask)
    table = sdata.tables["table"]
    assert set(table.obs["region"]) == {"S_labels"}
    assert table.uns["spatialdata_attrs"]["region"] == ["S_labels"]
    assert list(table.obs["cell_id"]) == [1, 2, 3, 4, 5]
    assert list(table.var_names) == ["DAPI", "CD3"]
    assert sdata.attrs["akoya"]["cell_tables"] == {"S": str(tmp_path / "cells.csv")}
    assert sdata.attrs["akoya"]["labels"] == {"S": str(tmp_path / "mask.tif")}


def test_labels_without_cells_table(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")

    sdata = phenocycler(path, labels=str(tmp_path / "mask.tif"), image_models_kwargs=_FAST)

    assert list(sdata.labels) == ["S_labels"]
    assert len(sdata.tables) == 0


def test_cells_table_without_mask_makes_circles(tmp_path: Path) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])
    _write_mcquant(tmp_path / "cells.csv")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(path, cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)

    assert list(sdata.shapes) == ["S_cells"]
    assert set(sdata.tables["table"].obs["region"]) == {"S_cells"}


def test_phenocycler_cells_mapping_per_region(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
        _write_mcquant(tmp_path / f"{region}.csv")
        _write_mask(tmp_path / f"{region}_mask.tif")

    sdata = phenocycler(
        tmp_path,
        cells_table={"A": tmp_path / "A.csv", "B": str(tmp_path / "B.csv")},
        labels={"A": tmp_path / "A_mask.tif", "B": tmp_path / "B_mask.tif"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert list(sdata.tables) == ["table"]
    table = sdata.tables["table"]
    assert table.n_obs == 10
    assert table.obs_names.is_unique
    assert table.uns["spatialdata_attrs"]["region"] == ["A_labels", "B_labels"]
    assert sorted(sdata.labels) == ["A_labels", "B_labels"]


def test_codex_format_table_uses_region_column(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI"])
    (tmp_path / "cells.csv").write_text("cell_id,Sample,x,y,DAPI\n1,A,1.0,2.0,3.0\n2,B,1.0,2.0,3.0\n3,B,4,5,6\n")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path,
            regions=["A"],
            cells_table=tmp_path / "cells.csv",
            cells_table_kwargs={"region_column": "Sample"},
            image_models_kwargs=_FAST,
        )

    assert list(sdata.tables["table"].obs["cell_id"]) == [1]
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["A_cells"]


def test_mapping_with_unknown_region_raises(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mcquant(tmp_path / "A.csv", markers=("DAPI",))

    with pytest.raises(ValueError, match="'Z'"):
        phenocycler(tmp_path, cells_table={"Z": tmp_path / "A.csv"})


def test_mapping_keys_of_unselected_regions_are_ignored(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
        _write_mcquant(tmp_path / f"{region}.csv")
        _write_mask(tmp_path / f"{region}_mask.tif")

    sdata = phenocycler(
        tmp_path,
        regions=["A"],
        cells_table={"A": tmp_path / "A.csv", "B": tmp_path / "B.csv"},
        labels={"A": tmp_path / "A_mask.tif", "B": tmp_path / "B_mask.tif"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert list(sdata.labels) == ["A_labels"]
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["A_labels"]
    assert sdata.attrs["akoya"]["cell_tables"] == {"A": str(tmp_path / "A.csv")}


def test_raw_region_names_select_sanitised_regions(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Slide 1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(tmp_path / "Slide 2.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mcquant(tmp_path / "cells.csv")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path,
            regions=["Slide 1"],
            cells_table={"Slide 1": tmp_path / "cells.csv"},
            image_models_kwargs=_FAST,
        )

    assert list(sdata.images) == ["Slide_1_image"]
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["Slide_1_cells"]


def test_mapping_keys_colliding_after_sanitising_raise(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Slide_1.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mcquant(tmp_path / "cells.csv")

    with pytest.raises(ValueError, match="same region"):
        phenocycler(
            tmp_path,
            cells_table={"Slide 1": tmp_path / "cells.csv", "Slide_1": tmp_path / "cells.csv"},
            image_models_kwargs=_FAST,
        )


def test_mask_shape_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif", shape=(200, 300))

    with pytest.warns(UserWarning, match=r"\(200, 300\).*\(256, 320\)"):
        sdata = phenocycler(tmp_path, labels=tmp_path / "mask.tif", image_models_kwargs=_FAST)

    assert list(sdata.labels) == ["S_labels"]


def test_mask_with_row_number_instance_ids_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")
    (tmp_path / "cells.csv").write_text("cx,cy,DAPI\n1,2,3\n4,5,6\n")

    with pytest.warns(UserWarning, match="row numbers"):
        phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.csv",
            cells_table_format="generic",
            labels=tmp_path / "mask.tif",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"x": "cx", "y": "cy"},
        )


@pytest.mark.parametrize(
    ("header", "kwargs"),
    [
        ("Cell ID,X,Y,CD3", {"x": "X", "y": "Y", "instance_id": "Cell ID"}),
        ("Region,X,Y,CD3", {"x": "X", "y": "Y", "region_column": "Region"}),
        ("CELL_ID,region,X,Y,CD3", {"x": "X", "y": "Y", "region_column": "region", "markers": ["CD3"]}),
    ],
)
def test_generic_columns_clashing_with_reserved_keys(tmp_path: Path, header: str, kwargs: dict[str, str]) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    rows = "\n".join(
        ",".join("S" if col.casefold() == "region" else str(i + 1) for col in header.split(",")) for i in range(2)
    )
    (tmp_path / "cells.csv").write_text(f"{header}\n{rows}\n")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.csv",
            cells_table_format="generic",
            image_models_kwargs=_FAST,
            cells_table_kwargs=kwargs,
        )

    table = sdata.tables["table"]
    assert list(table.obs["region"]) == ["S_cells", "S_cells"]
    assert table.obs["cell_id"].dtype == np.int64
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr").tables["table"].n_obs == 2


@pytest.mark.parametrize(
    ("header", "row", "fmt", "kwargs"),
    [
        (
            "CELL,X,Y,CD3,region,cell_id",
            "{i},1,2,3,other,{source}",
            "generic",
            {"x": "X", "y": "Y", "instance_id": "CELL"},
        ),
        ("CellID,DAPI,CD3,X_centroid,Y_centroid,Area,region,cell_id", "{i},1,2,3,4,5,other,{source}", "mcquant", {}),
    ],
)
def test_extra_region_and_cell_id_columns_are_kept(
    tmp_path: Path, header: str, row: str, fmt: str, kwargs: dict[str, object]
) -> None:
    # columns named like the region and instance keys, but not read as them, keep their values as `source_<name>`
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    rows = "\n".join(row.format(i=i, source=100 + i) for i in (1, 2))
    (tmp_path / "cells.csv").write_text(f"{header}\n{rows}\n")
    if fmt == "generic":
        kwargs = {**kwargs, "markers": ["CD3"]}

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.csv",
            cells_table_format=fmt,
            image_models_kwargs=_FAST,
            cells_table_kwargs=kwargs,
        )

    obs = sdata.tables["table"].obs
    assert list(obs["region"]) == ["S_cells", "S_cells"]
    assert list(obs["cell_id"]) == [1, 2]
    assert list(obs["source_region"]) == ["other", "other"]
    assert list(obs["source_cell_id"]) == [101, 102]


def test_duplicate_instance_ids_raise(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    (tmp_path / "cells.csv").write_text("CellID,CD3,X_centroid,Y_centroid\n1,1,1,1\n2,1,2,2\n2,1,3,3\n")

    with pytest.raises(ValueError, match=r"cells\.csv has 1 duplicate instance id.*\[2\].*region 'S'"):
        phenocycler(tmp_path, cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)


def test_qupath_without_markers_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    (tmp_path / "cells.tsv").write_text("Object ID\tCentroid X px\tCentroid Y px\nu1\t1\t2\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="no marker intensities"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    assert sdata.tables["table"].n_vars == 0


def test_single_table_many_regions_needs_region_column(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mcquant(tmp_path / "cells.csv")

    with pytest.raises(ValueError, match="needs a region column"):
        phenocycler(tmp_path, cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)


def test_single_qupath_table_many_regions(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    header = "Image\tObject ID\tCentroid X µm\tCentroid Y µm\tCD3: Cell: Mean"
    rows = ["A_Scan1.qptiff\tu1\t5\t5\t1", "B_Scan1.qptiff\tu2\t10\t10\t2", "C_Scan1.qptiff\tu3\t1\t1\t3"]
    (tmp_path / "cells.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="C_Scan1.qptiff"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    assert list(table.obs["region"]) == ["A_cells", "B_cells"]
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"]), np.array([[5, 5], [10, 10]]) / 0.5069)
    # QuPath column names are not valid spatialdata names, so they are sanitized and the result can be written
    assert {"Image", "Object_ID", "Centroid_X_µm"} <= set(table.obs.columns)
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr").tables["table"].n_obs == 2


@pytest.mark.parametrize("unit", ["px", "µm"])
def test_single_qupath_table_matches_raw_names_of_sanitised_regions(tmp_path: Path, unit: str) -> None:
    # the region ids are sanitised (`A_[1,2]` -> `A__1_2_`), the table's `Image` column holds the raw file names
    for name in ("A_[1,2]", "B [3,4]"):
        write_qptiff(tmp_path / f"{name}.qptiff", biomarkers=["DAPI"])
    header = f"Image\tObject ID\tCentroid X {unit}\tCentroid Y {unit}\tCD3: Cell: Mean"
    rows = ["A_[1,2].qptiff\tu1\t5\t5\t1", "B [3,4].qptiff\tu2\t10\t10\t2"]
    (tmp_path / "cells.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    assert list(table.obs["region"]) == ["A__1_2__cells", "B__3_4__cells"]
    scale = 1 / 0.5069 if unit == "µm" else 1
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"]), np.array([[5, 5], [10, 10]]) * scale)


def test_regions_colliding_after_sanitising_raise(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A [1].qptiff", biomarkers=["DAPI"])
    write_qptiff(tmp_path / "A_[1].qptiff", biomarkers=["DAPI"])

    with pytest.raises(ValueError, match=r"same region\(s\) \['A__1_'\]"):
        phenocycler(tmp_path, image_models_kwargs=_FAST)


def test_single_qupath_table_regions_with_different_pixel_sizes(tmp_path: Path) -> None:
    # scans of the same instrument differ in the 4th decimal; each cell is converted with its own scan's pixel size
    sizes = {"A": 0.5069, "B": 0.5079}
    for region, size in sizes.items():
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI", "CD3"], resolution_px_per_cm=1e4 / size)
    header = "Image\tObject ID\tCentroid X µm\tCentroid Y µm\tCell: Area µm^2\tCD3: Cell: Mean"
    rows = ["A_Scan1.qptiff\tu1\t5\t6\t3.14159\t1", "B_Scan1.qptiff\tu2\t10\t20\t3.14159\t2"]
    (tmp_path / "cells.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    expected = np.array([[5 / sizes["A"], 6 / sizes["A"]], [10 / sizes["B"], 20 / sizes["B"]]])
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"]), expected)
    for region, (x, y) in zip(sizes, expected, strict=True):
        circles = sdata.shapes[f"{region}_cells"]
        np.testing.assert_allclose([circles.geometry.x.iloc[0], circles.geometry.y.iloc[0]], [x, y])
        np.testing.assert_allclose(circles["radius"].iloc[0], 1 / sizes[region], rtol=1e-5)


def test_different_panels_make_per_region_tables(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI", "CD8", "CD4"])
    _write_mcquant(tmp_path / "A.csv", markers=("DAPI", "CD3"))
    _write_mcquant(tmp_path / "B.csv", markers=("DAPI", "CD8", "CD4"))

    with pytest.warns(UserWarning, match="circles"):
        sdata = phenocycler(
            tmp_path, cells_table={"A": tmp_path / "A.csv", "B": tmp_path / "B.csv"}, image_models_kwargs=_FAST
        )

    assert sorted(sdata.tables) == ["A_table", "B_table"]
    assert list(sdata.tables["B_table"].var_names) == ["DAPI", "CD8", "CD4"]
    assert sdata.tables["A_table"].uns["spatialdata_attrs"]["region"] == ["A_cells"]


def test_single_mask_many_regions_raises(tmp_path: Path) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif")

    with pytest.raises(ValueError, match="mapping"):
        phenocycler(tmp_path, labels=tmp_path / "mask.tif")


def test_cli_cells_table_and_labels_mappings(runner: CliRunner, tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI", "CD3"])
    _write_mcquant(tmp_path / "cells.csv")
    _write_mask(tmp_path / "mask.tif")
    output = tmp_path / "out.zarr"

    result = runner.invoke(
        phenocycler_wrapper,
        [
            "--input",
            str(tmp_path / "S_Scan1.qptiff"),
            "--output",
            str(output),
            "--cells-table",
            str(tmp_path / "cells.csv"),
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
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["S_labels"]


@pytest.mark.parametrize("value", ['{"S": 1}', "{not json"])
def test_cli_bad_mapping_is_bad_parameter(runner: CliRunner, tmp_path: Path, value: str) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])

    result = runner.invoke(
        phenocycler_wrapper, ["--input", str(tmp_path), "--output", str(tmp_path / "o.zarr"), "--cells-table", value]
    )

    assert result.exit_code == 2
    assert "cells_table" in result.output


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


@pytest.mark.parametrize("option", ["--image-models-kwargs", "--cells-table-kwargs"])
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


def _write_qupath_px(path: Path, images: dict[str, int]) -> None:
    rows = ["Image,Object ID,Centroid X px,Centroid Y px,CD3: Cell: Mean"]
    rows += [f"{image},{i},{i},{i},1" for image, n in images.items() for i in range(1, n + 1)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_region_values_match_image_names_not_prefixes(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Tonsil_Scan1.qptiff", biomarkers=["DAPI"])
    images = {"Tonsil_Scan1.qptiff - resolution #1": 3, "Tonsil_2_Scan1.qptiff": 2, "Spleen_Scan1.qptiff": 2}
    _write_qupath_px(tmp_path / "cells.csv", images)

    with pytest.warns(UserWarning, match=r"Ignoring the cells of \['Spleen_Scan1.qptiff', 'Tonsil_2_Scan1.qptiff'\]"):
        sdata = phenocycler(
            tmp_path / "Tonsil_Scan1.qptiff", cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST
        )

    assert list(sdata.tables["table"].obs["cell_id"]) == [1, 2, 3]


@pytest.mark.parametrize("regions", [None, ["A"]])
def test_region_values_naming_no_image_are_not_assigned_to_the_selected_one(
    tmp_path: Path, regions: list[str] | None
) -> None:
    for region in ("A", "B"):
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI"])
    _write_qupath_px(tmp_path / "cells.csv", {"Patient 7 tonsil": 2, "Patient 9 lymph node": 2})

    with pytest.warns(UserWarning, match="Ignoring the cells"), pytest.raises(ValueError, match="has no cells of"):
        phenocycler(tmp_path, regions=regions, cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)


def test_duplicate_instance_ids_are_checked_per_assigned_region(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    # two names of the same image
    _write_qupath_px(tmp_path / "cells.csv", {"S": 1, "S_Scan1.qptiff": 1})

    with pytest.raises(ValueError, match=r"cells\.csv has 1 duplicate instance id.*region 'S'"):
        phenocycler(tmp_path, cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)


def test_codex_region_values_match_qptiff_file_names(tmp_path: Path) -> None:
    for region in ("reg001", "reg002"):
        write_qptiff(tmp_path / f"exp_{region}.qptiff", biomarkers=["DAPI"], software="Codex Instrument Manager 1.30")
    _write_qupath_px(tmp_path / "cells.csv", {"exp_reg001.qptiff": 1, "exp_reg002.qptiff": 2})

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, regions=["reg001"], cells_table=tmp_path / "cells.csv", image_models_kwargs=_FAST)

    assert list(sdata.tables["table"].obs["cell_id"]) == [1]


@pytest.mark.parametrize("regions", [["A"], ["B"]])
def test_selected_region_of_qupath_table_uses_each_rows_pixel_size(tmp_path: Path, regions: list[str]) -> None:
    rows, features = ["Image,Object ID,Centroid X µm,Centroid Y µm,CD3: Cell: Mean"], []
    for region, pixel_size in (("A", 0.5), ("B", 0.25)):
        write_qptiff(
            tmp_path / f"{region}_Scan1.qptiff",
            biomarkers=["DAPI"],
            resolution_px_per_cm=1e4 / pixel_size,
            slide_id=region,
        )
        for i in range(4):
            x, y = 20 + 25 * i, 30 + 15 * i
            rows.append(f"{region}_Scan1.qptiff,{region}{i},{x * pixel_size},{y * pixel_size},{i}")
            ring = [[x - 3, y - 3], [x + 3, y - 3], [x + 3, y + 3], [x - 3, y + 3], [x - 3, y - 3]]
            features.append(
                {"type": "Feature", "id": f"{region}{i}", "geometry": {"type": "Polygon", "coordinates": [ring]}}
            )
    (tmp_path / "cells.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "cells.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    sdata = phenocycler(
        tmp_path,
        regions=regions,
        cells_table=tmp_path / "cells.csv",
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    assert list(sdata.labels) == [f"{regions[0]}_labels"]
    np.testing.assert_allclose(np.asarray(sdata.tables["table"].obsm["spatial"])[:, 0], [20, 45, 70, 95])
