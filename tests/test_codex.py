"""Tests for the CODEX reader on synthetic CODEX Processor/Uploader trees and CIM qptiff files."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.models import get_channel_names
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import codex, phenocycler
from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers.codex import _discover_legacy, _read_legacy_image
from tests._akoya_data import (
    cim_experiment_v4_xml,
    forbid_pixel_reads,
    write_fcs,
    write_legacy_processor_tree,
    write_qptiff,
)

if TYPE_CHECKING:
    from pathlib import Path

_FAST = {"scale_factors": [2]}


def _level0(sdata, name):  # type: ignore[no-untyped-def]
    return sdata.images[name]["scale0"]["image"]


def test_codex_per_marker_images(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker")

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.images) == ["reg001_image", "reg002_image"]
    for region in ("reg001", "reg002"):
        image = sdata.images[f"{region}_image"]
        assert get_channel_names(image) == dedupe_channel_names(data["markers"])
        np.testing.assert_array_equal(_level0(sdata, f"{region}_image").values, data["images"][region])
        attrs = sdata.attrs["akoya"]["regions"][region]
        assert attrs["generation"] == "legacy_processor"
        assert attrs["pixel_size_um"] is None
        assert [(c["cycle"], c["channel"]) for c in attrs["channels"][:5]] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]
    assert sdata.attrs["spatialdata_io_reader"] == "codex"


def test_codex_hyperstack_channel_names_txt(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack")

    sdata = codex(str(tmp_path), image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    image = _level0(sdata, "reg001_image")
    assert image.shape[0] == 8
    assert get_channel_names(sdata.images["reg001_image"]) == dedupe_channel_names(data["markers"])
    np.testing.assert_array_equal(image.values, data["images"]["reg001"].reshape(8, *image.shape[1:]))


@pytest.mark.parametrize("from_subdirectory", [False, True])
def test_codex_sidecar_names_are_case_insensitive(tmp_path: Path, from_subdirectory: bool) -> None:
    # SPACEc and the HuBMAP codex-pipeline write `channelnames.txt`; older exports `Experiment.json`
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack")
    (tmp_path / "channelNames.txt").rename(tmp_path / "channelnames.txt")
    (tmp_path / "experiment.json").unlink()
    (tmp_path / "Experiment.json").write_text(json.dumps({"xyResolution": 377.5}))
    source = tmp_path / "processed_2020-01-01" / "stitched" if from_subdirectory else tmp_path

    sdata = codex(source, image_models_kwargs=_FAST)

    assert get_channel_names(sdata.images["reg001_image"]) == dedupe_channel_names(data["markers"])
    assert sdata.attrs["akoya"]["regions"]["reg001"]["pixel_size_um"] == pytest.approx(0.3775)


def test_codex_hyperstack_bad_channel_names_warns(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=2, channels=4, layout="hyperstack")
    (tmp_path / "channelNames.txt").write_text("\n".join(data["markers"][:7]) + "\n")

    with pytest.warns(UserWarning, match="channelNames.txt"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    names = get_channel_names(sdata.images["reg001_image"])
    assert names[:5] == ["cyc001_ch001", "cyc001_ch002", "cyc001_ch003", "cyc001_ch004", "cyc002_ch001"]
    assert len(names) == 8


def test_codex_hyperstack_imagej_labels_win(tmp_path: Path) -> None:
    labels = [f"L{i}" for i in range(6)]
    tifffile.imwrite(
        tmp_path / "reg001_X01_Y01_Z01.tif",
        np.zeros((3, 2, 32, 48), dtype=np.uint16),
        imagej=True,
        metadata={"axes": "ZCYX", "Labels": labels},
    )
    (tmp_path / "channelNames.txt").write_text("\n".join(f"other{i}" for i in range(6)))

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert get_channel_names(sdata.images["reg001_image"]) == labels


@pytest.mark.parametrize("labels", [["", "", "", ""], ["cyc1.tif", "", "", ""]], ids=["blank", "partly_blank"])
def test_codex_hyperstack_blank_imagej_labels_ignored(tmp_path: Path, labels: list[str]) -> None:
    # ImageJ writes an empty label for every unlabelled slice; such labels must not replace channelNames.txt
    tifffile.imwrite(
        tmp_path / "reg001_montage.tif",
        np.zeros((4, 32, 48), dtype=np.uint16),
        imagej=True,
        metadata={"axes": "CYX", "Labels": labels},
    )
    (tmp_path / "channelNames.txt").write_text("DAPI\nCD3\nDAPI2\nCD8\n")

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert get_channel_names(sdata.images["reg001_image"]) == ["DAPI", "CD3", "DAPI2", "CD8"]


def test_codex_cim_qptiff(tmp_path: Path) -> None:
    path = tmp_path / "experiment_reg001.qptiff"
    data = write_qptiff(
        path,
        biomarkers=["DAPI", "CD3", "DAPI", "CD20"],
        software="Codex Instrument Manager 1.30.0.12",
        scan_profile=cim_experiment_v4_xml([["DAPI", "CD3"], ["DAPI", "CD20"]]),
        label_page=False,
    )

    for source in (tmp_path, path):
        sdata = codex(source, image_models_kwargs=_FAST)
        assert list(sdata.images) == ["reg001_image"]
        assert get_channel_names(sdata.images["reg001_image"]) == ["DAPI", "CD3", "DAPI (2)", "CD20"]
        np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data)
        assert sdata.attrs["akoya"]["regions"]["reg001"]["generation"] == "cim_qptiff"


def test_codex_and_phenocycler_name_a_qptiff_alike(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "Tonsil_Scan1.qptiff", biomarkers=["DAPI"], dtype="uint8")
    write_qptiff(tmp_path / "Tonsil_Scan1.er.qptiff", biomarkers=["DAPI"], dtype="uint16")

    for reader in (codex, phenocycler):
        sdata = reader(tmp_path, image_models_kwargs=_FAST)
        assert list(sdata.images) == ["Tonsil_image"]
        assert sdata.attrs["akoya"]["regions"]["Tonsil"]["path"].endswith("Tonsil_Scan1.er.qptiff")


def test_reader_modules_export_only_the_readers() -> None:
    from spatialdata_io.readers import codex as codex_module
    from spatialdata_io.readers import phenocycler as phenocycler_module

    assert codex_module.__all__ == ["codex"]
    assert phenocycler_module.__all__ == ["phenocycler"]


def test_codex_old_flat_layout_image(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, layout="old_flat")

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    # only the first plane of the stack is written to disk
    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data["images"]["reg001"][:1])


def test_codex_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reg"):
        codex(str(tmp_path))


def test_codex_ignores_raw_tiles(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    expected = sorted(codex(tmp_path, image_models_kwargs=_FAST).images)
    processed = tmp_path / "processed_2020-01-01"
    plane = np.zeros((16, 16), dtype=np.uint16)
    for raw in (
        tmp_path / "cyc001_reg001" / "1_00001_Z001_CH1.tif",
        tmp_path / "cyc001_reg002" / "reg002_X01_Y01_Z01.tif",
        processed / "tiles" / "reg002_X01_Y01" / "reg002_X01_Y01_t001_z001_c001.tif",
        processed / "bestFocus" / "reg002_X01_Y01_Z08.tif",
    ):
        raw.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(raw, plane)

    assert sorted(codex(tmp_path, image_models_kwargs=_FAST).images) == expected


def test_codex_regions_channel_names_and_pixel_size(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=2, layout="per_marker")
    (tmp_path / "experiment.json").write_text(json.dumps({"xyResolution": 377.5, "numChannels": 2}))

    sdata = codex(tmp_path, regions=["reg002"], channel_names=["a", "b"], image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg002_image"]
    assert get_channel_names(sdata.images["reg002_image"]) == ["a", "b"]
    region = sdata.attrs["akoya"]["regions"]["reg002"]
    assert region["pixel_size_um"] == pytest.approx(0.3775)
    assert region["raw"]["experiment_json"]["numChannels"] == 2
    with pytest.raises(ValueError, match="reg003"):
        codex(tmp_path, regions=["reg003"])
    with pytest.raises(ValueError, match="channel_names"):
        codex(tmp_path, channel_names=["a"])


def test_discover_legacy_tables_prefer_requested_format(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker", fcs=True, csv=True)

    assert {region: path.suffix for region, path in _discover_legacy(tmp_path, fcs=True).tables.items()} == {
        "reg001": ".fcs",
        "reg002": ".fcs",
    }
    assert {region: path.suffix for region, path in _discover_legacy(tmp_path, fcs=False).tables.items()} == {
        "reg001": ".csv",
        "reg002": ".csv",
    }


def test_discover_legacy_tables_ranking(tmp_path: Path) -> None:
    cells = pd.DataFrame({"cell_id": [1.0, 2.0], "x": [1.0, 2.0], "y": [3.0, 4.0], "cyc001_ch001": [5.0, 6.0]})
    tifffile.imwrite(tmp_path / "Kidney.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    cells.to_csv(tmp_path / "Kidney.csv", index=False)
    # an unrelated flow cytometry tube and a file that is no FCS at all do not replace the CODEX table
    (tmp_path / "flow").mkdir()
    write_fcs(tmp_path / "flow" / "tube1.fcs", pd.DataFrame({"FSC-A": [1.0], "SSC-A": [2.0]}))
    (tmp_path / "flow" / "broken.fcs").write_bytes(b"not an fcs file")
    assert _discover_legacy(tmp_path, fcs=True).tables == {"*": tmp_path / "Kidney.csv"}

    # a CODEX gate export next to the table: the reader cannot tell which one holds the cells
    (tmp_path / "gates").mkdir()
    write_fcs(tmp_path / "gates" / "CD3pos.fcs", cells.iloc[:1])
    with pytest.raises(ValueError, match=r"CD3pos\.fcs.*Kidney\.csv|Kidney\.csv.*CD3pos\.fcs"):
        _discover_legacy(tmp_path, fcs=True)

    # compensated beats uncompensated before the file format
    fcs_dir = tmp_path / "exp" / "segm" / "segm-1" / "fcs"
    for kind in ("compensated", "uncompensated"):
        (fcs_dir / kind).mkdir(parents=True)
    cells.to_csv(fcs_dir / "compensated" / "reg001_compensated.csv", index=False)
    write_fcs(fcs_dir / "uncompensated" / "reg001_uncompensated.fcs", cells)
    assert _discover_legacy(tmp_path / "exp", fcs=True).tables == {
        "reg001": fcs_dir / "compensated" / "reg001_compensated.csv"
    }

    # two segmentation runs of one region are not resolved by their path
    segm_10 = tmp_path / "exp" / "segm" / "segm-10" / "fcs" / "compensated"
    segm_10.mkdir(parents=True)
    cells.to_csv(segm_10 / "reg001_compensated.csv", index=False)
    with pytest.raises(ValueError, match=r"segm-1/.*segm-10/|segm-10/.*segm-1/"):
        _discover_legacy(tmp_path / "exp", fcs=True)


def test_codex_user_cells_table_replaces_discovered(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker", n_cells=50)
    pd.DataFrame(
        {"CellID": [1, 2, 3], "CD3": [1.0, 2.0, 3.0], "X_centroid": [1.0, 2.0, 3.0], "Y_centroid": [4.0, 5.0, 6.0]}
    ).to_csv(tmp_path / "mine.csv", index=False)
    mask = np.zeros((128, 160), dtype=np.uint16)
    mask[:3, :3] = [1, 2, 3]
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    sdata = codex(
        tmp_path,
        cells_table={"reg002": tmp_path / "mine.csv"},
        labels={"reg002": tmp_path / "mask.tif"},
        image_models_kwargs=_FAST,
        labels_models_kwargs=_FAST,
    )

    table = sdata.tables["table"]
    assert table.n_obs == 3
    assert list(table.var_names) == ["CD3"]
    assert table.uns["spatialdata_attrs"]["region"] == ["reg002_labels"]
    assert list(sdata.labels) == ["reg002_labels"]
    assert len(sdata.shapes) == 0
    assert sdata.attrs["akoya"]["cell_tables"] == {"reg002": str(tmp_path / "mine.csv")}


def test_codex_explicit_inputs_disable_discovery(tmp_path: Path) -> None:
    # HuBMAP-like: the mask has the region id of its image, and two region-less tables are discoverable
    for folder in ("expr", "mask"):
        (tmp_path / folder).mkdir()
    tifffile.imwrite(tmp_path / "expr" / "reg001_expr.ome.tiff", np.zeros((2, 16, 16), dtype=np.uint16))
    mask = tmp_path / "mask" / "reg001_mask.ome.tiff"
    tifffile.imwrite(mask, np.arange(256, dtype=np.uint16).reshape(16, 16))
    header = "cell_id,x,y,cyc001_ch001\n"
    (tmp_path / "cells_all.csv").write_text(header + "1,1.0,1.0,1.0\n2,2.0,2.0,2.0\n")
    (tmp_path / "cells_filtered.csv").write_text(header + "2,2.0,2.0,2.0\n")

    for labels in (mask, {"reg001": mask}):
        sdata = codex(
            tmp_path,
            cells_table=tmp_path / "cells_filtered.csv",
            labels=labels,
            image_models_kwargs=_FAST,
            labels_models_kwargs=_FAST,
        )
        assert list(sdata.images) == ["reg001_image"]
        assert list(sdata.labels) == ["reg001_labels"]
        assert sdata.tables["table"].n_obs == 1

    # an explicit table of an unselected region does not bring the discovered tables back
    write_legacy_processor_tree(tmp_path / "tree", regions=2, cycles=1, channels=1, n_cells=5)
    sdata = codex(tmp_path / "tree", regions=["reg001"], cells_table={"reg002": tmp_path / "cells_all.csv"})
    assert not sdata.tables


def test_codex_regions_of_regionless_table_are_available(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=1, fcs=False, csv=False, n_cells=5)
    pd.concat([data["cells"], data["cells"].assign(region=2.0)]).to_csv(tmp_path / "cells.csv", index=False)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, regions=["reg002"], image_models_kwargs=_FAST)

    assert not sdata.images
    assert list(sdata.shapes) == ["reg002_cells"]


def test_codex_labels_with_discovered_tables(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=1, layout="per_marker", n_cells=5)
    tifffile.imwrite(tmp_path / "mask.tif", np.zeros((128, 160), dtype=np.uint16))

    sdata = codex(tmp_path, labels=tmp_path / "mask.tif", image_models_kwargs=_FAST, labels_models_kwargs=_FAST)

    assert list(sdata.labels) == ["reg001_labels"]
    assert len(sdata.shapes) == 0
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["reg001_labels"]


def test_codex_qupath_geojson_labels(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path / "tree", regions=2, layout="per_marker", n_cells=5)
    header = "Image\tObject ID\tCentroid X px\tCentroid Y px\tCD3: Cell: Mean"
    rows = ["reg002.tif\tu1\t5\t5\t1", "reg002.tif\tu2\t25\t25\t2"]
    (tmp_path / "q.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    features = [
        {"type": "Feature", "id": oid, "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}}
        for oid, ring in [
            ("u2", [[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]),
            ("u1", [[0, 0], [9, 0], [9, 9], [0, 0]]),
        ]
    ]
    (tmp_path / "polygons.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    sdata = codex(
        tmp_path / "tree",
        regions=["reg002"],
        cells_table={"reg002": tmp_path / "q.tsv"},
        image_models_kwargs=_FAST,
        cells_table_kwargs={"geojson": {"reg002": tmp_path / "polygons.geojson"}},
    )

    assert list(sdata.labels) == ["reg002_labels"]
    assert list(sdata.shapes) == ["reg002_polygons"]
    values = sdata.labels["reg002_labels"]["scale0"]["image"].values
    assert values.shape == (128, 160)
    assert values[2, 7] == 1  # the triangle below the diagonal
    assert values[7, 2] == 0
    assert values[25, 25] == 2
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["reg002_labels"]
    assert sdata.attrs["akoya"]["geojson"] == {"reg002": str(tmp_path / "polygons.geojson")}

    with pytest.raises(ValueError, match="same form as the tables"):
        codex(
            tmp_path / "tree",
            cells_table=tmp_path / "q.tsv",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"geojson": {"reg002": tmp_path / "polygons.geojson"}},
        )


def test_codex_str_regions_and_channel_names(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=1, layout="per_marker", n_cells=5)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, regions="reg002", channel_names="nuclei", image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg002_image"]
    assert get_channel_names(sdata.images["reg002_image"]) == ["nuclei"]


def test_codex_regions_validation(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=1, layout="per_marker", fcs=False, csv=False)

    with pytest.raises(TypeError, match="sequence of region ids"):
        codex(tmp_path, regions={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        codex(tmp_path, regions=[])

    sdata = codex(tmp_path, regions=["reg002", "reg002"], image_models_kwargs=_FAST)
    assert list(sdata.images) == ["reg002_image"]


@pytest.mark.parametrize("imagej", [False, True])
def test_read_legacy_image_tiles_uncompressed_planes(tmp_path: Path, imagej: bool) -> None:
    # uncompressed single-strip planes are one zarr chunk each; they must be memory-mapped and tiled instead
    stack = np.random.default_rng(0).integers(0, 2**16, (2, 100, 130), dtype=np.uint16)
    files = [tmp_path / f"reg001_cyc001_ch{i + 1:03d}_M{i}.tif" for i in range(2)]
    for file, plane in zip(files, stack, strict=True):
        tifffile.imwrite(file, plane, imagej=imagej, rowsperstrip=plane.shape[0])
    hyperstack = tmp_path / "reg002_montage.tif"
    tifffile.imwrite(hyperstack, stack[None], imagej=True, metadata={"axes": "TCYX"})

    for source in (files, [hyperstack]):
        image, _ = _read_legacy_image(source, None, image_models_kwargs={"chunks": (1, 32, 64)})
        assert max(image.chunks[-2]) <= 32
        assert max(image.chunks[-1]) <= 64
        np.testing.assert_array_equal(image.compute(), stack)


def test_codex_single_tif_without_region(tmp_path: Path) -> None:
    path = tmp_path / "Kidney.tif"
    tifffile.imwrite(path, np.zeros((3, 32, 48), dtype=np.uint16), imagej=True, metadata={"axes": "CYX"})

    sdata = codex(path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["Kidney_image"]
    assert _level0(sdata, "Kidney_image").shape == (3, 32, 48)


def test_discover_legacy_regionless_table_ignores_sidecar_csv(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path / "exp", layout="per_marker", fcs=False, csv=False)
    (tmp_path / "antibody_list.csv").write_text("marker,clone\nCD3,UCHT1\n")
    cells = tmp_path / "exp" / "results" / "cells.csv"
    cells.parent.mkdir()
    cells.write_text("cell_id,x,y,cyc001_ch001:DAPI\n1,2.0,3.0,4.0\n")

    assert _discover_legacy(tmp_path, fcs=True).tables == {"*": cells}

    (tmp_path / "clusters.csv").write_text("cellid,cluster\n1,2\n")
    (tmp_path / "xy.csv").write_text("x,y\n1.0,2.0\n")
    assert _discover_legacy(tmp_path, fcs=True).tables == {"*": cells}

    (tmp_path / "other_cells.csv").write_text("cell_id:cell_id,x:x,y:y\n1,1.0,2.0\n")
    with pytest.raises(ValueError, match="cannot choose"):
        _discover_legacy(tmp_path, fcs=True)


@pytest.mark.parametrize("with_table", [True, False])
def test_codex_region_named_non_cell_csv_is_ignored(tmp_path: Path, with_table: bool) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=1, channels=2, fcs=False, csv=with_table, n_cells=5)
    (tmp_path / "reg001_antibodies.csv").write_text("Antibody,Clone\nCD3,UCHT1\n")
    # a HuBMAP SPRM output of the same region, which is only read when given as `cells_table`
    (tmp_path / "reg001_expr-cell_channel_mean.csv").write_text("ID,CD3\n1,2.0\n")

    if with_table:
        with pytest.warns(UserWarning, match="circles"):
            sdata = codex(tmp_path, image_models_kwargs=_FAST)
        assert sdata.tables["table"].n_obs == len(data["cells"])
        assert sdata.attrs["akoya"]["cell_tables"]["reg001"].endswith("reg001_compensated.csv")
    else:
        sdata = codex(tmp_path, image_models_kwargs=_FAST)
        assert not sdata.tables
    assert list(sdata.images) == ["reg001_image"]


@pytest.mark.parametrize("layout", ["per_marker", "hyperstack"])
def test_codex_is_lazy(tmp_path: Path, layout: Literal["per_marker", "hyperstack"]) -> None:
    data = write_legacy_processor_tree(tmp_path, layout=layout)

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    expected = data["images"]["reg001"]
    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, expected.reshape(-1, *expected.shape[-2:]))


@pytest.mark.parametrize("qptiff", [False, True])
def test_codex_imread_kwargs_warns_when_unused(tmp_path: Path, qptiff: bool) -> None:
    # every image is read lazily with tifffile, so `imread_kwargs` of the old dask_image reader are ignored
    if qptiff:
        write_qptiff(tmp_path / "Tonsil_Scan1.qptiff", biomarkers=["DAPI"])
    else:
        write_legacy_processor_tree(tmp_path, layout="per_marker")
    with pytest.warns(UserWarning, match="imread_kwargs"):
        codex(tmp_path, imread_kwargs={"nframes": 1}, image_models_kwargs=_FAST)


def test_codex_table_multi_region(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker", n_cells=50)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.tables) == ["table"]
    table = sdata.tables["table"]
    assert table.n_obs == 100
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert list(table.var_names) == dedupe_channel_names(data["markers"])
    assert sorted(sdata.shapes) == ["reg001_cells", "reg002_cells"]
    # the fixture returns the cells of the last region
    radius = sdata.shapes["reg002_cells"]["radius"].to_numpy()
    np.testing.assert_allclose(radius, np.sqrt(data["cells"]["size"] / np.pi), rtol=1e-6)
    reg002 = table[table.obs["region"] == "reg002_cells"]
    np.testing.assert_allclose(reg002.obsm["spatial"], data["cells"][["x", "y"]], rtol=1e-6)
    assert sdata.attrs["akoya"]["cell_tables"]["reg002"].endswith("reg002_compensated.fcs")


def test_codex_multi_region_own_coordinate_systems(tmp_path: Path) -> None:
    # CODEX Processor regions have no stage positions: each gets its own coordinate system
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, layout="per_marker", n_cells=5)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert sorted(sdata.coordinate_systems) == ["reg001", "reg002"]
    for region in ("reg001", "reg002"):
        for name in (f"{region}_image", f"{region}_cells"):
            assert get_transformation(sdata[name], get_all=True) == {region: Identity()}

    single = codex(tmp_path, regions=["reg002"], image_models_kwargs=_FAST)
    assert single.coordinate_systems == ["global"]


def test_codex_region_without_image_own_coordinate_system(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, layout="per_marker", n_cells=5)
    shutil.rmtree(tmp_path / "processed_2020-01-01" / "stitched" / "reg002")

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    assert get_transformation(sdata["reg001_image"], get_all=True) == {"reg001": Identity()}
    assert get_transformation(sdata["reg002_cells"], get_all=True) == {"reg002": Identity()}


@pytest.mark.parametrize("fcs", [True, False])
def test_codex_old_flat_layout_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcs: bool) -> None:
    # the original reader read the image (and the CSV) relative to the working directory
    data = write_legacy_processor_tree(tmp_path / "data", layout="old_flat", fcs=fcs)
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path / "data", fcs=fcs, image_models_kwargs=_FAST)

    np.testing.assert_array_equal(_level0(sdata, "reg001_image").values, data["images"]["reg001"][:1])
    table = sdata.tables["table"]
    assert table.n_obs == 50
    assert list(table.obs["region"].cat.categories) == ["reg001_cells"]
    np.testing.assert_allclose(table.X, data["cells"].filter(regex="^cyc").to_numpy(), rtol=1e-6)


def test_codex_fcs_false_falls_back_to_fcs_with_warning(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=True, csv=False)

    with pytest.warns(UserWarning, match="CSV"):
        sdata = codex(tmp_path, fcs=False, image_models_kwargs=_FAST)

    assert list(sdata.tables["table"].var_names) == dedupe_channel_names(data["markers"])


def test_codex_without_tables_reads_images_only(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=False)

    sdata = codex(tmp_path, image_models_kwargs=_FAST)

    assert list(sdata.images) == ["reg001_image"]
    assert not sdata.tables
    assert not sdata.shapes


def test_codex_table_float32_dense(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=True)

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, fcs=False, image_models_kwargs=_FAST)

    table = sdata.tables["table"]
    assert isinstance(table.X, np.ndarray)
    assert table.X.dtype == np.float32


def test_codex_regions_subsets_tables(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, layout="per_marker")

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path, regions=["reg002"], image_models_kwargs=_FAST)

    assert list(sdata.shapes) == ["reg002_cells"]
    assert list(sdata.tables["table"].obs["region"].cat.categories) == ["reg002_cells"]


def test_codex_single_fcs_file(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker")
    fcs = tmp_path / "processed_2020-01-01" / "segm" / "segm-1" / "fcs" / "compensated" / "reg001_compensated.fcs"

    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(fcs)

    assert not sdata.images
    assert list(sdata.shapes) == ["reg001_cells"]
    assert sdata.tables["table"].n_obs == 50


def test_codex_unreadable_regionless_table_raises(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "reg001_montage.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    # a region-less table that cannot be assigned to a region: there are two images and no `region` column
    tifffile.imwrite(tmp_path / "reg002_montage.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    (tmp_path / "cells.csv").write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,3.0\n")
    (tmp_path / "clusters.csv").write_text("cellid,cluster\n1,2\n")

    with pytest.raises(ValueError, match="cannot be assigned to a region"):
        codex(tmp_path, image_models_kwargs=_FAST)

    # a CODEX table next to a single image that cannot be read
    (tmp_path / "reg002_montage.tif").unlink()
    (tmp_path / "cells.csv").write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,3.0\n1,4.0,5.0,6.0\n")
    with pytest.raises(ValueError, match="duplicate instance id"):
        codex(tmp_path, image_models_kwargs=_FAST)

    # a single table file without a region column and no image
    write_fcs(tmp_path / "Kidney.fcs", pd.DataFrame({"cell_id": [1.0], "x": [1.0], "y": [2.0], "cyc001_ch001": [3.0]}))
    with pytest.raises(ValueError, match="cannot be assigned to a region"):
        codex(tmp_path / "Kidney.fcs")


def test_codex_regionless_table_belongs_to_single_stem_image(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "Kidney.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    (tmp_path / "cells.csv").write_text("cell_id,region,x,y,size,cyc001_ch001\n1,1,1.0,2.0,3.0,4.0\n")

    for regions in (None, ["Kidney"]):
        with pytest.warns(UserWarning, match="circles"):
            sdata = codex(tmp_path, regions=regions, image_models_kwargs=_FAST)
        assert list(sdata.tables["table"].obs["region"].cat.categories) == ["Kidney_cells"]
        assert list(sdata.shapes) == ["Kidney_cells"]
        assert sdata.attrs["akoya"]["cell_tables"] == {"*": str(tmp_path / "cells.csv")}


def test_codex_duplicate_cell_ids_raise(tmp_path: Path) -> None:
    data = write_legacy_processor_tree(tmp_path, cycles=1, channels=1, fcs=False, n_cells=3)
    table = next(tmp_path.rglob("reg001_compensated.csv"))
    cells = data["cells"].assign(cell_id=[1.0, 1.0, 2.0])
    cells.to_csv(table, index=False)

    with pytest.raises(ValueError, match="reg001_compensated.csv has 1 duplicate instance id"):
        codex(tmp_path, image_models_kwargs=_FAST)


def test_codex_overlapping_tables_raise(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, layout="per_marker", fcs=False, csv=True)
    (tmp_path / "all_cells.csv").write_text("cell_id,region,x,y,cyc001_ch001\n1,1,1.0,2.0,3.0\n")

    with pytest.raises(ValueError, match=r"Several cell tables contain the cells of region\(s\) \['reg001'\]"):
        codex(tmp_path, fcs=False, image_models_kwargs=_FAST)


def test_codex_write_roundtrip(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path / "data", regions=2, cycles=1, channels=2, n_cells=5)
    with pytest.warns(UserWarning, match="circles"):
        sdata = codex(tmp_path / "data", image_models_kwargs=_FAST)

    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")

    assert sorted(read.images) == ["reg001_image", "reg002_image"]
    assert sorted(read.shapes) == ["reg001_cells", "reg002_cells"]
    for region in ("reg001", "reg002"):
        name = f"{region}_image"
        assert get_channel_names(read.images[name]) == get_channel_names(sdata.images[name])
        np.testing.assert_array_equal(_level0(read, name).values, _level0(sdata, name).values)
        assert get_transformation(read.images[name], get_all=True) == {region: Identity()}
        assert get_transformation(read.shapes[f"{region}_cells"], get_all=True) == {region: Identity()}
    table = read.tables["table"]
    assert table.n_obs == 10
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert list(table.var_names) == list(sdata.tables["table"].var_names)
    assert read.attrs["akoya"] == sdata.attrs["akoya"]
    assert read.attrs["spatialdata_io_reader"] == "codex"


def test_codex_old_positional_call(tmp_path: Path) -> None:
    write_legacy_processor_tree(tmp_path, regions=2, cycles=1, channels=1, layout="per_marker", fcs=False, csv=False)

    # `codex(path, fcs, imread_kwargs)` is the signature of earlier releases
    for fcs in (True, False):
        sdata = codex(tmp_path, fcs, {})
        assert sorted(sdata.images) == ["reg001_image", "reg002_image"]

    with pytest.warns(UserWarning, match="imread_kwargs"):
        codex(tmp_path, True, {"nframes": 1})
