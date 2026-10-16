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

from spatialdata_io import phenocycler
from spatialdata_io.__main__ import phenocycler_wrapper
from tests._akoya_data import write_qptiff

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


def test_sidecars_hubmap_layout(tmp_path: Path) -> None:
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

    # DAPI repeats per cycle and blanks do not count as channels, so the sidecars match the two pages
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(tmp_path, image_models_kwargs=_FAST)

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


def test_mask_shape_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_mask(tmp_path / "mask.tif", shape=(128, 160))

    with pytest.warns(UserWarning, match=r"\(128, 160\).*\(256, 320\)"):
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


def test_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "S_Scan1.qptiff"
    write_qptiff(path, biomarkers=["DAPI", "CD3"])

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("full page decode during read")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    sdata = phenocycler(path, image_models_kwargs=_FAST)

    assert isinstance(_level0(sdata, "S_image").data, da.Array)
