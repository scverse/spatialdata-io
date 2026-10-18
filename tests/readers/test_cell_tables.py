from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import get_extent, read_zarr
from spatialdata.models import get_axes_names
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation
from tests._akoya_data import forbid_pixel_reads, write_fcs, write_qptiff

from spatialdata_io import phenocycler
from spatialdata_io.readers._utils._akoya import dedupe_channel_names
from spatialdata_io.readers._utils._cell_tables import (
    build_cells,
    read_cell_table,
    read_labels,
    resolve_per_region,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils._cell_tables import CellTable


def _cells(n: int = 5, region: int = 1, markers: tuple[str, ...] = ("cyc001_ch001", "cyc001_ch002")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, n + 1, dtype=float),
            "region": float(region),
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 80, n),
            "size": rng.uniform(10, 100, n),
            **{marker: rng.uniform(0, 1000, n) for marker in markers},
        }
    )


def _read_codex(path: Path, region: str | None = None) -> CellTable:
    return read_cell_table(path, default_region=region)


def test_read_codex_fcs_marker_names_from_pns(tmp_path: Path) -> None:
    cells = _cells(markers=("cyc001_ch001", "cyc001_ch002", "cyc002_ch001"))
    path = tmp_path / "reg001_compensated.fcs"
    write_fcs(path, cells, {"cyc001_ch001": "DAPI", "cyc001_ch002": "CD3", "cyc002_ch001": "DAPI"})

    table = _read_codex(path)

    assert list(table.var.index) == ["DAPI", "CD3", "DAPI (2)"]
    assert list(table.var["cycle"]) == [1, 1, 2]
    assert list(table.var["channel"]) == [1, 2, 1]
    assert list(table.var["source_column"]) == ["cyc001_ch001", "cyc001_ch002", "cyc002_ch001"]
    assert table.X.dtype == np.float32
    np.testing.assert_allclose(table.X, cells[["cyc001_ch001", "cyc001_ch002", "cyc002_ch001"]], rtol=1e-6)
    assert "cyc001_ch001" not in table.obs.columns
    assert list(table.obs.index) == ["1", "2", "3", "4", "5"]
    assert list(table.region.unique()) == ["reg001"]
    assert table.instance_id.dtype == np.int64
    assert table.centroids_px is not None
    np.testing.assert_allclose(table.centroids_px, cells[["x", "y"]], rtol=1e-6)
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(cells["size"] / np.pi), rtol=1e-6)


def test_read_codex_fcs_without_pns_keeps_column_names(tmp_path: Path) -> None:
    path = tmp_path / "reg001.fcs"
    write_fcs(path, _cells())

    assert list(_read_codex(path).var.index) == ["cyc001_ch001", "cyc001_ch002"]


def test_read_codex_csv_colon_header(tmp_path: Path) -> None:
    path = tmp_path / "reg002.csv"
    path.write_text(
        "cell_id:cell_id,region:region,x:x,y:y,size:size,cyc001_ch001:DAPI1,cyc001_ch002:CD3\n"
        "1,2,10,20,30,100.5,7\n"
        "2,2,11,21,31,200.5,8\n"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the markers are the cycCCC_chHHH columns, no heuristic
        table = _read_codex(path)

    assert list(table.var.index) == ["DAPI1", "CD3"]
    assert list(table.var["source_column"]) == ["cyc001_ch001:DAPI1", "cyc001_ch002:CD3"]
    np.testing.assert_array_equal(table.X, np.array([[100.5, 7], [200.5, 8]], dtype=np.float32))
    # the columns read as the instance ids and regions are replaced by the keys of the table element
    assert list(table.obs.columns) == ["x", "y", "size"]
    assert list(table.region) == ["reg002", "reg002"]
    np.testing.assert_array_equal(table.centroids_px, [[10, 20], [11, 21]])


def test_read_codex_csv_marker_name_columns(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,x,y,tile_num,CD3,CD8,label\n1,1.0,2.0,1,5.0,6.0,T\n")

    with pytest.warns(UserWarning, match=r"no 'cycCCC_chHHH' columns.*\['CD3', 'CD8'\]"):
        table = _read_codex(path, region="Kidney")

    assert list(table.var.index) == ["CD3", "CD8"]
    assert "cycle" not in table.var.columns
    assert list(table.obs.columns) == ["x", "y", "tile_num", "label"]
    assert list(table.region) == ["Kidney"]
    assert table.radius_px is None


def test_read_codex_table_errors(tmp_path: Path) -> None:
    no_xy = tmp_path / "a.csv"
    no_xy.write_text("cell_id,cyc001_ch001\n1,2.0\n")
    with pytest.raises(ValueError, match="'x' and 'y'"):
        _read_codex(no_xy)
    no_region = tmp_path / "b.csv"
    no_region.write_text("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,2.0\n")
    with pytest.raises(ValueError, match="region"):
        _read_codex(no_region)


@pytest.mark.parametrize("mangled", [False, True])
def test_repeated_column_names_are_deduplicated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mangled: bool) -> None:
    if mangled:  # pandas >= 3.1 renames repeated names (DAPI.1) with the pyarrow engine too
        read_csv = pd.read_csv

        def mangling_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
            df = read_csv(*args, **kwargs)
            seen: dict[str, int] = {}
            columns = []
            for column in df.columns:
                columns.append(f"{column}.{seen[column]}" if column in seen else column)
                seen[column] = seen.get(column, 0) + 1
            df.columns = pd.Index(columns)
            return df

        monkeypatch.setattr(pd, "read_csv", mangling_read_csv)
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,x,y,size,DAPI,CD3,DAPI\n1,2,3,4,5,6,7\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="no 'cycCCC_chHHH' columns"):
        table = _read_codex(path, region="S")

    assert list(table.var.index) == ["DAPI", "CD3", "DAPI (2)"]
    assert table.X.shape == (1, 3)


@pytest.mark.parametrize("cell_id", ["", "1.5", "a"])
def test_missing_or_fractional_instance_ids_raise(tmp_path: Path, cell_id: str) -> None:
    path = tmp_path / "cells.csv"
    path.write_text(f"cell_id,x,y,cyc001_ch001\n1,1,1,1\n{cell_id},2,2,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"1 cell\(s\) whose instance id in 'cell_id' is missing or not an integer"):
        _read_codex(path, region="S")


def test_build_cells_single_table_and_circles(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(n=3, region=1))
    write_fcs(tmp_path / "reg002.fcs", _cells(n=4, region=2))
    tables = [_read_codex(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]
    translation = Translation([10, 0], axes=("x", "y"))

    with pytest.warns(UserWarning, match=r"No segmentation mask found for region.*circles") as record:
        anndatas, shapes = build_cells(
            tables, {}, {"reg001": {"reg001": Identity()}, "reg002": {"reg002": translation}}
        )

    assert len(record) == 1
    assert list(anndatas) == ["table"]
    table = anndatas["table"]
    assert table.n_obs == 7
    assert table.obs_names.is_unique
    assert list(table.obs["region"].cat.categories) == ["reg001_cells", "reg002_cells"]
    assert table.uns["spatialdata_attrs"]["region"] == ["reg001_cells", "reg002_cells"]
    assert table.obs["cell_id"].dtype == np.int64
    assert tables[0].centroids_px is not None
    np.testing.assert_allclose(np.asarray(table.obsm["spatial"])[:3], tables[0].centroids_px)
    assert sorted(shapes) == ["reg001_cells", "reg002_cells"]
    assert list(shapes["reg002_cells"].index) == [1, 2, 3, 4]
    assert get_transformation(shapes["reg002_cells"], get_all=True) == {"reg002": translation}


def test_build_cells_different_panels_split_tables(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(region=1, markers=("cyc001_ch001",)))
    write_fcs(tmp_path / "reg002.fcs", _cells(region=2, markers=("cyc001_ch001", "cyc001_ch002")))
    tables = [_read_codex(tmp_path / f"reg00{r}.fcs") for r in (1, 2)]

    with pytest.warns(UserWarning, match="circles"):
        anndatas, _ = build_cells(tables, {}, {})

    assert sorted(anndatas) == ["reg001_table", "reg002_table"]
    assert anndatas["reg002_table"].n_vars == 2
    assert anndatas["reg002_table"].uns["spatialdata_attrs"]["region"] == ["reg002_cells"]


def test_dedupe_matches_akoya_convention(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": "X", "cyc001_ch002": "X"})

    assert list(_read_codex(tmp_path / "reg001.fcs").var.index) == dedupe_channel_names(["X", "X"])


def test_read_codex_fcs_blank_pns_falls_back_to_pnn(tmp_path: Path) -> None:
    write_fcs(tmp_path / "reg001.fcs", _cells(), {"cyc001_ch001": " ", "cyc001_ch002": "CD3"})

    assert list(_read_codex(tmp_path / "reg001.fcs").var.index) == ["cyc001_ch001", "CD3"]


@pytest.mark.parametrize("different_panels", [False, True])
def test_build_cells_overlapping_regions_raise(tmp_path: Path, different_panels: bool) -> None:
    write_fcs(tmp_path / "a.fcs", _cells(region=1))
    markers = ("cyc001_ch001",) if different_panels else ("cyc001_ch001", "cyc001_ch002")
    write_fcs(tmp_path / "b.fcs", _cells(region=1, markers=markers))
    tables = [_read_codex(tmp_path / name) for name in ("a.fcs", "b.fcs")]

    with pytest.raises(ValueError, match="Several cell tables"):
        build_cells(tables, {}, {})


def test_resolve_per_region(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.touch()
    b.touch()

    assert resolve_per_region(None, ["A"]) == {}
    assert resolve_per_region(str(a), ["A"]) == {"A": a}
    assert resolve_per_region(a, ["A", "B"]) == {"*": a}
    assert resolve_per_region({"B": str(b), "A": a}, ["A", "B"]) == {"A": a, "B": b}
    with pytest.raises(ValueError, match="'C'"):
        resolve_per_region({"C": a}, ["A", "B"])
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        resolve_per_region(tmp_path / "missing.csv", ["A"])


def test_read_labels_lazy_with_transformation(tmp_path: Path) -> None:
    mask = np.zeros((40, 60), dtype=np.uint32)
    mask[5:10, 5:10] = 1
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, mask)
    translation = Translation([3, 4], axes=("x", "y"))

    labels = read_labels(path, {"scale_factors": [2]}, {"global": translation})

    level0 = labels["scale0"]["image"]
    assert get_axes_names(labels) == ("y", "x")
    np.testing.assert_array_equal(level0.values, mask)
    assert get_transformation(labels, "global") == translation


def test_read_labels_first_channel(tmp_path: Path) -> None:
    masks = np.stack([np.full((16, 24), i + 1, dtype=np.int32) for i in range(2)])
    path = tmp_path / "mask.ome.tif"
    tifffile.imwrite(path, masks, metadata={"axes": "CYX", "Channel": {"Name": ["cells", "nuclei"]}})

    labels = read_labels(path, {"scale_factors": None}, {"global": Identity()})

    np.testing.assert_array_equal(labels.values, masks[0])


def test_read_labels_errors(tmp_path: Path) -> None:
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="integer"):
        read_labels(path, {}, {"global": Identity()})


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_labels_pyramidal_ome_is_lazy(tmp_path: Path, compression: str | None) -> None:
    mask = np.repeat(np.arange(64, dtype=np.uint16), 48).reshape(64, 48)
    path = tmp_path / "mask.ome.tif"
    tile = (16, 16) if compression else None
    with tifffile.TiffWriter(path) as tw:
        tw.write(mask, subifds=1, tile=tile, compression=compression, metadata={"axes": "YX"})
        tw.write(mask[::2, ::2], subfiletype=1, tile=tile, compression=compression)

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        labels = read_labels(path, {"scale_factors": None}, {"global": Identity()})

    assert labels.shape == (64, 48)
    np.testing.assert_array_equal(labels.values, mask)


@pytest.mark.parametrize(("mask_shape", "factor"), [((32, 40), 2), ((22, 27), 3)])
def test_mask_integer_downsample_autoscaled(tmp_path: Path, mask_shape: tuple[int, int], factor: int) -> None:
    # a mask downsampled by an integer factor (exactly, or with its shape rounded up) is scaled onto the image,
    # before the stage translation of its region; the full-resolution mask of the other region is unaffected
    image_shape = (64, 80)
    write_qptiff(tmp_path / "A_Scan1.qptiff", biomarkers=["DAPI"], shape=image_shape, stage_position_cm=(1.0, 2.0))
    write_qptiff(tmp_path / "B_Scan1.qptiff", biomarkers=["DAPI"], shape=image_shape, stage_position_cm=(1.01, 2.0))
    tifffile.imwrite(tmp_path / "A_mask.tif", np.zeros(image_shape, dtype=np.uint32))
    mask = np.zeros(mask_shape, dtype=np.uint32)
    mask[2:6, 2:6], mask[10:14, 8:12] = 1, 2
    tifffile.imwrite(tmp_path / "B_mask.tif", mask)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(
            tmp_path,
            labels={"A": tmp_path / "A_mask.tif", "B": tmp_path / "B_mask.tif"},
            image_models_kwargs={"scale_factors": [2]},
        )

    assert sdata.coordinate_systems == ["global"]
    assert get_transformation(sdata.labels["A_labels"], get_all=True) == get_transformation(
        sdata.images["A_image"], get_all=True
    )
    image_transformation = get_transformation(sdata.images["B_image"], "global")
    assert isinstance(image_transformation, Translation)
    labels_transformation = get_transformation(sdata.labels["B_labels"], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations == [Scale([factor, factor], axes=("y", "x")), image_transformation]
    image_extent = get_extent(sdata.images["B_image"], coordinate_system="global")
    labels_extent = get_extent(sdata.labels["B_labels"], coordinate_system="global")
    for axis, image_size, mask_size in zip(("y", "x"), image_shape, mask_shape, strict=True):
        assert labels_extent[axis][0] == pytest.approx(image_extent[axis][0])
        # a mask rounded up overhangs the image by less than one mask pixel
        expected_end = image_extent[axis][0] + mask_size * factor
        assert labels_extent[axis][1] == pytest.approx(expected_end)
        if image_size % mask_size == 0:
            assert labels_extent[axis][1] == pytest.approx(image_extent[axis][1])
    # writing may reorder the axes of the translation, so compare the composed affine maps
    sdata.write(tmp_path / "x.zarr")
    read = get_transformation(read_zarr(tmp_path / "x.zarr").labels["B_labels"], "global")
    assert isinstance(read, Sequence)
    np.testing.assert_allclose(
        read.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        labels_transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )


def test_mask_non_integer_mismatch_warns(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"], shape=(64, 80))
    tifffile.imwrite(tmp_path / "mask.tif", np.zeros((48, 60), dtype=np.uint32))

    with pytest.warns(UserWarning, match=r"\(48, 60\).*\(64, 80\).*not align"):
        sdata = phenocycler(
            tmp_path,
            labels=tmp_path / "mask.tif",
            image_models_kwargs={"scale_factors": [2]},
        )

    assert get_transformation(sdata.labels["S_labels"], "global") == Identity()
