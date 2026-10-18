from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import get_extent, read_zarr
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation
from tests._akoya_data import forbid_pixel_reads, write_fcs, write_qptiff

from spatialdata_io import phenocycler
from spatialdata_io.readers._utils._cell_tables import read_cell_table, read_labels

if TYPE_CHECKING:
    from pathlib import Path


def _cells(columns: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    meta = {"cell_id": [1.0, 2.0], "region": 1.0, "x": [1.0, 2.0], "y": [3.0, 4.0], "size": [3.0, 12.0]}
    return pd.DataFrame(meta | {column: rng.uniform(0, 1000, 2) for column in columns})


@pytest.mark.parametrize(
    ("pns", "names"),
    [
        ({"cyc001_ch001": "DAPI", "cyc001_ch002": "CD3", "cyc002_ch001": "DAPI"}, ["DAPI", "CD3", "DAPI (2)"]),
        ({"cyc001_ch001": " ", "cyc001_ch002": "CD3"}, ["cyc001_ch001", "CD3", "cyc002_ch001"]),
        (None, ["cyc001_ch001", "cyc001_ch002", "cyc002_ch001"]),
    ],
    ids=["pns", "blank_pns", "no_pns"],
)
def test_fcs_marker_names(tmp_path: Path, pns: dict[str, str] | None, names: list[str]) -> None:
    columns = ("cyc001_ch001", "cyc001_ch002", "cyc002_ch001")
    cells = _cells(columns)
    write_fcs(tmp_path / "reg001.fcs", cells, pns)

    table = read_cell_table(tmp_path / "reg001.fcs", default_region=None)

    assert list(table.var.index) == names
    assert list(table.var["cycle"]) == [1, 1, 2]
    assert list(table.var["source_column"]) == list(columns)
    np.testing.assert_allclose(table.X, cells[list(columns)], rtol=1e-6)
    assert list(table.region) == ["reg001", "reg001"]
    assert list(table.instance_id) == [1, 2]
    np.testing.assert_allclose(table.centroids_px, cells[["x", "y"]])
    assert table.radius_px is not None
    np.testing.assert_allclose(table.radius_px, np.sqrt(cells["size"] / np.pi))


def test_csv_header_names_the_markers(tmp_path: Path) -> None:
    path = tmp_path / "reg002.csv"
    path.write_text(
        "cell_id:cell_id,region:region,x:x,y:y,size:size,cyc001_ch001:DAPI1,cyc001_ch002:CD3\n1,2,10,20,30,1,7\n"
    )

    table = read_cell_table(path, default_region=None)

    assert list(table.var.index) == ["DAPI1", "CD3"]
    assert list(table.var["source_column"]) == ["cyc001_ch001:DAPI1", "cyc001_ch002:CD3"]
    # the instance ids and regions are stored under the keys of the table element
    assert list(table.obs.columns) == ["x", "y", "size"]
    assert list(table.region) == ["reg002"]


@pytest.mark.parametrize("mangled", [False, True])
def test_csv_without_cycle_columns_reads_numeric_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mangled: bool
) -> None:
    if mangled:  # pandas >= 3.1 renames repeated names (CD3.1) with the pyarrow engine too
        read_csv = pd.read_csv

        def mangling_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
            df = read_csv(*args, **kwargs)
            df.columns = pd.Index([*df.columns[:-2], "CD3.1", df.columns[-1]])
            return df

        monkeypatch.setattr(pd, "read_csv", mangling_read_csv)
    path = tmp_path / "cells.csv"
    path.write_text("cell_id,x,y,tile_num,CD3,CD8,CD3,label\n1,1.0,2.0,1,5.0,6.0,7.0,T\n")

    with pytest.warns(UserWarning, match=r"no 'cycCCC_chHHH' columns.*\['CD3', 'CD8', 'CD3 \(2\)'\]"):
        table = read_cell_table(path, default_region="Kidney")

    assert list(table.var.index) == ["CD3", "CD8", "CD3 (2)"]
    assert list(table.obs.columns) == ["x", "y", "tile_num", "label"]
    assert list(table.region) == ["Kidney"]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("cell_id,cyc001_ch001\n1,2.0\n", "'x' and 'y'"),
        ("cell_id,x,y,cyc001_ch001\n1,1.0,2.0,2.0\n", "cannot be assigned to regions"),
        ("cell_id,region,x,y,cyc001_ch001\n1,1,1,1,1\n1.5,1,2,2,1\n", r"1 cell\(s\) whose instance id in 'cell_id'"),
        (
            "cell_id,region,x,y,cyc001_ch001\n1,1,1,1,1\n1,1,2,2,1\n",
            r"1 duplicate instance id\(s\) \[1\] in region 'reg001'",
        ),
    ],
    ids=["no_centroids", "no_region", "fractional_id", "duplicate_id"],
)
def test_bad_tables_raise(tmp_path: Path, content: str, match: str) -> None:
    (tmp_path / "cells.csv").write_text(content)

    with pytest.raises(ValueError, match=match):
        read_cell_table(tmp_path / "cells.csv", default_region=None)


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_labels_is_lazy(tmp_path: Path, compression: str | None) -> None:
    mask = np.repeat(np.arange(64, dtype=np.uint16), 48).reshape(64, 48)
    path = tmp_path / "mask.ome.tif"
    tile = (16, 16) if compression else None
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.stack([mask, mask + 1]), subifds=1, tile=tile, compression=compression, metadata={"axes": "CYX"})
        tw.write(np.stack([mask, mask + 1])[:, ::2, ::2], subfiletype=1, tile=tile, compression=compression)
    translation = Translation([3, 4], axes=("x", "y"))

    with pytest.MonkeyPatch.context() as monkeypatch:
        forbid_pixel_reads(monkeypatch)
        labels = read_labels(path, {"scale_factors": None}, {"global": translation})

    # of several planes, the first
    np.testing.assert_array_equal(labels.values, mask)
    assert get_transformation(labels, "global") == translation
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="integer"):
        read_labels(path, {}, {"global": Identity()})


@pytest.mark.parametrize(("mask_shape", "factor"), [((32, 40), 2), ((22, 27), 3)])
def test_downsampled_mask_is_scaled_onto_its_image(tmp_path: Path, mask_shape: tuple[int, int], factor: int) -> None:
    # exactly, or with its shape rounded up; before the stage translation of its region
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"], shape=(64, 80), stage_position_cm=(1.0, 2.0))
    tifffile.imwrite(tmp_path / "mask.tif", np.zeros(mask_shape, dtype=np.uint32))

    sdata = phenocycler(tmp_path, labels=tmp_path / "mask.tif", image_models_kwargs={"scale_factors": [2]})

    image_transformation = get_transformation(sdata["S_image"], "global")
    labels_transformation = get_transformation(sdata["S_labels"], "global")
    assert isinstance(labels_transformation, Sequence)
    assert labels_transformation.transformations == [Scale([factor, factor], axes=("y", "x")), image_transformation]
    image_extent = get_extent(sdata["S_image"], coordinate_system="global")
    labels_extent = get_extent(sdata["S_labels"], coordinate_system="global")
    for axis, mask_size in zip(("y", "x"), mask_shape, strict=True):
        assert labels_extent[axis][0] == pytest.approx(image_extent[axis][0])
        # a mask rounded up overhangs the image by less than one mask pixel
        assert labels_extent[axis][1] == pytest.approx(image_extent[axis][0] + mask_size * factor)

    # writing may reorder the axes of the translation, so compare the composed affine maps
    sdata.write(tmp_path / "x.zarr")
    read = get_transformation(read_zarr(tmp_path / "x.zarr")["S_labels"], "global")
    np.testing.assert_allclose(
        read.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        labels_transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
