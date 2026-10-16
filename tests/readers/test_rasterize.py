from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pytest
import shapely
import tifffile
from skimage.draw import polygon as draw_polygon
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation
from tests._akoya_data import write_qptiff

from spatialdata_io import phenocycler
from spatialdata_io.readers._utils import _rasterize
from spatialdata_io.readers._utils._rasterize import rasterize_polygons_lazy

if TYPE_CHECKING:
    from pathlib import Path


def _eager_reference(polygons: list[shapely.Polygon], ids: list[int], shape: tuple[int, int]) -> np.ndarray:
    """Draw on the full canvas at once, in ascending id order; pixel (r, c) is covered when (c + .5, r + .5) is."""
    labels = np.zeros(shape, dtype=np.uint32)
    for i in np.argsort(ids, kind="stable"):
        inside = np.zeros(shape, dtype=bool)
        rings = [(polygons[i].exterior, True), *((ring, False) for ring in polygons[i].interiors)]
        for ring, fill in rings:
            xy = shapely.get_coordinates(ring)
            inside[draw_polygon(xy[:, 1] - 0.5, xy[:, 0] - 0.5, shape=shape)] = fill
        labels[inside] = ids[i]
    return labels


def test_rasterize_matches_eager_reference() -> None:
    polygons = [
        shapely.Polygon([(2.3, 3.1), (9.7, 2.2), (8.4, 11.6), (1.2, 9.9)]),
        # crosses the chunk borders at x = 16 and y = 16
        shapely.Polygon([(12.2, 10.4), (27.6, 13.3), (24.1, 30.8), (11.7, 25.2)]),
        # has a hole
        shapely.Polygon(
            [(30.1, 20.2), (48.3, 21.4), (47.6, 38.7), (31.4, 37.9)],
            holes=[[(35.2, 25.3), (42.4, 25.1), (41.7, 32.6), (36.1, 33.3)]],
        ),
    ]
    ids = [3, 1, 2]

    labels = rasterize_polygons_lazy(polygons, ids, shape=(40, 50), chunks=(16, 16))

    assert labels.dtype == np.uint32
    assert labels.chunks == ((16, 16, 8), (16, 16, 16, 2))
    result = labels.compute()
    np.testing.assert_array_equal(result, _eager_reference(polygons, ids, (40, 50)))
    assert set(np.unique(result)) == {0, 1, 2, 3}
    assert result[29, 38] == 0  # in the hole


def test_rasterize_pixel_convention_and_scale() -> None:
    square = shapely.box(1, 2, 3, 5)  # covers the pixels x 1..2, y 2..4

    expected = np.zeros((8, 8), dtype=np.uint32)
    expected[2:5, 1:3] = 7
    np.testing.assert_array_equal(rasterize_polygons_lazy([square], [7], shape=(8, 8)).compute(), expected)
    # coordinates in µm with 0.5 µm pixels
    half = shapely.box(0.5, 1, 1.5, 2.5)
    np.testing.assert_array_equal(rasterize_polygons_lazy([half], [7], shape=(8, 8), scale=2).compute(), expected)


def test_rasterize_overlap_higher_id_wins() -> None:
    left, right = shapely.box(0, 0, 6, 6), shapely.box(3, 3, 9, 9)
    # a cell inside the hole of a higher-id ring keeps its pixels
    ring = shapely.Polygon(shapely.box(10, 0, 20, 10).exterior, holes=[shapely.box(13, 3, 17, 7).exterior.coords])
    inner = shapely.box(14, 4, 16, 6)

    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        polygons = [[left, right, ring, inner][i] for i in order]
        ids = [[5, 2, 9, 4][i] for i in order]
        labels = rasterize_polygons_lazy(polygons, ids, shape=(10, 20), chunks=(4, 4)).compute()
        assert labels[4, 4] == 5
        assert labels[1, 1] == 5
        assert labels[8, 8] == 2
        assert labels[5, 15] == 4
        assert labels[3, 13] == 0
        assert labels[1, 11] == 9


def test_rasterize_multipolygon_and_empty() -> None:
    multi = shapely.MultiPolygon([shapely.box(0, 0, 2, 2), shapely.box(5, 5, 7, 7)])

    labels = rasterize_polygons_lazy([multi], [1], shape=(8, 8), chunks=(4, 4)).compute()

    assert labels[0, 0] == labels[6, 6] == 1
    assert labels.sum() == 8
    assert not rasterize_polygons_lazy([], [], shape=(5, 6)).compute().any()


def test_rasterize_errors() -> None:
    with pytest.raises(ValueError, match="2 polygons but 1 ids"):
        rasterize_polygons_lazy([shapely.box(0, 0, 1, 1)] * 2, [1], shape=(4, 4))
    with pytest.raises(ValueError, match="between 1 and"):
        rasterize_polygons_lazy([shapely.box(0, 0, 1, 1)], [0], shape=(4, 4))


def test_rasterize_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def counting_polygon(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        calls.append(1)
        return draw_polygon(*args, **kwargs)

    monkeypatch.setattr(_rasterize, "draw_polygon", counting_polygon)
    polygons = [shapely.box(i, i, i + 3, i + 3) for i in range(0, 60, 6)]

    labels = rasterize_polygons_lazy(polygons, list(range(1, 11)), shape=(64, 64), chunks=(16, 16))

    assert not calls
    labels.blocks[0, 0].compute()
    # only the polygons of the chunk are drawn
    assert len(calls) == 3
    assert labels.compute().max() == 10


def _write_qupath_cells(path: Path, object_ids: list[str]) -> None:
    # centroids in the 8 px squares `_cell(..., 20 * i, 20 * i)`
    header = "Image\tObject ID\tClass\tCentroid X px\tCentroid Y px\tCD3: Cell: Mean"
    rows = [f"S_Scan1.qptiff\t{oid}\tTumor\t{4 + 20 * i}\t{4 + 20 * i}\t{i}" for i, oid in enumerate(object_ids)]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _write_geojson(path: Path, features: list[dict], collection: bool = True) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features} if collection else features))


def _cell(object_id: str | None, x: float, y: float, object_type: str = "cell") -> dict:
    feature = {
        "type": "Feature",
        "geometry": shapely.geometry.mapping(shapely.box(x, y, x + 8, y + 8)),
        "properties": {"objectType": object_type, "classification": {"name": "Tumor"}},
    }
    if object_id is not None:
        feature["id"] = object_id
    return feature


_FAST = {"scale_factors": [2]}


def test_qupath_geojson_to_labels(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"], stage_position_cm=(1.0, 2.0))
    uuids = ["5b1f-aa", "5b1f-bb", "5b1f-cc"]
    _write_qupath_cells(tmp_path / "cells.tsv", uuids)
    # in any order; the annotation has an id the table does not know
    features = [_cell(uuids[2], 40, 40), _cell(uuids[0], 0, 0), _cell("tissue", 0, 0, "annotation")]
    features.append(_cell(uuids[1], 20, 20))
    _write_geojson(tmp_path / "cells.geojson", features)
    _write_geojson(tmp_path / "S-tissue-boundary.geojson", [_cell(None, 0, 0, "annotation")])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    assert list(sdata.labels) == ["S_labels"]
    assert list(sdata.shapes) == ["S_polygons"]
    labels = sdata.labels["S_labels"]["scale0"]["image"]
    assert isinstance(labels.data, da.Array)
    assert labels.shape == (256, 320)
    assert get_transformation(sdata.labels["S_labels"]) == get_transformation(sdata.images["S_image"])
    table = sdata.tables["table"]
    assert table.uns["spatialdata_attrs"]["region"] == ["S_labels"]
    # the UUIDs are not integers, so the instance ids are the row numbers; the mask values follow them
    assert list(table.obs["cell_id"]) == [1, 2, 3]
    assert list(table.obs["Object_ID"]) == uuids
    values = labels.values
    assert values[4, 4] == 1
    assert values[24, 24] == 2
    assert values[44, 44] == 3
    assert set(np.unique(values)) == {0, 1, 2, 3}
    assert sdata.attrs["akoya"]["geojson"] == {"S": str(tmp_path / "cells.geojson")}


def test_qupath_geojson_keeps_polygons(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_qupath_cells(tmp_path / "cells.tsv", ["a", "b", "c"])
    # cell "b" has no polygon: it is missing from the shapes and the labels
    _write_geojson(tmp_path / "cells.geojson", [_cell("c", 40, 40), _cell("a", 0, 0)])

    with pytest.warns(UserWarning, match="have no polygon"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    assert list(sdata.shapes) == ["S_polygons"]
    polygons = sdata.shapes["S_polygons"]
    assert list(polygons.index) == [1, 3]
    labels = sdata.labels["S_labels"]["scale0"]["image"].values
    assert sorted(set(np.unique(labels)) - {0}) == list(polygons.index)
    assert polygons.geometry.loc[3].equals(shapely.box(40, 40, 48, 48))
    assert get_transformation(polygons, get_all=True) == get_transformation(sdata.images["S_image"], get_all=True)
    assert sdata.tables["table"].uns["spatialdata_attrs"]["region"] == ["S_labels"]

    sdata.write(tmp_path / "out.zarr")
    reread = read_zarr(tmp_path / "out.zarr")
    assert reread.shapes["S_polygons"].geometry.equals(polygons.geometry)


@pytest.mark.parametrize("collection", [True, False])
def test_qupath_geojson_by_order_explicit_path_in_um(tmp_path: Path, collection: bool) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    header = "Image\tCentroid X µm\tCentroid Y µm\tCD3: Cell: Mean"
    rows = ["S_Scan1.qptiff\t4\t4\t1", "S_Scan1.qptiff\t24\t24\t2"]
    (tmp_path / "cells.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    # without ids, the cells and detections follow the row order of the table; annotations are skipped
    features = [_cell(None, 0, 0, "annotation"), _cell(None, 0, 0), _cell(None, 20, 20, "detection")]
    _write_geojson(tmp_path / "polygons.json", features, collection=collection)

    sdata = phenocycler(
        tmp_path,
        cells_table=tmp_path / "cells.tsv",
        image_models_kwargs=_FAST,
        cells_table_kwargs={"geojson": tmp_path / "polygons.json", "geojson_units": "um"},
    )

    values = sdata.labels["S_labels"]["scale0"]["image"].values
    # 8 µm squares at 0.5069 µm per pixel
    assert values[int(4 / 0.5069), int(4 / 0.5069)] == 1
    assert values[int(24 / 0.5069), int(24 / 0.5069)] == 2
    assert (values == 1).sum() == pytest.approx((8 / 0.5069) ** 2, rel=0.15)


def test_qupath_geojson_in_um_regions_with_different_pixel_sizes(tmp_path: Path) -> None:
    sizes = {"A": 0.5, "B": 0.25}
    for region, size in sizes.items():
        write_qptiff(tmp_path / f"{region}_Scan1.qptiff", biomarkers=["DAPI"], resolution_px_per_cm=1e4 / size)
    header = "Image\tCentroid X µm\tCentroid Y µm\tCD3: Cell: Mean"
    rows = ["A_Scan1.qptiff\t4\t4\t1", "B_Scan1.qptiff\t24\t24\t2"]
    (tmp_path / "cells.tsv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    _write_geojson(tmp_path / "polygons.json", [_cell(None, 0, 0), _cell(None, 20, 20)])

    sdata = phenocycler(
        tmp_path,
        cells_table=tmp_path / "cells.tsv",
        image_models_kwargs=_FAST,
        cells_table_kwargs={"geojson": tmp_path / "polygons.json", "geojson_units": "um"},
    )

    assert sdata.shapes["A_polygons"].geometry.iloc[0].equals(shapely.box(0, 0, 16, 16))
    assert sdata.shapes["B_polygons"].geometry.iloc[0].equals(shapely.box(80, 80, 112, 112))


def test_qupath_geojson_mask_wins(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_qupath_cells(tmp_path / "cells.tsv", ["a", "b"])
    _write_geojson(tmp_path / "cells_cells.geojson", [_cell("a", 0, 0), _cell("b", 20, 20)])
    mask = np.zeros((256, 320), dtype=np.uint32)
    mask[100:110, 100:110] = 1
    tifffile.imwrite(tmp_path / "mask.tif", mask)

    with pytest.warns(UserWarning, match="mask .* is used for the labels and the polygons are kept as 'S_polygons'"):
        sdata = phenocycler(
            tmp_path, cells_table=tmp_path / "cells.tsv", labels=tmp_path / "mask.tif", image_models_kwargs=_FAST
        )

    np.testing.assert_array_equal(sdata.labels["S_labels"]["scale0"]["image"].values, mask)
    assert "S_polygons" in sdata.shapes


def test_qupath_geojson_errors(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_qupath_cells(tmp_path / "cells.tsv", ["a", "b"])
    _write_geojson(tmp_path / "other.geojson", [_cell("x", 0, 0), _cell("y", 20, 20)])

    with pytest.raises(ValueError, match="none of the feature ids"):
        phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.tsv",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"geojson": tmp_path / "other.geojson"},
        )
    with pytest.raises(ValueError, match="geojson_units"):
        phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.tsv",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"geojson": tmp_path / "other.geojson", "geojson_units": "mm"},
        )
    with pytest.raises(FileNotFoundError, match="missing.geojson"):
        phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.tsv",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"geojson": tmp_path / "missing.geojson"},
        )
    (tmp_path / "cells.csv").write_text("CellID,DAPI,X_centroid,Y_centroid\n1,2,3,4\n")
    with pytest.raises(ValueError, match="QuPath"):
        phenocycler(
            tmp_path,
            cells_table=tmp_path / "cells.csv",
            image_models_kwargs=_FAST,
            cells_table_kwargs={"geojson": tmp_path / "other.geojson"},
        )

    # without ids, the number of cells must match
    (tmp_path / "noid.tsv").write_text("Image\tCentroid X px\tCentroid Y px\nS_Scan1.qptiff\t1\t1\n", encoding="utf-8")
    _write_geojson(tmp_path / "noid.geojson", [_cell(None, 0, 0), _cell(None, 20, 20)])
    with pytest.raises(ValueError, match="2 cell polygons .* 1 rows"), pytest.warns(UserWarning, match="marker"):
        phenocycler(tmp_path, cells_table=tmp_path / "noid.tsv", image_models_kwargs=_FAST)


def _write_order_table(path: Path) -> None:
    header = "Image\tCentroid X px\tCentroid Y px\tCD3: Cell: Mean"
    rows = ["S_Scan1.qptiff\t4\t4\t1", "S_Scan1.qptiff\t24\t24\t2"]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("features", "match"),
    [
        # without ids, polygons in another order than the rows
        ([_cell(None, 20, 20), _cell(None, 0, 0)], "2 of the 2 polygons .* are more than 1 px"),
        # ids, but no `Object ID` in the table: matched in order too
        ([_cell("u2", 20, 20), _cell("u1", 0, 0)], "another order"),
        # coordinates in µm read as pixels
        ([_cell(None, 0, 0), _cell(None, 10, 10)], "geojson_units"),
    ],
)
def test_qupath_geojson_polygons_far_from_centroids_raise(tmp_path: Path, features: list[dict], match: str) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_order_table(tmp_path / "cells.tsv")
    _write_geojson(tmp_path / "cells.geojson", features)

    with pytest.raises(ValueError, match=match):
        phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)


def test_qupath_geojson_partial_and_duplicate_ids(tmp_path: Path) -> None:
    write_qptiff(tmp_path / "S_Scan1.qptiff", biomarkers=["DAPI"])
    _write_qupath_cells(tmp_path / "cells.tsv", ["a", "b", "c"])
    _write_geojson(tmp_path / "cells.geojson", [_cell("a", 0, 0), _cell("c", 40, 40)])

    with pytest.warns(UserWarning, match="1 of the 3 cells .* have no polygon"):
        sdata = phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)

    assert set(np.unique(sdata.labels["S_labels"]["scale0"]["image"].values)) == {0, 1, 3}
    assert sdata.tables["table"].n_obs == 3

    _write_geojson(tmp_path / "cells.geojson", [_cell("a", 0, 0), _cell("b", 20, 20), _cell("a", 40, 40)])
    with pytest.raises(ValueError, match="same id"):
        phenocycler(tmp_path, cells_table=tmp_path / "cells.tsv", image_models_kwargs=_FAST)
