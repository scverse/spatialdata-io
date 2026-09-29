import dataclasses
import math
import os
import shutil
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely
import zarr
from click.testing import CliRunner
from spatialdata import get_extent, match_element_to_table, match_table_to_element, read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, Scale, Sequence, Translation, get_transformation
from xarray import DataTree

from spatialdata_io.__main__ import pyxa_wrapper
from spatialdata_io._constants._constants import PyxaKeys
from spatialdata_io.readers import pyxa as pyxa_module
from spatialdata_io.readers.pyxa import (
    _get_footprints,
    _get_image,
    _get_labels,
    _get_points,
    _get_shapes,
    _get_table,
    _get_voxel_size,
    _label_ids,
    _labels_level,
    _make_polygonal_valid,
    _mosaic_grid,
    _MosaicGrid,
    _plan_tiles,
    _rasterize_tile,
    _read_rings,
    _Rings,
    _validate_columns,
    pyxa,
)

# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
DATASETS = ["pyxa_xsmall"]
FIXTURE_DIR = Path("./data") / DATASETS[0]
MOSAIC_DIR = FIXTURE_DIR / "mosaic_3d.ome.zarr"


def _ensure_pyxa_xsmall_fixture() -> None:
    """Download the xsmall Pyxa fixture if the CI test-data artifact doesn't have it yet.

    Mirrors the `pyxa_xsmall` step of `.github/workflows/prepare_test_data.yaml`. Downloads and
    extracts into a uniquely named temp directory, then swaps it into place, so this is safe when
    several `pytest -n auto` workers import this module at once. Remove once the shared CI artifact
    includes `pyxa_xsmall`.
    """
    if FIXTURE_DIR.exists():
        return
    base_url = "https://huggingface.co/datasets/Stellaromics/demo/resolve/main/xsmall/"
    files = [
        "cell_assigned_gene_v1.csv",
        "cell_by_gene_v1.csv",
        "cell_metadata_v1.csv",
        "segmentation_geometries_v1.parquet",
        "mosaic_3d.ome.zarr.zip",
    ]
    tmp_dir = FIXTURE_DIR.parent / f".{FIXTURE_DIR.name}.tmp-{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir(parents=True)
        for name in files:
            with urllib.request.urlopen(base_url + name, timeout=60) as response, open(tmp_dir / name, "wb") as f:
                shutil.copyfileobj(response, f)
        zip_path = tmp_dir / "mosaic_3d.ome.zarr.zip"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        zip_path.unlink()
        os.replace(tmp_dir, FIXTURE_DIR)
    except OSError as err:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if FIXTURE_DIR.exists():
            return  # another worker downloaded it first
        pytest.skip(f"Could not download the pyxa_xsmall fixture: {err}", allow_module_level=True)


_ensure_pyxa_xsmall_fixture()


TINY_SCALE0 = np.arange(2 * 4 * 4, dtype="uint8").reshape(1, 1, 2, 4, 4)
# a constant rather than a downsampling of scale0, so a test can tell a loaded level from a recomputed one
TINY_SCALE1 = np.full((1, 1, 1, 2, 2), 7, dtype="uint8")


def _make_tiny_ome_zarr(path: Path) -> None:
    """Build a minimal two-level OME-NGFF v0.5 store, shapes (t=1, c=1, z=2, y=4, x=4) and (1, 1, 1, 2, 2)."""
    group = zarr.open_group(store=str(path), mode="w")
    for name, data in (("scale0/image", TINY_SCALE0), ("scale1/image", TINY_SCALE1)):
        array = group.create_array(name, shape=data.shape, dtype=data.dtype, dimension_names=["t", "c", "z", "y", "x"])
        array[:] = data
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "scale0/image",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.2, 0.2]},
                            {"type": "translation", "translation": [0.0, 0.0, 1.0, 2.0, 3.0]},
                        ],
                    },
                    {
                        "path": "scale1/image",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.4, 0.4]},
                            {"type": "translation", "translation": [0.0, 0.0, 1.25, 2.1, 3.1]},
                        ],
                    },
                ],
                "name": "image",
            }
        ],
        "omero": {"channels": [{"label": "DAPI"}]},
    }


def test_pyxa_keys_filenames() -> None:
    assert PyxaKeys.CELL_ASSIGNED_GENE_FILE == "cell_assigned_gene_v1.csv"
    assert PyxaKeys.CELL_BY_GENE_FILE == "cell_by_gene_v1.csv"
    assert PyxaKeys.CELL_METADATA_FILE == "cell_metadata_v1.csv"
    assert PyxaKeys.SEGMENTATION_GEOMETRIES_FILE == "segmentation_geometries_v1.parquet"
    assert PyxaKeys.PYXA_STUDIO_FILE == "pyxa_studio_v1.csv"


def test_pyxa_keys_columns() -> None:
    assert PyxaKeys.CELL_ID == "cell_id"
    assert PyxaKeys.GENE == "Gene"
    assert PyxaKeys.X_UM == "X_um"
    assert PyxaKeys.Y_UM == "Y_um"
    assert PyxaKeys.Z_UM == "Z_um"
    assert PyxaKeys.VOLUME_UM3 == "Volume_um3"
    assert PyxaKeys.ROI == "ROI"
    assert PyxaKeys.Z_INDEX == "ZIndex"
    assert PyxaKeys.BORDER == "Border"
    assert PyxaKeys.FOV == "FOV"
    assert PyxaKeys.UNASSIGNED_SUFFIX == "_-1"
    assert PyxaKeys.REGION_KEY == "region"
    assert PyxaKeys.REGION == "cell_boundaries"
    assert PyxaKeys.CELL_BOUNDARIES_Z == "cell_boundaries_z"
    assert PyxaKeys.INSTANCE_KEY == "cell_id"
    assert PyxaKeys.ASSIGNED == "assigned"


def test_validate_columns_passes_when_present() -> None:
    df = pd.DataFrame({"cell_id": [1], "Gene": ["A"]})
    _validate_columns(df, {"cell_id", "Gene"}, "test_file.csv")


def test_validate_columns_raises_when_missing() -> None:
    df = pd.DataFrame({"cell_id": [1]})
    with pytest.raises(ValueError, match=r"test_file\.csv is missing required column\(s\): \['Gene'\]"):
        _validate_columns(df, {"cell_id", "Gene"}, "test_file.csv")


def test_get_points_keeps_unassigned_transcripts() -> None:
    points = _get_points(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    assert isinstance(points, dd.DataFrame)
    computed = points.compute()
    assert "assigned" in computed.columns
    assert (~computed["assigned"]).sum() > 0
    assert computed[~computed["assigned"]]["cell_id"].str.endswith("_-1").all()
    assert computed["assigned"].sum() > 0


def test_get_points_has_required_coordinate_columns() -> None:
    points = _get_points(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    computed = points.compute()
    for col in ("X_um", "Y_um", "Z_um", "Gene", "cell_id"):
        assert col in computed.columns


def test_pyxa_reader_gene_categories_are_known(caplog: pytest.LogCaptureFixture) -> None:
    points = pyxa(FIXTURE_DIR)["transcripts"]
    # PointsModel.parse warns (and computes them itself) when the feature categories are unknown
    assert "unknown categories" not in caplog.text
    assert points["Gene"].cat.known
    raw = pd.read_csv(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    assert set(points["Gene"].cat.categories) == set(raw["Gene"])


def test_get_table_matches_raw_values() -> None:
    adata = _get_table(
        FIXTURE_DIR / "cell_by_gene_v1.csv",
        FIXTURE_DIR / "cell_metadata_v1.csv",
    )
    raw_by_gene = pd.read_csv(FIXTURE_DIR / "cell_by_gene_v1.csv", index_col="cell_id")
    raw_metadata = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", index_col="cell_id")

    assert adata.n_obs == len(raw_by_gene)
    sample_cell = raw_by_gene.index[0]
    sample_gene = raw_by_gene.columns[0]
    assert adata[sample_cell, sample_gene].to_df().iloc[0, 0] == raw_by_gene.loc[sample_cell, sample_gene]

    assert list(adata.obsm["spatial"][0]) == list(raw_metadata.loc[sample_cell, ["X_um", "Y_um", "Z_um"]])
    assert (adata.obs["region"] == "cell_boundaries").all()
    # an index named like the cell_id column breaks spatialdata's table joins
    assert adata.obs.index.name is None


def test_get_shapes_matches_raw_row_count() -> None:
    gdf = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size=0.114984751, z_size=0.5)
    raw = gpd.read_parquet(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    assert len(gdf) == len(raw)
    assert all(isinstance(c, str) for c in gdf["cell_id"])
    assert gdf.geometry.is_valid.all()
    assert set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}
    assert gdf.index.is_unique


def test_get_footprints_is_union_of_planes() -> None:
    planes = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size=0.114984751, z_size=0.5)
    footprints = _get_footprints(planes)
    assert footprints.index.name == "cell_id"
    assert footprints.index.is_unique
    assert set(footprints.index) == set(planes["cell_id"])
    assert footprints.geometry.is_valid.all()
    # every z-plane polygon lies inside its cell's footprint
    covered = footprints.loc[planes["cell_id"]].geometry.buffer(1e-9).covers(planes.geometry, align=False)
    assert covered.all()


def test_get_shapes_converts_to_um() -> None:
    gdf = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size=0.114984751, z_size=0.5)
    raw = gpd.read_parquet(FIXTURE_DIR / "segmentation_geometries_v1.parquet")
    np.testing.assert_allclose(gdf.total_bounds, raw.total_bounds * 0.114984751)
    np.testing.assert_allclose(gdf["Z_um"], (raw["ZIndex"] + 0.5) * 0.5)


def test_make_polygonal_valid_fixes_self_intersection() -> None:
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    square = shapely.box(0, 0, 1, 1)
    fixed = _make_polygonal_valid(np.array([bowtie, square]))
    assert shapely.is_valid(fixed).all()
    assert set(shapely.get_type_id(fixed)) <= {shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON}
    assert shapely.area(fixed[0]) == pytest.approx(2.0)
    assert fixed[1] is square  # valid geometries are passed through untouched


def test_make_polygonal_valid_drops_non_polygonal_parts() -> None:
    # a polygon with a zero-width spike: make_valid returns the square plus a dangling line
    spiky = shapely.Polygon([(0, 0), (1, 0), (1, 1), (1, 2), (1, 1), (0, 1)])
    (fixed,) = _make_polygonal_valid(np.array([spiky]))
    assert fixed.is_valid
    assert fixed.geom_type in {"Polygon", "MultiPolygon"}
    assert fixed.area == pytest.approx(1.0)


def test_get_voxel_size() -> None:
    xy, z = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    assert xy == pytest.approx(0.114984751)
    assert z == pytest.approx(0.5)


def _write_metadata(path: Path, n: int, xy: float, z: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    px = pd.DataFrame(rng.uniform(1, 1000, size=(n, 3)), columns=["X_pixels", "Y_pixels", "Z_pixels"])
    metadata = px.assign(X_um=px["X_pixels"] * xy, Y_um=px["Y_pixels"] * xy, Z_um=px["Z_pixels"] * z)
    metadata.to_csv(path, index=False)
    return metadata


def test_get_voxel_size_reads_only_a_sample(tmp_path: Path) -> None:
    path = tmp_path / "cell_metadata_v1.csv"
    metadata = _write_metadata(path, n=5_000, xy=0.25, z=1.5)
    # corrupt everything past the fitted sample: the fit must not read these rows
    metadata.loc[2_000:, ["X_um", "Y_um", "Z_um"]] = 1e6
    metadata.to_csv(path, index=False)
    assert _get_voxel_size(path, n_rows=1_000) == pytest.approx((0.25, 1.5))


def test_get_voxel_size_raises_when_not_a_pure_scale(tmp_path: Path) -> None:
    path = tmp_path / "cell_metadata_v1.csv"
    metadata = _write_metadata(path, n=100, xy=0.25, z=1.5)
    metadata["X_um"] += 10.0  # an offset between pixel and um coordinates
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not related by a pure scale"):
        _get_voxel_size(path)


def _area_weighted_centroids(shapes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-cell centroid of the polygon stack, weighting each z-plane polygon by its area."""
    centroids = shapes.geometry.centroid
    weighted = (
        pd.DataFrame(
            {
                "cell_id": shapes["cell_id"].to_numpy(),
                "area": shapes.geometry.area.to_numpy(),
                "x": (centroids.x * shapes.geometry.area).to_numpy(),
                "y": (centroids.y * shapes.geometry.area).to_numpy(),
                "z": (shapes["Z_um"] * shapes.geometry.area).to_numpy(),
            }
        )
        .groupby("cell_id")[["area", "x", "y", "z"]]
        .sum()
    )
    return weighted[["x", "y", "z"]].div(weighted["area"], axis=0)


def test_pyxa_reader_shapes_in_um_with_identity_transform() -> None:
    sdata = pyxa(FIXTURE_DIR)
    for name in ("cell_boundaries", "cell_boundaries_z"):
        assert isinstance(get_transformation(sdata[name], to_coordinate_system="global"), Identity)
        assert sdata[name].geometry.is_valid.all()


def test_pyxa_reader_shapes_aligned_with_cell_metadata() -> None:
    shapes = pyxa(FIXTURE_DIR)["cell_boundaries_z"]
    metadata = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", index_col="cell_id")
    # the per-cell metadata centroid is exactly the area-weighted centroid of the cell's polygon stack
    centroids = _area_weighted_centroids(shapes)
    expected = metadata.loc[centroids.index, ["X_um", "Y_um", "Z_um"]].to_numpy()
    np.testing.assert_allclose(centroids.to_numpy(), expected, atol=1e-6)


def test_pyxa_reader_builds_valid_sdata() -> None:
    sdata = pyxa(FIXTURE_DIR)

    assert "transcripts" in sdata.points
    assert "cell_boundaries" in sdata.shapes
    assert "cell_boundaries_z" in sdata.shapes
    assert "rna" in sdata.tables

    raw_transcripts = pd.read_csv(FIXTURE_DIR / "cell_assigned_gene_v1.csv")
    extent = get_extent(sdata["transcripts"])
    assert math.floor(extent["x"][0]) <= math.floor(raw_transcripts["X_um"].min())
    assert math.ceil(extent["x"][1]) >= math.ceil(raw_transcripts["X_um"].max())


def test_pyxa_reader_missing_file_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            pyxa(Path(tmpdir))


def test_get_image_loads_all_scales() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "tiny.ome.zarr"
        _make_tiny_ome_zarr(zarr_path)

        image = _get_image(zarr_path)
        assert isinstance(image, DataTree)
        assert list(image.keys()) == ["scale0", "scale1"]
        scale0, scale1 = image["scale0"]["image"], image["scale1"]["image"]
        assert scale0.dims == ("c", "z", "y", "x")
        assert list(scale0.coords["c"].values) == ["DAPI"]
        # both levels are read from the store as written, not recomputed from scale0
        np.testing.assert_array_equal(scale0.values, TINY_SCALE0[0])
        np.testing.assert_array_equal(scale1.values, TINY_SCALE1[0])

        expected = Sequence(
            [Scale([0.5, 0.2, 0.2], axes=("z", "y", "x")), Translation([1.0, 2.0, 3.0], axes=("z", "y", "x"))]
        )
        affine = get_transformation(image, to_coordinate_system="global").to_affine_matrix(
            ("z", "y", "x"), ("z", "y", "x")
        )
        np.testing.assert_allclose(affine, expected.to_affine_matrix(("z", "y", "x"), ("z", "y", "x")))
        # coarser levels map to the same physical extent as scale0
        assert get_extent(image) == get_extent(scale0)


def test_pyxa_reader_includes_image_when_given() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "tiny.ome.zarr"
        _make_tiny_ome_zarr(zarr_path)

        sdata = pyxa(FIXTURE_DIR, image=zarr_path)
        assert "mosaic_image" in sdata.images
        assert sdata["mosaic_image"]["scale0"]["image"].shape == (1, 2, 4, 4)


def test_pyxa_reader_example_mosaic() -> None:
    sdata = pyxa(FIXTURE_DIR, image=MOSAIC_DIR)
    image = sdata["mosaic_image"]
    # all five precomputed pyramid levels are loaded
    assert [image[k]["image"].shape for k in image] == [
        (1, 200, 217, 218),
        (1, 100, 109, 109),
        (1, 50, 54, 54),
        (1, 25, 27, 27),
        (1, 12, 14, 13),
    ]
    assert list(image["scale0"]["image"].coords["c"].values) == ["DAPI"]
    # the mosaic is cropped to the same 100 um cube as the cells
    extent = get_extent(image)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert extent == {"z": (20, 121), "y": (-5140, -5039), "x": (900, 1001)}


def test_pyxa_reader_missing_image_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            pyxa(FIXTURE_DIR, image=Path(tmpdir) / "does_not_exist.ome.zarr")


def _zip_dir(src: Path, zip_path: Path) -> Path:
    """Zip ``src`` so the archive holds one top-level ``src.name/`` directory, as on the Hub."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent).as_posix())
    return zip_path


def test_get_image_reads_zip_in_place(tmp_path: Path) -> None:
    zipped = _zip_dir(MOSAIC_DIR, tmp_path / "mosaic_3d.ome.zarr.zip")
    from_dir, from_zip = _get_image(MOSAIC_DIR), _get_image(zipped)
    assert list(from_zip.keys()) == list(from_dir.keys())
    for level in from_dir:
        np.testing.assert_array_equal(from_zip[level]["image"].values, from_dir[level]["image"].values)
    assert get_extent(from_zip) == get_extent(from_dir)


def test_get_image_reads_zip_with_group_at_root(tmp_path: Path) -> None:
    """A zip of the mosaic's contents (``zarr.json`` at its top level) reads the same image as the directory."""
    zip_path = tmp_path / "mosaic_3d.ome.zarr.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(MOSAIC_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(MOSAIC_DIR).as_posix())
    from_dir, from_zip = _get_image(MOSAIC_DIR), _get_image(zip_path)
    assert list(from_zip.keys()) == list(from_dir.keys())
    for level in from_dir:
        np.testing.assert_array_equal(from_zip[level]["image"].values, from_dir[level]["image"].values)


def test_get_image_ignores_macosx_entries_in_zip(tmp_path: Path) -> None:
    """A zip made on macOS carries a ``__MACOSX/`` tree beside the mosaic's directory; it is not the mosaic."""
    zip_path = _zip_dir(MOSAIC_DIR, tmp_path / "mosaic_3d.ome.zarr.zip")
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr(f"__MACOSX/{MOSAIC_DIR.name}/._zarr.json", b"\x00\x05\x16\x07")
    from_dir, from_zip = _get_image(MOSAIC_DIR), _get_image(zip_path)
    assert list(from_zip.keys()) == list(from_dir.keys())
    np.testing.assert_array_equal(from_zip["scale0"]["image"].values, from_dir["scale0"]["image"].values)


def test_pyxa_reader_finds_mosaic(tmp_path: Path) -> None:
    # the fixture holds the unzipped mosaic: found by default, skipped with image=False
    assert "mosaic_image" in pyxa(FIXTURE_DIR, cell_assigned_gene=False).images
    assert not pyxa(FIXTURE_DIR, cell_assigned_gene=False, image=False).images
    # a directory holding only the zip, as downloaded from the Hub
    hub = tmp_path / "hub"
    hub.mkdir()
    for name in ("cell_by_gene_v1.csv", "cell_metadata_v1.csv"):
        (hub / name).write_bytes((FIXTURE_DIR / name).read_bytes())
    _zip_dir(MOSAIC_DIR, hub / "mosaic_3d.ome.zarr.zip")
    assert "mosaic_image" in pyxa(hub).images
    with pytest.raises(FileNotFoundError, match="mosaic image not found"):
        pyxa(hub, image=tmp_path / "nope.ome.zarr")
    empty = tmp_path / "empty"
    empty.mkdir()
    for name in ("cell_by_gene_v1.csv", "cell_metadata_v1.csv"):
        (empty / name).write_bytes((FIXTURE_DIR / name).read_bytes())
    with pytest.raises(FileNotFoundError, match="mosaic_3d.ome.zarr"):
        pyxa(empty, image=True)


def test_pyxa_reader_has_no_image_path() -> None:
    with pytest.raises(TypeError):
        pyxa(FIXTURE_DIR, image_path=MOSAIC_DIR)  # type: ignore[call-arg]


# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
@pytest.mark.parametrize(
    "dataset,expected",
    [("pyxa_xsmall", "{'z': (20, 121), 'y': (-5150, -5029), 'x': (889, 1014)}")],
)
def test_example_data_data_extent(dataset: str, expected: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = pyxa(f, image=f / "mosaic_3d.ome.zarr")

    extent = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    assert str(extent) == expected


@pytest.mark.parametrize("dataset", DATASETS)
def test_example_data_index_integrity(dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = pyxa(f, image=f / "mosaic_3d.ome.zarr")

    if dataset == "pyxa_xsmall":
        # fmt: off
        # test elements
        assert sdata["mosaic_image"]["scale0"]["image"].sel(c="DAPI", z=0.5, y=0.5, x=0.5).data.compute() == 42
        assert sdata["mosaic_image"]["scale0"]["image"].sel(c="DAPI", z=100.5, y=108.5, x=109.5).data.compute() == 64
        assert sdata["mosaic_image"]["scale0"]["image"].sel(c="DAPI", z=199.5, y=216.5, x=217.5).data.compute() == 58
        transcripts = sdata["transcripts"].compute().loc[[0, 10000, 23494]]
        assert transcripts["Gene"].tolist() == ["Epb41l2", "Lamb1", "Id2"]
        assert transcripts["cell_id"].tolist() == ["Region_3645", "Region_4097", "Region_4776"]
        assert np.allclose(transcripts["x"], [982.285445, 942.385736, 907.775326])
        assert np.allclose(transcripts["z"], [29.5, 66.5, 111.5])
        footprint = sdata["cell_boundaries"].loc["Region_3645"].geometry
        assert np.isclose(footprint.centroid.x, 986.9741281150192)
        assert np.isclose(footprint.area, 275.81904506259013)
        planes = sdata["cell_boundaries_z"]
        plane = planes[(planes["cell_id"] == "Region_3645") & (planes["ZIndex"] == 62)].iloc[0]
        assert np.isclose(plane.geometry.centroid.x, 987.0888540837285)
        assert np.isclose(plane.geometry.centroid.y, -5109.209873190636)
        assert plane["Z_um"] == 31.25
        assert sdata["rna"]["Region_3645", "Epb41l2"].X[0, 0] == 3
        assert sdata["rna"]["Region_3645"].X.sum() == 132
        # fmt: on

        # test table annotation
        region, region_key, instance_key = get_table_keys(sdata["rna"])
        assert (region, region_key, instance_key) == ("cell_boundaries", "region", "cell_id")
        matched_table = match_table_to_element(sdata, element_name=region, table_name="rna")
        assert len(matched_table) == 187
        assert matched_table.obs["cell_id"][:3].tolist() == ["Region_3641", "Region_3645", "Region_3647"]
        elements, table = match_element_to_table(sdata, element_name=region, table_name="rna")
        assert len(elements[region]) == len(table) == 187


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa(dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    runner = CliRunner()
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            pyxa_wrapper,
            ["--input", str(f), "--output", str(output_zarr), "--image", str(f / "mosaic_3d.ome.zarr")],
        )
        assert result.exit_code == 0, result.output
        sdata = read_zarr(output_zarr)
        assert set(sdata.shapes) == {"cell_boundaries", "cell_boundaries_z"}
        assert "transcripts" in sdata.points
        assert "mosaic_image" in sdata.images


def _write_studio(path: Path, drop_every: int = 4) -> pd.DataFrame:
    """A Pyxa Studio export for the fixture: every ``drop_every``-th cell filtered out, as Studio does."""
    metadata = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv")
    kept = metadata[np.arange(len(metadata)) % drop_every != 0].reset_index(drop=True)
    rng = np.random.default_rng(0)
    studio = pd.DataFrame(
        {
            "cell_id": kept["cell_id"],
            "FOV": kept["FOV"],
            "Volume_um3": kept["Volume_um3"],
            "Z_pixels": kept["Z_pixels"],
            "Y_pixels": kept["Y_pixels"],
            "X_pixels": kept["X_pixels"],
            "Cluster": np.arange(len(kept)) % 11,
            "X_UMAP": rng.normal(size=len(kept)),
            "Y_UMAP": rng.normal(size=len(kept)),
            "Z_UMAP": rng.normal(size=len(kept)),
        }
    )
    studio.to_csv(path, index=False)
    return studio


def test_get_table_joins_pyxa_studio(tmp_path: Path) -> None:
    studio = _write_studio(tmp_path / "pyxa_studio_v1.csv").set_index("cell_id")
    adata = _get_table(
        FIXTURE_DIR / "cell_by_gene_v1.csv",
        FIXTURE_DIR / "cell_metadata_v1.csv",
        tmp_path / "pyxa_studio_v1.csv",
    )
    in_studio = adata.obs_names.isin(studio.index)
    assert 0 < in_studio.sum() < adata.n_obs

    cluster = adata.obs["Cluster"]
    assert isinstance(cluster.dtype, pd.CategoricalDtype)
    # numeric labels keep numeric order rather than string order ("10" after "9")
    assert list(cluster.cat.categories) == [str(i) for i in range(11)]
    assert cluster[~in_studio].isna().all()
    cell = adata.obs_names[in_studio][0]
    assert cluster[cell] == str(studio.loc[cell, "Cluster"])

    umap = np.asarray(adata.obsm["X_umap"])
    assert umap.shape == (adata.n_obs, 3)
    assert np.isnan(umap[~in_studio]).all()
    np.testing.assert_allclose(umap[in_studio][0], studio.loc[cell, ["X_UMAP", "Y_UMAP", "Z_UMAP"]])
    # cell_metadata wins where both files describe a cell
    assert "Volume_um3" in adata.obs and "UMAP" not in "".join(adata.obs.columns)


def test_pyxa_reader_optional_inputs(tmp_path: Path) -> None:
    studio_path = tmp_path / "pyxa_studio_v1.csv"
    _write_studio(studio_path)

    table_only = pyxa(
        FIXTURE_DIR, cell_assigned_gene=False, segmentation_geometries=False, pyxa_studio=studio_path, image=False
    )
    assert not table_only.points and not table_only.shapes and not table_only.images
    assert set(table_only.tables) == {"rna"}
    assert "Cluster" in table_only["rna"].obs and "X_umap" in table_only["rna"].obsm
    assert table_only["rna"].n_vars == pd.read_csv(FIXTURE_DIR / "cell_by_gene_v1.csv", nrows=1).shape[1] - 1

    # no directory: the required files are given explicitly
    explicit = pyxa(
        cell_by_gene=FIXTURE_DIR / "cell_by_gene_v1.csv",
        cell_metadata=FIXTURE_DIR / "cell_metadata_v1.csv",
        image=MOSAIC_DIR,
    )
    assert set(explicit.tables) == {"rna"} and set(explicit.images) == {"mosaic_image"}
    assert not explicit.points and not explicit.shapes

    # the table annotates the footprints only when the polygons are read too
    full = pyxa(FIXTURE_DIR, pyxa_studio=studio_path)
    assert get_table_keys(full["rna"])[0] == "cell_boundaries"
    assert "Cluster" in full["rna"].obs


def test_pyxa_reader_required_and_skip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cell_by_gene_v1.csv"):
        pyxa(cell_metadata=FIXTURE_DIR / "cell_metadata_v1.csv")
    with pytest.raises(TypeError, match="required"):
        pyxa(FIXTURE_DIR, cell_by_gene=False)
    with pytest.raises(FileNotFoundError, match="pyxa_studio_v1.csv"):
        pyxa(FIXTURE_DIR, pyxa_studio=True)
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        pyxa(FIXTURE_DIR, pyxa_studio=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="directory not found"):
        pyxa(tmp_path / "missing")


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa_skip(dataset: str, tmp_path: Path) -> None:
    f = Path("./data") / dataset
    studio_path = tmp_path / "studio.csv"
    _write_studio(studio_path)
    output_zarr = tmp_path / "data.zarr"
    result = CliRunner().invoke(
        pyxa_wrapper,
        [
            "--input", str(f), "--output", str(output_zarr),
            "--skip", "cell_assigned_gene", "--skip", "segmentation_geometries",
            "--pyxa-studio", str(studio_path), "--no-image",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    sdata = read_zarr(output_zarr)
    assert not sdata.points and not sdata.shapes
    assert not sdata.images
    assert "Cluster" in sdata["rna"].obs


def test_pyxa_keys_labels() -> None:
    assert PyxaKeys.MOSAIC_FILE.value == "mosaic_3d.ome.zarr"
    assert PyxaKeys.MOSAIC_ZIP_FILE.value == "mosaic_3d.ome.zarr.zip"
    assert PyxaKeys.CELL_LABELS.value == "cell_labels"
    assert PyxaKeys.LABEL_ID.value == "label_id"


def test_get_table_counts_are_sparse() -> None:
    from scipy import sparse

    adata = _get_table(FIXTURE_DIR / "cell_by_gene_v1.csv", FIXTURE_DIR / "cell_metadata_v1.csv")
    raw = pd.read_csv(FIXTURE_DIR / "cell_by_gene_v1.csv", index_col="cell_id")
    assert sparse.isspmatrix_csr(adata.X)
    assert adata.X.dtype == raw.to_numpy().dtype
    np.testing.assert_array_equal(adata.X.toarray(), raw.loc[adata.obs_names].to_numpy())
    assert list(adata.var_names) == list(raw.columns)


def test_mosaic_grid_matches_image() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    image = _get_image(MOSAIC_DIR)
    assert grid.shapes == tuple(image[k]["image"].shape[1:] for k in image)
    assert grid.step(0) == (1, 1, 1)

    # step must come from the levels' OME-NGFF scale ratio to level 0, not the array-shape ratio: read
    # the fixture's own scales and check every level's step against them directly.
    group = zarr.open_group(store=str(MOSAIC_DIR), mode="r")
    multiscale = cast("dict[str, Any]", group.attrs.asdict()["ome"])["multiscales"][0]
    axes = [a["name"] for a in multiscale["axes"]]
    zyx = [axes.index(a) for a in ("z", "y", "x")]
    datasets = multiscale["datasets"]
    scale0 = next(t for t in datasets[0]["coordinateTransformations"] if t["type"] == "scale")["scale"]
    for i, dataset in enumerate(datasets):
        scale_i = next(t for t in dataset["coordinateTransformations"] if t["type"] == "scale")["scale"]
        assert grid.step(i) == tuple(round(scale_i[a] / scale0[a]) for a in zyx)
    # the fixture's coarse levels are cropped with their own origins, so the array-shape ratio at
    # scale4 ((200, 217, 218) -> (12, 14, 13), rounding to (17, 16, 17)) is *not* the right stride;
    # the OME scale ratio is exactly 16x on every axis
    assert grid.step(4) == (16, 16, 16)

    affine = get_transformation(image, to_coordinate_system="global").to_affine_matrix(("z", "y", "x"), ("z", "y", "x"))
    np.testing.assert_allclose(grid.transformation.to_affine_matrix(("z", "y", "x"), ("z", "y", "x")), affine)


def test_mosaic_grid_step_from_colon_like_scales() -> None:
    """A synthetic grid reproducing the colon Region's mosaic: the array-shape ratio rounds z at
    levels 4-6 to 17 (284/17 = 16.7), one plane off; the OME scale ratios are the correct steps.
    """
    shapes = (
        (284, 9786, 11889),
        (142, 4893, 5944),
        (71, 2446, 2972),
        (35, 1223, 1486),
        (17, 611, 743),
        (17, 305, 371),
        (17, 152, 185),
    )
    scales = (
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        (4.0, 4.0, 4.0),
        (8.0, 8.0, 8.0),
        (16.0, 16.0, 16.0),
        (16.0, 32.0, 32.0),
        (16.0, 64.0, 64.0),
    )
    grid = _MosaicGrid(shapes=shapes, scale=scales[0], translation=(0.0, 0.0, 0.0), scales=scales)
    assert round(shapes[0][0] / shapes[4][0]) == 17  # the shape ratio would give the wrong stride
    assert grid.step(4) == (16, 16, 16)
    assert grid.step(6) == (16, 64, 64)


def test_mosaic_grid_step_raises_on_non_integer_scale_ratio() -> None:
    grid = _MosaicGrid(
        shapes=((10, 10, 10), (3, 3, 3)),
        scale=(1.0, 1.0, 1.0),
        translation=(0.0, 0.0, 0.0),
        scales=((1.0, 1.0, 1.0), (3.3, 3.3, 3.3)),
    )
    with pytest.raises(ValueError, match="not a positive integer"):
        grid.step(1)


def test_read_rings_on_mosaic_grid() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)

    n_polygons = pq.ParquetFile(FIXTURE_DIR / "segmentation_geometries_v1.parquet").metadata.num_rows
    assert 0 < len(rings) <= 2 * n_polygons  # multipolygons add parts; off-grid planes are dropped
    assert rings.label.dtype == np.uint32 and set(rings.label) <= set(ids)
    nz, ny, nx = grid.shapes[0]
    assert rings.plane.min() >= 0 and rings.plane.max() < nz
    assert rings.length.sum() == len(rings.coords)
    # the fixture's cells lie inside its mosaic crop, up to a cell radius at the edges
    assert rings.bounds[:, 0].min() > -50 and rings.bounds[:, 2].max() < nx + 50
    assert rings.bounds[:, 1].min() > -50 and rings.bounds[:, 3].max() < ny + 50


def test_read_rings_drops_cells_not_in_table() -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    one = pd.Series(np.array([7], dtype=np.uint32), index=[cells.iloc[0]])
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", one, grid, xy_size, z_size)
    assert len(rings) > 0 and set(rings.label) == {7}


def test_read_rings_empty_parquet_gives_empty_rings(tmp_path: Path) -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)

    empty_path = tmp_path / "segmentation_geometries_v1.parquet"
    pq.ParquetWriter(empty_path, pq.read_schema(FIXTURE_DIR / "segmentation_geometries_v1.parquet")).close()

    rings = _read_rings(empty_path, labels, grid, xy_size, z_size)
    assert len(rings) == 0
    assert rings.label.dtype == np.uint32 and rings.label.shape == (0,)
    assert rings.plane.dtype == np.int32 and rings.plane.shape == (0,)
    assert rings.length.dtype == np.int64 and rings.length.shape == (0,)
    assert rings.coords.dtype == np.float32 and rings.coords.shape == (0, 2)
    assert rings.bounds.dtype == np.float32 and rings.bounds.shape == (0, 4)


def test_read_rings_matches_across_row_group_counts(tmp_path: Path) -> None:
    # the pool path (joblib/loky) kicks in only above one row group; split the fixture into several
    # to exercise it, and check it gives the exact same rings as the single-row-group fixture file
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)

    multi_path = tmp_path / "multi.parquet"
    pq.write_table(pq.read_table(FIXTURE_DIR / "segmentation_geometries_v1.parquet"), multi_path, row_group_size=200)
    assert pq.ParquetFile(multi_path).metadata.num_row_groups > 1

    single = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)
    multi = _read_rings(multi_path, labels, grid, xy_size, z_size)
    assert len(single) == len(multi) > 0
    np.testing.assert_array_equal(single.label, multi.label)
    np.testing.assert_array_equal(single.plane, multi.plane)
    np.testing.assert_array_equal(single.length, multi.length)
    np.testing.assert_allclose(single.coords, multi.coords)
    np.testing.assert_allclose(single.bounds, multi.bounds)


def _multi_row_group_inputs(tmp_path: Path) -> tuple[Path, pd.Series, _MosaicGrid, float, float]:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    multi_path = tmp_path / "multi.parquet"
    pq.write_table(pq.read_table(FIXTURE_DIR / "segmentation_geometries_v1.parquet"), multi_path, row_group_size=2000)
    return multi_path, pd.Series(ids, index=cells), grid, xy_size, z_size


def test_read_rings_shuts_worker_processes_down(tmp_path: Path) -> None:
    """The decode's worker processes do not linger (holding memory) once the rings are read."""
    import multiprocessing

    path, labels, grid, xy_size, z_size = _multi_row_group_inputs(tmp_path)
    assert len(_read_rings(path, labels, grid, xy_size, z_size)) > 0
    assert multiprocessing.active_children() == []


def test_read_rings_follows_joblib_parallel_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``joblib.parallel_config(backend=...)`` chooses where row groups are decoded."""
    import os

    import joblib

    path, labels, grid, xy_size, z_size = _multi_row_group_inputs(tmp_path)
    pids: list[int] = []
    real = pyxa_module._rings_from_row_group

    def recording(*args: object) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        pids.append(os.getpid())
        return real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(pyxa_module, "_rings_from_row_group", recording)
    with joblib.parallel_config(backend="threading"):
        rings = _read_rings(path, labels, grid, xy_size, z_size)
    assert len(rings) > 0
    assert pids == [os.getpid()] * pq.ParquetFile(path).metadata.num_row_groups


def test_read_rings_matches_with_a_smaller_decode_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    # a batch size smaller than the fixture's single row group forces multiple batches per row group;
    # the concatenated result must be identical to decoding the whole row group in one batch
    monkeypatch.setattr("spatialdata_io.readers.pyxa._DECODE_BATCH_ROWS", 50)
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)
    batched = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)

    monkeypatch.undo()
    unbatched = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)

    assert len(batched) == len(unbatched) > 0
    np.testing.assert_array_equal(batched.label, unbatched.label)
    np.testing.assert_array_equal(batched.plane, unbatched.plane)
    np.testing.assert_array_equal(batched.length, unbatched.length)
    np.testing.assert_allclose(batched.coords, unbatched.coords)
    np.testing.assert_allclose(batched.bounds, unbatched.bounds)


def test_read_rings_drops_planes_off_the_mosaic_z_range(caplog: pytest.LogCaptureFixture) -> None:
    grid = _mosaic_grid(MOSAIC_DIR)
    shifted = dataclasses.replace(grid, translation=(grid.translation[0] + 1e6, *grid.translation[1:]))
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)

    with caplog.at_level("INFO"):
        rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, shifted, xy_size, z_size)
    assert len(rings) == 0
    assert "off the mosaic's z range" in caplog.text


def test_label_ids_trailing_integer() -> None:
    ids, rule = _label_ids(pd.Index(["Region_17", "Region_3", "ROI2_40"]))
    assert ids.dtype == np.uint32 and ids.tolist() == [17, 3, 40]
    assert "trailing integer" in rule


@pytest.mark.parametrize(
    ("cell_ids", "why"),
    [
        (["A_1", "B_1"], "not unique"),
        (["Region_1", "Region_x"], "no trailing integer"),
        (["Region_0", "Region_2"], "is 0"),
        (["Region_1", "Region_4294967296"], "2^31"),
    ],
)
def test_label_ids_fallback(cell_ids: list[str], why: str) -> None:
    ids, rule = _label_ids(pd.Index(cell_ids))
    assert ids.tolist() == [1, 2]
    assert why in rule


def _empty_rings() -> _Rings:
    return _Rings(
        label=np.empty(0, dtype=np.uint32),
        plane=np.empty(0, dtype=np.int32),
        length=np.empty(0, dtype=np.int64),
        coords=np.empty((0, 2), dtype=np.float32),
        bounds=np.empty((0, 4), dtype=np.float32),
    )


def _square_rings(squares: list[tuple[int, int, float, float, float, float]]) -> _Rings:
    """Rings from (label, plane, x0, y0, x1, y1) axis-aligned squares in level-0 voxel index space."""
    coords, bounds = [], []
    for _, _, x0, y0, x1, y1 in squares:
        coords.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float32))
        bounds.append([x0, y0, x1, y1])
    return _Rings(
        label=np.array([s[0] for s in squares], dtype=np.uint32),
        plane=np.array([s[1] for s in squares], dtype=np.int32),
        length=np.full(len(squares), 5, dtype=np.int64),
        coords=np.concatenate(coords),
        bounds=np.array(bounds, dtype=np.float32),
    )


def test_rasterize_square() -> None:
    rings = _square_rings([(5, 1, 2.0, 2.0, 5.0, 5.0)])
    (tile,) = _plan_tiles(rings, (4, 10, 10), (1, 1, 1))
    block = _rasterize_tile(tile)
    assert block.dtype == np.uint32 and block.shape == (4, 10, 10)
    assert (block[1, 2:6, 2:6] == 5).all()  # voxel centres 2..5 lie on or inside the square
    assert block[1, 0, 0] == 0 and block[1, 8, 8] == 0
    assert block[0].max() == 0 and block[2].max() == 0


def test_rasterize_higher_label_wins() -> None:
    for order in ([(3, 0, 0.0, 0.0, 5.0, 5.0), (7, 0, 3.0, 3.0, 8.0, 8.0)],
                  [(7, 0, 3.0, 3.0, 8.0, 8.0), (3, 0, 0.0, 0.0, 5.0, 5.0)]):  # fmt: skip
        (tile,) = _plan_tiles(_square_rings(order), (1, 10, 10), (1, 1, 1))
        block = _rasterize_tile(tile)
        assert block[0, 4, 4] == 7 and block[0, 1, 1] == 3


def test_labels_level_tiles_join_seamlessly() -> None:
    rings = _square_rings([(5, 1, 2.0, 2.0, 7.0, 7.0), (9, 3, 0.0, 6.0, 9.0, 9.0)])
    whole = _rasterize_tile(_plan_tiles(rings, (4, 10, 10), (1, 1, 1))[0])
    tiled = _labels_level(rings, (4, 10, 10), (1, 1, 1), tile=(2, 4, 4), chunks=(2, 3, 3))
    assert tiled.chunksize == (2, 3, 3)
    np.testing.assert_array_equal(tiled.compute(), whole)

    # rings ending (or starting) a fraction of a voxel from the tile edge at 4
    for edge in (3.2, 3.5, 3.7, 3.9, 4.0, 4.3, 4.5, 4.7):
        rings = _square_rings([(5, 0, 1.0, 1.0, edge, edge), (6, 0, edge, 5.0, 7.0, 7.0), (7, 0, 5.0, edge, 7.0, 4.9)])
        whole = _labels_level(rings, (1, 8, 8), (1, 1, 1), tile=(1, 8, 8), chunks=(1, 8, 8)).compute()
        tiled = _labels_level(rings, (1, 8, 8), (1, 1, 1), tile=(1, 4, 4), chunks=(1, 8, 8)).compute()
        np.testing.assert_array_equal(tiled, whole, err_msg=f"edge {edge}")


def test_labels_level_tiles_join_seamlessly_for_random_rings() -> None:
    """Arbitrary polygons with fractional vertices, on tiles of several sizes, draw as one whole tile does."""
    rng = np.random.default_rng(0)
    coords, lengths = [], []
    for _ in range(200):
        n = int(rng.integers(3, 9))
        centre, angle, radius = rng.uniform(-1, 41, 2), np.sort(rng.uniform(0, 2 * np.pi, n)), rng.uniform(0.1, 4, n)
        ring = np.column_stack((centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)))
        coords.append(np.vstack([ring, ring[:1]]).astype(np.float32))
        lengths.append(n + 1)
    rings = _Rings(
        label=np.arange(1, 201, dtype=np.uint32),
        plane=np.zeros(200, dtype=np.int32),
        length=np.array(lengths, dtype=np.int64),
        coords=np.concatenate(coords),
        bounds=np.array([[c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()] for c in coords]),
    )
    whole = _labels_level(rings, (1, 40, 40), (1, 1, 1), tile=(1, 40, 40), chunks=(1, 40, 40)).compute()
    for size in (4, 5, 8):
        tiled = _labels_level(rings, (1, 40, 40), (1, 1, 1), tile=(1, size, size), chunks=(1, 40, 40))
        np.testing.assert_array_equal(tiled.compute(), whole)


def test_labels_level_strides_level_zero() -> None:
    rings = _square_rings([(5, 2, 1.0, 1.0, 12.0, 12.0), (6, 3, 4.0, 4.0, 9.0, 9.0)])
    level0 = _labels_level(rings, (4, 16, 16), (1, 1, 1)).compute()
    level1 = _labels_level(rings, (2, 8, 8), (2, 2, 2)).compute()
    np.testing.assert_array_equal(level1, level0[::2, ::2, ::2])


def test_labels_level_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = pyxa_module._rasterize_tile

    def counting(tile: pyxa_module._Tile) -> np.ndarray:
        calls.append(1)
        return real(tile)

    monkeypatch.setattr(pyxa_module, "_rasterize_tile", counting)
    array = _labels_level(_square_rings([(5, 0, 1.0, 1.0, 3.0, 3.0)]), (1, 8, 8), (1, 1, 1))
    assert calls == []
    array.compute()
    assert calls == [1]


def test_plan_tiles_empty_rings_returns_no_tiles() -> None:
    assert _plan_tiles(_empty_rings(), (2, 5, 5), (1, 1, 1)) == []


def test_labels_level_empty_rings_is_all_zero() -> None:
    block = _labels_level(_empty_rings(), (2, 5, 5), (1, 1, 1)).compute()
    assert block.dtype == np.uint32 and block.shape == (2, 5, 5)
    assert (block == 0).all()


def _fixture_labels() -> tuple[DataTree, pd.Series, _Rings]:
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    cells = pd.read_csv(FIXTURE_DIR / "cell_metadata_v1.csv", usecols=["cell_id"])["cell_id"]
    ids, _ = _label_ids(pd.Index(cells))
    labels = pd.Series(ids, index=cells)
    rings = _read_rings(FIXTURE_DIR / "segmentation_geometries_v1.parquet", labels, grid, xy_size, z_size)
    return _get_labels(rings, grid), labels, rings


def _level_values(tree: DataTree, level: str) -> np.ndarray:
    """One level of a multiscale labels element, computed to a NumPy array."""
    return np.asarray(tree[level]["image"].data)


def test_get_labels_on_mosaic_grid() -> None:
    tree, labels, _ = _fixture_labels()
    image = _get_image(MOSAIC_DIR)
    assert list(tree.keys()) == list(image.keys())
    for level in image:
        assert tree[level]["image"].shape == image[level]["image"].shape[1:]
        assert tree[level]["image"].dtype == np.uint32

    def _affine(e):
        return get_transformation(e, to_coordinate_system="global").to_affine_matrix(("z", "y", "x"), ("z", "y", "x"))

    np.testing.assert_allclose(_affine(tree), _affine(image))
    level0 = _level_values(tree, "scale0")
    assert 0.05 < (level0 > 0).mean() < 0.95
    assert set(np.unique(level0)) - {0} <= set(labels.to_numpy())


def test_get_labels_logs_rings_tiles_and_levels(caplog: pytest.LogCaptureFixture) -> None:
    grid = _MosaicGrid(
        shapes=((1, 8, 8), (1, 4, 4)),
        scale=(1.0, 1.0, 1.0),
        translation=(0.0, 0.0, 0.0),
        scales=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    )
    with caplog.at_level("INFO"):
        _get_labels(_square_rings([(5, 0, 1.0, 1.0, 3.0, 3.0), (6, 0, 4.0, 4.0, 6.0, 6.0)]), grid)
    assert "2 rings in 1 level-0 tiles, 2 levels planned" in caplog.text


def test_get_labels_cell_voxels() -> None:
    """A cell's own polygon centre, on its plane, carries its label."""
    tree, labels, _ = _fixture_labels()
    level0 = _level_values(tree, "scale0")
    grid = _mosaic_grid(MOSAIC_DIR)
    xy_size, z_size = _get_voxel_size(FIXTURE_DIR / "cell_metadata_v1.csv")
    planes = _get_shapes(FIXTURE_DIR / "segmentation_geometries_v1.parquet", xy_size, z_size)
    sz, sy, sx = grid.scale
    tz, ty, tx = grid.translation
    checked = 0
    for _, row in planes.sort_values("cell_id").groupby("cell_id").head(1).head(40).iterrows():
        p = row.geometry.representative_point()  # micrometers
        if row.geometry.boundary.distance(p) < 0.5 * sx:
            continue  # too close to the polygon boundary for a voxel-centre check to be unambiguous
        z, y, x = (int(round((v - t) / s)) for v, t, s in ((row["Z_um"], tz, sz), (p.y, ty, sy), (p.x, tx, sx)))
        same_plane = planes[(planes["ZIndex"] == row["ZIndex"]) & (planes["cell_id"] != row["cell_id"])]
        if 0 <= z < level0.shape[0] and not same_plane.geometry.contains(p).any():
            assert level0[z, y, x] == labels[row["cell_id"]]
            checked += 1
    assert checked >= 10


def test_get_labels_levels_stride_level_zero() -> None:
    tree, _, _ = _fixture_labels()
    grid = _mosaic_grid(MOSAIC_DIR)
    level0 = _level_values(tree, "scale0")
    for i in range(1, len(grid.shapes)):
        dz, dy, dx = grid.step(i)
        nz, ny, nx = grid.shapes[i]
        # sampled at each coarse voxel's centre, as the mosaic's pyramid averages the block around it,
        # unless that would run off level 0's end (here only y at scale1: 217 voxels, 109 at stride 2)
        oz, oy, ox = (
            min(d // 2, a - 1 - (b - 1) * d)
            for a, b, d in zip(grid.shapes[0], grid.shapes[i], (dz, dy, dx), strict=True)
        )
        assert min(oz, oy, ox) >= 0
        strided = level0[oz::dz, oy::dy, ox::dx][:nz, :ny, :nx]
        assert strided.shape == (nz, ny, nx)
        level = _level_values(tree, f"scale{i}")
        # coarse levels are strided views of level 0, so they must match it exactly
        np.testing.assert_array_equal(level, strided)


def test_get_labels_levels_clamp_the_centre_offset_to_fit() -> None:
    """Where a centre offset would run a coarse level off level 0's end, the offset shrinks until it fits."""
    grid = _MosaicGrid(
        shapes=((1, 5, 5), (1, 3, 3)),
        scale=(1.0, 1.0, 1.0),
        translation=(0.0, 0.0, 0.0),
        scales=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    )
    tree = _get_labels(_square_rings([(4, 0, 0.0, 0.0, 0.2, 4.0), (9, 0, 2.0, 0.0, 2.2, 4.0)]), grid)
    level0 = _level_values(tree, "scale0")
    assert set(np.unique(level0[0, :, [0, 2]])) == {4, 9} and not level0[0, :, 1].any()
    np.testing.assert_array_equal(_level_values(tree, "scale1"), level0[:, ::2, ::2])


def test_get_labels_writes_each_level_zero_tile_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Writing the whole tree draws each level-0 tile once, shared by every coarser level."""
    from spatialdata import SpatialData

    grid = _mosaic_grid(MOSAIC_DIR)
    _, _, rings = _fixture_labels()

    calls: list[int] = []
    real = pyxa_module._rasterize_tile

    def counting(tile: pyxa_module._Tile) -> np.ndarray:
        calls.append(1)
        return real(tile)

    monkeypatch.setattr(pyxa_module, "_rasterize_tile", counting)

    tree = _get_labels(rings, grid)
    output = tmp_path / "data.zarr"
    SpatialData(labels={"cell_labels": tree}).write(output)

    n_tiles = len(_plan_tiles(rings, grid.shapes[0], (1, 1, 1)))
    assert len(calls) == n_tiles

    # compute the expected level 0 with the real (unpatched) drawing function, so this doesn't add calls
    monkeypatch.setattr(pyxa_module, "_rasterize_tile", real)
    level0_expected = _level_values(_get_labels(rings, grid), "scale0")

    written = read_zarr(output)
    np.testing.assert_array_equal(written["cell_labels"]["scale0"]["image"].values, level0_expected)


def test_get_labels_raises_when_a_level_is_shorter_than_any_stride() -> None:
    """No integer stride of a 4-voxel level 0 can reach a 6-voxel level 1: ``_get_labels`` must reject it."""
    grid = _MosaicGrid(
        shapes=((2, 4, 4), (2, 6, 6)),
        scale=(1.0, 1.0, 1.0),
        translation=(0.0, 0.0, 0.0),
        scales=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="shorter than the mosaic"):
        _get_labels(_empty_rings(), grid)


def test_pyxa_reader_labels(tmp_path: Path) -> None:
    sdata = pyxa(FIXTURE_DIR, cell_assigned_gene=False, labels=True)
    assert set(sdata.labels) == {"cell_labels"} and set(sdata.images) == {"mosaic_image"}
    assert not sdata.shapes  # labels replace the shapes by default
    table = sdata["rna"]
    assert get_table_keys(table) == ("cell_labels", "region", "label_id")
    assert table.obs["label_id"].dtype == np.uint32
    assert "cell_id" in table.obs

    sdata.write(tmp_path / "labels.zarr")
    back = read_zarr(tmp_path / "labels.zarr")
    drawn = set(np.unique(back["cell_labels"]["scale0"]["image"].values)) - {0}
    assert drawn <= set(back["rna"].obs["label_id"])
    assert get_table_keys(back["rna"])[0] == "cell_labels"


def test_pyxa_reader_labels_and_shapes() -> None:
    sdata = pyxa(FIXTURE_DIR, cell_assigned_gene=False, labels=True, shapes=True)
    assert set(sdata.shapes) == {"cell_boundaries", "cell_boundaries_z"}
    assert get_table_keys(sdata["rna"])[0] == "cell_labels"
    assert not pyxa(FIXTURE_DIR, cell_assigned_gene=False, shapes=False).shapes


def test_pyxa_reader_labels_need_image_and_geometries() -> None:
    with pytest.raises(ValueError, match="missing: a mosaic image"):
        pyxa(FIXTURE_DIR, labels=True, image=False)
    with pytest.raises(ValueError, match="missing: segmentation_geometries"):
        pyxa(FIXTURE_DIR, labels=True, segmentation_geometries=False)
    with pytest.raises(FileNotFoundError, match="segmentation_geometries_v1.parquet"):
        pyxa(FIXTURE_DIR, shapes=True, segmentation_geometries=False)


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa_labels(dataset: str, tmp_path: Path) -> None:
    output_zarr = tmp_path / "data.zarr"
    result = CliRunner().invoke(
        pyxa_wrapper,
        ["--input", str(Path("./data") / dataset), "--output", str(output_zarr),
         "--skip", "cell_assigned_gene", "--labels"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    sdata = read_zarr(output_zarr)
    assert set(sdata.labels) == {"cell_labels"} and not sdata.shapes


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa_no_shapes(dataset: str, tmp_path: Path) -> None:
    output_zarr = tmp_path / "data.zarr"
    result = CliRunner().invoke(
        pyxa_wrapper,
        ["--input", str(Path("./data") / dataset), "--output", str(output_zarr),
         "--skip", "cell_assigned_gene", "--no-image", "--no-shapes"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    sdata = read_zarr(output_zarr)
    assert not sdata.shapes and not sdata.labels and not sdata.images
    assert "rna" in sdata.tables


@pytest.mark.parametrize("dataset", DATASETS)
def test_cli_pyxa_image_and_no_image_conflict(dataset: str, tmp_path: Path) -> None:
    output_zarr = tmp_path / "data.zarr"
    result = CliRunner().invoke(
        pyxa_wrapper,
        ["--input", str(Path("./data") / dataset), "--output", str(output_zarr),
         "--image", str(MOSAIC_DIR), "--no-image"],
    )  # fmt: skip
    assert result.exit_code == 2, result.output  # click's usage error
    assert "--image" in result.output and "--no-image" in result.output
    assert not output_zarr.exists()
