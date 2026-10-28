from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from shapely.geometry import MultiPolygon, Polygon
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers.merscope import (
    _dask_image_load_merscope,
    _get_channel_names,
    _get_reader,
    _rioxarray_load_merscope,
    merscope,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def broken_rioxarray(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow `rioxarray` with a module that can be found but not imported.

    This is what a `rioxarray` installation with a broken `rasterio` looks like.
    """
    (tmp_path / "rioxarray.py").write_text("raise ModuleNotFoundError(\"No module named 'rasterio'\")\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "rioxarray", raising=False)


def test_get_reader_honours_an_explicit_backend() -> None:
    assert _get_reader("rioxarray") is _rioxarray_load_merscope
    assert _get_reader("dask_image") is _dask_image_load_merscope


@pytest.mark.usefixtures("broken_rioxarray")
def test_get_reader_falls_back_when_rioxarray_cannot_be_imported() -> None:
    """A `rioxarray` that is installed but raises on import must not select the rioxarray backend."""
    assert _get_reader(None) is _dask_image_load_merscope


GENES = ["GeneA", "GeneB", "GeneC"]
BLANKS = ["Blank-1", "Blank-2"]
CELL_IDS = ["1001", "1002", "1003", "1004"]


def _square(cx: float, cy: float, r: float = 1.0) -> Polygon:
    return Polygon([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)])


def _write_dataset(
    root: Path,
    *,
    with_images_transform: bool = True,
    polygon_geometry: bool = False,
) -> Path:
    """Write a tiny standard-layout MERSCOPE dataset and return the region directory."""
    region = root / "region0"
    region.mkdir(parents=True)

    rng = np.random.default_rng(0)
    xs = [10.0 * i for i in range(len(CELL_IDS))]
    ys = [5.0 * i for i in range(len(CELL_IDS))]

    # counts: cells x (genes + blanks)
    counts = pd.DataFrame(
        rng.integers(0, 5, size=(len(CELL_IDS), len(GENES) + len(BLANKS))),
        index=pd.Index(CELL_IDS, name="cell"),
        columns=GENES + BLANKS,
    )
    counts.to_csv(region / "cell_by_gene.csv")

    # metadata: same cell order as the counts (as Vizgen writes them)
    meta = pd.DataFrame(
        {"center_x": xs, "center_y": ys, "volume": 1.0},
        index=pd.Index(CELL_IDS, name="EntityID"),
    )
    meta.to_csv(region / "cell_metadata.csv")

    # transcripts: every gene appears; include the junk/local columns a real file carries
    tx = pd.DataFrame(
        {
            "barcode_id": range(len(GENES) * 2),
            "global_x": np.tile(xs[: len(GENES)], 2),
            "global_y": np.tile(ys[: len(GENES)], 2),
            "global_z": 0.0,
            "x": 0.0,
            "y": 0.0,
            "fov": 0,
            "gene": GENES + GENES,
            "transcript_id": [f"t{i}" for i in range(len(GENES) * 2)],
            "cell_id": -1,
        }
    )
    tx.to_csv(region / "detected_transcripts.csv")

    # boundaries parquet with a capital "Geometry" column (as Vizgen VPT writes)
    import geopandas as gpd

    geoms = []
    for cx, cy in zip(xs, ys, strict=True):
        sq = _square(cx, cy)
        geoms.append(sq if polygon_geometry else MultiPolygon([sq]))
    gdf = gpd.GeoDataFrame({"EntityID": CELL_IDS, "ZIndex": 0}, geometry=geoms).rename_geometry("Geometry")
    gdf.to_parquet(region / "cell_boundaries.parquet")

    if with_images_transform:
        images = region / "images"
        images.mkdir()
        (images / "micron_to_mosaic_pixel_transform.csv").write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")

    return region


def test_table_is_sparse_and_values_aligned(tmp_path: Path) -> None:
    region = _write_dataset(tmp_path)
    sdata = merscope(region, mosaic_images=False)
    adata = sdata["table"]

    # X must be sparse (like the other scverse readers), not a dense int64 array
    assert issparse(adata.X), f"table X should be sparse, got {type(adata.X)}"

    # values / var names stay correct (reader preserves the count-matrix row order)
    counts = pd.read_csv(region / "cell_by_gene.csv", index_col=0)
    np.testing.assert_array_equal(adata.to_df()[GENES].to_numpy(), counts[GENES].to_numpy())
    assert list(adata.var_names) == GENES
    assert adata.obsm["blank"].shape[1] == len(BLANKS)


def test_reading_does_not_warn_about_deprecated_dtype(tmp_path: Path) -> None:
    region = _write_dataset(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merscope(region, mosaic_images=False, transcripts=False, cells_boundaries=False)
    assert not any("dtype argument is deprecated" in str(w.message) for w in caught)


def test_transcript_gene_categories_known_across_partitions(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # A gene confined to a later dask partition must survive. Force >=2 partitions by shrinking the
    # reader's dd.read_csv blocksize; "GeneC" only appears in the tail, so the first partition does not
    # contain it -> on the unfixed reader its transcripts become NaN.
    import dask.dataframe as dd

    region = _write_dataset(tmp_path)
    genes = ["GeneA", "GeneB"] * 400 + ["GeneC"] * 20
    tx = pd.DataFrame({"global_x": 0.0, "global_y": 0.0, "global_z": 0.0, "gene": genes})
    tx.to_csv(region / "detected_transcripts.csv", index=False)
    read_csv = dd.read_csv
    monkeypatch.setattr(dd, "read_csv", lambda p, **kw: read_csv(p, blocksize="2kB", **kw))

    sdata = merscope(region, mosaic_images=False, cells_boundaries=False, cells_table=False)
    gene = next(iter(sdata.points.values()))["gene"]
    assert gene.npartitions >= 2
    assert gene.cat.known
    computed = gene.compute()
    assert not computed.isna().any()
    assert set(computed.astype(str).unique()) == set(GENES)


def test_read_without_images_directory(tmp_path: Path) -> None:
    # A dataset without the images/ dir must still read with the defaults (mosaic_images=True):
    # identity transform + no channels found, not a crash.
    region = _write_dataset(tmp_path, with_images_transform=False)
    sdata = merscope(region)
    assert "table" in sdata.tables
    assert len(sdata.images) == 0
    gdf = next(iter(sdata.shapes.values()))
    assert isinstance(get_transformation(gdf, "global"), Identity)


def test_vpt_dict_mode_accepts_string_paths(tmp_path: Path) -> None:
    # dict VPT mode with string paths must not crash on `.exists()`
    region = _write_dataset(tmp_path)
    vpt = {
        "cell_by_gene": str(region / "cell_by_gene.csv"),
        "cell_metadata": str(region / "cell_metadata.csv"),
        "cell_boundaries": str(region / "cell_boundaries.parquet"),
    }
    sdata = merscope(region, vpt_outputs=vpt, mosaic_images=False)
    assert "table" in sdata.tables
    assert len(sdata.shapes) == 1


def test_polygon_boundaries_supported(tmp_path: Path) -> None:
    # plain Polygon geometries must not crash the MultiPolygon normalisation
    region = _write_dataset(tmp_path, polygon_geometry=True)
    sdata = merscope(region, mosaic_images=False, transcripts=False, cells_table=False)
    gdf = next(iter(sdata.shapes.values()))
    assert (gdf.geom_type == "MultiPolygon").all()
    assert len(gdf) == len(CELL_IDS)


def test_invalid_boundaries_are_repaired_not_dropped(tmp_path: Path) -> None:
    # a self-intersecting (invalid) boundary must be repaired: dropping it leaves its table row
    # annotating a shape that does not exist
    import geopandas as gpd
    from shapely.geometry import Polygon

    region = _write_dataset(tmp_path)
    gdf = gpd.read_parquet(region / "cell_boundaries.parquet")
    bowtie = Polygon([(0, 0), (4, 4), (4, 0), (0, 4)])
    assert not bowtie.is_valid
    gdf.loc[[0], "Geometry"] = gpd.GeoSeries([MultiPolygon([bowtie])], index=[0])
    gdf.to_parquet(region / "cell_boundaries.parquet")

    sdata = merscope(region, mosaic_images=False, transcripts=False)
    shapes = next(iter(sdata.shapes.values()))
    assert len(shapes) == len(CELL_IDS)
    assert shapes.geometry.is_valid.all()
    assert (shapes.geom_type == "MultiPolygon").all()
    assert set(sdata["table"].obs["EntityID"].astype(str)) <= set(shapes.index)


def test_channel_names_sorted_deterministic(tmp_path: Path) -> None:
    # channel order must be deterministic (sorted), independent of set iteration order
    images = tmp_path / "images"
    images.mkdir()
    stains = ["PolyT", "DAPI", "Cellbound1", "Cellbound2"]
    for stain in stains:
        for z in range(3):
            (images / f"mosaic_{stain}_z{z}.tif").touch()
    result = _get_channel_names(images)
    assert result == sorted(stains)
