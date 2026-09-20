"""Display colours follow AnnData's existing uns convention."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from spatialdata_io.experimental.display_colors import (
    GENE_COLORS_KEY,
    add_cluster_colors,
    add_gene_colors,
    palette,
)


def _table(n_genes: int = 4) -> ad.AnnData:
    return ad.AnnData(X=sp.csr_matrix(np.zeros((3, n_genes), dtype=np.float32)))


def test_gene_colours_go_to_uns_not_var() -> None:
    # AnnData already has a place for colours: a `<name>_colors` list in uns aligned to an
    # ordering. Adding a var column instead would be a new convention for no gain.
    table = _table()
    assert add_gene_colors(table) == GENE_COLORS_KEY
    assert GENE_COLORS_KEY in table.uns
    assert "color" not in table.var
    assert len(table.uns[GENE_COLORS_KEY]) == table.n_vars


def test_colours_are_valid_hex_and_distinct() -> None:
    colors = palette(24)
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    assert all(all(ch in "0123456789abcdef" for ch in c[1:]) for c in colors)
    # Golden-ratio hue stepping should not repeat this early.
    assert len(set(colors)) == 24


def test_the_palette_is_deterministic() -> None:
    # A gene must keep its colour between runs, or every rebuild reshuffles the display.
    assert palette(10) == palette(10)
    assert palette(10) == palette(20)[:10]


def test_existing_colours_are_not_clobbered() -> None:
    table = _table()
    table.uns[GENE_COLORS_KEY] = ["#000000"] * table.n_vars
    assert add_gene_colors(table) is None
    assert table.uns[GENE_COLORS_KEY][0] == "#000000"
    assert add_gene_colors(table, overwrite=True) == GENE_COLORS_KEY
    assert table.uns[GENE_COLORS_KEY][0] != "#000000"


def test_cluster_colours_follow_the_scanpy_key() -> None:
    table = _table()
    table.obs["cell_type"] = pd.Categorical(["a", "b", "a"])
    assert add_cluster_colors(table, "cell_type") == "cell_type_colors"
    assert len(table.uns["cell_type_colors"]) == 2


def test_a_non_categorical_column_is_skipped() -> None:
    table = _table()
    table.obs["counts"] = [1, 2, 3]
    assert add_cluster_colors(table, "counts") is None
    assert add_cluster_colors(table, "absent") is None


def test_colours_round_trip_through_zarr(tmp_path) -> None:
    table = _table()
    add_gene_colors(table)
    table.write_zarr(tmp_path / "t.zarr")
    reloaded = ad.read_zarr(tmp_path / "t.zarr")
    assert list(reloaded.uns[GENE_COLORS_KEY]) == list(table.uns[GENE_COLORS_KEY])
