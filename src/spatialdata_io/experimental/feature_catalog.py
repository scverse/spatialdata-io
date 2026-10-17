"""Stable integer codes for transcript features.

Rendering a transcript layer needs a compact integer per point, not a string. This module
builds the mapping ``feature name <-> feature_code`` and pins two properties the rest of
the profile depends on:

1. Genes come first, in the annotating table's ``var_names`` order, so a gene's
   ``feature_code`` *is* its row-group index in the cell-by-gene file. No second lookup
   table, and no browser-side string join.
2. Non-gene features (negative controls, unassigned codewords) are kept, but are coded
   *above* every gene and flagged. They are never silently folded into a real gene, which
   would fabricate expression.

The split point is recorded as ``n_genes`` so a client can tell the two apart from the
manifest alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["FeatureCatalog", "CONTROL_PREFIXES"]

#: Feature-name prefixes that 10x uses for non-gene codewords. Used only for reporting;
#: catalog membership is decided by absence from the table's ``var_names``, not by prefix.
CONTROL_PREFIXES = (
    "NegControlProbe",
    "NegControlCodeword",
    "UnassignedCodeword",
    "antisense",
    "BLANK",
    "DeprecatedCodeword",
    "Intergenic",
)


@dataclass(frozen=True)
class FeatureCatalog:
    """An ordered feature vocabulary with genes first.

    Parameters
    ----------
    names
        All feature names. Position in this list is the feature code.
    n_genes
        Number of leading entries that are genes present in the annotating table. Codes
        ``>= n_genes`` are non-gene features.
    """

    names: tuple[str, ...]
    n_genes: int

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        if not 0 <= self.n_genes <= len(self.names):
            raise ValueError(f"n_genes={self.n_genes} out of range for {len(self.names)} features")

    def __len__(self) -> int:
        return len(self.names)

    @property
    def genes(self) -> tuple[str, ...]:
        """The gene names, in table ``var_names`` order."""
        return self.names[: self.n_genes]

    @property
    def controls(self) -> tuple[str, ...]:
        """The non-gene feature names."""
        return self.names[self.n_genes :]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Smallest unsigned integer dtype that can hold every code."""
        return np.dtype(np.uint16) if len(self.names) <= np.iinfo(np.uint16).max else np.dtype(np.uint32)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_features_and_table(
        cls,
        feature_names: Iterable[str],
        var_names: Sequence[str],
    ) -> FeatureCatalog:
        """Build a catalog from the observed transcript features and the table's genes.

        Genes are taken in ``var_names`` order. Any observed feature absent from
        ``var_names`` is appended as a control, sorted so the catalog is reproducible.

        Raises
        ------
        ValueError
            If ``var_names`` contains duplicates, which would make codes ambiguous.
        """
        genes = list(var_names)
        if len(set(genes)) != len(genes):
            raise ValueError("var_names contains duplicates; feature codes would be ambiguous")

        observed = set(feature_names)
        gene_set = set(genes)
        controls = sorted(observed - gene_set)
        return cls(names=tuple(genes) + tuple(controls), n_genes=len(genes))

    @classmethod
    def from_points_and_table(
        cls,
        points: Any,
        table: Any,
        feature_key: str = "feature_name",
    ) -> FeatureCatalog:
        """Build a catalog directly from a Points element and its annotating table.

        Handles the dask categorical read back from a zarr store, whose categories are
        lazily "unknown" and raise on access until realized.
        """
        col = points[feature_key]
        if hasattr(col, "cat"):
            try:
                features = list(col.cat.categories)
            except (NotImplementedError, AttributeError):
                # dask raises AttributeNotImplementedError (a subclass of both) for
                # unknown categories, which is the normal state straight after read_zarr.
                # Realize just the category list, which is tiny, rather than the column.
                features = list(col.cat.as_known().cat.categories)
        else:
            features = list(col.unique().compute() if hasattr(col, "compute") else col.unique())
        return cls.from_features_and_table(features, list(table.var_names))

    # -- encoding -------------------------------------------------------------

    def encode(self, values: pd.Series | pd.Categorical | NDArray[Any]) -> NDArray[Any]:
        """Map feature names to codes.

        Uses the categorical fast path when available, avoiding a per-row dict lookup over
        millions of transcripts.

        Raises
        ------
        ValueError
            If any value is absent from the catalog. Unknown features are never mapped to
            a fallback code, since that would silently attribute reads to the wrong feature.
        """
        index = pd.Index(self.names)
        cat = values.values if isinstance(values, pd.Series) else values

        if isinstance(cat, pd.Categorical):
            # Re-map the (small) category list, then take through the codes.
            mapped = index.get_indexer(pd.Index(cat.categories))
            self._raise_on_unknown(np.asarray(cat.categories)[mapped < 0])
            codes = np.asarray(cat.codes)
            if (codes < 0).any():
                raise ValueError("feature column contains missing (NaN) values")
            out = mapped[codes]
        else:
            arr = np.asarray(cat)
            out = index.get_indexer(pd.Index(arr))
            self._raise_on_unknown(np.unique(arr[out < 0]))

        return out.astype(self.dtype, copy=False)

    def _raise_on_unknown(self, unknown: NDArray[Any]) -> None:
        if len(unknown):
            shown = ", ".join(map(str, unknown[:5]))
            more = f" (and {len(unknown) - 5} more)" if len(unknown) > 5 else ""
            raise ValueError(
                f"{len(unknown)} feature name(s) are not in the catalog: {shown}{more}. "
                f"Rebuild the catalog from the same data, or pass the full feature list."
            )

    # -- serialization --------------------------------------------------------

    def to_frame(self, expression: Any | None = None) -> pd.DataFrame:
        """Return the catalog as the ``meta_gene.parquet`` table.

        The layout follows Celldega's own ``meta_gene.parquet``, because a client reads it
        by convention rather than through the manifest: the **index** is the gene name,
        and the columns are ``mean``, ``std``, ``max``, ``non-zero`` and ``color``. A
        client with no ``color`` column or no index finds no genes at all, which shows up
        as a viewer with no transcript controls rather than as an error.

        ``feature_code`` and ``is_gene`` are carried alongside as profile additions.

        Parameters
        ----------
        expression
            Optional cell-by-gene matrix (cells x features, in catalog order) used to
            compute the per-gene statistics. Without it the statistics are zero, which is
            valid but leaves the viewer's gene ranking flat.
        """
        import colorsys

        n = len(self.names)
        stats = {k: np.zeros(n, dtype=np.float64) for k in ("mean", "std", "max", "non-zero")}
        if expression is not None:
            stats.update(_expression_stats(expression, n))

        # A hue sweep, matching the look of Celldega's generated palette. Blank/control
        # features are white so they read as "not a gene" in the UI.
        colors = []
        for i, name in enumerate(self.names):
            if i >= self.n_genes or "Blank" in name:
                colors.append("#FFFFFF")
            else:
                r, g, b = colorsys.hsv_to_rgb(i / max(1, self.n_genes), 0.7, 0.9)
                colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

        frame = pd.DataFrame(
            {
                **stats,
                "color": colors,
                "feature_code": np.arange(n, dtype=self.dtype),
                "is_gene": np.arange(n) < self.n_genes,
            },
            # The index is deliberately unnamed: pandas then writes it as
            # '__index_level_0__', which is the only name a client looks for. A named
            # index becomes a column of that name instead, and the gene list reads empty.
            index=pd.Index(list(self.names)),
        )
        # Deliberately NOT sorted. A client builds its integer gene id from a feature's
        # *row position* in this file, and colours transcripts by indexing an array built
        # the same way. Sorting would break the correspondence with feature_code, so most
        # colour lookups would miss and the transcripts would render transparent.
        return frame

    def to_manifest_dict(self) -> dict[str, Any]:
        """Summary for the profile manifest. The full mapping lives in ``meta_gene.parquet``."""
        return {
            "n_features": len(self.names),
            "n_genes": self.n_genes,
            "feature_code_dtype": self.dtype.name,
            "gene_codes_match_cbg_row_groups": True,
        }


def _expression_stats(matrix: Any, n_features: int) -> dict[str, NDArray[np.float64]]:
    """Per-feature mean, std, max and non-zero fraction from a cells x features matrix.

    Computed column-wise on the sparse matrix rather than densifying it, which for a
    5,000-gene panel would otherwise be several GB.
    """
    import scipy.sparse as sp

    stats = {k: np.zeros(n_features, dtype=np.float64) for k in ("mean", "std", "max", "non-zero")}
    if matrix is None:
        return stats

    csc = matrix.tocsc() if sp.issparse(matrix) else sp.csc_matrix(np.asarray(matrix))
    n_cells = csc.shape[0]
    if n_cells == 0:
        return stats

    for col in range(min(n_features, csc.shape[1])):
        values = csc.data[csc.indptr[col] : csc.indptr[col + 1]]
        values = values[values != 0].astype(np.float64)
        if values.size == 0:
            continue
        mean = float(values.sum()) / n_cells
        # Variance over all cells, counting the implicit zeros.
        stats["mean"][col] = mean
        stats["std"][col] = float(np.sqrt(max(0.0, (values**2).sum() / n_cells - mean**2)))
        stats["max"][col] = float(values.max())
        stats["non-zero"][col] = values.size / n_cells
    return stats
