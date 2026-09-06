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
            except Exception:
                # Unknown categories (typical straight after read_zarr): realize just the
                # category list, which is tiny, rather than computing the whole column.
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

    def to_frame(self) -> pd.DataFrame:
        """Return the catalog as the ``meta_gene.parquet`` table."""
        return pd.DataFrame(
            {
                "name": list(self.names),
                "feature_code": np.arange(len(self.names), dtype=self.dtype),
                "is_gene": np.arange(len(self.names)) < self.n_genes,
            }
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        """Summary for the profile manifest. The full mapping lives in ``meta_gene.parquet``."""
        return {
            "n_features": len(self.names),
            "n_genes": self.n_genes,
            "feature_code_dtype": self.dtype.name,
            "gene_codes_match_cbg_row_groups": True,
        }
