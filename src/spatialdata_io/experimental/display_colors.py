"""Assign display colours to genes and categorical cell annotations.

Gene colours use ``uns["gene_colors"]`` in ``var_names`` order. This borrows the
shape of Scanpy's categorical palettes; the gene-specific key and alignment rule
are profile conventions, not an AnnData gene-colour standard. Unstructured lists
are not automatically realigned on variable subset/reorder. Callers must maintain
that association; existing lists are preserved without validation by default.

Cluster colours use ``uns["<column>_colors"]`` in stored category order.
The palette follows the same golden-ratio hue scheme as Celldega's fallback.
"""

from __future__ import annotations

import colorsys
from typing import Any

__all__ = ["add_gene_colors", "add_cluster_colors", "palette", "GENE_COLORS_KEY"]

#: ``uns`` key holding one hex colour per gene, in ``var_names`` order. Follows AnnData's
#: palette naming pattern; the gene-to-position association must be maintained explicitly.
GENE_COLORS_KEY = "gene_colors"

#: Successive hues are separated by the golden ratio, which keeps neighbouring entries
#: visually distinct instead of walking through a smooth ramp where adjacent genes look
#: identical.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

_SATURATION = 0.65
_LIGHTNESS = 0.55


def palette(n: int) -> list[str]:
    """``n`` visually distinct hex colours, deterministic in ``n`` and position."""
    colors = []
    for i in range(n):
        hue = (i * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, _LIGHTNESS, _SATURATION)
        colors.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return colors


def add_gene_colors(table: Any, key: str = GENE_COLORS_KEY, overwrite: bool = False) -> str | None:
    """Add one hex colour per gene to ``table.uns``, in place.

    Colours are ordered by position in ``var``, which is the order a client indexes with
    ``feature_code``. Controls are not in ``var`` and get the client's fallback colour.

    Returns
    -------
    The ``uns`` key, or ``None`` when one already exists and ``overwrite`` is False.
    """
    if key in table.uns and not overwrite:
        return None
    table.uns[key] = palette(table.n_vars)
    return key


def add_cluster_colors(table: Any, column: str, overwrite: bool = False) -> str | None:
    """Add ``uns["<column>_colors"]`` for a categorical ``obs`` column, in place.

    Follows the scanpy convention: one colour per category, in category order.

    Returns
    -------
    The ``uns`` key, or ``None`` when the column is missing or not categorical.
    """
    if column not in table.obs:
        return None

    values = table.obs[column]
    categories = getattr(getattr(values, "cat", None), "categories", None)
    if categories is None:
        return None

    key = f"{column}_colors"
    if key in table.uns and not overwrite:
        return None

    table.uns[key] = palette(len(categories))
    return key
