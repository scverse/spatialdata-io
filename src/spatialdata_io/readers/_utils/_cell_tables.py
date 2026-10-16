"""Cell-table adapters shared by the Akoya readers: read a table into a `CellTable` and build the table elements."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import readfcs
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity

from spatialdata_io._constants._constants import CodexKeys, PhenocyclerKeys
from spatialdata_io.readers._utils._akoya import dedupe_channel_names

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from geopandas import GeoDataFrame
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

__all__ = ["CellTable", "build_cells", "read_codex_table"]

# `cyc001_ch001` (FCS `$PnN`, plain CSV header) or the name part of the Processor CSV header `cyc001_ch001:DAPI1`
_MARKER_COLUMN = re.compile(r"cyc(\d+)_ch(\d+)", re.IGNORECASE)
# non-marker columns of CODEX Processor tables, for tables whose marker columns are named after the markers
_META_COLUMNS = frozenset(
    {"cell_id", "region", "tile_num", "x", "y", "z", "x_tile", "y_tile", "size", "tsne_x", "tsne_y", "homogeneity"}
)


@dataclass
class CellTable:
    """A cell table in reader-independent form.

    Attributes
    ----------
    obs
        The non-marker columns, indexed by the string instance ids.
    X
        The marker intensities, ``(n_cells, n_markers)`` float32.
    var
        Per-marker metadata indexed by the de-duplicated marker names: ``source_column`` and, for
        ``cycCCC_chHHH`` columns, ``cycle`` and ``channel``.
    region
        The region id of every cell.
    instance_id
        The instance id of every cell (int64).
    centroids_px
        The ``(x, y)`` centroids in level-0 pixels of the region image.
    radius_px
        The radius of a circle with the cell's area, or `None` if the table has no area.
    """

    obs: pd.DataFrame
    X: np.ndarray
    var: pd.DataFrame
    region: pd.Series
    instance_id: pd.Series
    centroids_px: np.ndarray
    radius_px: np.ndarray | None

    def take(self, rows: np.ndarray) -> CellTable:
        """The cells at the positional indices ``rows``."""
        return CellTable(
            obs=self.obs.iloc[rows],
            X=self.X[rows],
            var=self.var,
            region=self.region.iloc[rows],
            instance_id=self.instance_id.iloc[rows],
            centroids_px=self.centroids_px[rows],
            radius_px=None if self.radius_px is None else self.radius_px[rows],
        )


def read_codex_table(path: Path, region: str | None = None) -> CellTable:
    """Read a CODEX Processor ``.fcs`` or ``.csv`` cell table.

    Marker columns are the ``cycCCC_chHHH`` columns; their names come from the FCS ``$PnS`` keywords or from the
    display part of the Processor CSV header ``cycCCC_chHHH:<marker>``. A table without such columns has one
    marker column per numeric column that is not a known CODEX Processor column (``x``, ``size``, ...).

    Parameters
    ----------
    path
        The ``.fcs`` or ``.csv`` file.
    region
        The region id of all cells if the table has no ``region`` column. A numeric ``region`` column ``r`` is
        mapped to the region id ``regRRR``.

    Returns
    -------
    The cell table.
    """
    if path.suffix.lower() == CodexKeys.FCS_FILE.value:
        fcs = readfcs.ReadFCS(path)
        df = fcs.data  # positional columns, named by `$PnN`
        names = list(fcs.channels["PnN"])
        labels = list(fcs.channels["PnS"]) if "PnS" in fcs.channels.columns else [""] * len(names)
    else:
        df = pd.read_csv(path, engine="pyarrow")
        names = [str(column) for column in df.columns]
        labels = [""] * len(names)
    sources = names
    # the Processor CSV header form `cyc001_ch001:DAPI1` (also `cell_id:cell_id`)
    split = [name.partition(":") for name in names]
    df.columns = pd.Index([name for name, _, _ in split])
    displays = [label.strip() or display or name for (name, _, display), label in zip(split, labels, strict=True)]

    marker_matches = [_MARKER_COLUMN.fullmatch(column) for column in df.columns]
    if any(marker_matches):
        is_marker = [match is not None for match in marker_matches]
    else:
        is_marker = [
            column.casefold() not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
            for column in df.columns
        ]
    markers = [i for i, marker in enumerate(is_marker) if marker]
    var = pd.DataFrame(
        {"source_column": [sources[i] for i in markers]},
        index=pd.Index(dedupe_channel_names([displays[i] for i in markers])),
    )
    if any(marker_matches):
        var["cycle"] = [int(marker_matches[i].group(1)) for i in markers]  # type: ignore[union-attr]
        var["channel"] = [int(marker_matches[i].group(2)) for i in markers]  # type: ignore[union-attr]
    X = df.iloc[:, markers].to_numpy(dtype=np.float32)
    obs = df.iloc[:, [i for i, marker in enumerate(is_marker) if not marker]]

    columns = {column.casefold(): column for column in obs.columns}
    if "x" not in columns or "y" not in columns:
        raise ValueError(f"{path} has no 'x' and 'y' centroid columns; found {list(obs.columns)}.")
    centroids = obs[[columns["x"], columns["y"]]].to_numpy(dtype=np.float64)
    radius = np.sqrt(obs[columns["size"]].to_numpy(dtype=np.float64) / np.pi) if "size" in columns else None

    instance_key = CodexKeys.INSTANCE_KEY.value
    if instance_key in columns:
        instance_id = obs[columns[instance_key]].astype(np.int64)
    else:
        # label images use 0 for the background, so instance ids start at 1
        instance_id = pd.Series(np.arange(1, len(obs) + 1, dtype=np.int64))
    instance_id.index = pd.Index(instance_id.astype(str))
    obs.index = instance_id.index
    instance_id.name = instance_key

    region_key = CodexKeys.REGION_KEY.value
    if region_key in columns:
        regions = obs[columns[region_key]]
        if pd.api.types.is_numeric_dtype(regions):
            ids = {value: f"reg{int(value):03d}" for value in regions.unique()}
            regions = regions.map(ids)
        region_ids = regions.astype(str)
    elif region is not None:
        region_ids = pd.Series(region, index=obs.index)
    else:
        raise ValueError(
            f"{path} has no '{region_key}' column and no region id in its file name, so its cells cannot be "
            "assigned to a region."
        )
    region_ids.name = region_key
    return CellTable(
        obs=obs,
        X=X,
        var=var,
        region=region_ids,
        instance_id=instance_id,
        centroids_px=centroids,
        radius_px=radius,
    )


def build_cells(
    tables: Sequence[CellTable],
    region_elements: Mapping[str, str],
    labels: Mapping[str, DataArray | DataTree],
    transformations: Mapping[str, BaseTransformation],
) -> tuple[dict[str, ad.AnnData], dict[str, GeoDataFrame]]:
    """Build the table elements and, for regions without labels, circle shapes from cell tables.

    Parameters
    ----------
    tables
        The cell tables, each covering one or more regions.
    region_elements
        The element each region's cells annotate. Regions missing from it annotate ``<region>_labels`` if they are
        in ``labels``, else the circles ``<region>_cells`` built here.
    labels
        The labels elements by region id; their regions get no circles.
    transformations
        The transformation to ``"global"`` of each region's image; missing regions get ``Identity``.

    Returns
    -------
    The table elements, ``{"table": ...}`` if all tables have the same markers, else ``{"<region>_table": ...}``
    per region, and the circle shapes by element name. The circles have the cells' area (radius 1 px without one).
    """
    _check_disjoint_regions(tables)
    pieces = _split_by_region(tables) if len({tuple(table.var.index) for table in tables}) > 1 else None

    def element(region: str) -> str:
        if region in region_elements:
            return region_elements[region]
        suffix = PhenocyclerKeys.LABELS_SUFFIX if region in labels else PhenocyclerKeys.CELLS_SUFFIX
        return f"{region}{suffix}"

    shapes = {}
    circle_regions = []
    for table in tables:
        for region, rows in _region_rows(table).items():
            if region in labels:
                continue
            circle_regions.append(region)
            radius = 1.0 if table.radius_px is None else table.radius_px[rows]
            shapes[element(region)] = ShapesModel.parse(
                table.centroids_px[rows],
                geometry=0,
                radius=radius,
                index=pd.Index(table.instance_id.to_numpy()[rows]),
                transformations={"global": transformations.get(region, Identity())},
            )
    if circle_regions:
        warnings.warn(
            f"No segmentation mask found for region(s) {sorted(circle_regions)}; cells are represented as circles.",
            UserWarning,
            stacklevel=3,
        )

    if pieces is None:
        return {PhenocyclerKeys.TABLE_KEY.value: _to_anndata(tables, element)}, shapes
    return {f"{region}_table": _to_anndata([piece], element) for region, piece in sorted(pieces.items())}, shapes


def _region_rows(table: CellTable) -> dict[str, np.ndarray]:
    """The row indices of each region of ``table``, in order of first appearance."""
    codes, regions = pd.factorize(table.region)
    return {str(region): np.flatnonzero(codes == i) for i, region in enumerate(regions)}


def _check_disjoint_regions(tables: Sequence[CellTable]) -> None:
    seen: set[str] = set()
    for table in tables:
        regions = set(table.region.unique())
        if overlap := seen & regions:
            raise ValueError(f"Several cell tables contain the cells of region(s) {sorted(overlap)}.")
        seen |= regions


def _split_by_region(tables: Sequence[CellTable]) -> dict[str, CellTable]:
    return {region: table.take(rows) for table in tables for region, rows in _region_rows(table).items()}


def _to_anndata(tables: Sequence[CellTable], element: Callable[[str], str]) -> ad.AnnData:
    """Concatenate cell tables with the same markers into one table annotating the elements of their regions."""
    region_key, instance_key = CodexKeys.REGION_KEY.value, CodexKeys.INSTANCE_KEY.value
    obs = pd.concat([table.obs for table in tables]) if len(tables) > 1 else tables[0].obs.copy()
    regions = pd.concat([table.region for table in tables]) if len(tables) > 1 else tables[0].region
    elements = regions.map({region: element(region) for region in regions.unique()}).to_numpy()
    obs[region_key] = pd.Categorical(elements, categories=sorted(set(elements)))
    obs[instance_key] = np.concatenate([table.instance_id.to_numpy() for table in tables])
    if not obs.index.is_unique:
        obs.index = pd.Index(regions.to_numpy(dtype=object) + "_" + obs.index.to_numpy(dtype=object))
    X = np.concatenate([table.X for table in tables]) if len(tables) > 1 else tables[0].X
    adata = ad.AnnData(X=X, obs=obs, var=tables[0].var.copy())
    adata.obsm[CodexKeys.SPATIAL_KEY.value] = np.concatenate([table.centroids_px for table in tables])
    return TableModel.parse(
        adata,
        region=list(obs[region_key].cat.categories),
        region_key=region_key,
        instance_key=instance_key,
    )
