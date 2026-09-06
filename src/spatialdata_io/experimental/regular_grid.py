"""Deterministic regular-grid spatial tiling.

This module defines the tile geometry and the tile -> row-group -> file numbering
used by the ``celldega_regular_grid_v1`` visualization profile.

The grid is a non-overlapping regular square grid in *display pixel* space (level-0
pixels of a chosen reference image). Given an origin, a tile size and grid dimensions,
a point's tile is a pure function of its coordinates::

    tile_x = floor((x_px - origin_x) / tile_size_px)
    tile_y = floor((y_px - origin_y) / tile_size_px)
    tile_id = tile_x * num_tiles_y + tile_y

Tile bounds are half-open ``[min, max)``, except at the upper edge of the dataset where
the last tile is closed so that points lying exactly on ``x_max`` / ``y_max`` are kept.

``tile_id`` doubles as the *global row-group index*: one logical tile is written as
exactly one Parquet row group, including empty tiles, which are written as zero-row row
groups. Row groups are then split across files::

    file_index       = tile_id // max_row_groups_per_file
    local_row_group  = tile_id %  max_row_groups_per_file

Multi-file output is deliberate: a Parquet reader must fetch a file's entire footer
before it can read any row group, and footer size grows with row-group count. Splitting
keeps the browser's cold-start cost proportional to the viewport rather than to the
whole dataset.

This numbering matches Celldega's ``RowGroupTileReader`` exactly, but nothing here is
Celldega-specific -- it is the viewer-independent part of the profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["RegularGrid", "DEFAULT_MAX_ROW_GROUPS_PER_FILE"]

#: Default number of row groups per Parquet file. Matches Celldega's writer default.
DEFAULT_MAX_ROW_GROUPS_PER_FILE = 400


@dataclass(frozen=True)
class RegularGrid:
    """A deterministic non-overlapping regular square grid in display-pixel space.

    Parameters
    ----------
    origin_x
        X coordinate (display pixels) of the grid origin, i.e. the left edge of column 0.
    origin_y
        Y coordinate (display pixels) of the grid origin, i.e. the top edge of row 0.
    tile_size_px
        Edge length of a tile, in display pixels. Must be positive.
    num_tiles_x
        Number of tile columns. Must be positive.
    num_tiles_y
        Number of tile rows. Must be positive.
    """

    origin_x: float
    origin_y: float
    tile_size_px: float
    num_tiles_x: int
    num_tiles_y: int

    def __post_init__(self) -> None:
        if not self.tile_size_px > 0:
            raise ValueError(f"tile_size_px must be positive, got {self.tile_size_px}")
        if self.num_tiles_x < 1 or self.num_tiles_y < 1:
            raise ValueError(
                f"grid must have at least one tile per axis, got "
                f"num_tiles_x={self.num_tiles_x}, num_tiles_y={self.num_tiles_y}"
            )

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_bounds(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        tile_size_px: float,
    ) -> RegularGrid:
        """Build the smallest grid with the given tile size that covers ``[x_min, x_max] x [y_min, y_max]``.

        The origin is placed at ``(x_min, y_min)``. The grid always has at least one tile
        per axis, so a degenerate (zero-extent) bounding box still yields a valid grid.
        """
        if x_max < x_min or y_max < y_min:
            raise ValueError(f"invalid bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        num_tiles_x = max(1, math.ceil((x_max - x_min) / tile_size_px))
        num_tiles_y = max(1, math.ceil((y_max - y_min) / tile_size_px))
        return cls(
            origin_x=float(x_min),
            origin_y=float(y_min),
            tile_size_px=float(tile_size_px),
            num_tiles_x=int(num_tiles_x),
            num_tiles_y=int(num_tiles_y),
        )

    # -- geometry -------------------------------------------------------------

    @property
    def num_tiles(self) -> int:
        """Total number of logical tiles, which equals the total number of row groups."""
        return self.num_tiles_x * self.num_tiles_y

    @property
    def x_max(self) -> float:
        """Right edge of the grid in display pixels."""
        return self.origin_x + self.num_tiles_x * self.tile_size_px

    @property
    def y_max(self) -> float:
        """Bottom edge of the grid in display pixels."""
        return self.origin_y + self.num_tiles_y * self.tile_size_px

    def tile_xy(
        self, x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Map display-pixel coordinates to ``(tile_x, tile_y)`` indices.

        Coordinates on the upper edge of the grid are clamped into the last tile, so that
        a point at exactly ``x_max`` belongs to tile column ``num_tiles_x - 1`` rather
        than falling outside the grid.

        Raises
        ------
        ValueError
            If any coordinate falls below the grid origin, or more than one tile beyond
            the grid extent. Such points indicate a mismatched grid rather than a
            rounding artefact, and are not silently clamped.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

        tx = np.floor((x - self.origin_x) / self.tile_size_px).astype(np.int64)
        ty = np.floor((y - self.origin_y) / self.tile_size_px).astype(np.int64)

        self._check_in_range(tx, "x", self.num_tiles_x)
        self._check_in_range(ty, "y", self.num_tiles_y)

        # Clamp the closed upper edge into the last tile.
        np.clip(tx, 0, self.num_tiles_x - 1, out=tx)
        np.clip(ty, 0, self.num_tiles_y - 1, out=ty)
        return tx, ty

    def _check_in_range(self, t: NDArray[np.int64], axis: str, num_tiles: int) -> None:
        if t.size == 0:
            return
        lo = int(t.min())
        hi = int(t.max())
        # `hi == num_tiles` is the closed upper edge and is clamped; anything beyond is an error.
        if lo < 0 or hi > num_tiles:
            raise ValueError(
                f"{axis} coordinates map to tile indices [{lo}, {hi}], outside the grid "
                f"[0, {num_tiles - 1}] (upper edge {num_tiles} allowed). "
                f"The grid does not match the data; check origin, tile size and the "
                f"micron-to-pixel transform."
            )

    def tile_id(self, tile_x: NDArray[np.int64], tile_y: NDArray[np.int64]) -> NDArray[np.int64]:
        """Combine tile indices into a global tile id (x-major: ``tile_x * num_tiles_y + tile_y``)."""
        return np.asarray(tile_x, dtype=np.int64) * self.num_tiles_y + np.asarray(tile_y, dtype=np.int64)

    def assign(self, x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> NDArray[np.int64]:
        """Map display-pixel coordinates directly to global tile ids."""
        tx, ty = self.tile_xy(x, y)
        return self.tile_id(tx, ty)

    def tile_bounds(self, tile_x: int, tile_y: int) -> tuple[float, float, float, float]:
        """Return the half-open bounds ``(x_min, y_min, x_max, y_max)`` of one tile."""
        if not (0 <= tile_x < self.num_tiles_x and 0 <= tile_y < self.num_tiles_y):
            raise ValueError(f"tile ({tile_x}, {tile_y}) is outside the grid")
        x0 = self.origin_x + tile_x * self.tile_size_px
        y0 = self.origin_y + tile_y * self.tile_size_px
        return (x0, y0, x0 + self.tile_size_px, y0 + self.tile_size_px)

    # -- row-group / file numbering -------------------------------------------

    def chunk_location(
        self, tile_id: int, max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE
    ) -> tuple[int, int]:
        """Return ``(file_index, local_row_group_index)`` for a global tile id."""
        return divmod(int(tile_id), int(max_row_groups_per_file))

    def num_files(self, max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE) -> int:
        """Number of Parquet files needed to hold every tile's row group."""
        return math.ceil(self.num_tiles / max_row_groups_per_file)

    def chunk_filenames(
        self,
        max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
        prefix: str = "chunk",
    ) -> list[str]:
        """Return the ordered chunk filenames, zero-padded so lexicographic == numeric order.

        Padding matters because the two consumers disagree about ordering: Celldega indexes
        the manifest's ``files`` array by position, while dask's ``read_parquet`` globs a
        directory and sorts lexicographically. Without padding, ``chunk_10`` would sort
        before ``chunk_2`` and dask would silently reorder partitions.
        """
        n = self.num_files(max_row_groups_per_file)
        width = len(str(n - 1)) if n > 1 else 1
        return [f"{prefix}_{i:0{width}d}.parquet" for i in range(n)]

    # -- serialization --------------------------------------------------------

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize the grid into the ``tile_grid`` block of the profile manifest.

        Uses Celldega's established ``landscape_parameters.json`` key names so that an
        existing reader can consume it unchanged.
        """
        return {
            "num_tiles_x": self.num_tiles_x,
            "num_tiles_y": self.num_tiles_y,
            "tile_size": self.tile_size_px,
            "x_min": self.origin_x,
            "y_min": self.origin_y,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @classmethod
    def from_manifest_dict(cls, d: dict[str, Any]) -> RegularGrid:
        """Rebuild a grid from its manifest representation."""
        missing = {"num_tiles_x", "num_tiles_y", "tile_size", "x_min", "y_min"} - set(d)
        if missing:
            raise ValueError(f"tile_grid is missing required keys: {sorted(missing)}")
        return cls(
            origin_x=float(d["x_min"]),
            origin_y=float(d["y_min"]),
            tile_size_px=float(d["tile_size"]),
            num_tiles_x=int(d["num_tiles_x"]),
            num_tiles_y=int(d["num_tiles_y"]),
        )
