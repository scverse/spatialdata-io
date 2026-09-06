"""The visualization profile manifest.

The manifest is what turns a pile of Parquet files into a discoverable profile: it tells a
client the tile geometry, which files hold which row groups, and which columns to project.
Without it a client would have to infer the layout, which is exactly what this profile
exists to avoid.

Key names follow Celldega's existing ``landscape_parameters.json`` so that its reader can
consume the manifest unchanged (``use_row_groups``, ``tile_grid``, ``row_group_files``,
``technology``, ``image_info``). Profile-specific additions live under ``profile`` and in
each entry's column names, so a client that does not understand them still finds the keys
it expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spatialdata_io.experimental.regular_grid import RegularGrid

__all__ = [
    "PROFILE_NAME",
    "PROFILE_VERSION",
    "MANIFEST_FILENAME",
    "build_manifest",
    "validate_manifest",
    "write_manifest",
]

PROFILE_NAME = "celldega_regular_grid_v1"
PROFILE_VERSION = "0.1.0"
MANIFEST_FILENAME = "landscape_parameters.json"


def build_manifest(
    *,
    grid: RegularGrid,
    technology: str = "Xenium",
    transcripts: dict[str, Any] | None = None,
    cell_segmentation: dict[str, Any] | None = None,
    cbg: dict[str, Any] | None = None,
    images: dict[str, Any] | None = None,
    image_info: list[dict[str, Any]] | None = None,
    image_dimensions: dict[str, Any] | None = None,
    max_pyramid_zoom: int | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the profile manifest from the fragments returned by each writer.

    Parameters
    ----------
    grid
        The tile grid shared by transcripts and shapes.
    technology
        Celldega technology string, controlling viewer behaviour such as whether an image
        layer exists.
    transcripts, cell_segmentation, cbg, images
        Fragments returned by the corresponding writers. Omitted entries are simply absent,
        and Celldega skips initialising a reader for them.
    image_info
        Celldega image descriptors. An empty list is valid and means "no image layer".
    source
        Provenance recorded for invalidation: which store and elements this was built from.

    Returns
    -------
    The manifest as a plain dict.
    """
    row_group_files: dict[str, Any] = {}
    if transcripts is not None:
        row_group_files["transcripts"] = transcripts
    if cell_segmentation is not None:
        row_group_files["cell_segmentation"] = cell_segmentation
    if cbg is not None:
        row_group_files["cbg"] = cbg
    row_group_files["images"] = images or {}

    manifest: dict[str, Any] = {
        # -- keys Celldega's existing reader consumes -------------------------
        # Names and shapes follow a DegaFiles landscape_parameters.json so the reader
        # needs no special-casing for a SpatialData store.
        "technology": technology,
        "use_row_groups": True,
        "use_int_index": True,
        "segmentation_approach": ["default"],
        "tile_size": grid.tile_size_px,
        "tile_grid": grid.to_manifest_dict(),
        "row_group_files": row_group_files,
        "image_info": image_info or [],
        "image_format": ".webp",
        # -- profile identification -------------------------------------------
        "profile": PROFILE_NAME,
        "profile_version": PROFILE_VERSION,
    }
    if image_dimensions is not None:
        manifest["image_dimensions"] = image_dimensions
    if max_pyramid_zoom is not None:
        manifest["max_pyramid_zoom"] = max_pyramid_zoom
    if source is not None:
        manifest["source"] = source
    return manifest


def validate_manifest(manifest: dict[str, Any], base_path: str | Path | None = None) -> None:
    """Check a manifest is self-consistent, and that its files exist when ``base_path`` is given.

    Raises
    ------
    ValueError
        With a message naming the specific problem. Failing here is much cheaper than
        failing as a blank viewport in a browser.
    """
    for key in ("technology", "use_row_groups", "tile_grid", "row_group_files"):
        if key not in manifest:
            raise ValueError(f"manifest is missing required key {key!r}")

    grid = RegularGrid.from_manifest_dict(manifest["tile_grid"])
    files = manifest["row_group_files"]
    base = Path(base_path) if base_path is not None else None

    for name in ("transcripts", "cell_segmentation"):
        entry = files.get(name)
        if entry is None:
            continue
        expected = grid.num_tiles
        if entry.get("total_row_groups") != expected:
            raise ValueError(
                f"{name}: total_row_groups={entry.get('total_row_groups')} does not match the "
                f"tile grid ({grid.num_tiles_x} x {grid.num_tiles_y} = {expected} tiles). "
                f"The row-group index formula would address the wrong tiles."
            )
        _check_paths(name, entry, base)

    cbg = files.get("cbg")
    if cbg is not None:
        mapping = cbg.get("gene_to_row_group", {})
        if not mapping:
            raise ValueError("cbg: gene_to_row_group is empty; no gene could be selected")
        if cbg.get("total_row_groups") != len(mapping):
            raise ValueError(
                f"cbg: total_row_groups={cbg.get('total_row_groups')} does not match "
                f"{len(mapping)} entries in gene_to_row_group"
            )
        max_rg = cbg.get("max_row_groups_per_file", 1)
        n_files = len(cbg.get("files", []))
        if mapping and max(mapping.values()) >= n_files * max_rg:
            raise ValueError(
                f"cbg: gene_to_row_group references row group {max(mapping.values())} but only "
                f"{n_files} file(s) x {max_rg} row groups are listed"
            )
        _check_paths("cbg", cbg, base)


def _check_paths(name: str, entry: dict[str, Any], base: Path | None) -> None:
    """Verify the declared files exist, and that the entry names them coherently."""
    if "files" in entry:
        if "directory" not in entry:
            raise ValueError(f"{name}: entry lists 'files' but no 'directory'")
        n_files = len(entry["files"])
        expected = -(-entry["total_row_groups"] // entry["max_row_groups_per_file"])
        if n_files != expected:
            raise ValueError(
                f"{name}: lists {n_files} file(s) but {entry['total_row_groups']} row groups at "
                f"{entry['max_row_groups_per_file']} per file need {expected}"
            )
        if base is not None:
            for f in entry["files"]:
                if not (base / entry["directory"] / f).exists():
                    raise ValueError(f"{name}: declared file {entry['directory']}/{f} does not exist")
    elif "path" in entry:
        if base is not None and not (base / entry["path"]).exists():
            raise ValueError(f"{name}: declared path {entry['path']} does not exist")
    else:
        raise ValueError(f"{name}: entry has neither 'files' nor 'path'")


def write_manifest(manifest: dict[str, Any], directory: str | Path) -> Path:
    """Write the manifest as ``landscape_parameters.json`` and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    return path
