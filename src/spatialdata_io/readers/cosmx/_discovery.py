"""File discovery and dataset-ID inference for CosMx data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata._logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_FLAT_FILE_TARGETS = {"exprMat_file", "fov_positions_file", "metadata_file", "tx_file", "polygons"}
_DIR_TARGETS = [
    "AnalysisResults",
    "CellStatsDir",
    "RunSummary",
    "CellComposite",
    "CellLabels",
    "CellOverlay",
    "CellType_Accessory_Data",
    "CompartmentLabels",
    "ProteinDir",
    "Morphology2D",
    "Morphology3D",
]
_SCAN_TARGETS = list(_FLAT_FILE_TARGETS) + _DIR_TARGETS


def _infer_dataset_id(path: Path, dataset_id: str | None) -> str | None:
    """Infer the dataset ID from marker-file prefixes on disk.

    Handles multimodal layouts (e.g. ``S3RNA`` / ``S3Protein`` → ``S3``).

    Parameters
    ----------
    path
        Root directory to scan.
    dataset_id
        Explicit ID override.  Validated against what is found on disk.

    Returns
    -------
    The inferred or validated dataset ID, or ``None`` if nothing was found.
    """
    suffixes = [".csv", ".csv.gz", ".parquet"]
    found_ids: list[str] = []

    def _base_id(pid: str) -> str:
        for suf in ("RNA", "Protein"):
            if pid.endswith(suf) and len(pid) > len(suf):
                return pid[: -len(suf)]
        return pid

    for marker in _FLAT_FILE_TARGETS:
        for suffix in suffixes:
            for file in path.rglob(f"*{marker}{suffix}"):
                stem = file.name
                for s in suffixes:
                    if stem.endswith(s):
                        stem = stem.removesuffix(s)
                if stem.endswith(f"_{marker}"):
                    prefix = stem.removesuffix(f"_{marker}")
                elif stem.endswith(f"-{marker}"):
                    prefix = stem.removesuffix(f"-{marker}")
                else:
                    continue
                if prefix:
                    found_ids.append(prefix)
                    base = _base_id(prefix)
                    if base and base != prefix:
                        found_ids.append(base)

    unique_ids = list(dict.fromkeys(found_ids))
    unique_bases = list(dict.fromkeys(_base_id(x) for x in found_ids if x))

    if dataset_id is not None:
        if not unique_ids:
            raise ValueError(
                f"Provided dataset_id={dataset_id!r} but no CosMx marker files "
                f"with an ID prefix were found under {path}."
            )
        if dataset_id in unique_ids:
            return dataset_id
        if dataset_id in unique_bases and len(set(unique_bases)) == 1:
            return dataset_id
        raise ValueError(f"Provided dataset_id={dataset_id!r} not among inferred IDs {unique_ids}.")

    if not unique_ids:
        logger.warning("Could not infer dataset_id from marker files under %s.", path)
        return None
    if len(unique_ids) == 1:
        return unique_ids[0]
    if len(set(unique_bases)) == 1 and unique_bases:
        return unique_bases[0]

    raise ValueError(f"Found multiple possible dataset IDs {unique_ids}. Please specify dataset_id=... explicitly.")


# ---------------------------------------------------------------------------
# recursive file scanner
# ---------------------------------------------------------------------------

_WHITELIST_DIRS = {
    "flatFiles",
    "Flatfiles_RNA",
    "Flatfiles_Protein",
    "images",
    "AnalysisResults",
    "AnalysisResults_RNA",
    "AnalysisResults_Protein",
    "CellStatsDir",
    "RunSummary",
    "CellComposite",
    "CellLabels",
    "CellOverlay",
    "CellType_Accessory_Data",
    "CompartmentLabels",
    "ProteinImages",
    "Morphology2D",
    "Morphology3D",
}


def _scan_for_files_to_read(
    path: Path,
    targets: Iterable[str],
    dataset_id: str | None = None,
    max_depth: int = 4,
) -> dict[str, Path | None]:
    """Recursively scan *path* for CosMx data files matching *targets*."""
    found: dict[str, Path | None] = dict.fromkeys(targets)
    base = path.resolve()

    dataset_prefixes: set[str] = set()
    if dataset_id:
        dataset_prefixes = {dataset_id, f"{dataset_id}RNA", f"{dataset_id}Protein"}

    def _strip_all_extensions(p: Path) -> str:
        q = p
        while q.suffix:
            q = q.with_suffix("")
        return q.name

    def _matches_target(stem: str, target: str) -> bool:
        return stem.endswith(f"_{target}") or stem.endswith(f"-{target}") or stem == target

    def _recurse(curr: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in curr.iterdir():
            if dataset_id is not None:
                for t, val in found.items():
                    if val is not None:
                        continue
                    wanted_names = {
                        f"{dataset_id}_{t}",
                        f"{dataset_id}-{t}",
                        f"{dataset_id}RNA_{t}",
                        f"{dataset_id}RNA-{t}",
                        f"{dataset_id}Protein_{t}",
                        f"{dataset_id}Protein-{t}",
                        t,
                    }
                    if child.is_dir() and child.name in wanted_names:
                        found[t] = child
                        break
                    if child.is_file():
                        stem = _strip_all_extensions(child)
                        if stem in wanted_names:
                            found[t] = child
                            break
            for t, val in found.items():
                if val is not None:
                    continue
                if child.is_dir() and child.name == t:
                    found[t] = child
                    break
                if child.is_file() and _matches_target(_strip_all_extensions(child), t):
                    found[t] = child
                    break
            if child.is_dir() and any(v is None for v in found.values()):
                if dataset_prefixes:
                    if child.name not in _WHITELIST_DIRS and not any(
                        child.name.startswith(pref) for pref in dataset_prefixes
                    ):
                        continue
                _recurse(child, depth + 1)
            if all(v is not None for v in found.values()):
                return

    _recurse(base, 0)
    return found


# ---------------------------------------------------------------------------
# multimodal detection
# ---------------------------------------------------------------------------


def _modality_from_ancestors(file: Path, root: Path) -> str | None:
    """Infer a modality name (``RNA`` / ``Protein``) from ancestor directory names."""
    try:
        rel = file.relative_to(root)
    except ValueError:
        return None
    for part in rel.parts[:-1]:
        upper = part.upper()
        if "RNA" in upper and "PROTEIN" not in upper:
            return "RNA"
        if "PROTEIN" in upper:
            return "Protein"
    return None


def _discover_modalities(path: Path, base_id: str | None) -> dict[str, dict[str, Path]]:
    """Detect multiple modality prefixes (e.g. ``S3RNA`` / ``S3Protein``).

    Handles two conventions:

    * **Prefix-based** (V2): filenames encode the modality
      (``S3RNA_exprMat_file`` vs ``S3Protein_exprMat_file``).
    * **Directory-based** (breast multiomics): filenames share the same
      prefix but live under modality-specific directories
      (``Flatfiles_RNA/…/BreastCancer_exprMat_file`` vs
      ``Flatfiles_Protein/…/BreastCancer_exprMat_file``).

    For directory-based layouts the returned dict includes a ``flat_root``
    key pointing to the directory that contains the flat files for that
    modality so the scanner can be scoped correctly.
    """
    modalities: dict[str, dict[str, Path]] = {}
    expr_files = list(path.rglob("*exprMat_file.csv*"))
    for f in expr_files:
        stem = f.name
        for suf in (".csv.gz", ".csv", ".gz"):
            if stem.endswith(suf):
                stem = stem.removesuffix(suf)
        if stem.endswith("_exprMat_file"):
            prefix = stem.removesuffix("_exprMat_file")
        elif stem.endswith("-exprMat_file"):
            prefix = stem.removesuffix("-exprMat_file")
        else:
            continue
        if not prefix:
            continue
        if base_id and not prefix.startswith(base_id):
            continue
        mod = prefix
        if base_id and prefix.startswith(base_id):
            mod = prefix[len(base_id) :].lstrip("_-") or base_id
        # No modality suffix in the filename — try the directory tree.
        flat_root = None
        if mod == base_id:
            dir_mod = _modality_from_ancestors(f, path)
            if dir_mod:
                mod = dir_mod
                flat_root = f.parent
        if mod not in modalities:
            info: dict[str, Path] = {"prefix": prefix, "exprMat_file": f}
            if flat_root is not None:
                info["flat_root"] = flat_root
            modalities[mod] = info
    return modalities


# ---------------------------------------------------------------------------
# high-level dataset setup
# ---------------------------------------------------------------------------


def _label_dir_fallback(
    standalone: Path | None,
    cell_stats_dir: Path | None,
    prefix: str,
) -> Path | None:
    """Return the standalone label directory, or fall back to *cell_stats_dir*.

    Some datasets (e.g. breast multiomics) package ``CellLabels_F*.tif`` and
    ``CompartmentLabels_F*.tif`` inside per-FOV subdirectories of
    ``CellStatsDir/`` instead of separate top-level directories.  When the
    standalone directory is absent we check whether the per-FOV TIFFs exist
    inside *cell_stats_dir* and, if so, return it as the label root.  The
    reader uses ``rglob`` so it finds the TIFFs in either layout.
    """
    if standalone is not None:
        return standalone
    if cell_stats_dir is None:
        return None
    # Quick check: at least one per-FOV TIF present?
    if any(cell_stats_dir.rglob(f"{prefix}_F*.tif")):
        return cell_stats_dir
    return None


def _set_up_cosmx_dataset_for_conversion(
    path: Path,
    dataset_id: str | None = None,
):
    """Build a :class:`CosMxDataset` descriptor from a directory."""
    # import here to avoid circular dependency
    from ._reader import CosMxDataset

    path = path.resolve()
    inferred_id = _infer_dataset_id(path, dataset_id)
    modal_map = _discover_modalities(path, inferred_id)

    def _build_dataset(
        prefix: str | None,
        flat_root: Path | None = None,
        modality: str | None = None,
    ) -> CosMxDataset:
        if flat_root is not None:
            # Directory-based multimodal: scan flat files from the modality
            # subtree, shared directory targets from the dataset root.
            flat_files = _scan_for_files_to_read(
                path=flat_root,
                targets=_FLAT_FILE_TARGETS,
                dataset_id=prefix,
            )
            dir_files = _scan_for_files_to_read(
                path=path,
                targets=_DIR_TARGETS,
                dataset_id=prefix,
            )
            files = {**flat_files, **dir_files}
        else:
            files = _scan_for_files_to_read(path=path, targets=_SCAN_TARGETS, dataset_id=prefix)
        # Fall back to modality-suffixed AnalysisResults directory
        # (e.g. AnalysisResults_RNA, AnalysisResults_Protein).
        analysis_dir = files.get("AnalysisResults")
        if analysis_dir is None and modality is not None:
            candidate = path / f"AnalysisResults_{modality}"
            if candidate.is_dir():
                analysis_dir = candidate
        return CosMxDataset(
            path=path,
            dataset_id=prefix,
            exprMat_file=files.get("exprMat_file"),
            fov_positions_file=files.get("fov_positions_file"),
            metadata_file=files.get("metadata_file"),
            tx_file=files.get("tx_file"),
            polygons_file=files.get("polygons"),
            analysis_results_dir=analysis_dir,
            cell_stats_dir=files.get("CellStatsDir"),
            run_summary_dir=files.get("RunSummary"),
            cell_composite_dir=files.get("CellComposite"),
            cell_labels_dir=_label_dir_fallback(
                files.get("CellLabels"),
                files.get("CellStatsDir"),
                "CellLabels",
            ),
            cell_overlay_dir=files.get("CellOverlay"),
            celltype_accessory_data=files.get("CellType_Accessory_Data"),
            compartment_labels_dir=_label_dir_fallback(
                files.get("CompartmentLabels"),
                files.get("CellStatsDir"),
                "CompartmentLabels",
            ),
            protein_dir=files.get("ProteinDir"),
            morphology_2d_dir=files.get("Morphology2D"),
            morphology_3d_dir=files.get("Morphology3D"),
        )

    if len(modal_map) > 1:
        children = {
            mod_name: _build_dataset(
                info["prefix"],
                flat_root=info.get("flat_root"),
                modality=mod_name,
            )
            for mod_name, info in modal_map.items()
        }
        return CosMxDataset(path=path, dataset_id=inferred_id, modalities=children)

    return _build_dataset(inferred_id)
