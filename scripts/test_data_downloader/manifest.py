"""Dataset manifest loading and validation for optional test data downloads."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

DATASETS_TOML = Path(__file__).with_name("datasets.toml")
COMMON_REQUIRED_FIELDS = ("key", "group", "extracted_dir", "source")
STRING_SOURCE_FIELDS = ("test_path", "url", "known_hash", "doi")
ALLOWED_FIELDS = frozenset(COMMON_REQUIRED_FIELDS) | frozenset(STRING_SOURCE_FIELDS) | {"assets"}
ASSET_REQUIRED_FIELDS = frozenset({"url", "known_hash", "target"})
ASSET_OPTIONAL_FIELDS = frozenset({"extract"})
ASSET_ALLOWED_FIELDS = ASSET_REQUIRED_FIELDS | ASSET_OPTIONAL_FIELDS
KNOWN_HASH_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")
IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """Describe one independently verified file in a multi-asset dataset.

    Parameters
    ----------
    url
        Direct URL from which Pooch downloads the file.
    known_hash
        Expected digest in ``sha256:<hex-digest>`` form.
    target
        Relative destination below the dataset directory. For a regular file,
        this includes the destination filename. For an extracted tar archive,
        this names the extraction directory and may be ``"."``.
    extract
        Whether to treat the downloaded file as a tar archive and safely
        extract it below ``target``.
    """

    url: str
    known_hash: str
    target: str
    extract: bool = False


@dataclass(frozen=True, slots=True)
class TestDataset:
    """Describe an optional integration-test dataset.

    Exactly one download source must be configured: ``doi``; the ``url`` and
    ``known_hash`` archive pair; or one or more ``assets``.

    Parameters
    ----------
    key
        Stable command-line and pytest fixture identifier. Keys use lowercase
        letters, digits, underscores, and hyphens.
    group
        Reader-oriented selection group, such as ``"visium"`` or ``"xenium"``.
    extracted_dir
        Name of the dataset directory created directly below the selected
        downloader output directory.
    source
        Human-readable provenance and licensing description shown by
        ``--list``.
    test_path
        Optional relative directory below ``extracted_dir`` returned by the
        shared ``require_test_dataset`` pytest fixture.
    url
        Direct URL of a ZIP archive for a single-archive source.
    known_hash
        Expected ZIP archive digest in ``sha256:<hex-digest>`` form.
    doi
        Stable repository DOI without a ``doi:`` prefix. Pooch downloads every
        file in the repository's published registry.
    assets
        Independently downloaded and verified files assembled into one dataset.
    """

    key: str
    group: str
    extracted_dir: str
    source: str
    test_path: str = ""
    url: str = ""
    known_hash: str = ""
    doi: str = ""
    assets: tuple[DatasetAsset, ...] = ()


def load_datasets(path: str | Path = DATASETS_TOML) -> tuple[TestDataset, ...]:
    """Load and validate dataset entries from a TOML manifest.

    Parameters
    ----------
    path
        TOML manifest to load.

    Returns
    -------
    tuple of TestDataset
        Validated datasets in manifest order.

    Raises
    ------
    OSError
        If the manifest cannot be read.
    ValueError
        If the TOML is malformed or violates the manifest contract.
    """
    manifest_path = Path(path)
    try:
        raw_manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid dataset manifest TOML in {manifest_path}: {exc}") from exc
    unknown_root_fields = set(raw_manifest) - {"datasets"}
    if unknown_root_fields:
        unknown = ", ".join(sorted(unknown_root_fields))
        raise ValueError(f"Dataset manifest has unknown root field(s): {unknown}.")
    raw_datasets = raw_manifest.get("datasets")
    if not isinstance(raw_datasets, list):
        raise ValueError("Dataset manifest must define a [[datasets]] array.")
    datasets = tuple(_parse_dataset(raw_dataset, index) for index, raw_dataset in enumerate(raw_datasets))
    validate_datasets(datasets)
    return datasets


def validate_datasets(datasets: tuple[TestDataset, ...] | None = None) -> None:
    """Validate dataset identity, destination, and download-source contracts.

    Parameters
    ----------
    datasets
        Dataset entries to validate. The module-level manifest is validated
        when omitted.

    Raises
    ------
    ValueError
        If an entry is incomplete, ambiguous, unsafe, or duplicates another
        dataset key or destination.
    """
    if datasets is None:
        datasets = DATASETS
    seen_keys: set[str] = set()
    seen_extracted_dirs: set[str] = set()
    for dataset in datasets:
        _validate_dataset_strings(dataset)
        _validate_identifier(dataset.key, dataset_key=dataset.key, field="key")
        _validate_identifier(dataset.group, dataset_key=dataset.key, field="group")
        _validate_relative_path(
            dataset.key,
            "extracted_dir",
            dataset.extracted_dir,
            allow_current=False,
            require_single_component=True,
        )
        _validate_dataset_source(dataset)
        _validate_relative_path(dataset.key, "test_path", dataset.test_path, allow_current=True)
        if dataset.key in seen_keys:
            raise ValueError(f"Duplicate test dataset key: {dataset.key!r}.")
        if dataset.extracted_dir in seen_extracted_dirs:
            raise ValueError(f"Duplicate test dataset extracted_dir: {dataset.extracted_dir!r}.")
        seen_keys.add(dataset.key)
        seen_extracted_dirs.add(dataset.extracted_dir)


def get_dataset(key: str) -> TestDataset:
    """Return the registered dataset identified by ``key``.

    Parameters
    ----------
    key
        Dataset key from the manifest.

    Returns
    -------
    TestDataset
        Matching manifest entry.

    Raises
    ------
    KeyError
        If no registered dataset has ``key``.
    """
    try:
        return _datasets_by_key()[key]
    except KeyError as exc:
        available = ", ".join(sorted(_datasets_by_key()))
        raise KeyError(f"Unknown test dataset key {key!r}. Available keys: {available}") from exc


def datasets_by_group(group: str) -> tuple[TestDataset, ...]:
    """Return datasets in a selection group.

    Parameters
    ----------
    group
        Exact manifest group to select.

    Returns
    -------
    tuple of TestDataset
        Matching datasets in manifest order, or an empty tuple when the group
        is unknown.
    """
    return tuple(dataset for dataset in DATASETS if dataset.group == group)


def _validate_dataset_strings(dataset: TestDataset) -> None:
    for field_name in COMMON_REQUIRED_FIELDS:
        value = getattr(dataset, field_name)
        if not isinstance(value, str):
            raise ValueError(f"Dataset {dataset.key!r} field {field_name!r} must be a string.")
        if not value.strip():
            raise ValueError(f"Dataset {dataset.key!r} has empty {field_name}.")
    for field_name in STRING_SOURCE_FIELDS:
        if not isinstance(getattr(dataset, field_name), str):
            raise ValueError(f"Dataset {dataset.key!r} field {field_name!r} must be a string.")


def _validate_dataset_source(dataset: TestDataset) -> None:
    if not isinstance(dataset.assets, tuple):
        raise ValueError(f"Dataset {dataset.key!r} assets must be a tuple of DatasetAsset values.")
    has_archive = bool(dataset.url.strip() or dataset.known_hash.strip())
    has_doi = bool(dataset.doi.strip())
    has_assets = bool(dataset.assets)
    if sum((has_archive, has_doi, has_assets)) != 1:
        raise ValueError(
            f"Dataset {dataset.key!r} must define exactly one source: an archive, a DOI, or an assets array."
        )
    if has_archive:
        if not dataset.url.strip() or not dataset.known_hash.strip():
            raise ValueError(f"Dataset {dataset.key!r} archive requires both url and known_hash.")
        _validate_known_hash(dataset.key, dataset.known_hash)
    if has_doi and (not dataset.doi.startswith("10.") or any(character.isspace() for character in dataset.doi)):
        raise ValueError(f"Dataset {dataset.key!r} doi must be a DOI without a 'doi:' prefix.")
    if has_assets:
        _validate_assets(dataset)


def _validate_assets(dataset: TestDataset) -> None:
    seen_targets: set[str] = set()
    for index, asset in enumerate(dataset.assets):
        if not isinstance(asset, DatasetAsset):
            raise ValueError(f"Dataset {dataset.key!r} asset {index} must be a DatasetAsset.")
        if not isinstance(asset.url, str):
            raise ValueError(f"Dataset {dataset.key!r} asset {index} url must be a string.")
        if not isinstance(asset.known_hash, str):
            raise ValueError(f"Dataset {dataset.key!r} asset {index} known_hash must be a string.")
        if not isinstance(asset.target, str):
            raise ValueError(f"Dataset {dataset.key!r} asset {index} target must be a string.")
        if not isinstance(asset.extract, bool):
            raise ValueError(f"Dataset {dataset.key!r} asset {index} extract must be a boolean.")
        if not asset.url.strip():
            raise ValueError(f"Dataset {dataset.key!r} asset {index} has empty url.")
        _validate_known_hash(f"{dataset.key} asset {index}", asset.known_hash)
        _validate_relative_path(dataset.key, f"asset {index} target", asset.target, allow_current=asset.extract)
        if asset.target in seen_targets:
            raise ValueError(f"Dataset {dataset.key!r} has duplicate asset target {asset.target!r}.")
        seen_targets.add(asset.target)


def _validate_identifier(value: str, *, dataset_key: str, field: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"Dataset {dataset_key!r} {field} must contain only lowercase letters, digits, underscores, and hyphens."
        )


def _validate_relative_path(
    dataset_key: str,
    field: str,
    value: str,
    *,
    allow_current: bool,
    require_single_component: bool = False,
) -> None:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    paths = (posix_path, windows_path)
    if value and (any(path.is_absolute() for path in paths) or any(".." in path.parts for path in paths)):
        raise ValueError(f"Dataset {dataset_key!r} {field} must be a relative path inside extracted_dir.")
    if value in {"", "."} and not allow_current:
        raise ValueError(f"Dataset {dataset_key!r} {field} must name a destination inside extracted_dir.")
    if require_single_component and any(len(path.parts) != 1 for path in paths):
        raise ValueError(f"Dataset {dataset_key!r} {field} must be a single directory name.")


def _parse_dataset(raw_dataset: object, index: int) -> TestDataset:
    if not isinstance(raw_dataset, dict):
        raise ValueError(f"Dataset manifest entry at index {index} must be a table.")
    dataset_fields = set(raw_dataset)
    unknown_fields = dataset_fields - ALLOWED_FIELDS
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Dataset manifest entry at index {index} has unknown field(s): {unknown}.")
    missing_fields = set(COMMON_REQUIRED_FIELDS) - dataset_fields
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Dataset manifest entry at index {index} is missing required field(s): {missing}.")
    values: dict[str, Any] = dict(raw_dataset)
    raw_assets = values.pop("assets", [])
    for field_name in STRING_SOURCE_FIELDS:
        values.setdefault(field_name, "")
    values["assets"] = _parse_assets(raw_assets, dataset_index=index)
    dataset = TestDataset(**values)
    validate_datasets((dataset,))
    return dataset


def _parse_assets(raw_assets: object, *, dataset_index: int) -> tuple[DatasetAsset, ...]:
    if not isinstance(raw_assets, list):
        raise ValueError(f"Dataset manifest entry at index {dataset_index} assets must be an array of tables.")
    parsed: list[DatasetAsset] = []
    for asset_index, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise ValueError(f"Dataset {dataset_index} asset {asset_index} must be a table.")
        fields = set(raw_asset)
        unknown = fields - ASSET_ALLOWED_FIELDS
        missing = ASSET_REQUIRED_FIELDS - fields
        if unknown:
            raise ValueError(f"Dataset {dataset_index} asset {asset_index} has unknown field(s): {sorted(unknown)}.")
        if missing:
            raise ValueError(f"Dataset {dataset_index} asset {asset_index} is missing field(s): {sorted(missing)}.")
        values = dict(raw_asset)
        values.setdefault("extract", False)
        parsed.append(DatasetAsset(**values))
    return tuple(parsed)


def _validate_known_hash(dataset_key: str, known_hash: str) -> None:
    if not KNOWN_HASH_PATTERN.fullmatch(known_hash):
        raise ValueError(f"Dataset {dataset_key!r} known_hash must be formatted as 'sha256:<64 hex characters>'.")


DATASETS = load_datasets()


@cache
def _datasets_by_key() -> dict[str, TestDataset]:
    return {dataset.key: dataset for dataset in DATASETS}
