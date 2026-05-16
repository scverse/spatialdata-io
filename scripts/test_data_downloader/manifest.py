"""Dataset manifest loading and validation for optional test data downloads."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePath
from typing import Any

DATASETS_TOML = Path(__file__).with_name("datasets.toml")
REQUIRED_FIELDS = frozenset({"key", "group", "url", "archive_name", "extracted_dir", "source"})
OPTIONAL_FIELDS = frozenset({"test_path"})
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


@dataclass(frozen=True)
class TestDataset:
    """Test dataset to be downloaded.

    Parameters
    ----------
    key : str
        Unique identifier for the dataset, used for referencing in tests and
        the CLI.
    group : str
        Logical grouping of the dataset, for example ``"xenium"``,
        ``"visium_hd"``, ``"seqfish"``, or ``"macsima"``.
    url : str
        Direct URL to the dataset archive, for example a ZIP file.
    archive_name : str
        Expected filename of the downloaded archive.
    extracted_dir : str
        Expected name of the directory created when the archive is extracted.
    source : str
        Human-readable description of the dataset source and license.
    test_path : str
        Optional path inside the extracted directory that should be passed to
        integration tests.
    """

    key: str
    group: str
    url: str
    archive_name: str
    extracted_dir: str
    source: str
    test_path: str = ""


def load_datasets(path: str | Path = DATASETS_TOML) -> tuple[TestDataset, ...]:
    """Load dataset entries from a TOML manifest."""
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
    """Validate that dataset entries are internally consistent.

    Parameters
    ----------
    datasets : tuple[TestDataset, ...]
        Dataset entries to validate.

    Raises
    ------
    ValueError
        If a required field is empty, a key or extracted directory is
        duplicated, or ``test_path`` points outside the extracted directory.
    """
    if datasets is None:
        datasets = DATASETS

    seen_keys: set[str] = set()
    seen_extracted_dirs: set[str] = set()

    for dataset in datasets:
        for field_name in REQUIRED_FIELDS:
            value = getattr(dataset, field_name)
            if not isinstance(value, str):
                raise ValueError(f"Dataset {dataset.key!r} field {field_name!r} must be a string.")
            if not value.strip():
                raise ValueError(f"Dataset {dataset.key!r} has empty {field_name}.")
        if not isinstance(dataset.test_path, str):
            raise ValueError(f"Dataset {dataset.key!r} field 'test_path' must be a string.")

        test_path = PurePath(dataset.test_path)
        # Test paths are appended to extracted_dir by tests, so they must stay inside that directory.
        if dataset.test_path and (test_path.is_absolute() or ".." in test_path.parts):
            raise ValueError(f"Dataset {dataset.key!r} test_path must be a relative path inside extracted_dir.")
        if dataset.key in seen_keys:
            raise ValueError(f"Duplicate test dataset key: {dataset.key!r}.")
        if dataset.extracted_dir in seen_extracted_dirs:
            raise ValueError(f"Duplicate test dataset extracted_dir: {dataset.extracted_dir!r}.")
        seen_keys.add(dataset.key)
        seen_extracted_dirs.add(dataset.extracted_dir)


def get_dataset(key: str) -> TestDataset:
    """Return the dataset registered for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not a registered dataset key.
    """
    try:
        return _datasets_by_key()[key]
    except KeyError as exc:
        available = ", ".join(sorted(_datasets_by_key()))
        raise KeyError(f"Unknown test dataset key {key!r}. Available keys: {available}") from exc


def datasets_by_group(group: str) -> tuple[TestDataset, ...]:
    """Return datasets registered for ``group`` in manifest order."""
    return tuple(dataset for dataset in DATASETS if dataset.group == group)


def _parse_dataset(raw_dataset: object, index: int) -> TestDataset:
    if not isinstance(raw_dataset, dict):
        raise ValueError(f"Dataset manifest entry at index {index} must be a table.")

    dataset_fields = set(raw_dataset)
    unknown_fields = dataset_fields - ALLOWED_FIELDS
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Dataset manifest entry at index {index} has unknown field(s): {unknown}.")

    missing_fields = REQUIRED_FIELDS - dataset_fields
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Dataset manifest entry at index {index} is missing required field(s): {missing}.")

    values: dict[str, Any] = dict(raw_dataset)
    values.setdefault("test_path", "")
    dataset = TestDataset(**values)
    validate_datasets((dataset,))
    return dataset


# Keep keys and groups stable because CI workflows and pytest markers may refer to them.
DATASETS = load_datasets()


@cache
def _datasets_by_key() -> dict[str, TestDataset]:
    """Return cached lookup table for dataset keys."""
    return {dataset.key: dataset for dataset in DATASETS}
