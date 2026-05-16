from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from manifest import TestDataset as TestDatasetType


def _load_dataset_manifest() -> ModuleType:
    manifest_path = Path(__file__).parents[1] / "scripts" / "test_data_downloader" / "manifest.py"
    spec = importlib.util.spec_from_file_location("manifest", manifest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {manifest_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DATASET_MANIFEST = _load_dataset_manifest()

_READER_MARKS = {
    "codex",
    "cosmx",
    "curio",
    "dbit",
    "generic",
    "iss",
    "macsima",
    "mcmicro",
    "merscope",
    "seqfish",
    "steinbock",
    "stereoseq",
    "visium",
    "visium_hd",
    "xenium",
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the directory containing optional test datasets."""
    return Path(os.environ.get("SPATIALDATA_IO_TEST_DATA_DIR", "data"))


@pytest.fixture
def require_test_dataset(test_data_dir: Path) -> Callable[[str], Path]:
    """Return a dataset path or skip the test if the dataset is unavailable."""

    def _require_test_dataset(dataset_key: str) -> Path:
        dataset = cast("TestDatasetType", _DATASET_MANIFEST.get_dataset(dataset_key))
        path: Path = test_data_dir / dataset.extracted_dir
        if dataset.test_path:
            path = path / dataset.test_path
        if not path.is_dir():
            pytest.skip(
                f"Test data for {dataset_key!r} not found at {path!s}. "
                f"Download it with `uv run python scripts/test_data_downloader --dataset {dataset_key}` or set "
                "SPATIALDATA_IO_TEST_DATA_DIR."
            )
        return path

    return _require_test_dataset


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply scope and reader markers from the test path."""
    for item in items:
        path = Path(str(item.fspath))
        parts = path.parts
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "cli" in parts or "cli" in item.name:
            item.add_marker(pytest.mark.cli)
        if "require_test_dataset" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.data)
        for part in parts:
            if part in _READER_MARKS:
                item.add_marker(getattr(pytest.mark, part))
