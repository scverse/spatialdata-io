"""Download optional external datasets used by integration tests."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
import requests
from manifest import DATASETS, DatasetAsset, TestDataset, validate_datasets

if TYPE_CHECKING:
    from collections.abc import Sequence


DOWNLOAD_RETRIES = 2


class DatasetDownloadError(RuntimeError):
    """Report that a test dataset could not be downloaded or installed.

    Parameters
    ----------
    dataset
        Manifest entry whose download failed.
    reason
        Human-readable failure detail from the underlying operation.
    """

    def __init__(self, dataset: TestDataset, reason: str) -> None:
        self.dataset = dataset
        self.reason = reason
        super().__init__(f"{dataset.key}: {reason}")


def download_dataset(dataset: TestDataset, output: Path, force: bool) -> None:
    """Download, verify, and install one dataset.

    All network and extraction work is staged in a temporary directory below
    ``output``. An existing destination is changed only after the complete
    staged dataset has been produced.

    Parameters
    ----------
    dataset : TestDataset
        Validated manifest entry describing the source and destination.
    output : Path
        Parent directory where the extracted dataset directory should live.
    force : bool
        If ``True``, replace an existing extracted dataset directory.

    Raises
    ------
    DatasetDownloadError
        If the dataset files cannot be downloaded, extracted, or validated.
    """
    try:
        validate_datasets((dataset,))
        target = output / dataset.extracted_dir
        if (target.exists() or target.is_symlink()) and not force:
            print(f"Skipping {dataset.key}: {target} already exists")
            return

        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{dataset.key}-", dir=output) as tmpdir:
            tmp_path = Path(tmpdir)
            extracted_path = tmp_path / dataset.extracted_dir
            extracted_path.mkdir()

            # Stage work in a temporary directory so interrupted downloads do not leave partial datasets.
            _fetch_dataset(dataset, tmp_path, extracted_path)

            if not any(extracted_path.iterdir()):
                reason = f"download did not produce expected directory {dataset.extracted_dir!r}"
                raise DatasetDownloadError(dataset, reason)

            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            # The temporary directory is below output, so this is a same-filesystem rename in normal operation.
            shutil.move(extracted_path, target)
            print(f"Downloaded {dataset.key} to {target}")
    except DatasetDownloadError:
        raise
    except (requests.exceptions.RequestException, tarfile.TarError, ValueError, zipfile.BadZipFile, OSError) as exc:
        raise DatasetDownloadError(dataset, str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> None:
    """Run the optional test-data downloader command-line interface.

    Parameters
    ----------
    argv
        Arguments without the executable name. When omitted, parse
        ``sys.argv``.

    Raises
    ------
    SystemExit
        With status 1 after all selected datasets have been attempted if one
        or more downloads fail. Argument parsing can also raise ``SystemExit``.
    """
    validate_datasets(DATASETS)
    args = _parse_args(argv)
    if args.list:
        for dataset in DATASETS:
            print(f"{dataset.key}\t{dataset.group}\t{dataset.extracted_dir}\t{dataset.source}")
        return

    failures: list[DatasetDownloadError] = []
    for dataset in _selected_datasets(args.dataset, args.group):
        try:
            download_dataset(dataset, args.output, args.force)
        except DatasetDownloadError as exc:
            failures.append(exc)
            print(f"ERROR: {exc}", file=sys.stderr)

    if failures:
        print(f"Failed to download {len(failures)} dataset(s):", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse downloader selection, output, replacement, and listing options."""
    dataset_keys = sorted(dataset.key for dataset in DATASETS)
    groups = sorted({dataset.group for dataset in DATASETS})
    parser = argparse.ArgumentParser(description="Download optional test datasets used by spatialdata-io CI.")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Directory where datasets are extracted.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=dataset_keys,
        help="Dataset key to download. May be passed multiple times. Defaults to all datasets.",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=groups,
        help="Dataset group to download. May be passed multiple times.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and replace existing selected datasets.")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    return parser.parse_args(argv)


def _selected_datasets(dataset_keys: list[str] | None, groups: list[str] | None) -> list[TestDataset]:
    """Select the union of explicit keys and groups in manifest order."""
    selected_keys = set(dataset_keys or ())
    selected_groups = set(groups or ())
    if not selected_keys and not selected_groups:
        return list(DATASETS)
    return [dataset for dataset in DATASETS if dataset.key in selected_keys or dataset.group in selected_groups]


def _fetch_dataset(dataset: TestDataset, temporary_path: Path, extracted_path: Path) -> None:
    """Fetch and verify one validated DOI, multi-asset, or ZIP source."""
    if dataset.doi:
        manager = pooch.create(
            path=extracted_path,
            base_url=f"doi:{dataset.doi}/",
            registry={},
            retry_if_failed=DOWNLOAD_RETRIES,
        )
        # DOI repositories publish a per-file registry; Pooch loads its hashes and verifies each fetch.
        manager.load_registry_from_doi()
        if not manager.registry_files:
            raise ValueError(f"DOI repository {dataset.doi!r} contains no files")
        for file_name in manager.registry_files:
            manager.fetch(file_name)
        return

    if dataset.assets:
        _fetch_assets(dataset, temporary_path, extracted_path)
        return

    archive_name = f"{dataset.key}.zip"
    manager = pooch.create(
        path=temporary_path,
        base_url="",
        registry={archive_name: dataset.known_hash},
        urls={archive_name: dataset.url},
        retry_if_failed=DOWNLOAD_RETRIES,
    )
    manager.fetch(archive_name, processor=pooch.Unzip(extract_dir=dataset.extracted_dir))


def _fetch_assets(dataset: TestDataset, temporary_path: Path, extracted_path: Path) -> None:
    """Fetch independently hashed files into one staged dataset directory."""
    registry = {f"asset-{index}": asset.known_hash for index, asset in enumerate(dataset.assets)}
    urls = {f"asset-{index}": asset.url for index, asset in enumerate(dataset.assets)}
    manager = pooch.create(
        path=temporary_path,
        base_url="",
        registry=registry,
        urls=urls,
        retry_if_failed=DOWNLOAD_RETRIES,
    )
    for index, asset in enumerate(dataset.assets):
        fetched = manager.fetch(f"asset-{index}")
        if not isinstance(fetched, str):
            raise TypeError(f"Pooch returned an unexpected path collection for dataset asset {index}.")
        _install_asset(Path(fetched), asset, extracted_path)


def _install_asset(downloaded: Path, asset: DatasetAsset, extracted_path: Path) -> None:
    """Install one verified file or safely extract one tar archive.

    Tar extraction uses Python's restrictive data filter, which rejects
    members that would escape the declared destination or create unsafe
    filesystem objects.
    """
    target = extracted_path / asset.target
    if asset.extract:
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(downloaded, mode="r:*") as archive:
            archive.extractall(target, filter="data")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, target)
