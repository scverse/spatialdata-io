"""Download optional external datasets used by integration tests."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from manifest import DATASETS, TestDataset, validate_datasets

if TYPE_CHECKING:
    from collections.abc import Sequence


# Some providers reject Python's default user agent even for public files.
REQUEST_HEADERS = {
    "User-Agent": "curl/8.0.0",
    "Accept": "*/*",
}
DOWNLOAD_TIMEOUT_SEC = 60
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SEC = 2.0
TRANSIENT_HTTP_STATUS_CODES = {408, 429}


class DatasetDownloadError(RuntimeError):
    """Error raised when a test dataset cannot be downloaded or extracted."""

    def __init__(self, dataset: TestDataset, action: str, reason: str, suggestion: str | None = None) -> None:
        self.dataset = dataset
        self.action = action
        self.reason = reason
        self.suggestion = suggestion
        message = f"{dataset.key}: failed to {action} {dataset.url}: {reason}"
        if suggestion is not None:
            message = f"{message}. {suggestion}"
        super().__init__(message)


def download_dataset(dataset: TestDataset, output: Path, force: bool) -> None:
    """Download and extract a single dataset.

    Parameters
    ----------
    dataset : TestDataset
        Manifest entry describing the dataset archive and expected output
        directory.
    output : Path
        Parent directory where the extracted dataset directory should live.
    force : bool
        If ``True``, replace an existing extracted dataset directory.

    Raises
    ------
    DatasetDownloadError
        If the archive cannot be downloaded, extracted, or validated.
    """
    target = output / dataset.extracted_dir
    if target.exists() and not force:
        print(f"Skipping {dataset.key}: {target} already exists")
        return

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{dataset.key}-", dir=output) as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / dataset.archive_name
        extracted_path = tmp_path / dataset.extracted_dir
        extracted_path.mkdir()

        # Stage work in a temporary directory so interrupted downloads do not leave partial datasets.
        _download(dataset, archive_path)
        _extract_zip(dataset, archive_path, extracted_path)
        if not extracted_path.is_dir() or not any(extracted_path.iterdir()):
            raise DatasetDownloadError(
                dataset,
                "extract",
                f"archive did not produce expected directory {dataset.extracted_dir!r}",
            )

        if target.exists():
            shutil.rmtree(target)
        # Move the fully validated extraction into place only after all checks pass.
        shutil.move(str(extracted_path), target)
        print(f"Downloaded {dataset.key} to {target}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset downloader command-line interface."""
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
            print(f"ERROR: {exc}")

    if failures:
        print(f"Failed to download {len(failures)} dataset(s):")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for selecting and downloading datasets."""
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
    """Return manifest entries selected by explicit keys, groups, or all datasets."""
    selected_keys = set(dataset_keys or ())
    selected_groups = set(groups or ())
    if not selected_keys and not selected_groups:
        return list(DATASETS)
    return [dataset for dataset in DATASETS if dataset.key in selected_keys or dataset.group in selected_groups]


def _is_transient_error(error: BaseException) -> bool:
    """Return whether a download error is worth retrying."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP_STATUS_CODES or 500 <= error.code < 600
    return isinstance(error, urllib.error.URLError | TimeoutError)


def _download_error(dataset: TestDataset, error: BaseException) -> DatasetDownloadError:
    """Translate urllib and timeout failures into dataset-aware errors."""
    if isinstance(error, urllib.error.HTTPError):
        reason = f"HTTP {error.code} {error.reason}"
        suggestion = None
        if error.code == 403:
            suggestion = (
                "The server rejected the request. The downloader sends a curl-like User-Agent; "
                "if this persists, verify the URL in a browser or with curl and check whether the provider changed access rules"
            )
        return DatasetDownloadError(dataset, "download", reason, suggestion)
    if isinstance(error, urllib.error.URLError):
        return DatasetDownloadError(dataset, "download", f"URL error: {error.reason}")
    if isinstance(error, TimeoutError):
        return DatasetDownloadError(dataset, "download", f"timed out after {DOWNLOAD_TIMEOUT_SEC} seconds")
    return DatasetDownloadError(dataset, "download", str(error))


def _download(dataset: TestDataset, archive_path: Path) -> None:
    """Download ``dataset`` to ``archive_path`` with bounded retries."""
    print(f"Downloading {dataset.key} from {dataset.url}")
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        request = urllib.request.Request(dataset.url, headers=REQUEST_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
                with archive_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_DOWNLOAD_ATTEMPTS and _is_transient_error(exc):
                print(f"Retrying {dataset.key} after download failure ({attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {exc}")
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise _download_error(dataset, exc) from exc


def _extract_zip(dataset: TestDataset, archive_path: Path, destination: Path) -> None:
    """Extract ``archive_path`` into ``destination`` after validating member paths."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            destination_root = destination.resolve()
            for member in archive.infolist():
                # Guard against zip-slip archives that contain absolute paths or ``..`` components.
                member_path = (destination / member.filename).resolve()
                if not member_path.is_relative_to(destination_root):
                    raise DatasetDownloadError(
                        dataset,
                        "extract",
                        f"archive member {member.filename!r} would be extracted outside {destination}",
                    )
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(dataset, "extract", "downloaded file is not a valid zip archive") from exc
