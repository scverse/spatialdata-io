from __future__ import annotations

import importlib.util
import io
import sys
import urllib.error
import urllib.request
import zipfile
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_download_script() -> ModuleType:
    module_path = Path(__file__).parents[2] / "scripts" / "download_test_data.py"
    spec = importlib.util.spec_from_file_location("download_test_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


download_test_data = _load_download_script()


def _make_dataset(key: str = "example") -> Any:
    return download_test_data.TestDataset(
        key=key,
        group="group",
        url=f"https://example.com/{key}.zip",
        archive_name=f"{key}.zip",
        extracted_dir=key,
        source="example",
    )


def _write_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("payload.txt", "ok")


class _BytesResponse(io.BytesIO):
    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class TestDownload:
    def test_uses_dataset_manifest_module(self) -> None:
        assert download_test_data.TestDataset.__module__ == "download_test_data_datasets"
        assert all(isinstance(dataset, download_test_data.TestDataset) for dataset in download_test_data.DATASETS)

    def test_sends_explicit_headers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        seen_request: urllib.request.Request | None = None

        def urlopen(request: urllib.request.Request, timeout: int) -> _BytesResponse:
            nonlocal seen_request
            seen_request = request
            assert timeout == download_test_data.DOWNLOAD_TIMEOUT_SEC
            return _BytesResponse(b"payload")

        monkeypatch.setattr(download_test_data.urllib.request, "urlopen", urlopen)

        download_test_data._download(_make_dataset(), tmp_path / "archive.zip")

        assert seen_request is not None
        assert seen_request.get_header("User-agent") == "curl/8.0.0"
        assert seen_request.get_header("Accept") == "*/*"

    def test_reports_http_403_with_actionable_message(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("forbidden")

        def urlopen(request: object, timeout: int) -> None:
            raise urllib.error.HTTPError(dataset.url, 403, "Forbidden", hdrs=Message(), fp=None)

        monkeypatch.setattr(download_test_data.urllib.request, "urlopen", urlopen)

        with pytest.raises(download_test_data.DatasetDownloadError, match="HTTP 403 Forbidden") as exc_info:
            download_test_data._download(dataset, tmp_path / dataset.archive_name)

        message = str(exc_info.value)
        assert dataset.key in message
        assert dataset.url in message
        assert "rejected the request" in message

    def test_retries_transient_failures(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("retry")
        attempts = 0

        def urlopen(request: object, timeout: int) -> _BytesResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise urllib.error.HTTPError(dataset.url, 503, "Service Unavailable", hdrs=Message(), fp=None)
            return _BytesResponse(b"payload")

        monkeypatch.setattr(download_test_data.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(download_test_data.time, "sleep", lambda seconds: None)

        download_test_data._download(dataset, tmp_path / dataset.archive_name)

        assert attempts == 2


class TestDatasetManifest:
    def test_returns_dataset_by_key(self) -> None:
        dataset = download_test_data.get_dataset("seqfish")

        assert dataset.key == "seqfish"
        assert dataset.extracted_dir == "seqfish-2-test-dataset"

    def test_reports_unknown_dataset_key(self) -> None:
        with pytest.raises(KeyError, match="Unknown test dataset key 'missing'"):
            download_test_data.get_dataset("missing")

    def test_test_path_defaults_to_extracted_directory_root(self) -> None:
        dataset = download_test_data.get_dataset("xenium_breast")

        assert dataset.test_path == ""

    def test_can_filter_datasets_by_group(self) -> None:
        datasets = download_test_data.datasets_by_group("macsima")

        assert {dataset.key for dataset in datasets} == {"macsima_omap10", "macsima_omap23"}


class TestExtractZip:
    def test_reports_invalid_archive(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "bad.zip"
        archive_path.write_bytes(b"not a zip")

        with pytest.raises(download_test_data.DatasetDownloadError, match="not a valid zip archive"):
            download_test_data._extract_zip(_make_dataset(), archive_path, tmp_path / "extract")


class TestDownloadDataset:
    def test_skips_existing_dataset_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = _make_dataset("existing")
        target = tmp_path / dataset.extracted_dir
        target.mkdir()

        def fail_download(dataset: object, archive_path: Path) -> None:
            raise AssertionError("download should have been skipped")

        monkeypatch.setattr(download_test_data, "_download", fail_download)

        download_test_data.download_dataset(dataset, tmp_path, force=False)

        assert "Skipping existing" in capsys.readouterr().out

    def test_replaces_existing_dataset_with_force(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("force")
        target = tmp_path / dataset.extracted_dir
        target.mkdir()
        (target / "old.txt").write_text("old", encoding="utf-8")

        def write_archive(dataset: object, archive_path: Path) -> None:
            _write_zip(archive_path)

        monkeypatch.setattr(download_test_data, "_download", write_archive)

        download_test_data.download_dataset(dataset, tmp_path, force=True)

        assert not (target / "old.txt").exists()
        assert (target / "payload.txt").read_text(encoding="utf-8") == "ok"


class TestMain:
    def test_continues_after_failure_then_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = _make_dataset("first")
        second = _make_dataset("second")
        attempted: list[str] = []

        def download_dataset(dataset: Any, output: Path, force: bool) -> None:
            attempted.append(dataset.key)
            if dataset == first:
                raise download_test_data.DatasetDownloadError(dataset, "download", "HTTP 403 Forbidden")

        monkeypatch.setattr(download_test_data, "DATASETS", (first, second))
        monkeypatch.setattr(download_test_data, "download_dataset", download_dataset)
        monkeypatch.setattr(sys, "argv", ["download_test_data.py", "--output", str(tmp_path)])

        with pytest.raises(SystemExit) as exc_info:
            download_test_data.main()

        assert exc_info.value.code == 1
        assert attempted == ["first", "second"]
        output = capsys.readouterr().out
        assert "Failed to download 1 dataset(s)" in output
        assert "first" in output
