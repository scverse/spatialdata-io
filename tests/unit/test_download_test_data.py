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


SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "test_data_downloader"


def _load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest = _load_module("manifest", SCRIPT_DIR / "manifest.py")
download_test_data = _load_module("downloader", SCRIPT_DIR / "downloader.py")


def _make_dataset(
    key: str = "example",
    *,
    group: str = "group",
    url: str | None = None,
    archive_name: str | None = None,
    extracted_dir: str | None = None,
    source: str = "example",
    test_path: str = "",
) -> Any:
    return download_test_data.TestDataset(
        key=key,
        group=group,
        url=url or f"https://example.com/{key}.zip",
        archive_name=archive_name or f"{key}.zip",
        extracted_dir=extracted_dir or key,
        source=source,
        test_path=test_path,
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
        assert download_test_data.TestDataset.__module__ == "manifest"
        assert all(isinstance(dataset, manifest.TestDataset) for dataset in download_test_data.DATASETS)

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
    def test_manifest_is_valid(self) -> None:
        manifest.validate_datasets()

    def test_returns_dataset_by_key(self) -> None:
        dataset = manifest.get_dataset("seqfish")

        assert dataset.key == "seqfish"
        assert dataset.extracted_dir == "seqfish-2-test-dataset"

    def test_reports_unknown_dataset_key(self) -> None:
        with pytest.raises(KeyError, match="Unknown test dataset key 'missing'"):
            manifest.get_dataset("missing")

    def test_test_path_defaults_to_extracted_directory_root(self) -> None:
        dataset = manifest.get_dataset("xenium_breast")

        assert dataset.test_path == ""

    def test_can_filter_datasets_by_group(self) -> None:
        datasets = manifest.datasets_by_group("macsima")

        assert {dataset.key for dataset in datasets} == {"macsima_omap10", "macsima_omap23"}

    def test_loads_datasets_from_toml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            """
[[datasets]]
key = "example"
group = "group"
url = "https://example.com/example.zip"
archive_name = "example.zip"
extracted_dir = "example"
source = "example source"
test_path = "nested"
""",
            encoding="utf-8",
        )

        datasets = manifest.load_datasets(manifest_path)

        assert datasets == (
            manifest.TestDataset(
                key="example",
                group="group",
                url="https://example.com/example.zip",
                archive_name="example.zip",
                extracted_dir="example",
                source="example source",
                test_path="nested",
            ),
        )

    def test_rejects_duplicate_dataset_keys(self) -> None:
        first = _make_dataset("duplicate", extracted_dir="first")
        second = _make_dataset("duplicate", extracted_dir="second")

        with pytest.raises(ValueError, match="Duplicate test dataset key: 'duplicate'"):
            manifest.validate_datasets((first, second))

    def test_rejects_duplicate_extracted_dirs(self) -> None:
        first = _make_dataset("first", extracted_dir="duplicate")
        second = _make_dataset("second", extracted_dir="duplicate")

        with pytest.raises(ValueError, match="Duplicate test dataset extracted_dir: 'duplicate'"):
            manifest.validate_datasets((first, second))

    def test_rejects_empty_required_manifest_fields(self) -> None:
        dataset = _make_dataset("missing-group", group="")

        with pytest.raises(ValueError, match="Dataset 'missing-group' has empty group"):
            manifest.validate_datasets((dataset,))

    def test_rejects_test_path_outside_extracted_dir(self) -> None:
        dataset = _make_dataset("unsafe-test-path", test_path="../outside")

        with pytest.raises(ValueError, match="test_path must be a relative path"):
            manifest.validate_datasets((dataset,))

    def test_rejects_manifest_without_datasets_array(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text("datasets = { key = 'example' }\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\[\[datasets\]\] array"):
            manifest.load_datasets(manifest_path)

    def test_rejects_invalid_toml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text("[[datasets]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid dataset manifest TOML"):
            manifest.load_datasets(manifest_path)

    def test_rejects_unknown_root_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            """
version = 1

[[datasets]]
key = "example"
group = "group"
url = "https://example.com/example.zip"
archive_name = "example.zip"
extracted_dir = "example"
source = "example source"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unknown root field"):
            manifest.load_datasets(manifest_path)

    def test_rejects_missing_required_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            """
[[datasets]]
key = "example"
group = "group"
url = "https://example.com/example.zip"
archive_name = "example.zip"
extracted_dir = "example"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing required field"):
            manifest.load_datasets(manifest_path)

    def test_rejects_unknown_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            """
[[datasets]]
key = "example"
group = "group"
url = "https://example.com/example.zip"
archive_name = "example.zip"
extracted_dir = "example"
source = "example source"
checksum = "unexpected"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unknown field"):
            manifest.load_datasets(manifest_path)


class TestExtractZip:
    def test_reports_invalid_archive(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "bad.zip"
        archive_path.write_bytes(b"not a zip")

        with pytest.raises(download_test_data.DatasetDownloadError, match="not a valid zip archive"):
            download_test_data._extract_zip(_make_dataset(), archive_path, tmp_path / "extract")

    def test_rejects_archive_members_outside_destination(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "unsafe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("../escape.txt", "bad")

        with pytest.raises(download_test_data.DatasetDownloadError, match="outside"):
            download_test_data._extract_zip(_make_dataset(), archive_path, tmp_path / "extract")

        assert not (tmp_path / "escape.txt").exists()


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

    def test_reports_archive_without_expected_contents(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("empty")

        def write_empty_archive(dataset: object, archive_path: Path) -> None:
            with zipfile.ZipFile(archive_path, "w"):
                pass

        monkeypatch.setattr(download_test_data, "_download", write_empty_archive)

        with pytest.raises(download_test_data.DatasetDownloadError, match="expected directory"):
            download_test_data.download_dataset(dataset, tmp_path, force=False)

        assert not (tmp_path / dataset.extracted_dir).exists()


class TestMain:
    def test_downloads_multiple_selected_datasets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        first = _make_dataset("first")
        second = _make_dataset("second")
        third = _make_dataset("third")
        attempted: list[str] = []

        def download_dataset(dataset: Any, output: Path, force: bool) -> None:
            attempted.append(dataset.key)
            assert output == tmp_path
            assert not force

        monkeypatch.setattr(download_test_data, "DATASETS", (first, second, third))
        monkeypatch.setattr(download_test_data, "download_dataset", download_dataset)

        download_test_data.main(["--output", str(tmp_path), "--dataset", "first", "--dataset", "third"])

        assert attempted == ["first", "third"]

    def test_lists_available_datasets(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = _make_dataset("listed", group="group", extracted_dir="listed-dir", source="listed source")

        monkeypatch.setattr(download_test_data, "DATASETS", (dataset,))

        download_test_data.main(["--list"])

        assert capsys.readouterr().out == "listed\tgroup\tlisted-dir\tlisted source\n"

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

        with pytest.raises(SystemExit) as exc_info:
            download_test_data.main(["--output", str(tmp_path)])

        assert exc_info.value.code == 1
        assert attempted == ["first", "second"]
        output = capsys.readouterr().out
        assert "Failed to download 1 dataset(s)" in output
        assert "first" in output
