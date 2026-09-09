# Optional test-data downloader

This directory contains the manifest and command-line tool used to download the
external datasets required by some `spatialdata-io` integration tests. These
datasets are intentionally not stored in Git because they are large or are
maintained by another project.

The downloader provides one reproducible entry point for local development and
CI. It:

- reads dataset registrations from [`datasets.toml`](datasets.toml);
- verifies every downloaded file against a published or committed checksum;
- assembles multi-file datasets in a temporary staging directory;
- leaves an existing dataset unchanged if a download or extraction fails; and
- makes registered datasets available to tests through the shared
  `require_test_dataset` fixture.

The manifest is project metadata, not a copy of the Pooch registry. It records
stable test keys, reader groups, local directory names, provenance, and one of
the supported download-source configurations.

## Using the downloader

Run commands from the repository root with the development dependencies
installed (`uv sync --group dev --group test`).

List the registered keys, groups, output directories, and sources:

```bash
uv run python scripts/test_data_downloader --list
```

Download one dataset or all datasets in a reader group:

```bash
uv run python scripts/test_data_downloader --dataset visium_v1_human_lymph_node
uv run python scripts/test_data_downloader --group visium
```

`--dataset` and `--group` may each be repeated. When both are present, the
downloader installs the union of the selected keys and groups in manifest
order. With neither option, it downloads every registered dataset.

Datasets are installed below `data/` by default. Use another parent directory
with `--output` and point pytest at the same location:

```bash
uv run python scripts/test_data_downloader \
  --dataset visium_v1_human_lymph_node \
  --output /path/to/test-data
SPATIALDATA_IO_TEST_DATA_DIR=/path/to/test-data uv run pytest -m "visium and data"
```

An existing destination is skipped. Pass `--force` to download a fresh copy and
replace that destination after the new dataset has been completely staged:

```bash
uv run python scripts/test_data_downloader --dataset visium_v1_human_lymph_node --force
```

The downloader attempts every selected dataset and exits with status 1 after
reporting all failures. Downloads require network access; normal unit tests and
manifest validation do not.

## Manifest fields

Every `[[datasets]]` entry defines these fields:

| Field | Meaning |
| --- | --- |
| `key` | Stable lowercase identifier used by the CLI and pytest fixture. |
| `group` | Lowercase reader-oriented selector used by `--group`. |
| `extracted_dir` | Single directory name created below the output directory. |
| `source` | Human-readable upstream provenance, format version, and license. |
| `test_path` | Optional directory below `extracted_dir` returned to a test. |

Keys and groups may contain lowercase letters, digits, underscores, and
hyphens. Paths must be relative and may not contain `..`. Keep keys and output
directory names stable: tests, local caches, and CI workflows can refer to them.

Each dataset uses exactly one of the following source forms.

### One ZIP archive

Use a direct, stable URL and pin the complete ZIP by SHA-256. The archive's
contents are extracted into `extracted_dir` without removing a top-level
directory.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"
url = "https://example.org/releases/2.0/example.zip"
known_hash = "sha256:<64 hexadecimal characters>"
```

### DOI repository

Prefer a stable DOI when its repository publishes a Pooch-compatible file
registry. Pooch obtains the repository's file list and published hashes, then
downloads and verifies every registered file.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"
doi = "10.5281/zenodo.1234567"
```

Store the bare DOI, without a URL or `doi:` prefix. A DOI entry cannot select a
subset of repository files; use independently hashed assets if the test needs
only part of a large publication.

### Independently hashed assets

Use `[[datasets.assets]]` when one logical dataset is assembled from multiple
downloads or when retaining a small set of upstream artifacts avoids a much
larger archive. Every asset has its own URL, checksum, and destination.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"

[[datasets.assets]]
url = "https://example.org/releases/2.0/counts.h5"
known_hash = "sha256:<64 hexadecimal characters>"
target = "filtered_feature_bc_matrix.h5"

[[datasets.assets]]
url = "https://example.org/releases/2.0/spatial.tar.gz"
known_hash = "sha256:<64 hexadecimal characters>"
target = "."
extract = true
```

For a regular file, `target` includes the destination filename and parent
directories are created automatically. With `extract = true`, the asset must be
a tar archive and `target` is its extraction directory; `target = "."` extracts
at the dataset root. Tar members are filtered to reject path traversal and
unsafe filesystem objects. Asset targets must be unique within an entry.

## Adding or updating a dataset

1. Choose the smallest public, stable, and permissively licensed upstream data
   that faithfully covers the format behavior under test. Record the data
   producer, format/software version, and license in `source` and a nearby TOML
   comment with the authoritative dataset or license page.
2. Decide whether a DOI, one ZIP archive, or independently hashed assets best
   represents the source. Do not mix source forms in one dataset entry.
3. Download each direct URL once and calculate its SHA-256 digest. Pooch can
   produce the exact manifest value:

   ```bash
   uv run python -c 'import pooch; print("sha256:" + pooch.file_hash("/path/to/download", alg="sha256"))'
   ```

4. Append the entry to `datasets.toml`, keeping related reader datasets
   together. Never use an unknown hash, a mutable URL without a pinned digest,
   credentials, or a private machine path.
5. Validate the registration and perform a clean download into a disposable
   output directory:

   ```bash
   uv run python scripts/test_data_downloader --list
   test_output="$(mktemp -d)"
   uv run python scripts/test_data_downloader --dataset example --output "$test_output"
   ```

6. Inspect the resulting directory and connect the integration test to the
   stable key:

   ```python
   def test_example(require_test_dataset: Callable[[str], Path]) -> None:
       dataset_path = require_test_dataset("example")
       result = example_reader(dataset_path)
   ```

   The fixture skips cleanly when optional data is unavailable. Add the reader
   marker and let the fixture apply the repository's `data` marker.
7. Run the downloader unit tests and the affected data-backed integration test:

   ```bash
   uv run pytest tests/unit/test_download_test_data.py
   SPATIALDATA_IO_TEST_DATA_DIR="$test_output" uv run pytest -m "example_reader and data"
   ```

To add a file to an existing multi-asset dataset, append another
`[[datasets.assets]]` table with its own checksum and unused `target`, then
repeat the clean-download check. A ZIP or DOI entry cannot also contain assets;
either update that upstream source or deliberately convert the complete entry
to the multi-asset form. When changing an existing URL, file, archive contents,
or checksum, verify that the change is expected and rerun every integration test
that uses the dataset key.

Do not commit downloaded datasets, temporary output, credentials, or private
data. The repository tracks only the manifest and downloader implementation.
