# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

Release notes for `v0.7.1` and earlier are available on the [Releases][] page.

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/
[releases]: https://github.com/scverse/spatialdata-io/releases

## [Unreleased]

### Added

- `spatialdata_io` ships a `py.typed` marker, so downstream type checkers use its annotations.
- `atera` reader for 10x Genomics Atera bundles: cell-by-gene table, cell/nucleus labels and boundary
  polygons, transcripts, and morphology images, read natively via `anndata`'s zarr IO.
- `atera_to_parquet` converter: writes an Atera bundle's `transcripts.zarr.zip` to a `proseg`-compatible
  `transcripts.parquet` (readable via `proseg --xenium`), streaming one spatial tile at a time.

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
  and Dependabot updates.

### Removed

- Support for Python 3.11.
