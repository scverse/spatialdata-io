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

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
  and Dependabot updates.

### Fixed

- `visium()`: the circles are built again from the spot coordinates instead of from the raw `tissue_positions` table,
  which made the reader raise `TypeError: ShapesModel.parse() does not support the type
  <class 'pandas.core.frame.DataFrame'>`.
- `merscope()`: a `rioxarray` that is installed but cannot be imported (e.g. a broken `rasterio`) falls back to the
  `dask_image` backend again, instead of failing with a misleading "requires to install the rioxarray library".
- `dbit()`: without a `path` the reader no longer searches the current working directory; it raises unless the
  individual file paths are given.
- `macsima()`: files whose OME metadata cannot be parsed are skipped again rather than aborting the reader; a
  truncated TIFF raises `struct.error`, which is neither a `ValueError` nor one of the types the physical-size loop
  was catching.
- `macsima()`: the plane positions used as padding widths are rounded instead of truncated towards zero.

### Removed

- Support for Python 3.11.
