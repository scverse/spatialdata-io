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
- `pyxa`: `labels=True` rasterizes the segmentation polygons into lazy 3D `cell_labels` on the mosaic's grid, annotated by the table; `shapes=` controls the shapes; the mosaic is found in the Pyxa directory (`mosaic_3d.ome.zarr`, or its `.zip` read in place) via `image=`, which replaces `image_path`; counts are sparse.

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
  and Dependabot updates.

### Removed

- Support for Python 3.11.
