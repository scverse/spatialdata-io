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
- `xenium` reads pre-1.3.0 (XOA < 1.3.0) and CSV-only bundles: when the parquet files and
  `cells.zarr.zip` are absent, the table, boundaries, and transcripts are read from the `cells.csv`,
  `cell_boundaries.csv`, `nucleus_boundaries.csv` and `transcripts.csv` outputs (`.gz` or plain),
  and the raster cell/nucleus labels are reconstructed by rasterizing the boundary polygons. The
  cell feature matrix falls back to a `cell_feature_matrix/` MatrixMarket directory or
  `cell_feature_matrix.tar.gz` when `cell_feature_matrix.h5` is missing.
- `phenocycler()` reader (and `phenocycler` CLI command) for Akoya PhenoCycler-Fusion 1.x/2.x `.qptiff` /
  `.er.qptiff` scans, including PhenoImager (Opal) mode: lazy level-0 image with a recomputed pyramid, channel names
  from the per-page XML, structured and raw scan metadata in `sdata.attrs["akoya"]`, optional `.xpd` /
  `*channels.csv` sidecars, and optional cell tables (QuPath with GeoJSON outlines, MCMICRO `mcquant`, HuBMAP SPRM,
  inForm, CODEX Processor or generic CSV) with segmentation masks.

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
  and Dependabot updates.
- `codex()` was rewritten to read CODEX Processor, CODEX Uploader and CODEX Instrument Manager outputs. This changes
  its output and signature:
  - Element names are now `{region}_image`, `{region}_cells` (circles), `{region}_labels`, `{region}_polygons` and
    `table` (previously `images` and `{region}` for the shapes).
  - Every region is read (previously only the first file found); `regions=[...]` selects a subset.
  - Marker names in `table.var_names` come from the FCS `$PnS` keywords or the CSV header `cycCCC_chHHH:<marker>`
    (previously `cycCCC_chHHH`); the image channels are named too.
  - The positional third parameter is now `regions` (previously `imread_kwargs`); pass `imread_kwargs` by keyword.
    Code that still passes `imread_kwargs` positionally gets a `TypeError` (a mapping is not a valid `regions`)
    rather than a silently different read; an empty `regions` raises a `ValueError`. `imread_kwargs` is only used by
    the `dask_image.imread` fallback for TIFFs that `tifffile` cannot read lazily.
  - A single region is stored in `"global"`. Several regions are placed at their stage positions in `"global"` only
    when all of them share a sample space (same slide id, stage positions and pixel size); otherwise each region,
    including multi-region reads of unrelated regions, gets its own coordinate system named after the region.
  - New parameters `channel_names`, `cells_table`, `cells_table_format`, `labels`, `image_models_kwargs`,
    `labels_models_kwargs` and `cells_table_kwargs`; the `codex` CLI command exposes them.

### Fixed

- `codex()` read the CSV table and the image relative to the current working directory instead of `path`, and an
  operator-precedence bug in the table check made it ignore `fcs`.

### Removed

- Support for Python 3.11.
