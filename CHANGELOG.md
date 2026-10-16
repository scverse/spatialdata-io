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
  inForm, CODEX Processor or generic CSV) with segmentation masks. The lazy images and masks can be pickled (e.g. for
  `dask.distributed`): they open their file in each process on the first chunk read. A cell table for several
  regions converts µm coordinates with each region's own pixel size.
- `phenocycler(read_all_bit_depths=True)` (CLI `--read-all-bit-depths`) reads both the 8-bit `.qptiff` and the
  extended-range `.er.qptiff` of a scan: the 8-bit one becomes the image `<region>_8bit_image` with the transformations
  of the `.er` region. By default only the extended-range file is read.
- The Akoya readers (`phenocycler()`, `codex()`) scale a segmentation mask downsampled from its region image by an
  integer factor (exactly, or rounded up) onto the image (a `Scale` before the region's transformation), and scale
  the centroids (`obsm["spatial"]`) and circle radii of `mcquant`, SPRM and generic pixel tables, taken to be in
  mask pixels, by the same factor; the source columns in `obs` are kept as read. Other shape mismatches still warn.
- `phenocycler()` officially supports the PerkinElmer/Akoya QPTIFFs of Vectra Polaris and PhenoImager HT (generation
  `"qpi_other"`): Opal and unmixed (uint16) scans, brightfield RGB scans (channels `r`, `g`, `b`) and inForm component
  data (float32). A single `.tif` file is accepted if it is a QPI file. Region ids are made valid element names with
  `spatialdata.sanitize_name` (e.g. the `[x,y]` of inForm tiles: `A_[1,2]` becomes `A__1_2_`); the region column of a
  cell table may hold the raw file names.
- The Akoya cell-table adapters read real QuPath, inForm and mcquant exports without column mapping: R-exported
  QuPath names (`Object.ID`, `Centroid.X.um`, `<marker>_<compartment>_Mean`), QuPath `X`/`Y` centroids (in the unit
  of the other measurements with a warning, or `cells_table_kwargs["coordinates_unit"]`) with bare marker names
  (read with a warning, or chosen with `cells_table_kwargs["markers"]`), and inForm column names with trailing
  spaces. `mask_channel` accepts the page names of an inForm `*_binary_seg_maps.tif` (`"Tissue"`, `"Nucleus"`,
  `"Cytoplasm"`, `"Membrane"`). QuPath GeoJSON cells without a table row are skipped with a warning, and a region
  whose cells lie mostly outside its image raises a warning.

### Changed

- inForm cell tables whose `Cell X Position` names no unit are read in the unit of their other columns (e.g.
  microns, divided by the pixel size), no longer always in pixels; `cells_table_kwargs["coordinates_unit"]` sets it.
- An inForm `*_binary_seg_maps.tif` given as `labels` reads its `Nucleus` page (whose labels are the `Cell ID`s) by
  default, not its first page (the tissue class map).
- With `x` and `y` in `cells_table_kwargs`, `cells_table_format="auto"` reads a generic table.
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
  - The positional signature `codex(path, fcs, imread_kwargs)` is unchanged; all new parameters (`regions`,
    `channel_names`, `cells_table`, `cells_table_format`, `labels` and the `*_models_kwargs` / `cells_table_kwargs`)
    are keyword-only. An empty `regions` raises a `ValueError`. `imread_kwargs` is only used by the
    `dask_image.imread` fallback for TIFFs that `tifffile` cannot read lazily.
  - A single region is stored in `"global"`. Several regions are placed at their stage positions in `"global"` only
    when all of them share a sample space (same slide id, stage positions and pixel size); otherwise each region,
    including multi-region reads of unrelated regions, gets its own coordinate system named after the region.
  - New parameters `channel_names`, `cells_table`, `cells_table_format`, `labels`, `image_models_kwargs`,
    `labels_models_kwargs` and `cells_table_kwargs`; the `codex` CLI command exposes them.
  - `channelNames.txt` and `experiment.json` are found in any case (e.g. SPACEc's `channelnames.txt`); CSV files
    that are not CODEX Processor cell tables are ignored, also when their name holds a region id.
  - A CIM `.qptiff` without `regNNN` in its name is named as `phenocycler()` names it (without the suffix and a
    trailing `_Scan<N>`), and its `.er.qptiff` is preferred.

### Fixed

- `codex()` read the CSV table and the image relative to the current working directory instead of `path`, and an
  operator-precedence bug in the table check made it ignore `fcs`.

### Removed

- Support for Python 3.11.
