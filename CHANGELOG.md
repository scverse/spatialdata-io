# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.6] - 2024-11-26

-   (MERSCOPE) added `feature_key` attribute for points (i.e., the `'gene'` column) #210
-   (Visium HD) get transformation matrices even when only images are parsed #215
-   Support for `xarray.DataTree` (which was moved from `datatree.DataTree`) #232

## [0.1.5] - 2024-09-25

### Added

-   (Xenium) added `dims` parameter for more control in `xenium_aligned_image()`

### Fixed

-   Passing `rgb=None` to image model parser for both visium and visiumhd, leading to 3-4 channel images being
    interpreted as RGB(A)
-   Fix header bug Visium data #200
-   (Visium HD) Fix path parsing when images are missing #204 #206

## [0.1.4] - 2024-08-07

### Changed

-   (Xenium) changed default target of table to labels; radii of circles computed from cells, not nuclei #179
-   (Visium HD) changed default geometry to squares from circles for the bins; added parameter to choose #183
-   (CosMx) dropping points element with zero-length from the cosmx reader #191

## [0.1.3] - 2024-07-03

### Added

-   (Xenium) support reading multi-polygon selection files from the Xenium Explorer
-   (ISS) An experimental loader to load elemental ISS data objects, e.g. raw.tif, label.tif and anndata.h5ad
-   (Stereo-seq) Added reader @LLehner @timtreis @florianingelfinger #70
-   (MERSCOPE) Optional rioxarray backend for MERSCOPE data (reads chunks)
-   (MERSCOPE) Can choose which elements should be loaded

### Fixed

-   (Visium) Fixed issue with joining a SpatialElement with a table due to index values not being unique.
    obs_names_make_unique is now called internally to enforce unique index values allowing for join operations.

### Changed

-   (MERSCOPE) "global" coordinate system is used as a default instead of "microns"

## [0.1.2] - 2024-03-30

### Added

-   (Visium HD) added reader, coauthored by @LLehner

### Fixed

-   (Xenium) reader for 1.0.1 (paper data) and unknown versions
-   (Xenium) fix in reading "minimalistic" Xenium datasets #132

## [0.1.1] - 2024-03-24

### Added

-   (Xenium) support for post-xenium aligned images (IF, HE)
-   (Xenium) reader for the selection coordinates file from the Xenium Explorer
-   (Xenium) support for the new Xenium 2.0.0 (multimodal segmentation)
-   (Xenium) reading multiscale labels from cells.zarr.zip
-   (MCMICRO) support for TMAs (such as the data of exemplar-002)
-   (DBiT-seq) reader
-   converter functions `experimental.to_legacy_anndata()` and `experimental.from_legacy_anndata()`
-   (Visium) support for raw reads (capture locations not under tissue)

### Fixed

-   (Xenium) fixed index (fail on write)
-   (Xenium) renamed cells_as_shapes to cells_as_circles; set default to True
-   (MERSCOPE) don't try to load unexisting elements #87
-   (Visium) fixed axes ordering

## [0.0.9] - 2023-11-06

### Fixed

-   (Xenium) bug when converting feature_name #81, from @fbnrst
-   (Visium) visium() supports file counts without dataset_id #91

## [0.0.8] - 2023-10-02

### Fixed

-   (Xenium) coerce cell id to str #64
-   (MERSCOPE) fix coordinate transformation #68
-   (MERSCOPE) Improvements/fixes: merscope reader #73

## [0.0.7] - 2023-07-23

### Fixed

-   Bugs in Xenium and MERSCOPE

## [0.0.5] - 2023-06-21

### Added

-   MERFISH reader (from @quentinblampey)
-   CODEX reader (from @LLehner)

### Fixed

-   Issues on Visium reader (thanks @ilia-kats) and Xenium reader

## [0.0.4] - 2023-05-23

### Added

-   Curio reader

## [0.0.3] - 2023-05-22

### Merged

-   Merge pull request #40 from scverse/fix/categories

## [0.0.2] - 2023-05-04

### Changed

-   Revert version regex (#37)

## [0.0.1] - 2023-05-04

### Tested

-   Test installation from pypi
