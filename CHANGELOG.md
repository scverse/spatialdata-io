# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.10] - tbd

### Added

-   (MCMICRO) support for TMAs (such as the data of exemplar-002)
-   (Xenium) support for post-xenium aligned images (IF, HE)
-   (Xenium) reader for the selection coordinates file from the Xenium Explorer
-   DBiT-seq reader

### Fixed

-   (MERSCOPE) don't try to load unexisting elements #87
-   (Visium) fixed axes ordering
-   (Xenium) fixed index (fail on write)
-   (Xenium) renamed cells_as_shapes to cells_as_circles; set default to True

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
