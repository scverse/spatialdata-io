![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# spatialdata-io: convenient readers for loading common formats into SpatialData

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![DOI](https://zenodo.org/badge/544045123.svg)](https://zenodo.org/badge/latestdoi/544045123)

[badge-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata-io

This package contains reader functions to load common spatial omics formats into SpatialData. Currently, we provide support for:

-   NanoString CosMx
-   MCMICRO (output data)
-   Steinbock (output data)
-   10x Genomics Visium
-   10x Genomics Xenium
-   Curio Seeker
-   Akoya PhenoCycler (formerly CODEX)
-   Vizgen MERSCOPE (MERFISH)
-   DBiT-seq

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install spatialdata-io:

1. Install the latest release of `spatialdata-io` from [PyPI](https://pypi.org/project/spatialdata-io/):

```bash
pip install spatialdata-io
```

2. Install the latest development version:

```bash
pip install git+https://github.com/scverse/spatialdata-io.git@main
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Readers from third-party libraries

Technologies that can be read into `SpatialData` objects using third-party libraries:

-   METASPACE (MALDI, ...): [metaspace-converter](https://github.com/metaspace2020/metaspace-converter)
-   PhenoCycler: [SOPA](https://github.com/gustaveroussy/sopa)
-   MACSima: [SOPA](https://github.com/gustaveroussy/sopa)
-   Hyperion (Imaging Mass Cytometry): [SOPA](https://github.com/gustaveroussy/sopa)

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata-io/issues
[changelog]: https://spatialdata.scverse.org/projects/io/en/latest/changelog.html
[link-docs]: https://spatialdata.scverse.org/projects/io/en/latest/
[link-api]: https://spatialdata.scverse.org/projects/io/en/latest/api.html
