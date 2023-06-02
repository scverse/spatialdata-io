![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# spatialdata-io: convenient readers for loading common formats into SpatialData

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test_and_deploy.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test_and_deploy.yaml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata-io

This package contains reader functions to load common spatial omics formats into SpatialData. Currently, we provide support for:

-   NanoString CosMx
-   MCMICRO
-   Steinbock
-   10x Genomics Visium
-   10x Genomics Xenium

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

## Citation

[L Marconato*, G Palla*, KA Yamauchi*, I Virshup*, E Heidari, T Treis, M Toth, R Shrestha, H Vöhringer, W Huber, M Gerstung, J Moore, FJ Theis, O Stegle, bioRxiv, 2023](https://www.biorxiv.org/content/10.1101/2023.05.05.539647v1). \* = equal contribution

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata-io/issues
[changelog]: https://spatialdata-io.readthedocs.io/latest/changelog.html
[link-docs]: https://spatialdata-io.readthedocs.io
[link-api]: https://spatialdata-io.readthedocs.io/latest/api.html
