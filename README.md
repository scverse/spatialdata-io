![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# spatialdata-io: convenient readers for loading common formats into SpatialData

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![DOI](https://zenodo.org/badge/544045123.svg)](https://zenodo.org/badge/latestdoi/544045123)
[![Documentation][badge-pypi]][link-pypi]
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/spatialdata-io/badges/version.svg)](https://anaconda.org/conda-forge/spatialdata-io)

[badge-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata-io
[badge-pypi]: https://badge.fury.io/py/spatialdata-io.svg
[link-pypi]: https://pypi.org/project/spatialdata-io/

This package contains reader functions to load common spatial omics formats into SpatialData. Currently, we provide support for:

- 10x Genomics Visium®
- 10x Genomics Visium HD®
- 10x Genomics Xenium®
- Akoya PhenoCycler® (formerly CODEX®)
- Curio Seeker®
- DBiT-seq
- MCMICRO (output data)
- NanoString CosMx®
- Spatial Genomics GenePS® (seqFISH)
- Steinbock (output data)
- STOmics Stereo-seq®
- Vizgen MERSCOPE® (MERFISH)
- MACSima® (MACS® iQ View output)

Note: all mentioned technologies are registered trademarks of their respective companies.

## Known limitations

Contributions for addressing the below limitations are very welcomed.

- Only Stereo-seq 7.x is supported, 8.x is not currently supported. https://github.com/scverse/spatialdata-io/issues/161

### How to Contribute

1. **Open a GitHub Issue**: Start by opening a new issue or commenting on an existing one in the repository. Clearly describe the problem and your proposed changes to avoid overlapping efforts with others.

2. **Submit a Pull Request (PR)**: Once the issue is discussed, submit a PR to the `spatialdata-io` repository. Ensure your PR includes information about a suitable dataset for testing the reader, ideally no larger than 10 GB. Include clear instructions for accessing the data, preferably with a `curl` or `wget` command for easy downloading.

3. **Optional Enhancements**: To facilitate reproducibility and ease of data access, consider adding a folder in the [spatialdata-sandbox](https://github.com/giovp/spatialdata-sandbox) repository. Include a `download.py` and `to_zarr.py` script (refer to examples in the repository) to enable others to reproduce your reader by simply running these scripts sequentially.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api].
- [CLI documentation][link-cli].

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

- METASPACE (MALDI, ...): [metaspace-converter](https://github.com/metaspace2020/metaspace-converter)
- PhenoCycler®: [SOPA](https://github.com/gustaveroussy/sopa)
- MACSima®: [SOPA](https://github.com/gustaveroussy/sopa)
- Hyperion® (Imaging Mass Cytometry): [SOPA](https://github.com/gustaveroussy/sopa)

## Disclaimer

This library is community maintained and is not officially endorsed by the aforementioned spatial technology companies. As such, we cannot offer any warranty of the correctness of the representation. Furthermore, we cannot ensure the correctness of the readers for every data version as the technologies evolve and update their formats. If you find a bug or notice a misrepresentation of the data please report it via our [Bug Tracking System](https://github.com/scverse/spatialdata-io/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen) so that it can be addressed either by the maintainers of this library or by the community.

## Solutions to common problems

### Problem: I cannot visualize the data, everything is slow

Solution: after parsing the data with `spatialdata-io` readers, you need to write it to Zarr and read it again. Otherwise the performance advantage given by the SpatialData Zarr format will not available.

```python
from spatialdata_io import xenium
from spatialdata import read_zarr

sdata = xenium("raw_data")
sdata.write("data.zarr")
sdata = read_zarr("sdata.zarr")
```

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata-io/issues
[changelog]: https://spatialdata.scverse.org/projects/io/en/stable/changelog.html
[link-docs]: https://spatialdata.scverse.org/projects/io/en/stable/
[link-api]: https://spatialdata.scverse.org/projects/io/en/stable/api.html
[link-cli]: https://spatialdata.scverse.org/projects/io/en/stable/cli.html
[//]: # "numfocus-fiscal-sponsor-attribution"

spatialdata-io is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
