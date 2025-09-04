# Contributing guide
This guide is written for:

- **spatial technology companies** that would like to add a reader (or converter) for a new product, support a new version of an existing technology, or contribute with edits/bugfixes to one of the available readers
- **academic labs** that would like to contribute a reader (or converter) for in-house data generation pipelines
- **general users** that would like to improve the library

The `spatialdata-io` repository inherits the technical stack from `spatialdata`, therefore please read also the [contribution guide from the `spatialdata` repository](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md).

The rest of the document will be specific to `spatialdata-io`. Feedback on this document is highly welcomed: you are encouraged to open a PR to improve this guide.

## Adding a new reader
To contribute to a reader and make code review efficient we kindly ask for:
- the **specification** for the raw data: a link to a (ideally) *public* resource describing how the data is organized, and/or for a new version of an already supported technology, a changelog descring in details which changes are introduced;
- the **reader** itself: a Python function, generally supported by helper functions/classes/enums, that reads in the raw data and returns a `SpatialData` object;
- example **data**: preferably one or more (very) small *public* datasets to represent the data across the various versions and cover eventual edge cases of the raw data format. In alternative, scripts to easily download *public* data. Working without public data is also possible but discouraged because it makes development and testing harder;
- **test** functions: to ensure that the reader parses the data correctly. If helper functions are used, they should also be tested.
### The specification
In your issue or PR please include a link to a *public* specification and/or changelog, describing how the raw data is organized on disk and what changes across various version.

If the specification cannot be (yet) made public, you can reach out to scverse via [Zulip](https://scverse.zulipchat.com/#narrow/channel/443514-spatialdata-dev) via private message, or via email.
### The reader file
The reader files are located under [src/spatialdata_io/readers](https://github.com/scverse/spatialdata-io/tree/main/src/spatialdata_io/readers). For each supported technology, the reader file is a single function, named after the technology, that takes as input the path to the raw data and extra arguments, and returns a `SpatialData` object:

```python=
from pathlib import Path
from spatialdata import SpatialData


def technology(path: str | Path, ...) -> SpatialData:
    # `sdata` is constructed
    return sdata
```

For example, the reader for Xenium (a technology from 10x Genomics), is called `xenium()` and is located under [`src/spatialdata_io/readers/xenium.py`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/xenium.py).

#### Technical notes
> Feel free to skip this section if you just want an overview of the contribution process.

We recommend to study existing readers and reuse code from there. A few technical notes below.

- Large raster or points data is usually loaded from disk lazily (e.g. with `dask_image.imread()`), this allows to return a `SpatialData` object quickly, and defer the computation when saving the object to disk to the SpatialData Zarr format, with `sdata.write()`.
- When the raw data has multiple sample, we recommend to add a coordinate system for each sample, and if the samples are aligned in space, one common coordinate system. A single table containing the annotation for all the samples is preferred. See an example in the [`cosmx()`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/cosmx.py) reader.
- Small images should be represented as single-scale images (`xarray.DataArray`), large images as multiscale images (`xarray.DataTree`). The scale factors and chunk shape (`chunks`) should lead to chunks that fit the memory. See ane xample in [`visium()`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/visium.py).

#### Experimental readers
If a specification is not available a reader has less guarantee to work out-of-the-box. This is for instance the case for spatial omics datasets generated with in-house technologies. Even in this case users may find a reader beneficial: they may still have to adapt the reader for their use cases, but it could provide a good starting point to parse their data. We list these readers under the [`experimental` module](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/experimental/__init__.py). An example of this is the `[iss()](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/iss.py)` reader for In-Situ Sequencing data.

### Separating the constants from the reader logic

## Adding tests: small data vs real data
### When small public test datasets are available
### When only real (large) public data is available
### When only private data is available
#### The `download.py` script
#### The `to_zarr.py` script

## What to test
### Testing multiple versions
### Testing plotting capabilities
### Testing data integrity
### Testing auxiliary functions

## Adding a new converter

## Updating the CLI for readers and converters

# Bug tracking
- tracking bugs also outside spatialdata-io
- tracking bugs also in spatialdata

# Limitations and possible improvements
## Unsupported technologies
## Naming constistency across readers
## Pixel space vs physical space
## Transformations between coordinate systems
## Summary table of supported technlogies and specifications