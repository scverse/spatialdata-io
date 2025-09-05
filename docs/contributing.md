# Contribution guide

This guide is written for:

- **spatial technology companies** that would like to add a reader (or converter) for a new product, support a new version of an existing technology, or contribute with edits/bugfixes to one of the available readers
- **academic labs** that would like to contribute a reader (or converter) for in-house data generation pipelines
- **general users** that would like to improve the library

The `spatialdata-io` repository inherits the technical stack from `spatialdata`, therefore please read also the [contribution guide from the `spatialdata` repository](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md).

The rest of the document will be specific to `spatialdata-io`. Feedback on this document is highly welcomed: you are encouraged to open a PR to improve this guide.

## Adding a new reader

To contribute to a reader and make code review efficient we kindly ask for:

- the **specification** for the raw data: a link to a (ideally) _public_ resource describing how the data is organized, and/or for a new version of an already supported technology, a changelog descring in details which changes are introduced;
- the **reader** itself: a Python function, generally supported by helper functions/classes/enums, that reads in the raw data and returns a `SpatialData` object;
- example **data**: preferably one or more (very) small _public_ datasets, and with a permissive license, to represent the data across the various versions and cover eventual edge cases of the raw data format. In alternative, scripts to easily download _public_ data. Working without public data is also possible but discouraged because it makes development and testing harder;
- **test** functions: to ensure that the reader parses the data correctly. If helper functions are used, they should also be tested.

### The specification

In your issue or PR please include a link to a _public_ specification and/or changelog, describing how the raw data is organized on disk and what changes across various version.

If the specification cannot be (yet) made public, you can reach out to scverse via [Zulip](https://scverse.zulipchat.com/#narrow/channel/443514-spatialdata-dev) via private message, or via email.

The advantage of a public specification is that it makes contributions and bugfixes from the community easier.

### The reader file

The reader files are located under [src/spatialdata_io/readers](https://github.com/scverse/spatialdata-io/tree/main/src/spatialdata_io/readers). For each supported technology, the reader file is a single function, named after the technology, that takes as input the path to the raw data and extra arguments, and returns a `SpatialData` object:

```python=
from pathlib import Path
from spatialdata import SpatialData


def technology(path: str | Path, ...) -> SpatialData:
    # `sdata` is constructed
    return sdata
```

For example, the reader for Xenium data, is called `xenium()` and is located under [`src/spatialdata_io/readers/xenium.py`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/xenium.py).

#### Technical notes

> Feel free to skip this section (and all the "technical notes") if you are doing a first read and just want an overview of the contribution process.

We recommend to study existing readers and reuse code from there. A few technical notes below.

- Large raster or points data is usually loaded from disk lazily (e.g. with `dask_image.imread()`), this allows to return a `SpatialData` object quickly, and defer the computation when saving the object to disk to the SpatialData Zarr format, with `sdata.write()`.
- When the raw data has multiple sample, we recommend to add a coordinate system for each sample, and if the samples are aligned in space, one common coordinate system. A single table containing the annotation for all the samples is preferred. See an example in the [`cosmx()`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/cosmx.py) reader.
- Small images should be represented as single-scale images (`xarray.DataArray`), large images as multiscale images (`xarray.DataTree`). The scale factors and chunk shape (`chunks`) should lead to chunks that fit the memory. See ane xample in [`visium()`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/visium.py).

#### Experimental readers

If a specification is not available a reader has less guarantee to work out-of-the-box. This is for instance the case for spatial omics datasets generated with in-house technologies. Even in this case users may find a reader beneficial: they may still have to adapt the reader for their use cases, but it could provide a good starting point to parse their data. We list these readers under the [`experimental`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/experimental/__init__.py) module. An example of this is the [`iss()`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/readers/iss.py) reader for In-Situ Sequencing data.

### Separating the constants from the reader logic

Each raw data format introduces specific naming schemes. To reduce the risk of typos and maintain an overview of all the names used in a reader, we list all the string constants in the file [src/spatialdata_io.\_constants/\_constants.py](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/_constants/_constants.py).

#### Techical notes

By using the decorator `inject_docs` you can include constants in the docstring of reader functions. [See an example here](https://github.com/scverse/spatialdata-io/blob/d2fe0bc18349093ad2cafce752590117729baee8/src/spatialdata_io/readers/xenium.py#L55).

## Adding tests: small data vs real data

An effective way to make a specification easier to understand and to improve the code review process is to provide test data.

### When small public test datasets are available

Our preferred way to test a reader is by using (very) small test datasets (e.g. 100kB-10MB). By being small, we can afford to use the datasets in continuous integration tests that are run via GitHub actions. See here the [GitHub workflow](https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml) that we use to download the test datasets, and here an [example of a test](https://github.com/scverse/spatialdata-io/blob/d2fe0bc18349093ad2cafce752590117729baee8/tests/test_xenium.py#L47) using the small datasets.

Notes:

- The small test datasets do not need to represent real data and can be entirely artificial.
- If a format specification contains edge cases, we recommend to employ a dataset that covers the various edge cases, or to use multiple test datasets.

### When only real (large) public data is available

Even when a small dataset is not available, we encourage to produce one. If this is not possible a public real dataset can be used. A possibility is also to use both: use small test datasets for CI testing, and a real dataset for double-checking real world usability and performance.

To streamline the usage of datasets we kindly ask you to provide two scripts via a PR to the [`spatialdata-sandbox`](https://github.com/giovp/spatialdata-sandbox) repository (not `spatialdata-io`): a script to download the data and a script to convert the downloaded data to the SpatialData Zarr format.

#### Folder structure

Please create a folder in the [root directory of `spatialdata-sandbox`](https://github.com/giovp/spatialdata-sandbox/tree/main) named after your technology. Please include a format version if availble. We also add the suffix `_io`: we use this to distinguish between scripts that reuse `spatialdata_io` for converting the data to SpatialData Zarr and scripts that parse the data from scratch.

An example is [`visium_hd_3.1.1_io`](https://github.com/giovp/spatialdata-sandbox/tree/main/visium_hd_3.1.1_io). Here [`download.py`](https://github.com/giovp/spatialdata-sandbox/blob/main/visium_hd_3.1.1_io/download.py) fetches Visium HD v3.1.1 data, and [`to_zarr.py`](https://github.com/giovp/spatialdata-sandbox/blob/main/visium_hd_3.1.1_io/to_zarr.py) uses `spatialdata_io.visium_hd()` to parse the data.

A note: `spatialdata-sandbox` is primarily used by scverse developers and, as the name suggests, we mainly use it as a sandbox for experimenting and for small scripts. We do not have pre-commit installed nor CI set-up, so feel free not to worry about the code style.

#### The `download.py` script

The `download.py` script simply downloads the raw data inside the subfolder `data` and, if it is zipped, it unzips it. Example [`download.py`](https://github.com/giovp/spatialdata-sandbox/blob/main/visium_hd_3.1.1_io/download.py) for Visium HD data.

When the data does not follow any specification (for instance for in-house data), it sometimes also performs some manipulation on the data and, in such cases, it stores the result in any convenient way. Example [`download.py`](https://github.com/giovp/spatialdata-sandbox/blob/main/merfish/download.py) for in-house MERFISH data.

#### The `to_zarr.py` script

The `to_zarr.py` script: (1) imports a suitable reader from `spatialdata_io`; (2) reads in the raw data from the folder `data` to obtain a `SpatialData` object; (3) saves the object to disk to a new Zarr store called `data.zarr`; (4) as a consistency check, reads the store back in memory to a new `SpatialData` object.

### The importance of public data and permissive licenses

There are a series of reasons to prefer public datasets provided with a permissive license (e.g. 10x public datasets often uses the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)):

- It allow the community to use the `download.py` script to download the data. An example of data accessible to the public but when this is not possible is when the data can be downloaded only after completing a questionnaire. For private data, collaborations and community contributions are difficult.
- It allow the community to modify (e.g. subset, to make download faster) and rehost the data. This makes it easier to share results and collaborate effectively.

### When only private data is available

If the data cannot be public yet, please reach out to scverse via [Zulip](https://scverse.zulipchat.com/#narrow/channel/443514-spatialdata-dev) via private message, or via email. Please give us access to a private repository where you can upload a `download.py` and `to_zarr.py` script working as described above. In alternative, you can also send us the data directly instead of the `download.py` script.

#### Technical notes

If the `download.py` and `to_zarr.py` scripts require Python imports for packages that are not available in an environment where `spatialdata` is installed, please provide also a `requirement.txt` file (see for instance a [legacy one here](https://github.com/giovp/spatialdata-sandbox/blob/main/merfish/requirements.txt)), specify the dependencies in the scripts as [inline script metadata (PEP 723)](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

## What to test

We encourage to test the reader function and any helper function.

### Testing multiple versions

When multiple versions of the raw data format are present, we encourage to test the reader on all of them to ensure backward compatibility. This task is greatly simplified if small tests datasets are used for the CI tests. If this is not available, we suggest to run the tests locally on multiple versions of the data before the PR is ready for review.

### Testing that the visualization and table annotations are correct

Using `spatialdata-plot` and/or `napari-spatialdata` to visualize the data is a quick way to easily spot issues with spatial alignment or table annotations.

We encourage to manually check the visualization of the data. Please refer to the [documentation of `spatialdata-plot`](https://spatialdata.scverse.org/projects/plot/en/stable/api.html), to [its example notebooks](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html#technology-specific), and to the [documentation of `napari-spatialdata`](https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html). Please add such scripts to `spatialdata-sandbox` so that they are reproducible by others.

In `napari-spatialdata` and `spatialdata-plot` we have tests that visually compare programmatic visualizations with reference images. We may consider adding such tests also for `spatialdata-io`, but for the moment these are not available. We recommend the two following proxy tests:

1. Proxy test to check that the spatial aligment is correct. This can be achieved by testing that the data extent (i.e. the visual bounds of the data) is correct. The extent can be computed with [`spatialdata.get_extent()`](https://spatialdata.scverse.org/en/stable/api/operations.html#spatialdata.get_extent). Here is an [example of such a test](https://github.com/LucaMarconato/spatialdata-io/blob/1df4a2261ee7f82469fa67cb83bdb856d69f6dac/tests/test_xenium.py#L51).
2. Proxy test to check that the coloring of the spatial entities are correct. This can be achieved by checking that for each element, and for a small set of instance ids (i.e. data indices), and for a few feature columns, the values are correct. Here is an [example of such a test](https://github.com/LucaMarconato/spatialdata-io/blob/1df4a2261ee7f82469fa67cb83bdb856d69f6dac/tests/test_xenium.py#L79).

Note: writing these tests require some manual work to explore the data. If you find this too time-consuming, we could consider adding some external scripts to create these tests automatically: if you are interested please reach out to us.

### Testing auxiliary functions

Some technology make use of auxiliary functions to parse the data. We encourage to test these functions as well. Here is an [example of such an auxiliary function for Xenium data](https://github.com/LucaMarconato/spatialdata-io/blob/4ee33da99781ccb2ea284be0614bdaaf69bfb2ed/src/spatialdata_io/readers/xenium.py#L771) and the [corresponding test](https://github.com/LucaMarconato/spatialdata-io/blob/4ee33da99781ccb2ea284be0614bdaaf69bfb2ed/tests/test_xenium.py#L19).

## Adding a new converter

Readers are developed to parse raw data from spatial omics technologies. We also provide converters to convert data from/to other general data formats, i.e. data formats that are not specific to a single (or a small set) of spatial omics technologies.
An example of this is the legacy `AnnData` spatial format, used in early versions of `squidpy`.

Converters are not the primary scope of `spatialdata-io`, so we will just give some general indications: if you are interested in contributing to a converter, please adapt and follow the guidelines as for readers (in particular link to a specification, provide test data, and write extensive tests). Feedback is welcome: if you find it useful we can expand this section of the contribution guide.

## Updating the CLI for readers and converters

The readers and converters from `spatialdata-io` can be invoked via the command line (see the [CLI documentation](https://spatialdata.scverse.org/projects/io/en/stable/cli.html)). This Python file defines the CLI: [src/spatialdata_io/**main**.py](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/__main__.py). Please if you add or modify a reder or converter, update the CLI accordingly.

#### Technical notes

- In the future we may consider to automatically generate the CLI from the readers and converters, [see more here](https://github.com/scverse/spatialdata-io/pull/239#issuecomment-2588005228).
- Keeping the CLI code up-to-date could be a good task for the GitHub Copilot code agent. We will experiment with this in the future.

# Bug tracking

- tracking bugs also outside spatialdata-io
- tracking bugs also in spatialdata

# Limitations and possible improvements

## Unsupported technologies

## Naming constistency across readers

## Pixel space vs physical space

## Transformations between coordinate systems

## Summary table of supported technlogies and specifications

# Checklist
