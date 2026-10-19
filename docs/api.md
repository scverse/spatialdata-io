# API

This section documents the Application Programming Interface (API) for the `spatialdata_io` package.

```{eval-rst}
.. module:: spatialdata_io
```

## spatialdata-io

I/O for the `spatialdata` project.

### Readers

#### Spatial technologies

```{eval-rst}
.. autosummary::
    :toctree: generated

    codex
    cosmx
    curio
    dbit
    experimental.iss
    mcmicro
    merscope
    seqfish
    steinbock
    stereoseq
    visium
    visium_hd
    xenium
```

#### Data types

```{eval-rst}
.. autosummary::
    :toctree: generated

    generic
    image
    geojson
```

### Conversion functions

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    experimental.from_legacy_anndata
    experimental.to_legacy_anndata
```

### Experimental spatial tiling

These opt-in functions currently target Xenium-compatible stores. The canonical profile
rewrites Points and Shapes Parquet into spatial row groups, stores Shapes as GeoArrow,
adds table metadata and a gene-major `X_csc` layer, and publishes discovery metadata in
the root Zarr attributes. It creates no separate visualization directory.

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    experimental.add_spatial_tiling
    experimental.xenium_spatially_tiled
```

### Utility functions

```{eval-rst}
.. currentmodule:: spatialdata_io

.. autosummary::
    :toctree: generated

    xenium_aligned_image
    xenium_explorer_selection
```
