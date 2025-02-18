from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import anndata
import dask.dataframe as dd
import geopandas
import numpy as np
import pandas as pd
import xarray
from dask import array as da
from dask_image.imread import imread
from shapely.geometry import MultiPolygon
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation

from spatialdata_io._constants._constants import MerscopeKeys
from spatialdata_io._docs import inject_docs

SUPPORTED_BACKENDS = ["dask_image", "rioxarray"]


def _get_channel_names(images_dir: Path) -> list[str]:
    exp = r"mosaic_(?P<stain>[\w|-]+[0-9]?)_z(?P<z>[0-9]+).tif"
    matches = [re.search(exp, file.name) for file in images_dir.iterdir()]

    stainings = {match.group("stain") for match in matches if match}

    return list(stainings)


def _get_file_paths(path: Path, vpt_outputs: Path | str | dict[str, Any] | None) -> tuple[Path, Path, Path]:
    """
    Gets the MERSCOPE file paths when vpt_outputs is provided

    That is, (i) the file of transcript per cell, (ii) the cell metadata file, and (iii) the cell boundary file
    """
    if vpt_outputs is None:
        return (
            path / MerscopeKeys.COUNTS_FILE,
            path / MerscopeKeys.CELL_METADATA_FILE,
            path / MerscopeKeys.BOUNDARIES_FILE,
        )

    if isinstance(vpt_outputs, str) or isinstance(vpt_outputs, Path):
        vpt_outputs = Path(vpt_outputs)

        plausible_boundaries = [
            vpt_outputs / MerscopeKeys.CELLPOSE_BOUNDARIES,
            vpt_outputs / MerscopeKeys.WATERSHED_BOUNDARIES,
        ]
        valid_boundaries = [path for path in plausible_boundaries if path.exists()]

        assert (
            valid_boundaries
        ), f"Boundary file not found - expected to find one of these files: {', '.join(map(str, plausible_boundaries))}"

        return (
            vpt_outputs / MerscopeKeys.COUNTS_FILE,
            vpt_outputs / MerscopeKeys.CELL_METADATA_FILE,
            valid_boundaries[0],
        )

    if isinstance(vpt_outputs, dict):
        return (
            vpt_outputs[MerscopeKeys.VPT_NAME_COUNTS],
            vpt_outputs[MerscopeKeys.VPT_NAME_OBS],
            vpt_outputs[MerscopeKeys.VPT_NAME_BOUNDARIES],
        )

    raise ValueError(
        f"`vpt_outputs` has to be either `None`, `str`, `Path`, or `dict`. Found type {type(vpt_outputs)}."
    )


@inject_docs(ms=MerscopeKeys)
def merscope(
    path: str | Path,
    vpt_outputs: Path | str | dict[str, Any] | None = None,
    z_layers: int | list[int] | None = 3,
    region_name: str | None = None,
    slide_name: str | None = None,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    cells_boundaries: bool = True,
    cells_table: bool = True,
    mosaic_images: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *MERSCOPE* data from Vizgen.

    This function reads the following files:

        - ``{ms.COUNTS_FILE!r}``: Counts file.
        - ``{ms.TRANSCRIPTS_FILE!r}``: Transcript file.
        - ``{ms.CELL_METADATA_FILE!r}``: Per-cell metadata file.
        - ``{ms.BOUNDARIES_FILE!r}``: Cell polygon boundaries.
        - `mosaic_**_z*.tif` images inside the ``{ms.IMAGES_DIR!r}`` directory.

    Parameters
    ----------
    path
        Path to the region/root directory containing the *Merscope* files (e.g., `detected_transcripts.csv`).
    vpt_outputs
        Optional arguments to indicate the output of the vizgen-postprocessing-tool (VPT), when used.
        If a folder path is provided, it looks inside the folder for the following files:

            - ``{ms.COUNTS_FILE!r}``
            - ``{ms.CELL_METADATA_FILE!r}``
            - ``{ms.BOUNDARIES_FILE!r}``

        If a dictionary, then the following keys should be provided with the desired path:

            - ``{ms.VPT_NAME_COUNTS!r}``
            - ``{ms.VPT_NAME_OBS!r}``
            - ``{ms.VPT_NAME_BOUNDARIES!r}``
    z_layers
        Indices of the z-layers to consider. Either one `int` index, or a list of `int` indices. If `None`, then no image is loaded.
        By default, only the middle layer is considered (that is, layer 3).
    region_name
        Name of the region of interest, e.g., `'region_0'`. If `None` then the name of the `path` directory is used.
    slide_name
        Name of the slide/run. If `None` then the name of the parent directory of `path` is used (whose name starts with a date).
    backend
        Either `"dask_image"` or `"rioxarray"` (the latter uses less RAM, but requires `rioxarray` to be installed). By default, uses `"rioxarray"` if and only if the `rioxarray` library is installed.
    transcripts
        Whether to read transcripts.
    cells_boundaries
        Whether to read cell boundaries (polygons).
    cells_table
        Whether to read cells table.
    mosaic_images
        Whether to read the mosaic images.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    assert (
        backend is None or backend in SUPPORTED_BACKENDS
    ), f"Backend '{backend} not supported. Should be one of: {', '.join(SUPPORTED_BACKENDS)}"

    path = Path(path).absolute()
    count_path, obs_path, boundaries_path = _get_file_paths(path, vpt_outputs)
    images_dir = path / MerscopeKeys.IMAGES_DIR

    microns_to_pixels = Affine(
        np.genfromtxt(images_dir / MerscopeKeys.TRANSFORMATION_FILE), input_axes=("x", "y"), output_axes=("x", "y")
    )
    transformations = {"global": microns_to_pixels}

    vizgen_region = path.name if region_name is None else region_name
    slide_name = path.parent.name if slide_name is None else slide_name
    dataset_id = f"{slide_name}_{vizgen_region}"
    region = f"{dataset_id}_polygons"

    # Images
    images = {}

    if mosaic_images:
        z_layers = [z_layers] if isinstance(z_layers, int) else z_layers or []

        reader = _get_reader(backend)

        stainings = _get_channel_names(images_dir)
        if stainings:
            for z_layer in z_layers:
                images[f"{dataset_id}_z{z_layer}"] = reader(
                    images_dir,
                    stainings,
                    z_layer,
                    image_models_kwargs,
                    **imread_kwargs,
                )

    # Transcripts
    points = {}

    if transcripts:
        transcript_path = path / MerscopeKeys.TRANSCRIPTS_FILE
        if transcript_path.exists():
            points[f"{dataset_id}_transcripts"] = _get_points(transcript_path, transformations)
        else:
            logger.warning(f"Transcript file {transcript_path} does not exist. Transcripts are not loaded.")

    # Polygons
    shapes = {}

    if cells_boundaries:
        if boundaries_path.exists():
            shapes[f"{dataset_id}_polygons"] = _get_polygons(boundaries_path, transformations)
        else:
            logger.warning(f"Boundary file {boundaries_path} does not exist. Cell boundaries are not loaded.")

    # Tables
    tables = {}

    if cells_table:
        if count_path.exists() and obs_path.exists():
            tables["table"] = _get_table(count_path, obs_path, vizgen_region, slide_name, dataset_id, region)
        else:
            logger.warning(
                f"At least one of the following files does not exist: {count_path}, {obs_path}. The table is not loaded."
            )

    return SpatialData(shapes=shapes, points=points, images=images, tables=tables)


def _get_reader(backend: str | None) -> Callable:  # type: ignore[type-arg]
    if backend is not None:
        return _rioxarray_load_merscope if backend == "rioxarray" else _dask_image_load_merscope
    try:
        import rioxarray  # noqa: F401

        return _rioxarray_load_merscope
    except ModuleNotFoundError:
        return _dask_image_load_merscope


def _rioxarray_load_merscope(
    images_dir: Path,
    stainings: list[str],
    z_layer: int,
    image_models_kwargs: Mapping[str, Any],
    **kwargs: Any,
) -> Image2DModel:
    logger.info("Using rioxarray backend.")

    try:
        import rioxarray
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Using rioxarray backend requires to install the rioxarray library (`pip install rioxarray`)"
        )
    from rasterio.errors import NotGeoreferencedWarning

    warnings.simplefilter("ignore", category=NotGeoreferencedWarning)

    im = xarray.concat(
        [
            rioxarray.open_rasterio(
                images_dir / f"mosaic_{stain}_z{z_layer}.tif",
                chunks=image_models_kwargs["chunks"],
                **kwargs,
            )
            .rename({"band": "c"})
            .reset_coords("spatial_ref", drop=True)
            for stain in stainings
        ],
        dim="c",
    )

    return Image2DModel.parse(im, c_coords=stainings, rgb=None, **image_models_kwargs)


def _dask_image_load_merscope(
    images_dir: Path,
    stainings: list[str],
    z_layer: int,
    image_models_kwargs: Mapping[str, Any],
    **kwargs: Any,
) -> Image2DModel:
    im = da.stack(
        [imread(images_dir / f"mosaic_{stain}_z{z_layer}.tif", **kwargs).squeeze() for stain in stainings],
        axis=0,
    )

    return Image2DModel.parse(
        im,
        dims=("c", "y", "x"),
        c_coords=stainings,
        rgb=None,
        **image_models_kwargs,
    )


def _get_points(transcript_path: Path, transformations: dict[str, BaseTransformation]) -> dd.DataFrame:
    transcript_df = dd.read_csv(transcript_path)
    transcripts = PointsModel.parse(
        transcript_df,
        coordinates={"x": MerscopeKeys.GLOBAL_X, "y": MerscopeKeys.GLOBAL_Y},
        transformations=transformations,
        feature_key=MerscopeKeys.GENE_KEY,
    )
    transcripts["gene"] = transcripts["gene"].astype("category")
    return transcripts


def _get_polygons(boundaries_path: Path, transformations: dict[str, BaseTransformation]) -> geopandas.GeoDataFrame:
    geo_df = geopandas.read_parquet(boundaries_path)
    geo_df = geo_df.rename_geometry("geometry")
    geo_df = geo_df[geo_df[MerscopeKeys.Z_INDEX] == 0]  # Avoid duplicate boundaries on all z-levels
    geo_df = geo_df[geo_df.geometry.is_valid]  # Remove invalid geometries
    geo_df.geometry = geo_df.geometry.map(lambda x: MultiPolygon(x.geoms))
    geo_df.index = geo_df[MerscopeKeys.METADATA_CELL_KEY].astype(str)

    return ShapesModel.parse(geo_df, transformations=transformations)


def _get_table(
    count_path: Path,
    obs_path: Path,
    vizgen_region: str,
    slide_name: str,
    dataset_id: str,
    region: str,
) -> anndata.AnnData:
    data = pd.read_csv(count_path, index_col=0, dtype={MerscopeKeys.COUNTS_CELL_KEY: str})
    obs = pd.read_csv(obs_path, index_col=0, dtype={MerscopeKeys.METADATA_CELL_KEY: str})

    is_gene = ~data.columns.str.lower().str.contains("blank")
    adata = anndata.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)

    adata.obsm["blank"] = data.loc[:, ~is_gene]  # blank fields are excluded from adata.X
    adata.obsm["spatial"] = adata.obs[[MerscopeKeys.CELL_X, MerscopeKeys.CELL_Y]].values
    adata.obs["region"] = pd.Series(vizgen_region, index=adata.obs_names, dtype="category")
    adata.obs["slide"] = pd.Series(slide_name, index=adata.obs_names, dtype="category")
    adata.obs["dataset_id"] = pd.Series(dataset_id, index=adata.obs_names, dtype="category")
    adata.obs[MerscopeKeys.REGION_KEY] = pd.Series(region, index=adata.obs_names, dtype="category")
    adata.obs[MerscopeKeys.METADATA_CELL_KEY] = adata.obs.index

    table = TableModel.parse(
        adata,
        region_key=MerscopeKeys.REGION_KEY.value,
        region=region,
        instance_key=MerscopeKeys.METADATA_CELL_KEY.value,
    )
    return table
