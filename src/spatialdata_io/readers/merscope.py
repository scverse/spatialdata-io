from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import anndata
import dask.dataframe as dd
import geopandas
import numpy as np
import pandas as pd
from dask import array as da
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity

from spatialdata_io._constants._constants import MerscopeKeys
from spatialdata_io._docs import inject_docs


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

        If a dictionnary, then the following keys should be provided with the desired path:

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

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    count_path, obs_path, boundaries_path = _get_file_paths(path, vpt_outputs)
    images_dir = path / MerscopeKeys.IMAGES_DIR

    microns_to_pixels = np.genfromtxt(images_dir / MerscopeKeys.TRANSFORMATION_FILE)
    microns_to_pixels = Affine(microns_to_pixels, input_axes=("x", "y"), output_axes=("x", "y"))

    region_name = path.name if region_name is None else region_name
    slide_name = path.parent.name if slide_name is None else slide_name
    dataset_id = f"{slide_name}_{region_name}"

    # Images
    images = {}

    z_layers = [z_layers] if isinstance(z_layers, int) else z_layers or []

    stainings = _get_channel_names(images_dir)
    for z_layer in z_layers:
        im = da.stack([imread(images_dir / f"mosaic_{stain}_z{z_layer}.tif").squeeze() for stain in stainings], axis=0)
        parsed_im = Image2DModel.parse(
            im,
            dims=("c", "y", "x"),
            transformations={"pixels": Identity()},
            c_coords=stainings,
        )
        images[f"{dataset_id}_z{z_layer}"] = parsed_im

    # Transcripts
    transcript_df = dd.read_csv(path / MerscopeKeys.TRANSCRIPTS_FILE)
    transcripts = PointsModel.parse(
        transcript_df,
        coordinates={"x": MerscopeKeys.GLOBAL_X, "y": MerscopeKeys.GLOBAL_Y},
        transformations={"pixels": microns_to_pixels},
    )
    categories = transcripts["gene"].compute().astype("category")
    gene_categorical = dd.from_pandas(categories, npartitions=transcripts.npartitions).reset_index(drop=True)
    transcripts["gene"] = gene_categorical

    points = {f"{dataset_id}_transcripts": transcripts}

    # Polygons
    geo_df = geopandas.read_parquet(boundaries_path)
    geo_df = geo_df.rename_geometry("geometry")
    geo_df = geo_df[geo_df[MerscopeKeys.Z_INDEX] == 0]  # Avoid duplicate boundaries on all z-levels
    geo_df.index = geo_df[MerscopeKeys.INSTANCE_KEY].astype(str)

    polygons = ShapesModel.parse(geo_df, transformations={"pixels": microns_to_pixels})

    shapes = {f"{dataset_id}_polygons": polygons}

    # Table
    data = pd.read_csv(count_path, index_col=0, dtype={MerscopeKeys.COUNTS_CELL_KEY: str})
    obs = pd.read_csv(obs_path, index_col=0, dtype={MerscopeKeys.INSTANCE_KEY: str})

    is_gene = ~data.columns.str.lower().str.contains("blank")
    adata = anndata.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)

    adata.obsm["blank"] = data.loc[:, ~is_gene]  # blank fields are excluded from adata.X
    adata.obsm["spatial"] = adata.obs[[MerscopeKeys.CELL_X, MerscopeKeys.CELL_Y]].values
    adata.obs["region"] = pd.Series(region_name, index=adata.obs_names, dtype="category")
    adata.obs["slide"] = pd.Series(slide_name, index=adata.obs_names, dtype="category")
    adata.obs["dataset_id"] = pd.Series(dataset_id, index=adata.obs_names, dtype="category")
    adata.obs[MerscopeKeys.INSTANCE_KEY] = adata.obs.index

    table = TableModel.parse(
        adata,
        region_key="region",
        region=adata.obs["region"].cat.categories.tolist(),
        instance_key=MerscopeKeys.INSTANCE_KEY.value,
    )

    return SpatialData(shapes=shapes, points=points, images=images, table=table)
