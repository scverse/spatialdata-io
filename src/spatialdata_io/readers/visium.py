from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Tuple

import numpy as np
import pandas as pd
from dask_image.imread import imread
from imageio import imread as imread2
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Identity, Scale
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import _read_counts

__all__ = ["visium"]


@inject_docs(vx=VisiumKeys)
def visium(
    path: str | Path,
    dataset_id: str | None = None,
    counts_file: str = VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file: str | Path | None = None,
    tissue_positions_file: str | Path | None = None,
    scalefactors_file: str | Path | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> SpatialData:
    """
    Read *10x Genomics* Visium formatted dataset.

    This function reads the following files:

        - ``(<dataset_id>_)`{vx.FILTERED_COUNTS_FILE!r}```: Counts and metadata file.
        - ``{vx.IMAGE_HIRES_FILE!r}``: High resolution image.
        - ``{vx.IMAGE_LOWRES_FILE!r}``: Low resolution image.
        - ``{vx.SCALEFACTORS_FILE!r}``: Scalefactors file.
        - ``{vx.SPOTS_FILE_1!r}`` (SpaceRanger 1) or ``{vx.SPOTS_FILE_2!r}`` (SpaceRanger 2):
            Spots positions file.
        - ``fullres_image_file``: large microscopy image used as input for space ranger.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier to name the constructed `SpatialData` elements. The reader will try to infer it from the
        counts_file filen name (which defaults to ``{vx.FILTERED_COUNTS_FILE!r}``) file name. If the file name does not
        contain the dataset id, it needs to be provided.
    counts_file
        Name of the counts file, defaults to ``{vx.FILTERED_COUNTS_FILE!r}``; a common alternative is
        ``{vx.RAW_COUNTS_FILE!r}``. Also, use when the file names are not following the standard SpaceRanger
        conventions.
    fullres_image_file
        Path to the full-resolution image.
    tissue_positions_file
        Path to the tissue positions file.
    scalefactors_file
        Path to the scalefactors file.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    imread_kwargs = dict(imread_kwargs)
    image_models_kwargs = dict(image_models_kwargs)

    # try to infer library_id from the counts file
    library_id = None
    try:
        patt = re.compile(f".*{counts_file}")
        first_file = [i for i in os.listdir(path) if patt.match(i)][0]

        if f"_{counts_file}" in first_file:
            library_id = first_file.replace(f"_{counts_file}", "")
            counts_file = f"{library_id}_{counts_file}"
        elif counts_file == first_file:
            library_id = None
            counts_file = counts_file
        else:
            raise ValueError(
                f"Cannot determine the library_id. Expecting a file with format (<library_id>_)"
                f"{counts_file}. If the files have been renamed you may need to specify their file "
                f"names (not their paths), with some of the following arguments: `counts_file`, `fullres_image_file`, "
                f"`tissue_positions_file`, `scalefactors_file` arguments."
            )
    except IndexError as e:
        if counts_file is None:
            logger.error(
                f"{e}. \nError is due to the fact that the library id could not be found, if the counts file is `.mtx` "
                f"(or else), Please provide a `counts_file`.",
            )
            raise e
    assert counts_file is not None

    if library_id is None and dataset_id is None:
        raise ValueError("Cannot determine the `library_id`. Please provide `dataset_id`.")

    if dataset_id is not None:
        if dataset_id != library_id and library_id is not None:
            logger.warning(
                f"`dataset_id: {dataset_id}` does not match `library_id: {library_id}`. `dataset_id: {dataset_id}` "
                f"will be used to build SpatialData."
            )
        library_id = dataset_id
    else:
        dataset_id = library_id
    assert dataset_id is not None

    # The second element of the returned tuple is the full library as contained in the metadata of
    # VisiumKeys.FILTERED_COUNTS_FILE. For instance, for the spatialdata-sandbox/visium dataset it is:
    #     spaceranger100_count_30458_ST8059048_mm10-3_0_0_premrna
    # We discard this value and use the one inferred from the filename of VisiumKeys.FILTERED_COUNTS_FILE, or the one
    # provided by the user in dataset_id
    adata, _ = _read_counts(path, counts_file=counts_file, library_id=library_id, **kwargs)

    if (path / "spatial" / VisiumKeys.SPOTS_FILE_1).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_1) in str(tissue_positions_file)
    ):
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_1
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    elif (path / "spatial" / VisiumKeys.SPOTS_FILE_2).exists() or (
        tissue_positions_file is not None and str(VisiumKeys.SPOTS_FILE_2) in str(tissue_positions_file)
    ):
        tissue_positions_file = (
            path / "spatial" / VisiumKeys.SPOTS_FILE_2
            if tissue_positions_file is None
            else path / tissue_positions_file
        )
    else:
        raise ValueError(f"Cannot find `tissue_positions` file in `{path}`.")
    coords = pd.read_csv(tissue_positions_file, header=None, index_col=0)

    if tissue_positions_file.name == VisiumKeys.SPOTS_FILE_1:
        # spaceranger_version < 2.0.0 but header detected: https://github.com/scverse/spatialdata-io/issues/146
        if "in_tissue" in coords.iloc[0].tolist():
            coords = pd.read_csv(tissue_positions_file, header=0, index_col=0)
        else:
            assert "in_tissue" not in coords.columns
            coords.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    else:
        assert tissue_positions_file.name == VisiumKeys.SPOTS_FILE_2
        coords = pd.read_csv(tissue_positions_file, header=0, index_col=0)

    adata.obs = pd.merge(adata.obs, coords, how="left", left_index=True, right_index=True)
    coords = adata.obs[[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y]].values
    adata.obsm["spatial"] = coords
    adata.obs = pd.DataFrame(adata.obs)
    adata.obs.drop(columns=[VisiumKeys.SPOTS_X, VisiumKeys.SPOTS_Y], inplace=True)
    adata.obs["spot_id"] = np.arange(len(adata))
    adata.var_names_make_unique()

    if not adata.obs_names.is_unique:
        logger.info("Non-unique obs names detected, calling `obs_names_make_unique`.")
        # This is required for napari-spatialdata because of the join operation that would otherwise fail
        adata.obs_names_make_unique()

    if (path / "spatial" / VisiumKeys.SCALEFACTORS_FILE).exists() or (
        scalefactors_file is not None and (path / scalefactors_file).exists()
    ):
        scalefactors_file = (
            path / "spatial" / VisiumKeys.SCALEFACTORS_FILE if scalefactors_file is None else path / scalefactors_file
        )
        scalefactors = json.loads(scalefactors_file.read_bytes())

    transform_original = Identity()
    transform_lowres = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_LOWRES], scalefactors[VisiumKeys.SCALEFACTORS_LOWRES]]),
        axes=("y", "x"),
    )
    transform_hires = Scale(
        np.array([scalefactors[VisiumKeys.SCALEFACTORS_HIRES], scalefactors[VisiumKeys.SCALEFACTORS_HIRES]]),
        axes=("y", "x"),
    )
    shapes = {}
    circles = ShapesModel.parse(
        coords,
        geometry=0,
        radius=scalefactors["spot_diameter_fullres"] / 2.0,
        index=adata.obs["spot_id"].copy(),
        transformations={
            "global": transform_original,
            "downscaled_hires": transform_hires,
            "downscaled_lowres": transform_lowres,
        },
    )
    shapes[dataset_id] = circles
    adata.obs["region"] = dataset_id
    table = TableModel.parse(adata, region=dataset_id, region_key="region", instance_key="spot_id")

    images = {}
    if fullres_image_file is not None:
        fullres_image_file = path / Path(fullres_image_file)
        if fullres_image_file.exists():
            image = _read_image(fullres_image_file, imread_kwargs)
            full_image = DataArray(image, dims=("c", "y", "x"))
            images[dataset_id + "_full_image"] = Image2DModel.parse(
                full_image,
                scale_factors=[2, 2, 2, 2],
                transformations={"global": transform_original},
                rgb=None,
                **image_models_kwargs,
            )
        else:
            logger.warning(f"File {fullres_image_file} does not exist, skipping...")

    if (path / VisiumKeys.IMAGE_HIRES_FILE).exists():
        image_hires = imread(path / VisiumKeys.IMAGE_HIRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
        image_hires = DataArray(image_hires, dims=("c", "y", "x"))
        images[dataset_id + "_hires_image"] = Image2DModel.parse(
            image_hires, transformations={"downscaled_hires": Identity()}, rgb=None
        )
    if (path / VisiumKeys.IMAGE_LOWRES_FILE).exists():
        image_lowres = imread(path / VisiumKeys.IMAGE_LOWRES_FILE, **imread_kwargs).squeeze().transpose(2, 0, 1)
        image_lowres = DataArray(image_lowres, dims=("c", "y", "x"))
        images[dataset_id + "_lowres_image"] = Image2DModel.parse(
            image_lowres,
            transformations={"downscaled_lowres": Identity()},
            rgb=None,
        )

    return SpatialData(images=images, shapes=shapes, table=table)


def _read_image(image_file: Path, imread_kwargs: dict[str, Any]) -> Any:
    if "MAX_IMAGE_PIXELS" in imread_kwargs:
        from PIL import Image as ImagePIL

        ImagePIL.MAX_IMAGE_PIXELS = imread_kwargs.pop("MAX_IMAGE_PIXELS")
    if image_file.suffix != ".btf":
        im = imread(image_file, **imread_kwargs)
    else:
        # dask_image doesn't recognize .btf automatically
        im = imread2(image_file, **imread_kwargs)
        # Depending on the versions of the pipeline, the axes of the image file from the tiff data is ordered in
        # different ways; here let's implement a simple check on the shape to determine the axes ordering.
        # Note that a more robust check could be implemented; this could be the work of a future PR. Unfortunately,
        # the tif data does not (or does not always) have OME metadata, so even such more general parser could lead
        # to edge cases that could be addressed by a more interoperable file format.
    if len(im.shape) not in [3, 4]:
        raise ValueError(f"Image shape {im.shape} is not supported.")
    if len(im.shape) == 4:
        if im.shape[0] == 1:
            im = im.squeeze(0)
        else:
            raise ValueError(f"Image shape {im.shape} is not supported.")
    # for immunofluerence images there could be an arbitrary number of channels (usually, 2, 3 or 4); we can detect this
    # as the dimension which has the minimum size
    min_size = np.argmin(im.shape)
    if min_size == 0:
        image = im
    elif min_size == 2:
        image = im.transpose(2, 0, 1)
    else:
        raise ValueError(f"Image shape {im.shape} is not supported.")
    return image


def get_sdata_res(sdata: SpatialData):
    """
    Retrieve the image resolution from the Visium SpatialData object.

    This function extracts the shape (resolution) of the highest resolution image (scale0)
    from the Visium SpatialData object. The shape is returned as the number of channels (c),
    height (y), and width (x) in pixels.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object containing images and spatial data, with image resolutions stored
        in a multi-scale format.

    Returns
    -------
    shape : tuple
        A tuple containing the image shape in the format (channels, height, width):
        - c: Number of image channels (typically RGB).
        - y: Image height in pixels.
        - x: Image width in pixels.
    """

    image_name = list(sdata.images.keys())[0]
    dimensions = sdata.images[image_name]["scale0"].dims
    shape = dimensions["c"], dimensions["y"], dimensions["x"]
    return shape


class SpotPacking(Enum):
    """Types of ST spots disposition,
    for Orange Crate Packing see:
    https://kb.10xgenomics.com/hc/en-us/articles/360041426992-Where-can-I-find-the-Space-Ranger-barcode-whitelist-and-their-coordinates-on-the-slide
    """

    ORANGE_CRATE_PACKING = 0
    GRID_PACKING = 1


def find_pixel_size_visium(
    my_df: pd.DataFrame, inter_spot_dist: float = 100.0, packing: SpotPacking = SpotPacking.ORANGE_CRATE_PACKING
) -> tuple[float, int]:
    """Estimate the pixel size of an image in um/px given a dataframe containing the spot coordinates in that image

    Args:
        my_df (pd.DataFrame): dataframe containing the coordinates of each spot in an image, it must contain the following columns:
            ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_col', 'array_row']
        inter_spot_dist (float, optional): the distance in um between two spots on the same row. Defaults to 100..
        packing (SpotPacking, optional): disposition of the spots on the slide. Defaults to SpotPacking.ORANGE_CRATE_PACKING.

    Raises:
        Exception: if cannot find two spots on the same row

    Returns:
        Tuple[float, int]: approximation of the pixel size in um/px and over how many spots that pixel size was estimated
    """

    def _cart_dist(start_spot, end_spot):
        """cartesian distance in pixel between two spots"""
        d = np.sqrt(
            (start_spot["pxl_col_in_fullres"] - end_spot["pxl_col_in_fullres"]) ** 2
            + (start_spot["pxl_row_in_fullres"] - end_spot["pxl_row_in_fullres"]) ** 2
        )
        return d

    df = my_df.copy()

    max_dist_col = 0
    approx_nb = 0
    best_approx = 0
    df = df.sort_values("array_row")
    for _, row in df.iterrows():
        y = row["array_col"]
        x = row["array_row"]
        if len(df[df["array_row"] == x]) > 1:
            b = df[df["array_row"] == x]["array_col"].idxmax()
            start_spot = row
            end_spot = df.loc[b]
            dist_px = _cart_dist(start_spot, end_spot)

            div = 1 if packing == SpotPacking.GRID_PACKING else 2
            dist_col = abs(df.loc[b, "array_col"] - y) // div

            approx_nb += 1

            if dist_col > max_dist_col:
                max_dist_col = dist_col
                best_approx = inter_spot_dist / (dist_px / dist_col)
            if approx_nb > 3:
                break

    if approx_nb == 0:
        raise Exception("Pixel size estimation failed. Couldn't find two spots on the same row")

    return best_approx, max_dist_col


def create_df_coord_visium(data: SpatialData):
    """
    Create a DataFrame with coordinates and array indices from Visium SpatialData.

    This function processes the spatial shapes and table data from a Visium SpatialData object
    to generate a DataFrame containing pixel coordinates (row and column) for each spot
    on the tissue image at full resolution. It also includes the corresponding array row and column
    indices from the data tables.

    Parameters
    ----------
    data : SpatialData
        A SpatialData object containing Visium spatial information, including shapes and table data
        (spot coordinates and array indices).

    Returns
    -------
    df_coord : pandas.DataFrame
        A DataFrame with the following columns:
        - 'pxl_row_in_fullres': Pixel row coordinates in full-resolution tissue image.
        - 'pxl_col_in_fullres': Pixel column coordinates in full-resolution tissue image.
        - 'array_row': Row index of the spot in the Visium array.
        - 'array_col': Column index of the spot in the Visium array.
    """
    tissue_name = list(data.shapes.keys())[0]
    shapes_df = data.shapes[tissue_name]
    shapes_df["pxl_col_in_fullres"] = shapes_df.geometry.apply(lambda geom: geom.x)
    shapes_df["pxl_row_in_fullres"] = shapes_df.geometry.apply(lambda geom: geom.y)

    shapes_df["array_row"] = list(data.tables["table"].obs["array_row"])
    shapes_df["array_col"] = list(data.tables["table"].obs["array_col"])

    # Now, you have the necessary DataFrame in the correct format:
    df_coord = shapes_df[["pxl_row_in_fullres", "pxl_col_in_fullres", "array_row", "array_col"]]
    return df_coord


def calculate_pixel_size_from_visium(
    path: str,
    dataset_id: str,
    counts_file: str,
    fullres_image_file: str,
    tissue_positions_file: str,
    scalefactors_file: str,
    inter_spot_dist: float = 100.0,
) -> SpatialData:
    """
    Main function to load data into a spatialdata class and calculate scale0 image shape and pixel size.

    Parameters
    ----------
    path : str
        Path to the directory containing the data.
    dataset_id : str
        ID of the dataset to use.
    counts_file : str
        Path to the filtered feature barcode matrix (counts file).
    fullres_image_file : str
        Path to the full-resolution image file (usually tissue_hires_image.png).
    tissue_positions_file : str
        Path to the tissue positions file (usually tissue_positions_list.csv).
    scalefactors_file : str
        Path to the scalefactors file (usually scalefactors_json.json).
    inter_spot_dist : float, optional
        Distance between 2 spots in a visium field. Default value = 100um.

    Returns
    -------
    visium_sdata : SpatialData
        SpatialData object that includes both image shape and pixel size stored in tables['table'].uns
    """

    # Load the SpatialData object using the visium function. 6 files are expected to be passed from Visium raw data
    visium_sdata = visium(
        path=path,
        dataset_id=dataset_id,
        counts_file=counts_file,
        fullres_image_file=fullres_image_file,
        tissue_positions_file=tissue_positions_file,
        scalefactors_file=scalefactors_file,
    )

    df_coord = create_df_coord_visium(visium_sdata)

    pixel_size, _ = find_pixel_size_visium(df_coord, inter_spot_dist)

    image_shape = get_sdata_res(visium_sdata)

    visium_sdata.tables["table"].uns["image_shape"] = image_shape
    visium_sdata.tables["table"].uns["pixel_size"] = pixel_size

    return visium_sdata
