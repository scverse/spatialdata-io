from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import shapely
import spatialdata as sd
from dask_image.imread import imread
from numpy.typing import NDArray
from spatialdata import SpatialData
from xarray import DataArray

from spatialdata_io._constants._constants import DBiTKeys

__all__ = ["DBiT"]


def xy2edges(xy: list[int], scale: float = 1.0, border: bool = True, border_scale: float = 1) -> NDArray:
    """
    Construct vertex coordinate of a square from the barcode coordinates.
    The constructed square can have a border, that is scalable

    Parameters
    ----------
    xy : list[int]
        coordinate of the spot identified by its barcode.
    scale : float, optional
        Resize the square.
        The default is 1.0.
    border : bool, optional
        If True, the square is shrinked toward its center, leaving an empty border.
        The default is True.
    border_scale : float, optional
        The factor by which the border is scaled.
        The default is 1. It corresponds to a border length of 1/4 * length of the square's edge

    Returns
    -------
    ndarray
        The resized square derived from the barcoded reads coordinates of a certain spot.

    """
    # unpack coordinates
    x, y = xy
    # create border grid
    border_grid = np.array([[0.25, 0.25], [0.25, -0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
    # create square grid
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    # position square on the right place following the barcodes coordinates
    # barcode coordinates starts at 1, so we subtract 1 to start at 0
    square[:, 0] += x - 1
    square[:, 1] += y - 1
    # check if user wants border, and scale accordingly
    if border:
        return (np.array(square) * scale) + (border_grid * border_scale)

    else:
        return np.array(square) * scale


def DBiT(
    path: str | Path,
    anndata_path: Optional[str | Path] = None,
    barcode_position: Optional[str | Path] = None,
    dataset_id: Optional[str] = None,
    image_path: Optional[str | Path] = None,
) -> SpatialData:
    """
    Read DBiT experiment data (Deterministic Barcoding in Tissue)
    As published here: https://www.cell.com/cell/fulltext/S0092-8674(20)31390-8
    DOI: https://doi.org/10.1016/j.cell.2020.10.026

    Parameters
    ----------
    path : str | Path
        Path to the directory containing the data.
    anndata_path : Optional[str | Path], optional
        path to the counts and metadata file.
        The default is None.
    barcode_position : Optional[str | Path], optional
        path to the barcode coordinates file.
        The default is None.
    dataset_id : Optional[str], optional
        Dataset identifier to name the constructed `SpatialData` elements.
        If not given, filename is used as dataset_id
        The default is None.
    image_path : Optional[str | Path], optional
        path to the low resolution image.
        It expect the image to be correctly cropped and transformed.
        The default is None.

    Returns
    -------
    SpatialData
        :class:`spatialdata.SpatialData`.

    """

    path = Path(path)
    # if path is invalid, raise error
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"The path you have passed: {path} has not been found. A correct path to the data directory is needed."
        )
    # compile regex pattern to find file name in path, according to _constants.DBiTKeys()
    else:
        patt_h5ad = re.compile(f".*{DBiTKeys.COUNTS_FILE}")
        patt_barcode = re.compile(f".*{DBiTKeys.BARCODE_POSITION}.*")
        patt_lowres = re.compile(f".*{DBiTKeys.IMAGE_LOWRES_FILE}")

        # search for files paths. Gives priority to files mathing the pattern found in path.
        # check for .h5ad counts file
        try:
            counts_file = [i for i in os.listdir(path) if patt_h5ad.match(i)][0]  # this is the filename
            anndata_path = Path.joinpath(path, counts_file)
        except IndexError:
            # handle case in which the anndata path is not in the same directory as path
            if anndata_path is not None:
                if os.path.isfile(anndata_path):
                    anndata_path = Path(anndata_path)
                else:
                    raise FileNotFoundError(f"{anndata_path} is not a valid path for a .h5ad file.")
            else:
                raise FileNotFoundError(f"No file with extension .h5ad found in folder {path}.")
        # check for barcode file
        try:
            barcode_file = [i for i in os.listdir(path) if patt_barcode.match(i)][0]
            barcode_position = Path.joinpath(path, barcode_file)
        except IndexError:
            if barcode_position is not None:
                if os.path.isfile(barcode_position):
                    pass
                else:
                    raise FileNotFoundError(f"{barcode_position} is not a valid path for the barcode file.")
            else:
                raise FileNotFoundError(f"No file named {DBiTKeys.BARCODE_POSITION} found in folder {path}.")
        # check for image file
        # use hasimage flag to track image availability
        hasimage = False
        try:
            image_file = [i for i in os.listdir(path) if patt_lowres.match(i)][0]
            image_path = Path.joinpath(path, image_file)
            hasimage = True
        except IndexError:
            if image_path is not None:
                if os.path.isfile(image_path):
                    hasimage = True
                else:
                    warnings.warn(
                        f"{image_path} is not a valid path for the tissue_lowres_image. No image will be used.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"No file named {DBiTKeys.IMAGE_LOWRES_FILE} found in folder {path}. No image will be used.",
                    stacklevel=2,
                )

    # read annData.
    adata = ad.read(anndata_path)

    # Read barcode. We want it to accept 2 columns: [Barcode index, Barcode sequence]
    try:
        df = pd.read_csv(barcode_position, header=None, sep="\t")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The path you have passed:\n {barcode_position}\nhas not been found.\nA correct file path to the barcode file has to be provided."
        )

    # check if there are 2 columns
    if len(df.columns) != 2:
        raise ValueError(
            f"The barcode file you passed at {barcode_position} does not have 2 columns.\nYour file has to be formatted with 2 columns, the first for positions, the second for the barcode, as follows:\n\nA1 AACCTTGG\nA2 GGCATGTA\nA3 GCATATGC\n..."
        )
    # check if the data in the columns are correct. What do we want:
    # df[0] : str, composed by "A" or "B", followed by a number, like 'A6', 'A22','B7'
    # df[1] : str, of 8 chars representing nucleotides, like 'AACTGCTA'
    patt_position = re.compile(
        r"(^[A|B])([\d]{1,2}$)"
    )  # match A or B at the start, then match 1 or 2 numbers at the end
    patt_barcode = re.compile(r"^[A|a|T|t|C|c|G|g]{8}$")  # match nucleotides string of 8 char. Case insensitive.
    bc_positions: dict[str, dict[str, str]] = {}  # dict, used to collect data after matching
    for (
        line
    ) in (
        df.iterrows()
    ):  # line[0]: row index, line[1] row values. line[1][0] : barcode coordinates, line[1][1] : barcode
        if not bool(patt_position.fullmatch(line[1][0])):
            raise ValueError(
                f"Row {line[0]}, has an incorrect positional id: {line[1][0]}, \nThe correct pattern for the position is a str, containing a letter between A or B, and one or two digits. Case insensitive."
            )
        if not bool(patt_barcode.fullmatch(line[1][1])):
            raise ValueError(
                f"Row {line[0]} has an incorrect barcode: {line[1][1]}, \nThe correct pattern for a barcode is a str of 8 nucleotides, each char is a letter between A,T,C,G. Case insensitive."
            )
        barcode = line[1][1]
        letter = line[1][0][0]
        try:
            bc_positions[barcode][letter] = line[1][0][1:]
        except KeyError:
            bc_positions[barcode] = {}
            bc_positions[barcode][letter] = line[1][0][1:]
    # convert to pandas.DataFrame, (pseudo)long form
    bc_positions = pd.DataFrame(bc_positions).transpose()
    # add barcode positions to annData
    adata.obs["array_B"] = [int(bc_positions.loc[x[:8], "B"]) for x in adata.obs_names]
    adata.obs["array_A"] = [int(bc_positions.loc[x[8:16], "A"]) for x in adata.obs_names]
    # sort annData by barcode position. Barcode A first, then Barcode B
    adata.obs.sort_values(by=["array_A", "array_B"], inplace=True)

    # populate annData
    if dataset_id is None:  # if no dataset_id, use file name as id.
        print("No dataset_id received as input.")
        dataset_id = ".".join(
            anndata_path.name.split(".")[:-1]
        )  # this is the filename stripped from the file extension
        print(f"{dataset_id} is used as dataset_id.")

    adata.obs["region"] = dataset_id
    adata.obs["sample"] = dataset_id
    adata.obs["region"] = adata.obs["region"].astype("category")
    # assignment of pixel id follow the row ordering of adata.obs
    adata.obs["pixel_id"] = np.arange(len(adata.obs_names))
    adata.uns["spatialdata_attrs"] = {"region_key": "region", "region": dataset_id, "instance_key": "pixel_id"}
    # the array that we will create below has 2 columns without header
    # so it is up to the user to know that:
    # the first columns is 'array_A'
    # the second column in 'array_B'.
    adata.obsm["spatial"] = adata.obs[["array_A", "array_B"]].values
    # parse data from annData using SpatialData parser
    table_data = sd.models.TableModel.parse(adata)

    # read and convert image for SpatialData
    # check if image exist, and has been passed as argument
    if hasimage:
        try:
            image = imread(image_path).squeeze().transpose(2, 0, 1)  # channel, y, x
            image = DataArray(image, dims=("c", "y", "x"), name=dataset_id)
            image_sd = sd.models.Image2DModel.parse(image)
        except Exception as e:
            print(e)  # TODO: should we handle incorrect images in some other way?
    # calculate scale factor of the grid wrt the histological image.
    # this is needed because we want to mantain the original histological image
    # dimensions, and scale the grid accordingly in such a way that the grid
    # overlaps with the histological image.
    # TODO: Option #1
    # should we hardcode 50, or infer it from data?
    # We assume that the grid is a square, and indeed it is if we follow the DBiT protocol.
    grid_length = 50  # hardcoded lines number
    # TODO: we are passing an image, that is supposed to be perfectly
    # cropped to match the tissue part from which the data are collected.
    # We assume the user has already cropped and transformed everything correctly.
    if hasimage:
        scale_factor = max(image_sd.shape) / grid_length
    else:
        scale_factor = 1
    # contruct polygon grid with grid coordinates
    xy = adata.obs[["array_A", "array_B"]].values.astype(int)
    f = np.array(
        [xy2edges(x, scale=scale_factor, border=True, border_scale=scale_factor * 0.5) for x in xy]
    )  # some of these arguments can be exposed
    ra = shapely.to_ragged_array([shapely.Polygon(x) for x in f])
    grid = sd.models.ShapesModel.parse(ra[1], geometry=ra[0], offsets=ra[2], index=adata.obs["pixel_id"].copy())
    # create SpatialData object!
    sdata = sd.SpatialData(table=table_data, shapes={dataset_id: grid})
    # TODO: how to name images? dataset_id not usable.
    # option (not used): concat sha256(timestamp)[:6] to dataset_id, instead of concat '_img'
    if hasimage:
        imgname = dataset_id + "_img"
        sdata.add_image(name=imgname, image=image_sd)
    return sdata
