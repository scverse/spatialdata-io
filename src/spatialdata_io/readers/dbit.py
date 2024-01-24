from __future__ import annotations

import os
import re
from pathlib import Path
from re import Pattern
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import shapely
import spatialdata as sd
from dask_image.imread import imread
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata._logging import logger
from xarray import DataArray

from spatialdata_io._constants._constants import DbitKeys
from spatialdata_io._docs import inject_docs

__all__ = ["dbit"]


def _check_path(
    path: Path,
    pattern: Pattern[str],
    key: DbitKeys,
    path_specific: Optional[str | Path] = None,
    return_flag: bool = False,
    optional_arg: bool = False,
) -> Optional[Path] | None | tuple[Path | None, bool]:
    """
    Check that the path is valid and match a regex pattern.

    Parameters
    ----------
    path :
        The path of the main directory where to search for the path.
    path_specific :
        path to the file, if it is not in the main directory.
    pattern :
        regex pattern.
    key :
        String to match in the path or path_specific path.
    return_flag :
        If True it returns a bool that indicate if the path have been matched.
    optional_arg :
        User specify if the file to search in mandatory (optional_arg=True, raise an Error if not found)
        or optional (optional_arg=False, rais a warning if not found).

    Raises
    ------
    FileNotFoundError
        The error is raised if no match is found in the given paths and optional_arg=True.

    Returns
    -------
    pathlib.PosixPath
        pathlib.PosixPath is returned if the path is found
        if return_flag=True return a tuple(pathlib.PosixPath, bool). The bool is a flag that indicate if one of the supplied path arguments points to a file that match the key.
    """
    flag = False
    file_path = None
    try:
        checked_file = [i for i in os.listdir(path) if pattern.match(i)][0]  # this is the filename
        file_path = Path.joinpath(path, checked_file)
        flag = True
    except IndexError:
        # handle case in which the searched file is not in the same directory as path
        if path_specific is not None:
            if os.path.isfile(path_specific):
                file_path = Path(path_specific)
                flag = True
            else:
                if optional_arg:
                    logger.warning(f"{path_specific} is not a valid path for {key}. No {key} will be used.")
                else:
                    raise FileNotFoundError(f"{path_specific} is not a valid path for a {key} file.")
        else:
            if optional_arg:
                logger.warning(f"No file named {key} found in folder {path}. No {key} will be used.")
            else:
                raise FileNotFoundError(f"No file with extension {key} found in folder {path}.")
    if return_flag:
        return file_path, flag
    return file_path


def _barcode_check(barcode_position: str | Path) -> pd.DataFrame:
    """
    Check that the barcode file is formatted as expected.

    What do we expect :
        A tab separated file, headless, with 2 columns:
            column 0 : str, composed by "A" or "B", followed by a number, like 'A6', 'A22','B7'
            column 1 : str, of 8 chars representing nucleotides, like 'AACTGCTA'

    Parameters
    ----------
    barcode_position :
        The path to the barcode file.

    Raises
    ------
    ValueError :
        ValueError is raised if a field of the file does not comply with the expected pattern.
        Appropriate error message is printed.

    Returns
    -------
    pd.DataFrame :
        A pandas.DataFrame with 2 columns, named 'A' and 'B', with a barcode as row index.
        Columns 'A' and 'B' contains an int each, that are the spatial coordinate of the barcode.
        The columns are ordered in ascending order.

    """
    df = pd.read_csv(barcode_position, header=None, sep="\t")
    # check if there are 2 columns
    if len(df.columns) != 2:
        raise ValueError(
            f"The barcode file you passed at {barcode_position} does not have 2 columns.\nYour file has to be formatted with 2 columns, the first for positions, the second for the barcode, as follows:\n\nA1 AACCTTGG\nA2 GGCATGTA\nA3 GCATATGC\n..."
        )
    # check if the data in the columns are correct.
    # Pattern 1: match A or B at the start, then match 1 or 2 numbers at the end. Case sensitive.
    patt_position = re.compile(r"(^[A|B])([\d]{1,2}$)")
    # Pattern 2: match nucleotides string of 8 char. Case insensitive.
    patt_barcode = re.compile(r"^[A|a|T|t|C|c|G|g]{8}$")
    # dict, used to collect data after matching
    bc_positions: dict[str, dict[str, str]] = {}
    # line[0]: row index, line[1] row values. line[1][0] : barcode coordinates, line[1][1] : barcode
    for line in df.iterrows():
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
    # return pandas.DataFrame, in (pseudo)long form
    return pd.DataFrame(bc_positions).transpose()


def _xy2edges(xy: list[int], scale: float = 1.0, border: bool = True, border_scale: float = 1) -> NDArray[np.double]:
    """
    Construct vertex coordinate of a square from the barcode coordinates.

    The constructed square has a scalable border.

    Parameters
    ----------
    xy :
        coordinate of the spot identified by its barcode.
    scale :
        Resize the square.
    border :
        If True, the square is shrinked toward its center, leaving an empty border.
    border_scale :
        The factor by which the border is scaled.
        The default is 1. It corresponds to a border length of 0.125 * length of the square's edge

    Returns
    -------
    ndarray
        The resized square derived from the barcoded reads coordinates of a certain spot.

    """
    # unpack coordinates
    x, y = xy
    # create border grid
    border_grid = np.array([[0.125, 0.125], [0.125, -0.125], [-0.125, -0.125], [-0.125, 0.125], [0.125, 0.125]])
    # create square grid
    square = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    # position square on the right place following the barcodes coordinates
    # barcode coordinates starts at 1, so we subtract 1 to start at 0
    square[:, 0] += x - 1
    square[:, 1] += y - 1
    # check if user wants border, and scale accordingly
    if border:
        return (np.array(square) * scale) + (border_grid * border_scale)

    return np.array(square) * scale


@inject_docs(vx=DbitKeys)
def dbit(
    path: str | Path,
    anndata_path: Optional[str | Path] = None,
    barcode_position: Optional[str | Path] = None,
    dataset_id: Optional[str] = None,
    image_path: Optional[str | Path] = None,
    border: bool = True,
    border_scale: float = 1,
) -> SpatialData:
    """
    Read DBiT experiment data (Deterministic Barcoding in Tissue)

    This function reads the following files:

        - ``{vx.COUNTS_FILE!r}`` : Counts matrix.
        - ``{vx.BARCODE_POSITION!r}`` : Barcode file.
        - ``{vx.IMAGE_LOWRES_FILE!r}`` : Histological image | Optional.

    .. seealso::

        - `High-Spatial-Resolution Multi-Omics Sequencing via Deterministic Barcoding in Tissue <https://www.cell.com/cell/fulltext/S0092-8674(20)31390-8/>`_.

    Parameters
    ----------
    path :
        Path to the directory containing the data.
    anndata_path :
        path to the counts and metadata file.
    barcode_position :
        path to the barcode coordinates file.
    dataset_id :
        Dataset identifier to name the constructed `SpatialData` elements.
        If not given, filename is used as dataset_id
    image_path :
        path to the low resolution image.
        It expect the image to be correctly cropped and transformed.
    border :
        Value passed internally to _xy2edges()
        If True, the square is shrinked toward its center, leaving an empty border.
    border_scale :
        Value passed internally to _xy2edges()
        The factor by which the border is scaled.
        The default is 1. It corresponds to a border length of (0.125 * length of the square's edge)

    Returns
    -------
    :class:`spatialdata.SpatialData`.
    """
    path = Path(path)
    # if path is invalid, raise error
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"The path you have passed: {path} has not been found. A correct path to the data directory is needed."
        )
    # compile regex pattern to find file name in path, according to _constants.DbitKeys()
    patt_h5ad = re.compile(f".*{DbitKeys.COUNTS_FILE}")
    patt_barcode = re.compile(f".*{DbitKeys.BARCODE_POSITION}.*")
    patt_lowres = re.compile(f".*{DbitKeys.IMAGE_LOWRES_FILE}")

    # search for files paths. Gives priority to files matching the pattern found in path.
    anndata_path_checked = _check_path(path=path, path_specific=anndata_path, pattern=patt_h5ad, key=DbitKeys.COUNTS_FILE)
    barcode_position_checked = _check_path(
        path=path, path_specific=barcode_position, pattern=patt_barcode, key=DbitKeys.BARCODE_POSITION
    )
    image_path_checked, hasimage = _check_path(
        path=path,
        path_specific=image_path,
        pattern=patt_lowres,
        key=DbitKeys.IMAGE_LOWRES_FILE,
        return_flag=True,
        optional_arg=True,
    )

    # read annData.
    adata = ad.read_h5ad(anndata_path_checked)
    # Read barcode. We want it to accept 2 columns: [Barcode index, Barcode sequence]
    bc_df = _barcode_check(barcode_position=barcode_position_checked)

    # add barcode positions to annData.
    # A and B naming follow original publication and protocol
    adata.obs["array_A"] = [int(bc_df.loc[x[8:16], "A"]) for x in adata.obs_names]
    adata.obs["array_B"] = [int(bc_df.loc[x[:8], "B"]) for x in adata.obs_names]
    # sort annData by barcode position. Barcode A first, then Barcode B
    adata.obs.sort_values(by=["array_A", "array_B"], inplace=True)

    # populate annData
    if dataset_id is None:  # if no dataset_id, use file name as id.
        logger.warning("No dataset_id received as input.")
        dataset_id = ".".join(
            anndata_path.name.split(".")[:-1]
        )  # this is the filename stripped from the file extension
        logger.warning(f"{dataset_id} is used as dataset_id.")

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
        image = imread(image_path_checked).squeeze().transpose(2, 0, 1)  # channel, y, x
        image = DataArray(image, dims=("c", "y", "x"), name=dataset_id)
        image_sd = sd.models.Image2DModel.parse(image)
    # calculate scale factor of the grid wrt the histological image.
    # this is needed because we want to mantain the original histological image
    # dimensions, and scale the grid accordingly in such a way that the grid
    # overlaps with the histological image.
    # We assume that the grid is a square, and indeed it is if we follow the DBiT protocol.
    grid_length = 50  # hardcoded lines number
    # we are passing an image, that is supposed to be perfectly
    # cropped to match the tissue part from which the data are collected.
    # We assume the user has already cropped and transformed everything correctly.
    if hasimage:
        scale_factor = max(image_sd.shape) / grid_length
    else:
        scale_factor = 1
    # contruct polygon grid with grid coordinates
    xy = adata.obs[["array_A", "array_B"]].values.astype(int)
    f = np.array(
        [_xy2edges(x, scale=scale_factor, border=border, border_scale=scale_factor * border_scale) for x in xy]
    )
    ra = shapely.to_ragged_array([shapely.Polygon(x) for x in f])
    grid = sd.models.ShapesModel.parse(ra[1], geometry=ra[0], offsets=ra[2], index=adata.obs["pixel_id"].copy())
    # create SpatialData object!
    sdata = sd.SpatialData(table=table_data, shapes={dataset_id: grid})
    if hasimage:
        imgname = dataset_id + "_image"
        sdata.images[imgname] = image_sd
    return sdata
