from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import pandas as pd
import spatialdata as sd
from aicsimageio import AICSImage
from dask_image.imread import imread
from ome_types import from_tiff
from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._constants import MacsimaKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils._utils import calc_scale_factors, parse_physical_size

__all__ = ["macsima"]


@inject_docs(vx=MacsimaKeys)
def macsima(
    path: str | Path,
    metadata: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = True,
    scale_factors: list[int] | None = None,
    default_scale_factor: int = 2,
) -> SpatialData:
    """
    Read *MACSima* formatted dataset.

    This function reads images from a MACSima cyclic imaging experiment. Metadata of the cycles is either parsed from a metadata file or image names.
    For parsing .qptiff files, installation of bioformats is adviced.

    .. seealso::

        - `MACSima output <https://application.qitissue.com/getting-started/naming-your-datasets>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    metadata
        Whether to search for a .txt file with metadata in the folder. If False, the metadata in the image names is used.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    subset
        Subset the image to the first ``subset`` pixels in x and y dimensions.
    c_subset
        Subset the image to the first ``c_subset`` channels.
    max_chunk_size
        Maximum chunk size for x and y dimensions.
    c_chunks_size
        Chunk size for c dimension.
    multiscale
        Whether to create a multiscale image.
    transformations
        Whether to add a transformation from pixels to microns to the image.
    scale_factors
        Scale factors to use for downsampling. If None, scale factors are calculated based on image size.
    default_scale_factor
        Default scale factor to use for downsampling.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    path_files = []
    pixels_to_microns = None
    if metadata:
        # read metadata to get list of images and channel names
        path_files = list(path.glob(f"*{MacsimaKeys.METADATA_SUFFIX}"))
        if len(path_files) > 0:
            if len(path_files) > 1:
                logger.warning(
                    f"Cannot determine metadata file. Expecting a single file with format .txt. Got multiple files: {path_files}"
                )
            path_metadata = list(path.glob(f"*{MacsimaKeys.METADATA_SUFFIX}"))[0]
            df = pd.read_csv(path_metadata, sep="\t", header=0, index_col=None)
            logger.debug(df)
            df["channel"] = df["ch1"].str.split(" ").str[0]
            df["round_channel"] = df["Round"] + " " + df["channel"]
            path_files = [path / p for p in df.filename.values]
            assert all(
                [p.exists() for p in path_files]
            ), f"Cannot find all images in metadata file. Missing: {[p for p in path_files if not p.exists()]}"
            round_channels = df.round_channel.values
            stack, sorted_channels = get_stack(path_files, round_channels, imread_kwargs)
        else:
            logger.warning(f"Cannot find metadata file. Will try to parse from image names.")
    if not metadata or len(path_files) == 0:
        # get list of image paths, get channel name from OME data and cycle number from filename
        # look for OME-TIFF files
        ome_patt = f"*{MacsimaKeys.IMAGE_OMETIF}*"
        path_files = list(path.glob(ome_patt))
        if not path_files:
            # look for .qptiff files
            qptif_patt = f"*{MacsimaKeys.IMAGE_QPTIF}*"
            path_files = list(path.glob(qptif_patt))
            logger.debug(path_files)
            if not path_files:
                raise ValueError("Cannot determine data set. Expecting '{ome_patt}' or '{qptif_patt}' files")
            # TODO: warning if not 1 ROI with 1 .qptiff per cycle
            # TODO: robuster parsing of {name}_cycle{round}_{scan}.qptiff
            rounds = [f"R{int(p.stem.split('_')[1][5:])}" for p in path_files]
            # parse .qptiff files
            imgs = [AICSImage(img, **imread_kwargs) for img in path_files]
            # sort based on cycle number
            rounds, imgs = zip(*sorted(zip(rounds, imgs), key=lambda x: int(x[0][1:])))
            channels_per_round = [img.channel_names for img in imgs]
            # take first image and first channel to get physical size
            ome_data = imgs[0].ome_metadata
            logger.debug(ome_data)
            pixels_to_microns = parse_physical_size(ome_pixels=ome_data.images[0].pixels)
            da_per_round = [img.dask_data[0, :, 0, :, :] for img in imgs]
            sorted_channels = []
            for r, cs in zip(rounds, channels_per_round):
                for c in cs:
                    sorted_channels.append(f"{r} {c}")
            stack = da.stack(da_per_round).squeeze()
            # Parse OME XML
            # img.ome_metadata
            # arr = img.dask_data[0, :, 0, :, :]
            # channel_names = img.channel_names
            logger.debug(sorted_channels)
            logger.debug(stack)
        else:
            logger.debug(path_files[0])
            # make sure not to remove round 0 when parsing!
            rounds = [f"R{int(p.stem.split('_')[0])}" for p in path_files]
            channels = [from_tiff(p).images[0].pixels.channels[0].name for p in path_files]
            round_channels = [f"{r} {c}" for r, c in zip(rounds, channels)]
            stack, sorted_channels = get_stack(path_files, round_channels, imread_kwargs)

    # do subsetting if needed
    if subset:
        stack = stack[:, :subset, :subset]
    if c_subset:
        stack = stack[:c_subset, :, :]
        sorted_channels = sorted_channels[:c_subset]
    if multiscale and not scale_factors:
        scale_factors = calc_scale_factors(stack, default_scale_factor=default_scale_factor)
    if not multiscale:
        scale_factors = None
    logger.debug(f"Scale factors: {scale_factors}")

    t_dict = None
    if transformations:
        pixels_to_microns = pixels_to_microns or parse_physical_size(path_files[0])
        t_pixels_to_microns = sd.transformations.Scale([pixels_to_microns, pixels_to_microns], axes=("x", "y"))
        # 'microns' is also used in merscope example
        # no inverse needed as the transformation is already from pixels to microns
        t_dict = {"microns": t_pixels_to_microns}
    # # chunk_size can be 1 for channels
    chunks = {
        "x": max_chunk_size,
        "y": max_chunk_size,
        "c": c_chunks_size,
    }
    stack = sd.models.Image2DModel.parse(
        stack,
        # TODO: make sure y and x locations are correct
        dims=["c", "y", "x"],
        scale_factors=scale_factors,
        chunks=chunks,
        c_coords=sorted_channels,
        transformations=t_dict,
    )
    sdata = sd.SpatialData(images={path.stem: stack}, table=None)

    return sdata


def get_stack(path_files: list[Path], round_channels: list[str], imread_kwargs: Mapping[str, Any]) -> Any:
    imgs_channels = list(zip(path_files, round_channels))
    logger.debug(imgs_channels)
    # sort based on round number
    imgs_channels = sorted(imgs_channels, key=lambda x: int(x[1].split(" ")[0][1:]))
    logger.debug(f"Len imgs_channels: {len(imgs_channels)}")
    # read in images and merge channels
    sorted_paths, sorted_channels = list(zip(*imgs_channels))
    imgs = [imread(img, **imread_kwargs) for img in sorted_paths]
    stack = da.stack(imgs).squeeze()
    return stack, sorted_channels
