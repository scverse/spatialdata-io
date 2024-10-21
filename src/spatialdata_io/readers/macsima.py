from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import dask.array as da
import pandas as pd
import spatialdata as sd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._enum import ModeEnum
from spatialdata_io.readers._utils._utils import (
    calc_scale_factors,
    parse_channels,
    parse_physical_size,
)

__all__ = ["macsima"]


class MACSimaParsingStyle(ModeEnum):
    """Different parsing styles for MACSima data."""

    PROCESSED_SINGLE_FOLDER = "processed_single_folder"
    PROCESSED_MULTIPLE_FOLDERS = "processed_multiple_folders"
    RAW = "raw"
    AUTO = "auto"


def macsima(
    path: str | Path,
    parsing_style: MACSimaParsingStyle | str = MACSimaParsingStyle.AUTO,
    filter_folder_names: list[str] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = True,
    scale_factors: list[int] | None = None,
    default_scale_factor: int = 2,
    nuclei_channel_name: str = "DAPI",
    skip_rounds: list[int] | None = None,
) -> SpatialData:
    """
    Read *MACSima* formatted dataset.

    This function reads images from a MACSima cyclic imaging experiment. Metadata of the cycle rounds is parsed from the image names. The channel names are parsed from the OME metadata.

    .. seealso::

        - `MACSima output <https://application.qitissue.com/getting-started/naming-your-datasets>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    parsing_style
        Parsing style to use. If ``auto``, the parsing style is determined based on the contents of the path.
    filter_folder_names
        List of folder names to filter out when parsing multiple folders.
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
    nuclei_channel_name
        Common string of the nuclei channel to separate nuclei from other channels.
    skip_rounds
        List of round numbers to skip when parsing the data.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    if not isinstance(parsing_style, MACSimaParsingStyle):
        parsing_style = MACSimaParsingStyle(parsing_style)

    if parsing_style == MACSimaParsingStyle.AUTO:
        assert path.is_dir(), f"Path {path} is not a directory."

        if any(p.suffix in [".tif", ".tiff"] for p in path.iterdir()):
            # if path contains tifs, do parse_processed_folder on path
            parsing_style = MACSimaParsingStyle.PROCESSED_SINGLE_FOLDER
        elif all(p.is_dir() for p in path.iterdir() if not p.name.startswith(".")):
            # if path contains only folders or hidden files, do parse_processed_folder on each folder
            parsing_style = MACSimaParsingStyle.PROCESSED_MULTIPLE_FOLDERS
        else:
            raise ValueError(f"Cannot determine parsing style for path {path}. Please specify the parsing style.")

    if parsing_style == MACSimaParsingStyle.PROCESSED_SINGLE_FOLDER:
        return parse_processed_folder(
            path,
            imread_kwargs,
            subset,
            c_subset,
            max_chunk_size,
            c_chunks_size,
            multiscale,
            transformations,
            scale_factors,
            default_scale_factor,
            nuclei_channel_name,
            skip_rounds,
        )
    if parsing_style == MACSimaParsingStyle.PROCESSED_MULTIPLE_FOLDERS:
        sdatas = {}
        # iterate over all non-filtered folders in path and parse each folder
        for p in [
            p
            for p in path.iterdir()
            if p.is_dir() and (not filter_folder_names or not any(f in p.name for f in filter_folder_names))
        ]:
            sdatas[p.stem] = parse_processed_folder(
                p,
                imread_kwargs,
                subset,
                c_subset,
                max_chunk_size,
                c_chunks_size,
                multiscale,
                transformations,
                scale_factors,
                default_scale_factor,
                nuclei_channel_name,
                skip_rounds,
            )
        return sd.concatenate(list(sdatas.values()))
    if parsing_style == MACSimaParsingStyle.RAW:
        # TODO: see https://github.com/scverse/spatialdata-io/issues/155
        raise NotImplementedError("Parsing raw MACSima data is not yet implemented.")


def parse_name_to_cycle(name: str) -> int:
    """Parse the cycle number from the name of the image."""
    cycle = name.split("_")[0]
    if "-" in cycle:
        cycle = cycle.split("-")[1]
    return int(cycle)


def parse_processed_folder(
    path: Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = True,
    scale_factors: list[int] | None = None,
    default_scale_factor: int = 2,
    nuclei_channel_name: str = "DAPI",
    skip_rounds: list[int] | None = None,
    file_pattern: str = "*.tif*",
) -> SpatialData:
    """Parse a single folder containing images from a cyclical imaging platform."""
    # get list of image paths, get channel name from OME data and cycle round number from filename
    # look for OME-TIFF files
    path_files = list(path.glob(file_pattern))
    logger.debug(path_files[0])
    # make sure not to remove round 0 when parsing!
    cycles = []
    channels = []
    for p in path_files:
        cycle = parse_name_to_cycle(p.stem)
        cycles.append(cycle)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                channel_names = parse_channels(p)
            if len(channel_names) > 1:
                logger.warning(f"Found multiple channels in OME-TIFF file {p}. Only the first one will be used.")
            channels.append(channel_names[0])
        except ValueError as e:
            logger.warning(f"Cannot parse OME metadata from {p}. Error: {e}. Skipping this file.")
    stack, sorted_cycles, sorted_channels = get_stack(
        path_files,
        cycles,
        channels,
        imread_kwargs,
        skip_rounds=skip_rounds,
    )

    sorted_cycle_channels: list[str] = []
    for c, ch in zip(sorted_cycles, sorted_channels):
        sorted_cycle_channels.append(f"R{str(c)} {ch}")

    # do subsetting if needed
    if subset:
        stack = stack[:, :subset, :subset]
    if c_subset:
        stack = stack[:c_subset, :, :]
        sorted_cycle_channels = sorted_cycle_channels[:c_subset]
    if multiscale and not scale_factors:
        scale_factors = calc_scale_factors(stack, default_scale_factor=default_scale_factor)
    if not multiscale:
        scale_factors = None
    logger.debug(f"Scale factors: {scale_factors}")

    filtered_name = path.stem.replace(" ", "_")

    return create_sdata(
        stack,
        sorted_cycle_channels,
        path_files,
        max_chunk_size,
        c_chunks_size,
        transformations,
        scale_factors,
        nuclei_channel_name,
        filtered_name,
    )


def create_sdata(
    stack: da.Array,
    sorted_cycle_channels: list[str],
    path_files: list[Path],
    max_chunk_size: int,
    c_chunks_size: int,
    transformations: bool,
    scale_factors: list[int] | None,
    nuclei_channel_name: str,
    filtered_name: str,
) -> SpatialData:
    # for stack and sorted_cycle_channels, if channel name is nuclei_channel_name, add to seperate nuclei stack
    # keep the first nuclei channel in both the stack and the nuclei stack
    nuclei_sorted_cycle_channels = []
    nuclei_idx = []
    for i, c in enumerate(sorted_cycle_channels):
        if nuclei_channel_name in c:
            nuclei_sorted_cycle_channels.append(c)
            nuclei_idx.append(i)
    if len(nuclei_idx) > 2:
        has_nuclei_stack = True
        # More than two nuclei channels found, keep only the first and last one in the stack
        nuclei_stack = stack[nuclei_idx]
        nuclei_idx_without_first_and_last = nuclei_idx[1:-1]
        stack = stack[[i for i in range(len(sorted_cycle_channels)) if i not in nuclei_idx_without_first_and_last]]
        sorted_cycle_channels = [
            c for i, c in enumerate(sorted_cycle_channels) if i not in nuclei_idx_without_first_and_last
        ]
    else:
        has_nuclei_stack = False
    # Only one or two nuclei channels found, keep all in the stack
    pixels_to_microns = parse_physical_size(path_files[0])
    image_element = create_image_element(
        stack,
        sorted_cycle_channels,
        max_chunk_size,
        c_chunks_size,
        transformations,
        pixels_to_microns,
        scale_factors,
        name=filtered_name,
    )
    if has_nuclei_stack:
        nuclei_image_element = create_image_element(
            nuclei_stack,
            nuclei_sorted_cycle_channels,
            max_chunk_size,
            c_chunks_size,
            transformations,
            pixels_to_microns,
            scale_factors,
            name=filtered_name,
        )
        table_nuclei = create_table(nuclei_sorted_cycle_channels)
    table_channels = create_table(sorted_cycle_channels)

    sdata = sd.SpatialData(
        images={
            f"{filtered_name}_image": image_element,
        },
        tables={
            f"{filtered_name}_table": table_channels,
        },
    )
    if has_nuclei_stack:
        sdata.images[f"{filtered_name}_nuclei_image"] = nuclei_image_element
        sdata.tables[f"{filtered_name}_nuclei_table"] = table_nuclei

    return sdata


def create_table(sorted_cycle_channels: list[str]) -> ad.AnnData:
    df = pd.DataFrame(
        {
            "name": sorted_cycle_channels,
            "cycle": [int(c.split(" ")[0][1:]) for c in sorted_cycle_channels],
        }
    )
    table = ad.AnnData(var=df)
    table.var_names = sorted_cycle_channels
    return sd.models.TableModel.parse(table)


def create_image_element(
    stack: da.Array,
    sorted_channels: list[str],
    max_chunk_size: int,
    c_chunks_size: int,
    transformations: bool,
    pixels_to_microns: float,
    scale_factors: list[int] | None,
    name: str | None = None,
) -> sd.models.Image2DModel:
    t_dict = None
    if transformations:
        t_pixels_to_microns = sd.transformations.Scale([pixels_to_microns, pixels_to_microns], axes=("x", "y"))
        # 'microns' is also used in merscope example
        # no inverse needed as the transformation is already from pixels to microns
        t_dict = {name: t_pixels_to_microns}
    # # chunk_size can be 1 for channels
    chunks = {
        "x": max_chunk_size,
        "y": max_chunk_size,
        "c": c_chunks_size,
    }
    if t_dict:
        logger.debug("Adding transformation: %s", t_dict)
    el = sd.models.Image2DModel.parse(
        stack,
        # TODO: make sure y and x locations are correct
        dims=["c", "y", "x"],
        scale_factors=scale_factors,
        chunks=chunks,
        c_coords=sorted_channels,
        transformations=t_dict,
    )
    return el


def get_stack(
    path_files: list[Path],
    cycles: list[int],
    channels: list[str],
    imread_kwargs: Mapping[str, Any],
    skip_rounds: list[int] | None = None,
) -> tuple[da.Array, list[int], list[str]]:
    if len(path_files) != len(cycles) or len(path_files) != len(channels):
        raise ValueError("Length of path_files, cycles and channels must be the same.")
    # if any of round_channels is in skip_rounds, remove that round from the list and from path_files
    if skip_rounds:
        logger.info("Skipping cycles: %d", skip_rounds)
        path_files, cycles, channels = map(
            list,
            zip(*[(p, c, ch) for p, c, ch in zip(path_files, cycles, channels) if c not in skip_rounds]),
        )
    imgs = [imread(img, **imread_kwargs) for img in path_files]
    for img, path in zip(imgs, path_files):
        if img.shape[1:] != imgs[0].shape[1:]:
            raise ValueError(
                f"Images are not all the same size. Image {path} has shape {img.shape[1:]} while the first image {path_files[0]} has shape {imgs[0].shape[1:]}"
            )
    # sort imgs, cycles and channels based on cycles
    imgs, cycles, channels = map(
        list,
        zip(*sorted(zip(imgs, cycles, channels), key=lambda x: (x[1]))),
    )
    stack = da.stack(imgs).squeeze()
    return stack, cycles, channels
