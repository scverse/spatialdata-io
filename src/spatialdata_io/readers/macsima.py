from __future__ import annotations

import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["macsima"]


class MACSimaParsingStyle(ModeEnum):
    """Different parsing styles for MACSima data."""

    PROCESSED_SINGLE_FOLDER = "processed_single_folder"
    PROCESSED_MULTIPLE_FOLDERS = "processed_multiple_folders"
    RAW = "raw"
    AUTO = "auto"


@dataclass
class ChannelMetadata:
    """Metadata for a channel in a multichannel dataset."""

    name: str
    cycle: int


@dataclass
class MultiChannelImage:
    """Multichannel image with metadata."""

    data: list[da.Array]
    metadata: list[ChannelMetadata]
    include_cycle_in_channel_name: bool = False

    @classmethod
    def from_paths(
        cls,
        path_files: list[Path],
        imread_kwargs: Mapping[str, Any],
        skip_rounds: list[int] | None = None,
    ) -> MultiChannelImage:
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
                    warnings.warn(
                        f"Found multiple channels in OME-TIFF file {p}. Only the first one will be used.",
                        UserWarning,
                        stacklevel=2,
                    )
                channels.append(channel_names[0])
            except ValueError as e:
                warnings.warn(
                    f"Cannot parse OME metadata from {p}. Error: {e}. Skipping this file.", UserWarning, stacklevel=2
                )

        if len(path_files) != len(cycles) or len(path_files) != len(channels):
            raise ValueError("Length of path_files, cycles and channels must be the same.")
        # if any of round_channels is in skip_rounds, remove that round from the list and from path_files
        if skip_rounds:
            logger.info(f"Skipping cycles: {skip_rounds}")
            path_files, cycles, channels = map(
                list,
                zip(
                    *[
                        (p, c, ch)
                        for p, c, ch in zip(path_files, cycles, channels, strict=True)
                        if c not in skip_rounds
                    ],
                    strict=True,
                ),
            )
        imgs = [imread(img, **imread_kwargs) for img in path_files]
        for img, path in zip(imgs, path_files, strict=True):
            if img.shape[1:] != imgs[0].shape[1:]:
                raise ValueError(
                    f"Images are not all the same size. Image {path} has shape {img.shape[1:]} while the first image "
                    f"{path_files[0]} has shape {imgs[0].shape[1:]}"
                )
        # create MultiChannelImage object with imgs and metadata
        output = cls(
            data=imgs,
            metadata=[ChannelMetadata(name=ch, cycle=c) for c, ch in zip(cycles, channels, strict=True)],
        )
        return output

    @classmethod
    def subset_by_channel(cls, mci: MultiChannelImage, c_name: str) -> MultiChannelImage:
        """Create new MultiChannelImage with only the channels that contain the string c_name."""
        indices = [i for i, c in enumerate(mci.metadata) if c_name in c.name]
        return MultiChannelImage.subset_by_index(mci, indices)

    @classmethod
    def subset_by_index(cls, mci: MultiChannelImage, indices: list[int]) -> MultiChannelImage:
        """Create new MultiChannelImage with only the channels selected by the indices. The underlying data will still be the same reference, use copy.deepcopy to make a new copy."""
        metadata = [c for i, c in enumerate(mci.metadata) if i in indices]
        data = [d for i, d in enumerate(mci.data) if i in indices]
        return cls(
            data=data,
            metadata=metadata,
            include_cycle_in_channel_name=mci.include_cycle_in_channel_name,
        )

    def get_channel_names(self) -> list[str]:
        """Get the channel names."""
        if self.include_cycle_in_channel_name:
            return [f"R{c.cycle} {c.name}" for c in self.metadata]
        else:
            # if name is duplicated, add (i) to the name
            names = [c.name for c in self.metadata]
            name_dict: dict[str, int] = defaultdict(int)
            name_counter: dict[str, int] = defaultdict(int)
            for name in names:
                name_dict[name] += 1
            output = []
            for name in names:
                name_counter[name] += 1
                output.append(f"{name} ({name_counter[name]})" if name_dict[name] > 1 else name)
            return output

    def get_cycles(self) -> list[int]:
        """Get the cycle numbers."""
        return [c.cycle for c in self.metadata]

    def sort_by_channel(self) -> None:
        """Sort the channels by cycle number."""
        self.data = [d for _, d in sorted(zip(self.metadata, self.data, strict=True), key=lambda x: x[0].cycle)]
        self.metadata = sorted(self.metadata, key=lambda x: x.cycle)

    def subset(self, subset: int | None = None) -> MultiChannelImage:
        """Subsets the images to keep only the first `subset` x `subset` pixels."""
        if subset:
            self.data = [d[:, :subset, :subset] for d in self.data]
        return self

    def calc_scale_factors(self, default_scale_factor: int = 2) -> list[int]:
        lower_scale_limit = min(self.data[0].shape[1:])
        return calc_scale_factors(lower_scale_limit, default_scale_factor=default_scale_factor)

    def get_stack(self) -> da.Array:
        return da.stack(self.data, axis=0).squeeze(axis=1)


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
    split_threshold_nuclei_channel: int | None = 2,
    skip_rounds: list[int] | None = None,
    include_cycle_in_channel_name: bool = False,
) -> SpatialData:
    """Read *MACSima* formatted dataset.

    This function reads images from a MACSima cyclic imaging experiment. Metadata of the cycle rounds is parsed from
    the image names. The channel names are parsed from the OME metadata.

    .. seealso::

        - `MACSima output <https://application.qimagingsys.com/getting-started/naming-your-datasets>`_.

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
    split_threshold_nuclei_channel
        Threshold for splitting nuclei channels. If the number of channels that include nuclei_channel_name is
        greater than this threshold, the nuclei channels are split into a separate stack.
    skip_rounds
        List of round numbers to skip when parsing the data. Rounds or cycles are counted from 0 e.g. skip_rounds=[1, 2]
         will parse only the first round 0 when there are only 3 cycles.
    include_cycle_in_channel_name
        Whether to include the cycle number in the channel name.

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
            path=path,
            imread_kwargs=imread_kwargs,
            subset=subset,
            c_subset=c_subset,
            max_chunk_size=max_chunk_size,
            c_chunks_size=c_chunks_size,
            multiscale=multiscale,
            transformations=transformations,
            scale_factors=scale_factors,
            default_scale_factor=default_scale_factor,
            nuclei_channel_name=nuclei_channel_name,
            split_threshold_nuclei_channel=split_threshold_nuclei_channel,
            skip_rounds=skip_rounds,
            include_cycle_in_channel_name=include_cycle_in_channel_name,
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
                path=p,
                imread_kwargs=imread_kwargs,
                subset=subset,
                c_subset=c_subset,
                max_chunk_size=max_chunk_size,
                c_chunks_size=c_chunks_size,
                multiscale=multiscale,
                transformations=transformations,
                scale_factors=scale_factors,
                default_scale_factor=default_scale_factor,
                nuclei_channel_name=nuclei_channel_name,
                split_threshold_nuclei_channel=split_threshold_nuclei_channel,
                skip_rounds=skip_rounds,
                include_cycle_in_channel_name=include_cycle_in_channel_name,
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
    split_threshold_nuclei_channel: int | None = 2,
    skip_rounds: list[int] | None = None,
    file_pattern: str = "*.tif*",
    include_cycle_in_channel_name: bool = False,
) -> SpatialData:
    """Parse a single folder containing images from a cyclical imaging platform."""
    # get list of image paths, get channel name from OME data and cycle round number from filename
    # look for OME-TIFF files
    # TODO: replace this pattern and the p.suffix in [".tif", ".tiff"] with a single function based on a regexp, like
    # this one re.compile(r".*\.tif{1,2}$", re.IGNORECASE)
    path_files = list(path.glob(file_pattern))
    logger.debug(path_files[0])

    mci = MultiChannelImage.from_paths(
        path_files,
        imread_kwargs,
        skip_rounds,
    )
    mci.include_cycle_in_channel_name = include_cycle_in_channel_name

    mci.sort_by_channel()

    # do subsetting if needed
    if subset:
        mci = mci.subset(subset)
    if c_subset:
        mci = MultiChannelImage.subset_by_index(mci, indices=list(range(0, c_subset)))
    if multiscale and not scale_factors:
        scale_factors = mci.calc_scale_factors(default_scale_factor=default_scale_factor)
    if not multiscale:
        scale_factors = None
    logger.debug(f"Scale factors: {scale_factors}")

    filtered_name = path.stem.replace(" ", "_")

    return create_sdata(
        mci=mci,
        path_files=path_files,
        max_chunk_size=max_chunk_size,
        c_chunks_size=c_chunks_size,
        transformations=transformations,
        scale_factors=scale_factors,
        nuclei_channel_name=nuclei_channel_name,
        split_threshold_nuclei_channel=split_threshold_nuclei_channel,
        filtered_name=filtered_name,
    )


def create_sdata(
    mci: MultiChannelImage,
    path_files: list[Path],
    max_chunk_size: int,
    c_chunks_size: int,
    transformations: bool,
    scale_factors: list[int] | None,
    nuclei_channel_name: str,
    split_threshold_nuclei_channel: int | None,
    filtered_name: str,
) -> SpatialData:
    nuclei_idx = [i for i, c in enumerate(mci.get_channel_names()) if nuclei_channel_name in c]
    n_nuclei_channels = len(nuclei_idx)
    if not split_threshold_nuclei_channel:
        # if split_threshold_nuclei_channel is None, do not split nuclei channels
        split_nuclei = False
    else:
        split_nuclei = n_nuclei_channels > split_threshold_nuclei_channel
    if split_nuclei:
        # if channel name is nuclei_channel_name, add to separate nuclei stack
        nuclei_mci = deepcopy(MultiChannelImage.subset_by_index(mci, indices=nuclei_idx))
        # keep the first nuclei channel in both the stack and the nuclei stack
        nuclei_idx_without_first_and_last = nuclei_idx[1:-1]
        mci = MultiChannelImage.subset_by_index(
            mci,
            [i for i in range(len(mci.metadata)) if i not in nuclei_idx_without_first_and_last],
        )

    pixels_to_microns = parse_physical_size(path_files[0])

    image_element = create_image_element(
        mci,
        max_chunk_size,
        c_chunks_size,
        transformations,
        pixels_to_microns,
        scale_factors,
        coordinate_system=filtered_name,
    )
    table_channels = create_table(mci)

    if split_nuclei:
        nuclei_image_element = create_image_element(
            nuclei_mci,
            max_chunk_size,
            c_chunks_size,
            transformations,
            pixels_to_microns,
            scale_factors,
            coordinate_system=filtered_name,
        )
        table_nuclei = create_table(nuclei_mci)

    sdata = sd.SpatialData(
        images={
            f"{filtered_name}_image": image_element,
        },
        tables={
            f"{filtered_name}_table": table_channels,
        },
    )

    if split_nuclei:
        sdata.images[f"{filtered_name}_nuclei_image"] = nuclei_image_element
        sdata.tables[f"{filtered_name}_nuclei_table"] = table_nuclei

    return sdata


def create_table(mci: MultiChannelImage) -> ad.AnnData:
    cycles = mci.get_cycles()
    names = mci.get_channel_names()
    df = pd.DataFrame(
        {
            "name": names,
            "cycle": cycles,
        }
    )
    table = ad.AnnData(var=df)
    table.var_names = names
    return sd.models.TableModel.parse(table)


def create_image_element(
    mci: MultiChannelImage,
    max_chunk_size: int,
    c_chunks_size: int,
    transformations: bool,
    pixels_to_microns: float,
    scale_factors: list[int] | None,
    coordinate_system: str | None = None,
) -> sd.models.Image2DModel:
    t_dict = None
    if transformations:
        t_pixels_to_microns = sd.transformations.Scale([pixels_to_microns, pixels_to_microns], axes=("x", "y"))
        # 'microns' is also used in merscope example
        # no inverse needed as the transformation is already from pixels to microns
        t_dict = {coordinate_system: t_pixels_to_microns}
    # # chunk_size can be 1 for channels
    chunks = {
        "y": max_chunk_size,
        "x": max_chunk_size,
        "c": c_chunks_size,
    }
    if t_dict:
        logger.debug("Adding transformation: %s", t_dict)
    el = sd.models.Image2DModel.parse(
        mci.get_stack(),
        # the data on disk is not always CYX, but imread takes care of parsing things correctly, so that we can assume
        # mci to be CYX. Still, to make the code more robust, we could consider using a different backend, for instance
        # bioio-ome-tiff, read both the data and its dimensions from disk, and let Image2DModel.parse() rearrange the
        # dimensions into CYX.
        dims=["c", "y", "x"],
        scale_factors=scale_factors,
        chunks=chunks,
        c_coords=mci.get_channel_names(),
        transformations=t_dict,
    )
    return el
