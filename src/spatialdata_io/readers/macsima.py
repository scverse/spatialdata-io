from __future__ import annotations

import os
import re
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
from ome_types import OME, from_tiff
from spatialdata import SpatialData
from spatialdata._logging import logger

from spatialdata_io._constants._enum import ModeEnum
from spatialdata_io.readers._utils._utils import (
    _set_reader_metadata,
    calc_scale_factors,
    parse_channels,
    parse_physical_size,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["macsima"]

# Dictionary to harmonize imagetype across metadata versions
IMAGETYPE_DICT = {
    "BleachCycle": "bleach",  # v0
    "B": "bleach",  # v1
    "AntigenCycle": "stain",  # v0
    "S": "stain",  # v1
}


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
    imagetype: str
    well: str
    roi: int
    fluorophore: str
    exposure: float
    clone: str | None = None  # For example DAPI doesnt have a clone


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
        valid_files: list[Path] = []
        channel_metadata: list[ChannelMetadata] = []
        for p in path_files:
            try:
                metadata = parse_metadata(p)
            except ValueError as e:
                warnings.warn(
                    f"Cannot parse OME metadata from {p}. Error: {e}. Skipping this file.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            valid_files.append(p)
            channel_metadata.append(
                ChannelMetadata(
                    name=metadata["name"],
                    cycle=metadata["cycle"],
                    imagetype=metadata["imagetype"],
                    well=metadata["well"],
                    roi=metadata["roi"],
                    fluorophore=metadata["fluorophore"],
                    clone=metadata["clone"],
                    exposure=metadata["exposure"],
                )
            )

        if not valid_files:
            raise ValueError("No valid files were found.")
        if len(valid_files) != len(channel_metadata):
            raise ValueError("Length of valid files and metadata must be the same.")
        # if any of round_channels is in skip_rounds, remove that round from the list and from valid_files
        if skip_rounds:
            logger.info(f"Skipping cycles: {skip_rounds}")
            valid_files, channel_metadata = map(
                list,
                zip(
                    *[
                        (p, ch_meta)
                        for p, ch_meta in zip(valid_files, channel_metadata, strict=True)
                        if ch_meta.cycle not in skip_rounds
                    ],
                    strict=True,
                ),
            )
        imgs = [imread(img, **imread_kwargs) for img in valid_files]

        # Pad images to same dimensions if necessary
        if cls._check_for_differing_xy_dimensions(imgs):
            imgs = cls._pad_images(imgs)

        # create MultiChannelImage object with imgs and metadata
        output = cls(data=imgs, metadata=channel_metadata)
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

    def get_image_types(self) -> list[str | None]:
        """Get the staining types (stain or bleach)."""
        return [c.imagetype for c in self.metadata]

    def get_wells(self) -> list[str | None]:
        """Get the wells."""
        return [c.well for c in self.metadata]

    def get_rois(self) -> list[int | None]:
        """Get the ROIs."""
        return [c.roi for c in self.metadata]

    def get_fluorophores(self) -> list[str | None]:
        """Get the fluorophores."""
        return [c.fluorophore for c in self.metadata]

    def get_clones(self) -> list[str | None]:
        """Get the clones."""
        return [c.clone for c in self.metadata]

    def get_exposures(self) -> list[float | None]:
        """Get the exposures."""
        return [c.exposure for c in self.metadata]

    def sort_by_channel(self) -> None:
        """Sort the channels by cycle number.

        Use channel name as tie breaker.
        """
        pairs = sorted(zip(self.metadata, self.data, strict=True), key=lambda x: (x[0].cycle, x[0].name))

        self.metadata = [m for m, _ in pairs]
        self.data = [d for _, d in pairs]

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

    @staticmethod
    def _check_for_differing_xy_dimensions(imgs: list[da.Array]) -> bool:
        """Checks whether any of the images have differing extent in dimensions X and Y."""
        # Shape has order CYX
        dims_x = [x.shape[2] for x in imgs]
        dims_y = [x.shape[1] for x in imgs]

        dims_x_different = False if len(set(dims_x)) == 1 else True
        dims_y_different = False if len(set(dims_y)) == 1 else True

        different_dimensions = any([dims_x_different, dims_y_different])

        warnings.warn(
            "Supplied images have different dimensions!",
            UserWarning,
            stacklevel=2,
        )

        return different_dimensions

    @staticmethod
    def _pad_images(imgs: list[da.Array]) -> list[da.Array]:
        """Pad all images to the same dimensions in X and Y with 0s."""
        dims_x_max = max([x.shape[2] for x in imgs])
        dims_y_max = max([x.shape[1] for x in imgs])

        warnings.warn(
            f"Padding images with 0s to same size of ({dims_y_max}, {dims_x_max})",
            UserWarning,
            stacklevel=2,
        )

        padded_imgs = []
        for img in imgs:
            pad_y = dims_y_max - img.shape[1]
            pad_x = dims_x_max - img.shape[2]
            # Only pad if necessary
            if (pad_y, pad_y) != (0, 0):
                # Always pad to the right/bottom
                pad_width = (
                    (0, 0),
                    (0, pad_y),
                    (0, pad_x),
                )

                img = da.pad(img, pad_width, mode="constant", constant_values=0)
            padded_imgs.append(img)

        return padded_imgs


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

    This function reads images from a MACSima cyclic imaging experiment. MACSima data follows the OME-TIFF specificiation.
    All metadata is parsed from the OME metadata. The exact metadata schema can change between software versions of MACSiQView.
    As there is no public specification of the metadata fields used, please consider the provided test data sets as ground truth to guide development.

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


def _collect_map_annotation_values(ome: OME) -> dict[str, Any]:
    """Collapse structured_annotations from OME into dictionary.

    Collects all key/value pairs from all map_annotations in structured_annotations into a single flat dictionary.
    If a key appears multiple times across annotations, the *first*
    occurrence wins and later occurrences are ignored.
    """
    merged: dict[str, Any] = {}

    sa = getattr(ome, "structured_annotations", None)
    map_annotations = getattr(sa, "map_annotations", []) if sa else []

    for ma in map_annotations:
        raw_value = ma.value
        value = raw_value.dict()

        for k, v in value.items():
            if k not in merged:
                merged[k] = v
            else:
                # We do expect repeated keys with different values, because the same key is reused for different annotations.
                # But the order is fixed and fine for what we need.
                # Therefore log this for debugging, if it becomes a problem, but don't throw warnings to the user.
                if v != merged[k]:
                    logger.debug(
                        f"Found different value for {k}: {v}. The parser will only use the first found value, which is {merged[k]}!"
                    )

    return merged


def _get_software_version(ma_values: dict[str, Any]) -> str:
    """Extract the software version string from the flattened map-annotation values.

    Supports both:
      - 'Software version'  (v0)
      - 'SoftwareVersion'   (v1)
    """
    for key in ("SoftwareVersion", "Software version"):
        v = ma_values.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    raise ValueError("Could not extract Software Version from OME metadata.")


def _get_software_major_version(version: str) -> int:
    """Parse the major component of a semantic version string."""
    s = version.strip()
    if s.startswith(("v", "V")):
        s = s[1:]
    parts = s.split(".")
    if not parts:
        raise ValueError("Could not extract major software version part from version string.")

    major = int(parts[0])
    logger.debug(f"Found major software version {major}")

    return major


def _parse_v0_ome_metadata(ome: OME) -> dict[str, Any]:
    """Parse Legacy Format of OME Metadata (software version 0.x.x)."""
    logger.debug("Parsing OME metadata expecting version 0 format")

    metadata: dict[str, Any] = {
        "name": None,
        "clone": None,
        "fluorophore": None,
        "cycle": None,
        "imagetype": None,
        "well": None,
        "roi": None,
        "exposure": None,
    }

    antigen = None
    clone = None

    if ome.screens:
        screen0 = ome.screens[0]
        reagents = getattr(screen0, "reagents", [])
        if reagents:
            r0 = reagents[0]
            name = getattr(r0, "name", None)
            if isinstance(name, str) and name:
                if "__" in name:
                    antigen, clone = name.split("__", 1)
                else:
                    antigen = name
                    clone = None

    metadata["name"] = antigen
    metadata["clone"] = clone

    ma_values = _collect_map_annotation_values(ome)

    if "Fluorochrome" in ma_values:
        metadata["fluorophore"] = ma_values["Fluorochrome"]

    if "Exposure time" in ma_values:
        exp_time = ma_values["Exposure time"]
        try:
            metadata["exposure"] = float(exp_time)
        except (TypeError, ValueError):
            metadata["exposure"] = None

    cyc = None
    if "Cycle" in ma_values:
        cyc = ma_values["Cycle"]
    elif "MICS cycle ID" in ma_values:  # Very old formats do not have "Cycle", then use "MICS cycle ID"
        cyc = ma_values["MICS cycle ID"]
    if cyc:
        try:
            metadata["cycle"] = int(cyc)
        except (TypeError, ValueError):
            metadata["cycle"] = None

    if "ROI ID" in ma_values:
        roi = ma_values["ROI ID"]
        try:
            metadata["roi"] = int(roi)
        except (TypeError, ValueError):
            metadata["roi"] = None

    if "MICS cycle type" in ma_values:
        metadata["imagetype"] = ma_values["MICS cycle type"]

    well = None
    if ome.plates:
        plate0 = ome.plates[0]
        wells = getattr(plate0, "wells", [])
        if wells:
            w0 = wells[0]
            ext_id = getattr(w0, "external_identifier", None)
            if isinstance(ext_id, str) and ext_id:
                well = ext_id

    metadata["well"] = well

    # Add _background suffix to marker name of bleach images, to distinguish them from stain image
    if metadata["imagetype"] == "BleachCycle":
        metadata["name"] = metadata["name"] + "_background"

    # Harmonize imagetype across versions
    if metadata["imagetype"]:
        metadata["imagetype"] = IMAGETYPE_DICT[metadata["imagetype"]]

    return metadata


def _parse_v1_ome_metadata(ome: OME) -> dict[str, Any]:
    """Parse v1 format of OME metadata (software version 1.x.x)."""
    logger.debug("Parsing OME metadata expecting version 1 format")

    metadata: dict[str, Any] = {
        "name": None,
        "clone": None,
        "fluorophore": None,
        "cycle": None,
        "imagetype": None,
        "well": None,
        "roi": None,
        "exposure": None,
    }

    ma_values = _collect_map_annotation_values(ome)

    if "Clone" in ma_values:
        metadata["clone"] = ma_values["Clone"]

    antigen_name = None
    if "Biomarker" in ma_values and ma_values["Biomarker"]:
        antigen_name = ma_values["Biomarker"]
    elif "Dye" in ma_values and ma_values["Dye"]:
        antigen_name = ma_values["Dye"]

    metadata["name"] = antigen_name

    if "Fluorochrome" in ma_values and ma_values["Fluorochrome"]:
        metadata["fluorophore"] = ma_values["Fluorochrome"]
    elif "Dye" in ma_values and ma_values["Dye"]:
        metadata["fluorophore"] = ma_values["Dye"]

    if "ExposureTime" in ma_values:
        exp_time = ma_values["ExposureTime"]
        try:
            metadata["exposure"] = float(exp_time)
        except (TypeError, ValueError):
            metadata["exposure"] = None

    if "Cycle" in ma_values:
        cyc = ma_values["Cycle"]
        try:
            metadata["cycle"] = int(cyc)
        except (TypeError, ValueError):
            metadata["cycle"] = None

    if "RoiId" in ma_values:
        roi = ma_values["RoiId"]
        try:
            metadata["roi"] = int(roi)
        except (TypeError, ValueError):
            metadata["roi"] = None

    if "ScanType" in ma_values:
        metadata["imagetype"] = ma_values["ScanType"]

    well = None
    if ome.plates:
        plate0 = ome.plates[0]
        wells = getattr(plate0, "wells", [])
        if wells:
            w0 = wells[0]
            ext_id = getattr(w0, "external_identifier", None)
            if isinstance(ext_id, str) and ext_id:
                well = ext_id

    metadata["well"] = well

    # Add _background suffix to marker name of bleach images, to distinguis them from stain image
    if metadata["imagetype"] == "B":
        metadata["name"] = metadata["name"] + "_background"

    # Harmonize imagetype across versions
    if metadata["imagetype"]:
        metadata["imagetype"] = IMAGETYPE_DICT[metadata["imagetype"]]

    return metadata


def _parse_ome_metadata(ome: OME) -> dict[str, Any]:
    """Extract the software version from OME metadata and parse with appropriate parser."""
    ma_values = _collect_map_annotation_values(ome)
    version_str = _get_software_version(ma_values)
    major = _get_software_major_version(version_str)

    if major == 0:
        return _parse_v0_ome_metadata(ome)
    elif major == 1:
        return _parse_v1_ome_metadata(ome)
    else:
        raise ValueError("Unknown software version, cannot determine parser")


def parse_metadata(path: Path) -> dict[str, Any]:
    """Parse metadata for a file.

    All metadata is extracted from the OME metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ome = from_tiff(path)

    return _parse_ome_metadata(ome)


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
    # get list of image paths, look for OME-TIFF files
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Iterate over path files, as it may still contain invalid files
        pixels_to_microns = None
        for p in path_files:
            try:
                pixels_to_microns = parse_physical_size(p)
            except Exception:
                logger.debug(f"Could not parse physical size from {p}. Trying next file.")
                continue
        if pixels_to_microns is None:
            raise ValueError("Could not parse physical size from any file")

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

    return _set_reader_metadata(sdata, "macsima")


def create_table(mci: MultiChannelImage) -> ad.AnnData:
    cycles = mci.get_cycles()
    names = mci.get_channel_names()
    imagetypes = mci.get_image_types()
    wells = mci.get_wells()
    rois = mci.get_rois()
    fluorophores = mci.get_fluorophores()
    clones = mci.get_clones()
    exposures = mci.get_exposures()

    df = pd.DataFrame(
        {
            "name": names,
            "cycle": cycles,
            "imagetype": imagetypes,
            "well": wells,
            "ROI": rois,
            "fluorophore": fluorophores,
            "clone": clones,
            "exposure": exposures,
        }
    )

    # Replace missing data. This happens mostly in the clone column.
    df = df.replace({None: pd.NA, "": pd.NA})
    df.index = df.index.astype(str)

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
