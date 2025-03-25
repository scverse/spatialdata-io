from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from anndata.io import read_text
from h5py import File
from ome_types import from_tiff
from ome_types.model import Pixels, UnitsLength
from spatialdata._logging import logger

from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData

PathLike = os.PathLike | str  # type:ignore[type-arg]


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: str | None = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if counts_file.endswith(".h5"):
        adata: AnnData = _read_10x_h5(path / counts_file, **kwargs)
        with File(path / counts_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
                except ValueError:
                    raise KeyError(
                        "Unable to extract library id from attributes. Please specify one explicitly."
                    ) from None

            adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id]["metadata"][key] = metadata

        return adata, library_id
    if library_id is None:
        raise ValueError("Please explicitly specify `library id`.")

    if counts_file.endswith((".csv", ".txt")):
        adata = read_text(path / counts_file, **kwargs)
    elif counts_file.endswith(".mtx.gz"):
        try:
            from scanpy.readwrite import read_10x_mtx
        except ImportError:
            raise ImportError("Please install scanpy to read 10x mtx files, `pip install scanpy`.") from None
        prefix = counts_file.replace("matrix.mtx.gz", "")
        adata = read_10x_mtx(path, prefix=prefix, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id


def _initialize_raster_models_kwargs(
    image_models_kwargs: Mapping[str, Any], labels_models_kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_models_kwargs = dict(image_models_kwargs)
    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    labels_models_kwargs = dict(labels_models_kwargs)
    if "chunks" not in labels_models_kwargs:
        labels_models_kwargs["chunks"] = (4096, 4096)
    if "scale_factors" not in labels_models_kwargs:
        labels_models_kwargs["scale_factors"] = [2, 2, 2, 2]
    return image_models_kwargs, labels_models_kwargs


def calc_scale_factors(lower_scale_limit: float, min_size: int = 1000, default_scale_factor: int = 2) -> list[int]:
    """Calculate scale factors based on image size to get lowest resolution under min_size pixels."""
    # get lowest dimension, ignoring channels
    scale_factor: int = default_scale_factor
    scale_factors = [scale_factor]
    lower_scale_limit /= scale_factor
    while lower_scale_limit >= min_size:
        # scale_factors are cumulative, so we don't need to do e.g. scale_factor *= 2
        scale_factors.append(scale_factor)
        lower_scale_limit /= scale_factor
    return scale_factors


def parse_channels(path: Path) -> list[str]:
    """Parse channel names from an OME-TIFF file."""
    images = from_tiff(path).images
    if len(images) > 1:
        logger.warning("Found multiple images in OME-TIFF file. Only the first one will be used.")
    channels = images[0].pixels.channels
    logger.debug(channels)
    names = [c.name for c in channels if c.name is not None]
    return names


def parse_physical_size(path: Path | None = None, ome_pixels: Pixels | None = None) -> float:
    """Parse physical size from OME-TIFF to micrometer."""
    pixels = ome_pixels or from_tiff(path).images[0].pixels
    logger.debug(pixels)
    if pixels.physical_size_x_unit != pixels.physical_size_y_unit:
        logger.error("Physical units for x and y dimensions are not the same.")
        raise NotImplementedError
    if pixels.physical_size_x != pixels.physical_size_y:
        logger.error("Physical sizes for x and y dimensions are not the same.")
        raise NotImplementedError
    # convert to micrometer if needed
    if pixels.physical_size_x_unit == UnitsLength.NANOMETER:
        physical_size = pixels.physical_size_x / 1000
    elif pixels.physical_size_x_unit == UnitsLength.MICROMETER:
        physical_size = pixels.physical_size_x
    else:
        logger.error(f"Physical unit not recognized: '{pixels.physical_size_x_unit}'.")
        raise NotImplementedError
    return float(physical_size)
