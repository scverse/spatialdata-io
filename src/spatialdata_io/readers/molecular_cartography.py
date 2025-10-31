from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import pandas as pd
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel

from spatialdata_io._constants._constants import MolecularCartographyKeys

__all__ = ["molecular_cartography"]


def molecular_cartography(
    path: str | Path,
    region: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *Molecular Cartography* data from *Resolve Bioscience* as a `SpatialData` object.

    This function reads the following files:

        - {dataset_id}_*.tiff: The images for the given region.
        - {dataset_id}_results.txt: The transcript locations for the given region.

    Parameters
    ----------
        path: Path to the Molecular Cartography directory containing one or multiple region(s).
        region: Name of the region to read. The region name can be found before the `_results.txt` file, e.g. `A2-1`.
        image_models_kwargs: Keyword arguments passed to `spatialdata.models.Image2DModel`.
        imread_kwargs: Keyword arguments passed to `dask_image.imread.imread`.

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

    path = Path(path)
    dataset_id = _get_dataset_id(path, region)

    # Read the points
    transcripts = pd.read_csv(path / f"{dataset_id}{MolecularCartographyKeys.POINTS_SUFFIX}", sep="\t", header=None)
    transcripts.columns = ["x", "y", "z", MolecularCartographyKeys.FEATURE_KEY, "unnamed"]

    transcripts = PointsModel.parse(transcripts, feature_key=MolecularCartographyKeys.FEATURE_KEY)
    transcripts_name = f"{dataset_id}_points"

    # Read the images
    images_paths = list(path.glob(f"{dataset_id}_*.tiff"))
    c_coords = [image_path.stem.split("_")[-1] for image_path in images_paths]

    image = Image2DModel.parse(
        da.concatenate([imread(image_path, **imread_kwargs) for image_path in images_paths], axis=0),
        dims=("c", "y", "x"),
        c_coords=c_coords,
        rgb=None,
        **image_models_kwargs,
    )
    image_name = f"{dataset_id}_image"

    return SpatialData(images={image_name: image}, points={transcripts_name: transcripts})


def _get_dataset_id(path: Path, region: str) -> str:
    _dataset_ids = [path.name[:-12] for path in path.glob("*_results.txt")]
    region_to_id = {dataset_id.split("_")[-1]: dataset_id for dataset_id in _dataset_ids}

    assert region in region_to_id, f"Region {region} not found. Must be one of {list(region_to_id.keys())}"

    return region_to_id[region]
