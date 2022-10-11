from anndata import AnnData
from spatialdata._types import ArrayLike
from typing import Optional, Iterable, Any, Union
import numpy as np

__all__ = ["circles_anndata_from_coordinates"]


def circles_anndata_from_coordinates(
    coordinates: ArrayLike,
    radii: Union[float, ArrayLike],
    instance_key: Optional[str] = None,
    instance_values: Optional[Iterable[Any]] = None,
) -> AnnData:
    if isinstance(radii, float):
        radii = np.array([radii] * coordinates.shape[0])
    assert len(coordinates) == len(radii)
    adata = AnnData(shape=(len(coordinates), 0), obsm={"spatial": coordinates, "region_radius": radii})
    assert bool(instance_key is None) == bool(instance_values is None)
    if instance_key is not None:
        adata.obs[instance_key] = instance_values
    return adata
