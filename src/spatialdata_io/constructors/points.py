from anndata import AnnData
from spatialdata._types import ArrayLike
from typing import Optional, Sequence, Any

__all__ = ["points_anndata_from_coordinates"]


def points_anndata_from_coordinates(
    coordinates: ArrayLike,
    points_types: Optional[Sequence[Any]] = None
) -> AnnData:
    adata = AnnData(shape=(len(coordinates), 0), obsm={"spatial": coordinates})
    if points_types is not None:
        assert len(coordinates) == len(points_types)
        adata.obsm["points_types"] = points_types
    return adata
