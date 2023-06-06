from anndata import AnnData
from spatialdata import SpatialData

__all__ = ["get_table"]


def get_table(sdata: SpatialData) -> AnnData:
    """Retrieve table from SpatialData."""
    return sdata.table
