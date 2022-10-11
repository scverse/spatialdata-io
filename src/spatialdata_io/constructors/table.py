from anndata import AnnData
from spatialdata._types import ArrayLike
from typing import Sequence, Any, List, Union, Optional
from loguru import logger

__all__ = ["table_update_anndata"]


def table_update_anndata(
    adata: AnnData,
    regions: Optional[Union[str, List[str]]] = None,
    regions_key: Optional[str] = None,
    instance_key: Optional[str] = None,
    regions_values: Optional[Union[str, Sequence[Any]]] = None,
    instance_values: Optional[Sequence[Any]] = None,
):
    if "spatial" in adata.obsm or "spatial" in adata.obs:
        logger.warning(
            "this annotation table seems to contain spatial information; to assign a coordinate space to "
            "spatial information please use the points or circles elements"
        )
    if regions is not None or regions_key is not None or regions_values is not None:
        if "mapping_info" not in adata.uns:
            adata.uns["mapping_info"] = {}
    for var in ["regions", "regions_key", "instance_key"]:
        if locals()[var] is not None and "mapping_info" in adata.uns and var in adata.uns["mapping_info"]:
            raise ValueError(f"{var} is already defined")
        adata.uns["mapping_info"][var] = locals()[var]
    if regions_values is not None:
        if regions_key in adata.obs:
            raise ValueError(f"this annotation table already contains the {regions_key} (regions_key) column")
        assert isinstance(regions_values, str) or len(adata) == len(regions_values)
        adata.obs[regions_key] = regions_values
    if instance_values is not None:
        if instance_key in adata.obs:
            raise ValueError(f"this annotation table already contains the {instance_key} (instance_key) column")
        assert len(adata) == len(instance_values)
        adata.obs[instance_key] = instance_values
    return adata
