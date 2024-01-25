import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation


def to_legacy_anndata(sdata: SpatialData) -> AnnData:
    """
    Convert SpatialData object to a (legacy) spatial AnnData object.

    This is useful for converting a Spatialdata object with Visium data for the use
    with packages using spatial information in AnnData as used by scanpy and older versions of Squidpy.
    Using this format for any new package is not recommended.

    Parameters
    ----------
    sdata
        SpatialData object
    """
    adata = sdata.table.copy()
    for dataset_id in adata.uns["spatial"]:
        adata.uns["spatial"][dataset_id]["images"] = {
            "hires": np.array(sdata.images[f"{dataset_id}_hires_image"]).transpose([1, 2, 0]),
            "lowres": np.array(sdata.images[f"{dataset_id}_lowres_image"]).transpose([1, 2, 0]),
        }
        adata.uns["spatial"][dataset_id]["scalefactors"] = {
            "tissue_hires_scalef": get_transformation(
                sdata.shapes[dataset_id], to_coordinate_system="downscaled_hires"
            ).scale[0],
            "tissue_lowres_scalef": get_transformation(
                sdata.shapes[dataset_id], to_coordinate_system="downscaled_lowres"
            ).scale[0],
            "spot_diameter_fullres": sdata.shapes[dataset_id]["radius"][0] * 2,
        }

    return adata
