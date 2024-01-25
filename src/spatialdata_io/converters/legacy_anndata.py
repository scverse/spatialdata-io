import warnings

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale, get_transformation


def to_legacy_anndata(sdata: SpatialData) -> AnnData:
    """
    Convert SpatialData object to a (legacy) spatial AnnData object.

    This is useful for converting a Spatialdata object with Visium data for the use
    with packages using spatial information in AnnData as used by Scanpy and older versions of Squidpy.
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


def from_legacy_anndata(adata: AnnData) -> SpatialData:
    """
    Convert (legacy) spatial AnnData object to SpatialData object.

    This is useful for parsing a (legacy) spatial AnnData object, for example the ones produced by Scanpy and older
    version of Squidpy.

    Parameters
    ----------
    adata
        (legacy) spatial AnnData object
    """
    # AnnData keys
    SPATIAL = "spatial"

    SCALEFACTORS = "scalefactors"
    TISSUE_HIRES_SCALEF = "tissue_hires_scalef"
    TISSUE_LORES_SCALEF = "tissue_lowres_scalef"
    SPOT_DIAMETER_FULLRES = "spot_diameter_fullres"

    IMAGES = "images"
    HIRES = "hires"
    LORES = "lowres"

    # SpatialData keys
    REGION = "locations"
    REGION_KEY = "region"
    INSTANCE_KEY = "instance_id"
    SPOT_DIAMETER_FULLRES_DEFAULT = 10

    images = {}
    shapes = {}
    spot_diameter_fullres_list = []
    shapes_transformations = {}

    if SPATIAL in adata.uns:
        dataset_ids = list(adata.uns[SPATIAL].keys())
        for dataset_id in dataset_ids:
            # read the image data and the scale factors for the shapes
            keys = set(adata.uns[SPATIAL][dataset_id].keys())
            tissue_hires_scalef = None
            tissue_lores_scalef = None
            if SCALEFACTORS in keys:
                scalefactors = adata.uns[SPATIAL][dataset_id][SCALEFACTORS]
                if TISSUE_HIRES_SCALEF in scalefactors:
                    tissue_hires_scalef = scalefactors[TISSUE_HIRES_SCALEF]
                if TISSUE_LORES_SCALEF in scalefactors:
                    tissue_lores_scalef = scalefactors[TISSUE_LORES_SCALEF]
                if SPOT_DIAMETER_FULLRES in scalefactors:
                    spot_diameter_fullres_list.append(scalefactors[SPOT_DIAMETER_FULLRES])
            if IMAGES in keys:
                image_data = adata.uns[SPATIAL][dataset_id][IMAGES]
                if HIRES in image_data:
                    hires = image_data[HIRES]
                if LORES in image_data:
                    lores = image_data[LORES]

            # construct the spatialdata elements
            if hires is not None:
                # prepare the hires image
                assert (
                    tissue_hires_scalef is not None
                ), "tissue_hires_scalef is required when an the hires image is present"
                hires_image = Image2DModel.parse(
                    hires, dims=("y", "x", "c"), transformations={f"{dataset_id}_downscaled_hires": Identity()}
                )
                images[f"{dataset_id}_hires_image"] = hires_image

                # prepare the transformation to the hires image for the shapes
                scale_hires = Scale([tissue_hires_scalef, tissue_hires_scalef], axes=("x", "y"))
                shapes_transformations[f"{dataset_id}_downscaled_hires"] = scale_hires
            if lores is not None:
                # prepare the lores image
                assert (
                    tissue_lores_scalef is not None
                ), "tissue_lores_scalef is required when an the lores image is present"
                lores_image = Image2DModel.parse(
                    lores, dims=("y", "x", "c"), transformations={f"{dataset_id}_downscaled_lowres": Identity()}
                )
                images[f"{dataset_id}_lowres_image"] = lores_image

                # prepare the transformation to the lores image for the shapes
                scale_lores = Scale([tissue_lores_scalef, tissue_lores_scalef], axes=("x", "y"))
                shapes_transformations[f"{dataset_id}_downscaled_lowres"] = scale_lores

    # validate the spot_diameter_fullres value
    if len(spot_diameter_fullres_list) > 0:
        d = np.array(spot_diameter_fullres_list)
        if not np.allclose(d, d[0]):
            warnings.warn(
                "spot_diameter_fullres is not constant across datasets. Using the average value.",
                UserWarning,
                stacklevel=2,
            )
            spot_diameter_fullres = d.mean()
        else:
            spot_diameter_fullres = d[0]
    else:
        warnings.warn(
            f"spot_diameter_fullres is not present. Using {SPOT_DIAMETER_FULLRES_DEFAULT} as default value.",
            UserWarning,
            stacklevel=2,
        )
        spot_diameter_fullres = SPOT_DIAMETER_FULLRES_DEFAULT

    # parse and prepare the shapes
    if SPATIAL in adata.obsm:
        xy = adata.obsm[SPATIAL]
        radius = spot_diameter_fullres / 2
        shapes[REGION] = ShapesModel.parse(xy, geometry=0, radius=radius, transformations=shapes_transformations)

        # link the shapes to the table
        new_table = adata.copy()
        new_table.obs[REGION_KEY] = REGION
        new_table.obs[REGION_KEY] = new_table.obs[REGION_KEY].astype("category")
        new_table.obs[INSTANCE_KEY] = shapes[REGION].index.values
        new_table = TableModel.parse(new_table, region=REGION, region_key=REGION_KEY, instance_key=INSTANCE_KEY)
    else:
        new_table = adata.copy()
        # workaround for https://github.com/scverse/spatialdata/issues/306, not anymore needed as soona ss the
        # multi_table branch is merged
        new_table.obs[REGION_KEY] = "dummy"
        new_table.obs[REGION_KEY] = new_table.obs[REGION_KEY].astype("category")
        new_table.obs[INSTANCE_KEY] = np.arange(len(new_table))

        new_table = TableModel.parse(new_table, region="dummy", region_key=REGION_KEY, instance_key=INSTANCE_KEY)
    return SpatialData(table=new_table, images=images, shapes=shapes)
