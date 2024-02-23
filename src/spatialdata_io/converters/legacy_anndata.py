from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData, join_sdata_spatialelement_table
from spatialdata.models import Image2DModel, ShapesModel, TableModel, get_table_keys
from spatialdata.transformations import Identity, Scale


def to_legacy_anndata(
    sdata: SpatialData, coordinate_system: str | None = None, table_name: str | None = None
) -> AnnData:
    """
    Convert a SpatialData object to a (legacy) spatial AnnData object.

    This is useful for using packages expecting spatial information in AnnData, for example Scanpy and older versions
    of Squidpy. Using this format for any new package is not recommended.

    Parameters
    ----------
    sdata
        SpatialData object
    coordinate_system
        The coordinate system to consider. The AnnData object will be populated with the data transformed to this
        coordinate system.
    table_name
        The name of the table in the SpatialData object to consider. If None and the SpatialData object has only one
        table, that table will be used.

    Returns
    -------
    The legacy spatial AnnData object

    Notes
    -----
    Edge cases and limitations:

        - The Table can only annotate Shapes or Labels.
        - Labels will be approximated to circles by using the centroids of each label and an average approximated
          radius.
        - The Table cannot annotate more than one Shapes element. If multiple Shapes elements are present, please merge
          them into a single Shapes element and make the table annotate the new Shapes element.
        - Table rows not annotating geometries in the coordinate system will be dropped. Similarly, Shapes rows (in the
          case of Labels, the circle rows approximating the Labels) not annotated by Table rows will be dropped.

    How resolutions, coordinates, indices and Table rows will be affected:

        - The Images will be scaled (see later) and will not maintain the original resolution.
        - The Shapes centroids will likely change to match the alignment in the specified coordinate system, and to the
          images bounding boxes (see below).
        - The Table and Shapes rows will have the same indices as before the conversion, which is useful for plugging
          results back to the SpatialData object, but as mentioned above, some rows may be dropped.

    The generated AnnData object will contain only low-resolution images and their origin will be reset to the pixel
    (0, 0). In particular, the ImageContainer class used by Squidpy is not used. Eventual transformations are applied
    before conversion. Precisely, this is how the Images are prepared for the AnnData object:

        - For each Image aligned to the specified coordinate system, the coordinate transformation to the coordinate
          system is applied; all the Images are rendered to a common target bounding box, which is the bounding box of
          the union of the Images aligned to the target coordinate system.
        - The origins of the new images will match the pixel (0, 0) of the target bounding box.
        - Each Image is downscaled during rendering to fit a 2000x2000 pixels image (downscaled_hires) and a 600x600
          pixels image (downscaled_lowres).
        - Practically the above implies that if the Images have very dissimilar locations, most the rendered Images will
          have empty spaces; in such cases it is recommended to drop or crop some images before the conversion.

    Matching of spatial coordinates and pixel coordinates:

        - The coordinates in `obsm['spatial']` will match the high-resolution (generated) images (i.e. coordinate (x, y)
          will match with the pixel (y, x) in the image).
        - The above matching is not perfect due to this bug: https://github.com/scverse/spatialdata/issues/165, which
          will eventually be addressed.

    Final consideration:

        - Due to the legacy nature of the AnnData object generated, this function may not handle all the edge cases,
          please report any issue you may find.
    """
    # check that coordinate system is valid
    css = sdata.coordinate_systems
    if coordinate_system is None:
        assert len(css) == 1, "The SpatialData object has more than one coordinate system. Please specify one."
        coordinate_system = css[0]
    else:
        assert (
            coordinate_system in css
        ), f"The SpatialData object does not have the coordinate system {coordinate_system}."
    sdata = sdata.filter_by_coordinate_system(coordinate_system)

    if table_name is None:
        assert (
            len(sdata.tables) == 1
        ), "When the table_name is not specified, the SpatialData object must have exactly one table."
        table_name = next(iter(sdata.tables))
    else:
        assert table_name in sdata.tables, f"The table {table_name} is not present in the SpatialData object."

    table = sdata[table_name]
    (
        region,
        _,
        instance_key,
    ) = get_table_keys(table)
    if not isinstance(region, list):
        region = [region]

    # the table needs to annotate exactly one Shapes element
    if len(region) != 1:
        raise ValueError(f"The table needs to annotate exactly one element. Found {len(region)}.")
    if region[0] not in sdata.shapes and region[0] not in sdata.labels:
        raise ValueError(f"The table needs to annotate a Shapes or Labels element, not Points.")
    element = sdata[region[0]]

    # TODO: call get_centroids() here and convert polygons/multipolygons/labels to circles

    if isinstance(element, (SpatialImage, MultiscaleSpatialImage)):
        table.obs[instance_key] = table.obs[instance_key].astype(element.dtype)
    joined_elements, new_table = join_sdata_spatialelement_table(
        sdata=sdata, spatial_element_name=region[0], table_name=table_name, how="inner", match_rows="left"
    )
    joined_elements[region[0]]

    # the table after the filtering must not be empty
    assert len(new_table) > 0, (
        "The table does not annotate any geometry in the Shapes element. This could also be caused by a mismatch "
        "between the type of the indices of the geometries and the type of the INSTANCE_KEY column of the table."
    )

    # get the average radius; approximates polygons/multipolygons as circles
    if new_shapes.iloc[0] == "Point":
        np.mean(new_shapes["radius"])
    else:
        np.mean(np.sqrt(new_shapes.geometry.area / np.pi))

    adata = new_table.copy()
    adata.obsm["spatial"] = xy
    # # process the images
    # sdata_images = sdata.subset(element_names=list(sdata.images.keys()))
    # bb = get_extent(sdata_images)

    #     adata = sdata.table.copy()
    #     for dataset_id in adata.uns["spatial"]:
    #         adata.uns["spatial"][dataset_id]["images"] = {
    #             "hires": np.array(sdata.images[f"{dataset_id}_hires_image"]).transpose([1, 2, 0]),
    #             "lowres": np.array(sdata.images[f"{dataset_id}_lowres_image"]).transpose([1, 2, 0]),
    #         }
    #         adata.uns["spatial"][dataset_id]["scalefactors"] = {
    #             "tissue_hires_scalef": get_transformation(
    #                 sdata.shapes[dataset_id], to_coordinate_system="downscaled_hires"
    #             ).scale[0],
    #             "tissue_lowres_scalef": get_transformation(
    #                 sdata.shapes[dataset_id], to_coordinate_system="downscaled_lowres"
    #             ).scale[0],
    #             "spot_diameter_fullres": sdata.shapes[dataset_id]["radius"][0] * 2,
    #         }
    #
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

    Returns
    -------
    The SpatialData object

    Notes
    -----
    The SpatialData object will have one hires and one lores image for each dataset in the AnnData object.
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
