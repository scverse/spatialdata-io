from __future__ import annotations

import warnings

import numpy as np
from anndata import AnnData
from spatialdata import (
    SpatialData,
    get_centroids,
    get_extent,
    join_spatialelement_table,
    to_circles,
)
from spatialdata._core.operations._utils import transform_to_data_extent
from spatialdata.models import Image2DModel, ShapesModel, TableModel, get_table_keys
from spatialdata.transformations import Identity, Scale


def to_legacy_anndata(
    sdata: SpatialData,
    coordinate_system: str | None = None,
    table_name: str | None = None,
    include_images: bool = False,
) -> AnnData:
    """
    Convert a SpatialData object to a (legacy) spatial AnnData object.

    This is useful for using packages expecting spatial information in AnnData, for example Scanpy and older versions
    of Squidpy. Using this format for any new package is not recommended.

    This function by default ignores images (recommended); setting the `include_images` parameter to `True` will
    include a downscaled version of the images in the output AnnData object.

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
    include_images
        If True, includes downscaled versions of the images in the output AnnData object. It is recommended to handle
        the full resolution images separately and keep them in the SpatialData object.

    Returns
    -------
    The legacy spatial AnnData object.

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

        - When `include_images` is `True`, the Images will be scaled (see later) and will not maintain the original
          resolution.
        - The Shapes centroids will likely change to match the alignment in the specified coordinate system, and to the
          images bounding boxes when `include_images` is `True` (see below).
        - The Table and Shapes rows will have the same indices as before the conversion, which is useful for plugging
          results back to the SpatialData object, but as mentioned above, some rows may be dropped.

    When `include_images` is `True`, the generated AnnData object will contain low-resolution images and their origin
    will be reset to the pixel (0, 0). In particular, the ImageContainer class used by Squidpy is not used. Eventual
    transformations are applied before conversion. Precisely, this is how the Images are prepared for the AnnData
    object:

        - For each Image aligned to the specified coordinate system, the coordinate transformation to the coordinate
          system is applied; all the Images are rendered to a common target bounding box, which is the bounding box of
          the union of the Images aligned to the target coordinate system.
        - The origins of the new images will match the pixel (0, 0) of the target bounding box.
        - Each Image is downscaled (or upscaled) during rendering to fit a 2000x2000 pixels image (downscaled_hires) and
          a 600x600 pixels image (downscaled_lowres).
        - Practically the above implies that if the Images have very dissimilar locations, most the rendered Images will
          have empty spaces; in such cases it is recommended to drop or crop some images before the conversion.

    Matching of spatial coordinates and pixel coordinates:

        - When `include_images` is `True`, the coordinates in `obsm['spatial']` will match the downscaled (generated)
          image "hires" (i.e. coordinate (x, y) will match with the pixel (y, x) in the image). Please note that the
          "hires" naming is used for legacy reasons, but the image is actually downscaled and fairly low resolution.
        - The above matching is not perfect due to this bug: https://github.com/scverse/spatialdata/issues/165, which
          will eventually be addressed.

    Imperfect downscaling:

        - Due to the bug mentioned above, https://github.com/scverse/spatialdata/issues/165, when the downscaling factor
          is large, the error between the original and the downscale image can be significant. This is another reason to
          not recommend using include_images=True. Note, however, that the error can be large only between the original
          and the downscaled images, not between the downscaled images and the new spatial coordinates. Thus the use
          may be acceptable for some use cases.

    Final consideration:

        - Due to the legacy nature of the AnnData object generated, this function may not handle all the edge cases,
          please report any issue you may find.
    """
    DOWNSCALED_HIRES_LENGTH = 2000
    DOWNSCALED_LOWRES_LENGTH = 600
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
        assert len(sdata.tables) == 1, (
            "When the table_name is not specified, the SpatialData object must have exactly one table. The specified "
            f"sdata object, after being filtered for the coordinate system {coordinate_system}, has "
            f"{len(sdata.tables)} tables."
        )
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
        raise ValueError("The table needs to annotate a Shapes or Labels element, not Points.")
    element = sdata[region[0]]
    region_name = region[0]

    # convert polygons, multipolygons and labels to circles
    shapes = to_circles(element)
    circles_sdata = SpatialData(tables={table_name: table}, shapes={region_name: shapes})

    joined_elements, new_table = join_spatialelement_table(
        sdata=circles_sdata, spatial_element_names=region_name, table_name=table_name, how="inner", match_rows="left"
    )

    # the table after the filtering must not be empty
    assert len(new_table) > 0, (
        "The table does not annotate any geometry in the Shapes element. This could also be caused by a mismatch "
        "between the type of the indices of the geometries and the type of the INSTANCE_KEY column of the table."
    )

    sdata_pre_rasterize = SpatialData(
        tables={table_name: new_table}, shapes={region_name: joined_elements[region_name]}
    )

    # process the images
    downscaled_images_hires = {}
    downscaled_images_lowres = {}
    sdata_post_rasterize = None
    if not include_images:
        sdata_post_rasterize = sdata_pre_rasterize
    else:
        sdata_images = sdata.subset(element_names=list(sdata.images.keys()))
        for image_name, image in sdata_images.images.items():
            sdata_pre_rasterize[image_name] = image
        bb = get_extent(sdata_images, coordinate_system=coordinate_system)

        parameters_dict = {"x": "target_width", "y": "target_height", "z": "target_depth"}
        longest_side = max(bb, key=bb.get)
        parameter_hires = {parameters_dict[longest_side]: DOWNSCALED_HIRES_LENGTH}
        parameter_lowres = {parameters_dict[longest_side]: DOWNSCALED_LOWRES_LENGTH}
        downscaled_hires = transform_to_data_extent(
            sdata_pre_rasterize, coordinate_system=coordinate_system, maintain_positioning=False, **parameter_hires
        )
        downscaled_lowres = transform_to_data_extent(
            sdata_pre_rasterize, coordinate_system=coordinate_system, maintain_positioning=False, **parameter_lowres
        )
        for image_name in sdata_images.images.keys():
            downscaled_images_hires[image_name] = downscaled_hires[image_name]
            downscaled_images_lowres[image_name] = downscaled_lowres[image_name]

        if sdata_post_rasterize is None:
            sdata_post_rasterize = downscaled_hires

    adata = sdata_post_rasterize[table_name].copy()

    if len(downscaled_images_hires) > 0:
        for image_name in sdata_images.images.keys():
            if "spatial" not in adata.uns:
                adata.uns["spatial"] = {}
            adata.uns["spatial"][image_name] = {}
            adata.uns["spatial"][image_name]["images"] = {
                "hires": downscaled_images_hires[image_name].data.compute().transpose([1, 2, 0]),
                "lowres": downscaled_images_lowres[image_name].data.compute().transpose([1, 2, 0]),
            }
            try:
                adata.uns["spatial"][image_name]["scalefactors"] = {
                    "tissue_hires_scalef": 1.0,
                    "tissue_lowres_scalef": DOWNSCALED_LOWRES_LENGTH / DOWNSCALED_HIRES_LENGTH,
                    "spot_diameter_fullres": sdata_post_rasterize.shapes[region_name]["radius"].iloc[0] * 2,
                }
            except KeyError:
                pass

    adata.obsm["spatial"] = (
        get_centroids(sdata_post_rasterize[region_name], coordinate_system=coordinate_system).compute().values
    )
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
    The SpatialData object will have one hires and one lowres image for each dataset in the AnnData object.
    """
    # AnnData keys
    SPATIAL = "spatial"

    SCALEFACTORS = "scalefactors"
    TISSUE_HIRES_SCALEF = "tissue_hires_scalef"
    TISSUE_LOWRES_SCALEF = "tissue_lowres_scalef"
    SPOT_DIAMETER_FULLRES = "spot_diameter_fullres"

    IMAGES = "images"
    HIRES = "hires"
    LOWRES = "lowres"

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
            tissue_lowres_scalef = None
            hires = None
            lowres = None
            if SCALEFACTORS in keys:
                scalefactors = adata.uns[SPATIAL][dataset_id][SCALEFACTORS]
                if TISSUE_HIRES_SCALEF in scalefactors:
                    tissue_hires_scalef = scalefactors[TISSUE_HIRES_SCALEF]
                if TISSUE_LOWRES_SCALEF in scalefactors:
                    tissue_lowres_scalef = scalefactors[TISSUE_LOWRES_SCALEF]
                if SPOT_DIAMETER_FULLRES in scalefactors:
                    spot_diameter_fullres_list.append(scalefactors[SPOT_DIAMETER_FULLRES])
            if IMAGES in keys:
                image_data = adata.uns[SPATIAL][dataset_id][IMAGES]
                if HIRES in image_data:
                    hires = image_data[HIRES]
                if LOWRES in image_data:
                    lowres = image_data[LOWRES]

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
            if lowres is not None:
                # prepare the lowres image
                assert (
                    tissue_lowres_scalef is not None
                ), "tissue_lowres_scalef is required when an the lowres image is present"
                lowres_image = Image2DModel.parse(
                    lowres, dims=("y", "x", "c"), transformations={f"{dataset_id}_downscaled_lowres": Identity()}
                )
                images[f"{dataset_id}_lowres_image"] = lowres_image

                # prepare the transformation to the lowres image for the shapes
                scale_lowres = Scale([tissue_lowres_scalef, tissue_lowres_scalef], axes=("x", "y"))
                shapes_transformations[f"{dataset_id}_downscaled_lowres"] = scale_lowres

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
        if TableModel.ATTRS_KEY in new_table.uns:
            del new_table.uns[TableModel.ATTRS_KEY]
        new_table.obs[REGION_KEY] = REGION
        new_table.obs[REGION_KEY] = new_table.obs[REGION_KEY].astype("category")
        new_table.obs[INSTANCE_KEY] = shapes[REGION].index.values
        new_table = TableModel.parse(new_table, region=REGION, region_key=REGION_KEY, instance_key=INSTANCE_KEY)
    else:
        new_table = adata.copy()
    return SpatialData(table=new_table, images=images, shapes=shapes)
