from typing import Literal

import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from spatialdata import SpatialData, get_element_instances
from spatialdata.datasets import blobs
from spatialdata.models import TableModel
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import get_transformation, set_transformation

from spatialdata_io.experimental import from_legacy_anndata, to_legacy_anndata

BlobsTypes = Literal["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"]


def blobs_annotating_element(name: BlobsTypes) -> SpatialData:
    sdata = blobs(length=50)
    if name == "blobs_labels":
        instance_id = get_element_instances(sdata[name]).tolist()
    else:
        instance_id = sdata[name].index.tolist()
    n = len(instance_id)
    new_table = AnnData(shape=(n, 0), obs={"region": [name for _ in range(n)], "instance_id": instance_id})
    new_table = TableModel.parse(new_table, region=name, region_key="region", instance_key="instance_id")
    del sdata.table
    sdata.table = new_table
    return sdata


def _adjust_table_for_idempotency_check(sdata: SpatialData, include_images: bool) -> SpatialData:
    """We need to adjust the coordinate systems from from_legacy_anndata() to perform the idempotency check."""
    if not include_images:
        # nothing needs to be adjusted
        return sdata
    CS_NAME = "blobs_multiscale_image_downscaled_hires"
    sdata = sdata.filter_by_coordinate_system(CS_NAME)
    sdata.rename_coordinate_systems({CS_NAME: "global"})
    t = get_transformation(sdata["locations"], "global")
    set_transformation(sdata["locations"], {"global": t}, set_all=True)
    IMG_NAME = "blobs_multiscale_image_hires_image"
    im = sdata.images[IMG_NAME]
    del sdata.images[IMG_NAME]
    sdata.images["blobs_multiscale_image"] = im
    return sdata


def idempotency_check_to_anndata(sdata0: SpatialData, include_images: bool) -> None:
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    adata2 = to_legacy_anndata(sdata1, include_images=include_images)
    assert_equal(adata0, adata2)


def idempotency_check_from_anndata(adata0: SpatialData, include_images: bool) -> None:
    sdata0 = from_legacy_anndata(adata0)
    sdata0 = _adjust_table_for_idempotency_check(sdata0, include_images=include_images)
    adata1 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata1)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    assert_spatial_data_objects_are_identical(sdata0, sdata1)


@pytest.mark.parametrize("name", ["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"])
@pytest.mark.parametrize("include_images", [False, True])
def test_bidectional_convesion(name: BlobsTypes, include_images: bool) -> None:
    if include_images:
        pytest.skip(
            "include_images=True can't be tested because the bug "
            "https://github.com/scverse/spatialdata/issues/165 causes a large error"
        )
    sdata0 = blobs_annotating_element(name)
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)

    idempotency_check_to_anndata(sdata1, include_images=include_images)
    idempotency_check_from_anndata(adata0, include_images=include_images)
