from typing import Literal

from anndata import AnnData
from anndata.tests.helpers import assert_equal
from spatialdata.datasets import blobs
from spatialdata.models import TableModel

from spatialdata_io import from_legacy_anndata, to_legacy_anndata

# from spatialdata.


def blobs_annotating_element(name: Literal["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"]):
    sdata = blobs()
    n = len(sdata[name])
    new_table = AnnData(shape=(n, 0), obs={"region": [name for _ in range(n)], "instance_id": sdata[name].index})
    new_table = TableModel.parse(new_table, region=name, region_key="region", instance_key="instance_id")
    del sdata.table
    sdata.table = new_table
    # TODO: call helper function to shuffle the order of the rows of the table and of the shapes
    return sdata


def test_invalid_coordinate_system():
    pass
    # coordinate system not passed but multiple present
    # coordinate system passed but multiple present, not matching


def test_invalid_annotations():
    pass
    # table annotating labels
    # table annotating multiple shapes
    # table not annotating any shapes
    # table annotating a shapes but with instance_key not matching


# valid coordinate systems combinations:
# not passed but only one present
# passed and multiple present, matching with one of them
# valid shapes combinations: polygons, multipolygons, circles
# images: no images, one image, multiple images, negative translation and rotation
def test_bidectional_convesion():
    pass
    # test idempotency


def idempotency_check_to_anndata(sdata):
    adata = to_legacy_anndata(sdata)
    adata2 = to_legacy_anndata(from_legacy_anndata(adata))
    assert_equal(adata, adata2)


def idempotency_check_from_anndata(adata):
    sdata = from_legacy_anndata(adata)
    from_legacy_anndata(to_legacy_anndata(sdata))

    # assert_spatialdata_objects_seem_equal(sdata, sdata2)


# new branch in spatialdata
# TODO: make assert spatialdata objects seem equal public
# TODO: make get_centroids public
# TODO: make a simple helper function in relational query
# TODO: create _util function transform_to_data_extent() in spatialdata
