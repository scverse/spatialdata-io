from typing import Literal

import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from spatialdata._core.query.relational_query import _get_unique_label_values_as_index
from spatialdata._utils import _assert_spatialdata_objects_seem_identical
from spatialdata.datasets import blobs
from spatialdata.models import TableModel

from spatialdata_io import from_legacy_anndata, to_legacy_anndata


def blobs_annotating_element(name: Literal["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"]):
    sdata = blobs()
    if name == "blobs_labels":
        instance_id = _get_unique_label_values_as_index(sdata[name]).tolist()
    else:
        instance_id = sdata[name].index.tolist()
    n = len(instance_id)
    new_table = AnnData(shape=(n, 0), obs={"region": [name for _ in range(n)], "instance_id": instance_id})
    new_table = TableModel.parse(new_table, region=name, region_key="region", instance_key="instance_id")
    del sdata.table
    sdata.table = new_table
    return sdata


def idempotency_check_to_anndata(sdata):
    adata = to_legacy_anndata(sdata)
    adata2 = to_legacy_anndata(from_legacy_anndata(adata))
    assert_equal(adata, adata2)


def idempotency_check_from_anndata(adata):
    sdata = from_legacy_anndata(adata)
    sdata1 = from_legacy_anndata(to_legacy_anndata(sdata))
    _assert_spatialdata_objects_seem_identical(sdata, sdata1)


@pytest.mark.parametrize("name", ["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"])
def test_bidectional_convesion(name):
    sdata0 = blobs_annotating_element(name)
    adata0 = to_legacy_anndata(sdata0)
    sdata1 = from_legacy_anndata(adata0)

    idempotency_check_to_anndata(sdata1)
    idempotency_check_from_anndata(adata0)
