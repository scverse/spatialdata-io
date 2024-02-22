import numpy as np

from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    cell_id_prefix = np.array([1, 1437536272, 1437536273])
    dataset_suffix = np.array([1, 1, 2])

    cell_id_str = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    assert np.array_equal(cell_id_str, np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"]))
