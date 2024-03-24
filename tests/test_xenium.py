import numpy as np

from spatialdata_io.readers.xenium import (
    cell_id_str_from_prefix_suffix_uint32,
    prefix_suffix_uint32_from_cell_id_str,
)


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    cell_id_prefix = np.array([1, 1437536272, 1437536273], dtype=np.uint32)
    dataset_suffix = np.array([1, 1, 2])

    cell_id_str = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    assert np.array_equal(cell_id_str, np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"]))


def test_prefix_suffix_uint32_from_cell_id_str() -> None:
    cell_id_str = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    cell_id_prefix, dataset_suffix = prefix_suffix_uint32_from_cell_id_str(cell_id_str)
    assert np.array_equal(cell_id_prefix, np.array([1, 1437536272, 1437536273], dtype=np.uint32))
    assert np.array_equal(dataset_suffix, np.array([1, 1, 2]))


def test_roundtrip_with_data_limits() -> None:
    # min and max values for uint32
    cell_id_prefix = np.array([0, 4294967295], dtype=np.uint32)
    dataset_suffix = np.array([1, 1])
    cell_id_str = np.array(["aaaaaaaa-1", "pppppppp-1"])
    f0 = cell_id_str_from_prefix_suffix_uint32
    f1 = prefix_suffix_uint32_from_cell_id_str
    assert np.array_equal(cell_id_prefix, f1(f0(cell_id_prefix, dataset_suffix))[0])
    assert np.array_equal(dataset_suffix, f1(f0(cell_id_prefix, dataset_suffix))[1])
    assert np.array_equal(cell_id_str, f0(*f1(cell_id_str)))
