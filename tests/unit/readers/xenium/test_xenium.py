from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from spatialdata_io.__main__ import xenium_wrapper
from spatialdata_io.readers.xenium import (
    _cell_id_str_from_prefix_suffix_uint32_reference,
    cell_id_str_from_prefix_suffix_uint32,
    prefix_suffix_uint32_from_cell_id_str,
)


def test_cell_id_str_from_prefix_suffix_uint32() -> None:
    cell_id_prefix = np.array([1, 1437536272, 1437536273], dtype=np.uint32)
    dataset_suffix = np.array([1, 1, 2])
    expected = np.array(["aaaaaaab-1", "ffkpbaba-1", "ffkpbabb-2"])

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, expected)
    assert np.array_equal(reference, expected)


def test_cell_id_str_optimized_matches_reference() -> None:
    rng = np.random.default_rng(42)
    cell_id_prefix = rng.integers(0, 2**32, size=10_000, dtype=np.uint32)
    dataset_suffix = rng.integers(0, 10, size=10_000)

    result = cell_id_str_from_prefix_suffix_uint32(cell_id_prefix, dataset_suffix)
    reference = _cell_id_str_from_prefix_suffix_uint32_reference(cell_id_prefix, dataset_suffix)
    assert np.array_equal(result, reference)


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


@pytest.mark.parametrize(
    "kwarg_name",
    ["--imread-kwargs", "--image-models-kwargs", "--labels-models-kwargs"],
)
def test_cli_xenium_invalid_json_rejected(runner: CliRunner, tmp_path: Path, kwarg_name: str) -> None:
    """Invalid JSON for any kwargs option must produce a non-zero exit and a clear error."""
    result = runner.invoke(
        xenium_wrapper,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            "not-valid-json{",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output


@pytest.mark.parametrize(
    ("kwarg_name", "kwarg_param"),
    [
        ("--imread-kwargs", "imread_kwargs"),
        ("--image-models-kwargs", "image_models_kwargs"),
        ("--labels-models-kwargs", "labels_models_kwargs"),
    ],
)
def test_cli_xenium_valid_json_forwarded(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture, kwarg_name: str, kwarg_param: str
) -> None:
    """Valid JSON kwargs must be parsed and forwarded to the xenium reader as a dict."""
    mock_xenium = mocker.patch("spatialdata_io.readers.xenium.xenium")
    mock_xenium.return_value = mocker.MagicMock()
    result = runner.invoke(
        xenium_wrapper,
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.zarr"),
            kwarg_name,
            '{"chunks": 512}',
        ],
    )
    assert result.exit_code == 0, result.output
    call_kwargs = mock_xenium.call_args.kwargs
    assert call_kwargs[kwarg_param] == {"chunks": 512}
