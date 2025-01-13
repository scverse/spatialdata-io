import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner
from spatialdata import read_zarr

from spatialdata_io.__main__ import seqfish_wrapper
from spatialdata_io.readers.seqfish import seqfish
from tests._utils import skip_if_below_python_version


# See https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml for instructions on
# how to download and place the data on disk
@skip_if_below_python_version()
@pytest.mark.parametrize(
    "dataset,expected", [("seqfish-2-test-dataset/instrument 2 official", "{'y': (0, 108), 'x': (0, 108)}")]
)
@pytest.mark.parametrize("rois", [[1], None])
@pytest.mark.parametrize("cells_as_circles", [False, True])
def test_example_data(dataset: str, expected: str, rois: list[int] | None, cells_as_circles: bool) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    sdata = seqfish(f, cells_as_circles=cells_as_circles, rois=rois)
    from spatialdata import get_extent

    extent = get_extent(sdata, exact=False)
    extent = {ax: (math.floor(extent[ax][0]), math.ceil(extent[ax][1])) for ax in extent}
    if cells_as_circles:
        # manual correction required to take into account for the circle radii
        expected = "{'y': (-2, 109), 'x': (-2, 109)}"
    assert str(extent) == expected


@skip_if_below_python_version()
@pytest.mark.parametrize("dataset", ["seqfish-2-test-dataset/instrument 2 official"])
def test_cli_seqfish(runner: CliRunner, dataset: str) -> None:
    f = Path("./data") / dataset
    assert f.is_dir()
    with TemporaryDirectory() as tmpdir:
        output_zarr = Path(tmpdir) / "data.zarr"
        result = runner.invoke(
            seqfish_wrapper,
            [
                "--input",
                f,
                "--output",
                output_zarr,
            ],
        )
        assert result.exit_code == 0, result.output
        _ = read_zarr(output_zarr)
