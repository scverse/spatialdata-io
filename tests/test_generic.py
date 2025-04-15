import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image
from spatialdata import SpatialData
from spatialdata.datasets import blobs

from spatialdata_io.__main__ import read_generic_wrapper
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr


@contextmanager
def save_temp_files() -> Generator[tuple[Path, Path, Path], None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        sdata = blobs()
        # save the image as jpg
        x = sdata["blobs_image"].data.compute()
        x = np.clip(x * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        im = Image.fromarray(x)
        jpg_path = Path(tmpdir) / "blobs_image.jpg"
        im.save(jpg_path)

        # save the shapes as geojson
        gdf = sdata["blobs_multipolygons"]
        geojson_path = Path(tmpdir) / "blobs_multipolygons.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")

        yield jpg_path, geojson_path, Path(tmpdir)


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("element_name", [None, "test_element"])
def test_read_generic_image(runner: CliRunner, cli: bool, element_name: str | None) -> None:
    with save_temp_files() as (image_path, geojson_path, tmpdir):
        output_zarr_path = tmpdir / "output.zarr"
        if cli:
            result = runner.invoke(
                read_generic_wrapper,
                [
                    "--input",
                    image_path,
                    "--output",
                    output_zarr_path,
                    "--name",
                    element_name,
                    "--data-axes",
                    "cyx",
                    "--coordinate-system",
                    "global",
                ],
            )
            assert result.exit_code == 0, result.output
            assert f"Data written to {tmpdir}" in result.output
        else:
            generic_to_zarr(
                input=image_path,
                output=output_zarr_path,
                name=element_name,
                data_axes="cyx",
                coordinate_system="global",
            )
        sdata = SpatialData.read(output_zarr_path)
        if element_name is None:
            assert "blobs_image" in sdata
        else:
            assert element_name in sdata


def test_cli_read_generic_image_invalid_data_axes(runner: CliRunner) -> None:
    with save_temp_files() as (image_path, geojson_path, tmpdir):
        output_zarr_path = tmpdir / "output.zarr"

        result = runner.invoke(
            read_generic_wrapper,
            [
                "--input",
                image_path,
                "--output",
                output_zarr_path,
                "--data-axes",
                "invalid_axes",
            ],
        )
        assert result.exit_code != 0, result.output
        assert "data_axes must be a permutation of 'cyx' or 'czyx'." in result.exc_info[1].args[0]


@pytest.mark.parametrize("cli", [True, False])
def test_read_generic_geojson(runner: CliRunner, cli: bool) -> None:
    with save_temp_files() as (image_path, geojson_path, tmpdir):
        output_zarr_path = tmpdir / "output.zarr"

        if cli:
            result = runner.invoke(
                read_generic_wrapper,
                [
                    "--input",
                    geojson_path,
                    "--output",
                    output_zarr_path,
                    "--coordinate-system",
                    "global",
                ],
            )
            assert result.exit_code == 0, result.output
            assert f"Data written to {tmpdir}" in result.output
        else:
            generic_to_zarr(input=geojson_path, output=output_zarr_path, coordinate_system="global")

        sdata = SpatialData.read(output_zarr_path)
        assert "blobs_multipolygons" in sdata
