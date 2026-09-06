"""Tests for the WebP display pyramid.

Zoom numbering follows DeepZoom so the output is interchangeable with `vips dzsave`
tiles; the numbering tests below encode that convention explicitly.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr

from spatialdata_io.experimental.webp_parquet import write_webp_pyramid

TILE = 128
WIDTH, HEIGHT = 500, 300


def _image(width: int = WIDTH, height: int = HEIGHT, channels: int = 2) -> xr.DataArray:
    y, x = np.mgrid[0:height, 0:width]
    base = ((np.sin(x / 17.0) * np.cos(y / 11.0) + 1) * 1000).astype(np.uint16)
    stack = np.stack([base * (i + 1) for i in range(channels)])
    return xr.DataArray(stack, dims=("c", "y", "x"), coords={"c": [f"ch{i}" for i in range(channels)]})


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, dict]:
    out = tmp_path / "dapi"
    manifest = write_webp_pyramid(_image(), out, tile_size=TILE, source_element="morphology")
    return out, manifest


def _tile(directory: Path, manifest: dict, zoom: int, tx: int, ty: int):
    zi = manifest["zoom_info"][str(zoom)]
    rg = zi["row_group_offset"] + tx * zi["num_tiles_y"] + ty
    file_index, local = divmod(rg, manifest["max_row_groups_per_file"])
    return pq.ParquetFile(directory / manifest["files"][file_index]).read_row_group(local)


# -- zoom numbering -----------------------------------------------------------


def test_max_zoom_follows_deepzoom(written: tuple[Path, dict]) -> None:
    _, manifest = written
    assert manifest["max_zoom"] == math.ceil(math.log2(max(WIDTH, HEIGHT)))


def test_every_level_down_to_zero_exists(written: tuple[Path, dict]) -> None:
    """vips dzsave emits levels 0..max; matching it keeps the two interchangeable."""
    _, manifest = written
    assert sorted(int(z) for z in manifest["zoom_info"]) == list(range(manifest["max_zoom"] + 1))


def test_level_dimensions_halve(written: tuple[Path, dict]) -> None:
    _, manifest = written
    max_zoom = manifest["max_zoom"]
    for zoom in range(max_zoom, 0, -1):
        w = math.ceil(WIDTH / 2 ** (max_zoom - zoom))
        h = math.ceil(HEIGHT / 2 ** (max_zoom - zoom))
        zi = manifest["zoom_info"][str(zoom)]
        assert zi["num_tiles_x"] == max(1, math.ceil(w / TILE)), f"zoom {zoom}"
        assert zi["num_tiles_y"] == max(1, math.ceil(h / TILE)), f"zoom {zoom}"


def test_row_group_offsets_are_cumulative(written: tuple[Path, dict]) -> None:
    _, manifest = written
    running = 0
    for zoom in sorted(manifest["zoom_info"], key=int):
        zi = manifest["zoom_info"][zoom]
        assert zi["row_group_offset"] == running
        assert zi["num_tiles"] == zi["num_tiles_x"] * zi["num_tiles_y"]
        running += zi["num_tiles"]
    assert running == manifest["total_row_groups"]


def test_index_formula_addresses_the_right_tile(written: tuple[Path, dict]) -> None:
    """Mirror of ImageRowGroupReader.computeRowGroupIndex."""
    directory, manifest = written
    for zoom_s, zi in manifest["zoom_info"].items():
        for tx in range(zi["num_tiles_x"]):
            for ty in range(zi["num_tiles_y"]):
                t = _tile(directory, manifest, int(zoom_s), tx, ty)
                assert (t["zoom"][0].as_py(), t["tile_x"][0].as_py(), t["tile_y"][0].as_py()) == (
                    int(zoom_s),
                    tx,
                    ty,
                )


# -- payload ------------------------------------------------------------------


def test_tiles_decode_as_webp(written: tuple[Path, dict]) -> None:
    from PIL import Image

    directory, manifest = written
    t = _tile(directory, manifest, manifest["max_zoom"], 0, 0)
    img = Image.open(io.BytesIO(t["image_data"][0].as_py()))
    assert img.format == "WEBP"
    assert img.size == (TILE, TILE)


def test_edge_tile_is_cropped_not_padded(written: tuple[Path, dict]) -> None:
    from PIL import Image

    directory, manifest = written
    max_zoom = manifest["max_zoom"]
    zi = manifest["zoom_info"][str(max_zoom)]
    t = _tile(directory, manifest, max_zoom, zi["num_tiles_x"] - 1, 0)
    img = Image.open(io.BytesIO(t["image_data"][0].as_py()))
    assert img.size[0] == WIDTH - (zi["num_tiles_x"] - 1) * TILE


def test_schema_matches_celldega_reader(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    schema = pq.ParquetFile(directory / manifest["files"][0]).schema_arrow
    assert schema.names == ["zoom", "tile_x", "tile_y", "image_data"]
    meta = schema.metadata
    assert json.loads(meta[b"zoom_info"]) == manifest["zoom_info"]
    assert meta[b"storage_mode"] == b"row_groups_image_chunked"


def test_one_tile_per_row_group(written: tuple[Path, dict]) -> None:
    directory, manifest = written
    for f in manifest["files"]:
        md = pq.ParquetFile(directory / f).metadata
        assert all(md.row_group(i).num_rows == 1 for i in range(md.num_row_groups))


# -- channels and windowing ---------------------------------------------------


def test_channel_can_be_selected_by_name(tmp_path: Path) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = write_webp_pyramid(_image(), a_dir, tile_size=TILE, channel="ch0")
    b = write_webp_pyramid(_image(), b_dir, tile_size=TILE, channel="ch1")
    assert a["channel"] == "ch0" and b["channel"] == "ch1"
    # ch1 is twice ch0, so the encoded pixels must differ even though the window matches.
    pa_bytes = _tile(a_dir, a, a["max_zoom"], 0, 0)["image_data"][0].as_py()
    pb_bytes = _tile(b_dir, b, b["max_zoom"], 0, 0)["image_data"][0].as_py()
    assert pa_bytes != pb_bytes


def test_default_window_is_the_full_dtype_range(tmp_path: Path) -> None:
    """No stretch by default, matching Celldega's pipeline.

    The viewer applies its own intensity slider; a percentile stretch here would be
    applied on top of it and blow out the mid-tones.
    """
    m = write_webp_pyramid(_image(), tmp_path / "d", tile_size=TILE)
    assert (m["display_min"], m["display_max"]) == (0.0, float(np.iinfo(np.uint16).max))


def test_explicit_window_is_recorded_and_used(tmp_path: Path) -> None:
    m = write_webp_pyramid(_image(), tmp_path / "w", tile_size=TILE, display_min=100, display_max=900)
    assert (m["display_min"], m["display_max"]) == (100.0, 900.0)


def test_degenerate_window_does_not_divide_by_zero(tmp_path: Path) -> None:
    flat = xr.DataArray(np.full((1, 200, 200), 7, dtype=np.uint16), dims=("c", "y", "x"))
    m = write_webp_pyramid(flat, tmp_path / "flat", tile_size=TILE)
    assert m["display_max"] > m["display_min"]


def test_multiscale_input_uses_full_resolution(tmp_path: Path) -> None:
    """A DataTree must be tiled from scale0, not from a downsampled level."""
    from xarray import DataTree

    full = _image(400, 200, channels=1)
    half = full.isel(y=slice(None, None, 2), x=slice(None, None, 2))
    tree = DataTree.from_dict({"scale0": full.to_dataset(name="image"), "scale1": half.to_dataset(name="image")})
    m = write_webp_pyramid(tree, tmp_path / "ms", tile_size=TILE)
    assert (m["source_width"], m["source_height"]) == (400, 200)


# -- metadata and safety ------------------------------------------------------


def test_source_metadata_is_recorded_for_invalidation(written: tuple[Path, dict]) -> None:
    _, m = written
    assert m["source_element"] == "morphology"
    assert (m["source_width"], m["source_height"]) == (WIDTH, HEIGHT)
    assert m["source_dtype"] == "uint16"
    assert m["downsampling"] == "box-2x2-mean"
    assert m["image_format"] == ".webp"


def test_lossless_is_recorded(tmp_path: Path) -> None:
    m = write_webp_pyramid(_image(), tmp_path / "ll", tile_size=TILE, lossless=True)
    assert m["webp_lossless"] is True
    assert m["webp_quality"] is None


def test_overwrite_guard(written: tuple[Path, dict]) -> None:
    directory, _ = written
    with pytest.raises(FileExistsError):
        write_webp_pyramid(_image(), directory, tile_size=TILE)


def test_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    out = tmp_path / "boom"
    bad = xr.DataArray(np.zeros((2, 2, 10, 10), dtype=np.uint16), dims=("t", "c", "y", "x"))
    with pytest.raises(ValueError, match="expected a 2D plane"):
        write_webp_pyramid(bad, out, tile_size=TILE)
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()
