from __future__ import annotations

import os
import pickle
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
import tifffile
from spatialdata.transformations import Identity, Translation

from spatialdata_io.readers._utils._akoya import (
    apply_channel_names,
    dedupe_channel_names,
    read_tiff,
    region_transformations,
)
from spatialdata_io.readers._utils._qptiff import AkoyaChannel, AkoyaScan


def _scan(
    region: str,
    stage: tuple[float, float] | None = None,
    pixel_size_um: float | None = 0.5,
    slide_id: str | None = "slide",
) -> AkoyaScan:
    channels = [AkoyaChannel(name=name) for name in ("DAPI", "DAPI", "CD3")]
    return AkoyaScan(
        Path(f"{region}.qptiff"), region, "fusion2", None, slide_id, pixel_size_um, None, stage, channels, False, {}
    )


def test_channel_names() -> None:
    # a generated suffix never repeats a name already in the list
    assert dedupe_channel_names(["DAPI", "DAPI", "DAPI (2)"]) == ["DAPI", "DAPI (3)", "DAPI (2)"]
    scan = _scan("a")
    assert apply_channel_names(scan, None) == ["DAPI", "DAPI (2)", "CD3"]
    assert apply_channel_names(scan, ["a", "b", "a"]) == ["a", "b", "a (2)"]
    with pytest.raises(ValueError, match="expected 3"):
        apply_channel_names(scan, ["a", "b"])
    # the keys of a mapping or the order of a set would silently become the channel names
    with pytest.raises(TypeError, match="sequence"):
        apply_channel_names(scan, {"a", "b", "c"})  # type: ignore[arg-type]


_GLOBAL = {
    "a": {"global": Translation([0.0, 0.0], axes=("x", "y"))},
    "b": {"global": Translation([200.0, 100.0], axes=("x", "y"))},
}
_OWN = {"a": {"a": Identity()}, "b": {"b": Identity()}}


@pytest.mark.parametrize(
    ("scans", "expected"),
    [
        ([_scan("a", (0.0, 0.0)), _scan("b", (100.0, 50.0))], _GLOBAL),
        # pixel sizes equal up to rounding share the sample space
        ([_scan("a", (0.0, 0.0)), _scan("b", (100.0, 50.0), pixel_size_um=0.5004)], None),
        ([_scan("a", (0.0, 0.0), slide_id="s1"), _scan("b", (100.0, 50.0), slide_id="s2")], _OWN),
        ([_scan("a", (0.0, 0.0)), _scan("b", (1.0, 1.0), pixel_size_um=0.25)], _OWN),
        ([_scan("a"), _scan("b")], _OWN),
        ([_scan("a", (0.0, 0.0), pixel_size_um=None), _scan("b", (1.0, 1.0))], _OWN),
        ([_scan("a", (0.0, 0.0), slide_id=None), _scan("b", (1.0, 1.0), slide_id=None)], _OWN),
    ],
    ids=[
        "same_slide",
        "rounded_pixel_size",
        "other_slide",
        "other_pixel_size",
        "no_stage",
        "no_pixel_size",
        "no_slide",
    ],
)
def test_region_transformations(scans: list[AkoyaScan], expected: dict | None) -> None:
    result = region_transformations(scans)
    if expected is None:
        assert set(result["b"]) == {"global"}
    else:
        assert result == expected
    # a region without an image is unrelated to the scans
    assert region_transformations(scans, ["c"])["c"] == {"c": Identity()}
    assert region_transformations([_scan("a", (5.0, 5.0))]) == {"a": {"global": Identity()}}


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_tiff_is_picklable_and_keys_depend_on_chunks(tmp_path: Path, compression: str | None) -> None:
    # compressed tiles are read through tifffile's zarr store, uncompressed data is memory-mapped
    path = tmp_path / "x.tif"
    data = np.arange(2 * 64 * 96, dtype=np.uint16).reshape(2, 64, 96)
    tifffile.imwrite(path, data, compression=compression, tile=(16, 16) if compression else None)

    small, _, _ = read_tiff(path, (16, 16))
    large, _, _ = read_tiff(path, (32, 32))

    np.testing.assert_array_equal(pickle.loads(pickle.dumps(small)).compute(), data)
    # arrays of one file with other chunks must not share dask keys, or computing both mixes their chunks
    assert small.name != large.name
    np.testing.assert_array_equal(da.stack([small.rechunk(large.chunks), large]).compute(), [data, data])


@pytest.mark.parametrize("imagej", [False, True])
def test_read_tiff_big_endian(tmp_path: Path, imagej: bool) -> None:
    # ImageJ/Fiji write big-endian tiffs by default; the memory-mapped path must use the file's byte order
    data = np.array([[[1, 2], [300, 0]]], np.uint16)
    tifffile.imwrite(tmp_path / "x.tif", data, byteorder=">", imagej=imagej)

    np.testing.assert_array_equal(read_tiff(tmp_path / "x.tif", (16, 16))[0].compute(), data)


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to count open files")
@pytest.mark.parametrize("compression", [None, "zlib"])
def test_read_tiff_does_not_keep_files_open(tmp_path: Path, compression: str | None) -> None:
    # computed chunks and arrays kept alive must not hold a file descriptor each (EMFILE on large reads)
    data = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    images = []
    for i in range(20):
        tifffile.imwrite(tmp_path / f"{i}.tif", data, compression=compression, tile=(16, 16) if compression else None)
        images.append(read_tiff(tmp_path / f"{i}.tif", (16, 16))[0])
    n_open = len(os.listdir("/dev/fd"))

    chunks = dask.compute(*[chunk for image in images for chunk in image.to_delayed().ravel()])

    assert len(os.listdir("/dev/fd")) - n_open <= 2
    np.testing.assert_array_equal(chunks[-1][0], data[48:, 80:])
