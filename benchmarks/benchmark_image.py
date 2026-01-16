"""Benchmarks for SpatialData IO operations for large images.

Instructions:
    See benchmark_xenium.py for instructions.
"""

import logging
import logging.handlers
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from spatialdata import SpatialData
from spatialdata._logging import logger
from xarray import DataArray

from spatialdata_io import image  # type: ignore[attr-defined]

# =============================================================================
# CONFIGURATION - Edit these values to match your setup
# =============================================================================
# Image dimensions: (channels, height, width)
IMAGE_SHAPE = (3, 30000, 30000)
# =============================================================================


class IOBenchmarkImage:
    """Benchmark IO read operations with different parameter combinations."""

    timeout = 3600
    repeat = 3
    number = 1
    warmup_time = 0
    processes = 1

    # Parameter combinations: scale_factors, use_tiff_memmap, chunks
    params = [
        [None, [2, 2]],  # scale_factors
        [True, False],  # use_tiff_memmap
        [(1, 5000, 5000), (1, 1000, 1000)],  # chunks
    ]
    param_names = ["scale_factors", "use_tiff_memmap", "chunks"]

    # Class-level temp directory for image files (persists across all benchmarks)
    _images_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    _path_read_uncompressed: Path | None = None
    _path_read_compressed: Path | None = None

    @classmethod
    def _setup_images(cls) -> None:
        """Create fake image data once for all benchmarks."""
        if cls._images_temp_dir is not None:
            return

        cls._images_temp_dir = tempfile.TemporaryDirectory()
        images_dir = Path(cls._images_temp_dir.name)
        cls._path_read_uncompressed = images_dir / "image_uncompressed.tif"
        cls._path_read_compressed = images_dir / "image_compressed.tif"

        # Generate fake image data
        rng = np.random.default_rng(42)
        data = rng.integers(0, 255, size=IMAGE_SHAPE, dtype=np.uint8)

        # Write uncompressed TIFF (memmappable)
        tifffile.imwrite(cls._path_read_uncompressed, data, compression=None)
        # Write compressed TIFF (not memmappable)
        tifffile.imwrite(cls._path_read_compressed, data, compression="zlib")

    def setup(self, *_: Any) -> None:
        """Set up paths for benchmarking."""
        # Create images once (shared across all benchmark runs)
        self._setup_images()
        self.path_read_uncompressed = self._path_read_uncompressed
        self.path_read_compressed = self._path_read_compressed

        # Create a separate temp directory for output (cleaned up after each run)
        self._output_temp_dir = tempfile.TemporaryDirectory()
        self.path_write = Path(self._output_temp_dir.name) / "data_benchmark.zarr"

    def teardown(self, *_: Any) -> None:
        """Clean up output directory after each benchmark run."""
        if hasattr(self, "_output_temp_dir"):
            self._output_temp_dir.cleanup()

    def _convert_image(
        self, scale_factors: list[int] | None, use_tiff_memmap: bool, chunks: tuple[int, ...]
    ) -> SpatialData:
        """Read image data with specified parameters."""
        # Use uncompressed (memmappable) for use_tiff_memmap=True, compressed otherwise
        path_read = self.path_read_uncompressed if use_tiff_memmap else self.path_read_compressed

        # Capture log messages to verify memmappable warning behavior
        log_capture = logging.handlers.MemoryHandler(capacity=100)
        log_capture.setLevel(logging.WARNING)
        logger.addHandler(log_capture)
        original_propagate = logger.propagate
        logger.propagate = True

        try:
            im = image(
                input=path_read,
                data_axes=("c", "y", "x"),
                coordinate_system="global",
                use_tiff_memmap=use_tiff_memmap,
                chunks=chunks,
                scale_factors=scale_factors,
            )
        finally:
            logger.removeHandler(log_capture)
            logger.propagate = original_propagate

        # Check warning behavior: when use_tiff_memmap=True, no compression warning should be raised
        log_messages = [record.getMessage() for record in log_capture.buffer]
        has_memmap_warning = any("image data is not memory-mappable" in msg for msg in log_messages)
        if use_tiff_memmap:
            assert not has_memmap_warning, "Uncompressed TIFF should not trigger memory-mappable warning"

        sdata = SpatialData.init_from_elements({"image": im})
        # sanity check: chunks is (c, y, x)
        if scale_factors is None:
            assert isinstance(sdata["image"], DataArray)
            if chunks is not None:
                assert (
                    sdata["image"].chunksizes["x"][0] == chunks[2]
                    or sdata["image"].chunksizes["x"][0] == sdata["image"].shape[2]
                )
                assert (
                    sdata["image"].chunksizes["y"][0] == chunks[1]
                    or sdata["image"].chunksizes["y"][0] == sdata["image"].shape[1]
                )
        else:
            assert len(sdata["image"].keys()) == len(scale_factors) + 1
            if chunks is not None:
                assert (
                    sdata["image"]["scale0"]["image"].chunksizes["x"][0] == chunks[2]
                    or sdata["image"]["scale0"]["image"].chunksizes["x"][0]
                    == sdata["image"]["scale0"]["image"].shape[2]
                )
                assert (
                    sdata["image"]["scale0"]["image"].chunksizes["y"][0] == chunks[1]
                    or sdata["image"]["scale0"]["image"].chunksizes["y"][0]
                    == sdata["image"]["scale0"]["image"].shape[1]
                )

        return sdata

    def time_io(self, scale_factors: list[int] | None, use_tiff_memmap: bool, chunks: tuple[int, ...]) -> None:
        """Walltime for data parsing."""
        sdata = self._convert_image(scale_factors, use_tiff_memmap, chunks)
        sdata.write(self.path_write)

    def peakmem_io(self, scale_factors: list[int] | None, use_tiff_memmap: bool, chunks: tuple[int, ...]) -> None:
        """Peak memory for data parsing."""
        sdata = self._convert_image(scale_factors, use_tiff_memmap, chunks)
        sdata.write(self.path_write)


# if __name__ == "__main__":
#     # Run a single test case for quick verification
#     bench = IOBenchmarkImage()
#
#     bench.setup()
#     bench.time_io(None, True, (1, 5000, 5000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io(None, True, (1, 1000, 1000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io(None, False, (1, 5000, 5000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io(None, False, (1, 1000, 1000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io([2, 2, 2], True, (1, 5000, 5000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io([2, 2, 2], True, (1, 1000, 1000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io([2, 2, 2], False, (1, 5000, 5000))
#     bench.teardown()
#
#     bench.setup()
#     bench.time_io([2, 2, 2], False, (1, 1000, 1000))
#     bench.teardown()
#
#     # Clean up the shared images temp directory at the end
#     if IOBenchmarkImage._images_temp_dir is not None:
#         IOBenchmarkImage._images_temp_dir.cleanup()
#         IOBenchmarkImage._images_temp_dir = None
