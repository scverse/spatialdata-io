"""Benchmarks for SpatialData IO operations for large images.

Instructions:
    See benchmark_xenium.py for instructions.
"""

import shutil
from pathlib import Path

from spatialdata import SpatialData
from xarray import DataArray

from spatialdata_io import image  # type: ignore[attr-defined]

# =============================================================================
# CONFIGURATION - Edit these paths to match your setup
# =============================================================================
SANDBOX_DIR = Path(__file__).parent.parent.parent / "spatialdata-sandbox"
DATASET = "xenium_2.0.0_io"
# =============================================================================


def get_paths() -> tuple[Path, Path]:
    """Get paths for benchmark data."""
    path = SANDBOX_DIR / DATASET
    # TODO: this is not a good image because it's compressed (not memmappable)
    path_read = path / "data" / "morphology.ome.tif"
    path_write = path / "data_benchmark.zarr"

    if not path_read.exists():
        raise ValueError(f"Data directory not found: {path_read}")

    return path_read, path_write


class IOBenchmarkImage:
    """Benchmark IO read operations with different parameter combinations."""

    timeout = 3600
    repeat = 3
    number = 1
    warmup_time = 0
    processes = 1

    # Parameter combinations: scale_factors, use_tiff_memmap, chunks
    params = [
        [None, [2, 2, 2]],  # scale_factors
        [True, False],  # use_tiff_memmap
        [(5000, 5000), (1000, 1000)],  # chunks
    ]
    param_names = ["scale_factors", "use_tiff_memmap", "chunks"]

    def setup(self, *_) -> None:
        """Set up paths for benchmarking."""
        self.path_read, self.path_write = get_paths()
        if self.path_write.exists():
            shutil.rmtree(self.path_write)

    def _convert_image(self, scale_factors, use_tiff_memmap, chunks) -> SpatialData:
        """Read image data with specified parameters."""
        im = image(
            input=self.path_read,
            data_axes=("c", "y", "x"),
            coordinate_system="global",
            use_tiff_memmap=use_tiff_memmap,
            chunks=chunks,
            scale_factors=scale_factors,
        )
        sdata = SpatialData.init_from_elements({"image": im})
        # sanity check
        if scale_factors is None:
            assert isinstance(sdata["image"], DataArray)
        else:
            assert len(sdata["image"].keys()) == len(scale_factors)

        if chunks is not None:
            # TODO: bug here!
            assert sdata["image"].chunksizes["x"] == chunks[0]
            assert sdata["image"].chunksizes["y"] == chunks[1]
        return sdata

    def time_io(self, scale_factors, use_tiff_memmap, chunks) -> None:
        """Walltime for data parsing."""
        sdata = self._convert_image(scale_factors, use_tiff_memmap, chunks)
        sdata.write(self.path_write)

    def peakmem_io(self, scale_factors, use_tiff_memmap, chunks) -> None:
        """Peak memory for data parsing."""
        sdata = self._convert_image(scale_factors, use_tiff_memmap, chunks)
        sdata.write(self.path_write)


if __name__ == "__main__":
    # Run a single test case for quick verification
    bench = IOBenchmarkImage()
    bench.setup(None, True, (1000, 1000))
    bench.time_io(None, True, (1000, 1000))
