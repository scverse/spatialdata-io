"""Benchmarks for SpatialData IO operations.

Configuration:
    Edit SANDBOX_DIR and DATASET below to point to your data.

Setup:
    cd <SANDBOX_DIR>/<DATASET>
    python download.py  # use the same env where spatialdata is installed

Running:
    cd /path/to/spatialdata-io

    # Quick benchmark (single run, for testing):
    asv run --python=same -b IOBenchmark --quick --show-stderr -v

    # Full benchmark (multiple runs, for accurate results):
    asv run --python=same -b IOBenchmark --show-stderr -v

Comparing branches:
    # Run on specific commits:
    asv run main^! -b IOBenchmark --show-stderr -v
    asv run xenium-labels-dask^! -b IOBenchmark --show-stderr -v

    # Or compare two branches directly:
    asv continuous main xenium-labels-dask -b IOBenchmark --show-stderr -v

    # View comparison:
    asv compare main xenium-labels-dask

Results:
    - Console output shows timing and memory after each run
    - JSON results saved to: .asv/results/
    - Generate HTML report: asv publish && asv preview
"""

import inspect
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io import xenium

# =============================================================================
# CONFIGURATION - Edit these paths to match your setup
# =============================================================================
SANDBOX_DIR = Path(__file__).parent.parent.parent / "spatialdata-sandbox"
DATASET = "xenium_2.0.0_io"
# =============================================================================


def get_paths() -> tuple[Path, Path]:
    """Get paths for benchmark data."""
    path = SANDBOX_DIR / DATASET
    path_read = path / "data"
    path_write = path / "data_benchmark.zarr"

    if not path_read.exists():
        raise ValueError(f"Data directory not found: {path_read}")

    return path_read, path_write


class IOBenchmark:
    """Benchmark IO read operations."""

    timeout = 3600
    repeat = 3
    number = 1
    warmup_time = 0
    processes = 1

    def setup(self) -> None:
        """Set up paths for benchmarking."""
        self.path_read, self.path_write = get_paths()
        if self.path_write.exists():
            shutil.rmtree(self.path_write)

    def _read_xenium(self) -> SpatialData:
        """Read xenium data with version-compatible kwargs."""
        signature = inspect.signature(xenium)
        kwargs = {}
        if "cleanup_labels_zarr_tmpdir" in signature.parameters:
            kwargs["cleanup_labels_zarr_tmpdir"] = False

        return xenium(
            path=str(self.path_read),
            n_jobs=8,
            cell_boundaries=True,
            nucleus_boundaries=True,
            morphology_focus=True,
            cells_as_circles=True,
            **kwargs,
        )

    def time_io(self) -> None:
        """Walltime for data parsing."""
        sdata = self._read_xenium()
        sdata.write(self.path_write)

    def peakmem_io(self) -> None:
        """Peak memory for data parsing."""
        sdata = self._read_xenium()
        sdata.write(self.path_write)


if __name__ == "__main__":
    IOBenchmark().time_io()
