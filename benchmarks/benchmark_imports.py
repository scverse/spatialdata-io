"""ASV benchmarks for spatialdata-io import times.

Measures how long it takes to import the package and individual readers
in a fresh subprocess, isolating import overhead from runtime work.

Running (with the current environment, no virtualenv rebuild):
    # Quick sanity check (single iteration):
    asv run --python=same --quick --show-stderr -v -b ImportBenchmark

    # Full benchmark on current commit:
    asv run --python=same --show-stderr -v -b ImportBenchmark HEAD^!

    # Compare two branches (tip commits):
    asv continuous --python=same --show-stderr -v -b ImportBenchmark main faster-imports

    # Run on specific commits and then compare:
    asv run --python=same -b ImportBenchmark <commit_main>^!
    asv run --python=same -b ImportBenchmark <commit_pr>^!
    asv compare <commit_main> <commit_pr>

    # Generate an HTML report:
    asv publish && asv preview
"""

import subprocess
import sys


def _import_time(statement: str) -> float:
    """Time an import in a fresh subprocess. Returns seconds."""
    code = f"import time; t0=time.perf_counter(); {statement}; print(time.perf_counter()-t0)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout.strip())


class ImportBenchmark:
    """Import-time benchmarks for spatialdata-io.

    Each ``time_*`` method is a separate ASV benchmark.
    They run in isolated subprocesses so that one import
    does not warm the cache for the next.
    """

    # ASV settings tuned for subprocess-based import timing:
    timeout = 120  # seconds before ASV kills a benchmark; generous since each
    # call spawns a subprocess (~2s each × 10 repeats = ~20s worst case)
    repeat = 10  # number of timing samples ASV collects; high because import
    # times have variance from OS caching / disk I/O / background load;
    # ASV reports the median and IQR from these samples
    number = 1  # calls per sample; must be 1 because each call spawns a fresh
    # subprocess — running >1 would just re-import in a warm process
    warmup_time = 0  # seconds of warm-up iterations before timing; disabled because
    # each call is already a cold subprocess — warming up the parent
    # process is meaningless
    processes = 1  # number of ASV worker processes; 1 avoids parallel subprocesses
    # competing for CPU / disk and inflating timings

    # -- top-level package -------------------------------------------------

    def time_import_spatialdata_io(self) -> float:
        """Wall time: ``import spatialdata_io`` (lazy, no readers loaded)."""
        return _import_time("import spatialdata_io")

    # -- single reader via the public API ----------------------------------

    def time_from_spatialdata_io_import_xenium(self) -> float:
        """Wall time: ``from spatialdata_io import xenium``."""
        return _import_time("from spatialdata_io import xenium")

    def time_from_spatialdata_io_import_visium(self) -> float:
        """Wall time: ``from spatialdata_io import visium``."""
        return _import_time("from spatialdata_io import visium")

    def time_from_spatialdata_io_import_visium_hd(self) -> float:
        """Wall time: ``from spatialdata_io import visium_hd``."""
        return _import_time("from spatialdata_io import visium_hd")

    def time_from_spatialdata_io_import_merscope(self) -> float:
        """Wall time: ``from spatialdata_io import merscope``."""
        return _import_time("from spatialdata_io import merscope")

    def time_from_spatialdata_io_import_cosmx(self) -> float:
        """Wall time: ``from spatialdata_io import cosmx``."""
        return _import_time("from spatialdata_io import cosmx")

    # -- key dependencies (reference) --------------------------------------

    def time_import_spatialdata(self) -> float:
        """Wall time: ``import spatialdata`` (reference)."""
        return _import_time("import spatialdata")

    def time_import_anndata(self) -> float:
        """Wall time: ``import anndata`` (reference)."""
        return _import_time("import anndata")
