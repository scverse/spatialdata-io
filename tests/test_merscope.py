import sys
from pathlib import Path

import pytest

from spatialdata_io.readers.merscope import (
    _dask_image_load_merscope,
    _get_reader,
    _rioxarray_load_merscope,
)


@pytest.fixture
def broken_rioxarray(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow `rioxarray` with a module that can be found but not imported.

    This is what a `rioxarray` installation with a broken `rasterio` looks like.
    """
    (tmp_path / "rioxarray.py").write_text("raise ModuleNotFoundError(\"No module named 'rasterio'\")\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "rioxarray", raising=False)


def test_get_reader_honours_an_explicit_backend() -> None:
    assert _get_reader("rioxarray") is _rioxarray_load_merscope
    assert _get_reader("dask_image") is _dask_image_load_merscope


@pytest.mark.usefixtures("broken_rioxarray")
def test_get_reader_falls_back_when_rioxarray_cannot_be_imported() -> None:
    """A `rioxarray` that is installed but raises on import must not select the rioxarray backend."""
    assert _get_reader(None) is _dask_image_load_merscope
