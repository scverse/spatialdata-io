import re
from pathlib import Path

import pytest

from spatialdata_io._constants._constants import DbitKeys
from spatialdata_io.readers.dbit import _check_path


def test_check_path_without_a_directory_raises() -> None:
    """Without a directory to search, `_check_path` must not fall back to the current one."""
    with pytest.raises(ValueError, match="Either `path` or a specific path"):
        _check_path(
            path=None,
            pattern=re.compile(f".*{DbitKeys.COUNTS_FILE}"),
            key=DbitKeys.COUNTS_FILE,
        )


def test_check_path_uses_the_specific_path_without_a_directory(tmp_path: Path) -> None:
    """A file given explicitly is used even when no directory is given."""
    counts_file = tmp_path / f"counts{DbitKeys.COUNTS_FILE}"
    counts_file.touch()

    file_path, flag = _check_path(
        path=None,
        pattern=re.compile(f".*{DbitKeys.COUNTS_FILE}"),
        key=DbitKeys.COUNTS_FILE,
        path_specific=counts_file,
    )

    assert file_path == counts_file
    assert flag
