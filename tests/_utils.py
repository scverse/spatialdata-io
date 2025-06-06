import sys

import pytest


def skip_if_below_python_version() -> pytest.mark.skipif:
    """Decorator to skip tests if the Python version is below a specified version.

    This decorator prevents running tests on unsupported Python versions. Update the `MIN_VERSION`
    constant to change the minimum Python version required for the tests.

    Returns
    -------
    pytest.mark.skipif
        A pytest marker that skips the test if the current Python version is below the specified `MIN_VERSION`.

    Notes
    -----
    The current minimum version is set to Python 3.10. Adjust the `MIN_VERSION` constant as needed
    to accommodate newer Python versions.

    Examples
    --------
    >>> @skip_if_below_python_version()
    >>> def test_some_feature():
    >>>     assert True
    """
    MIN_VERSION = (3, 12)
    reason = f"Test requires Python {'.'.join(map(str, MIN_VERSION))} or higher"
    return pytest.mark.skipif(sys.version_info < MIN_VERSION, reason=reason)
