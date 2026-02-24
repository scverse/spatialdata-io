import spatialdata_io


def test_package_has_version() -> None:
    assert spatialdata_io.__version__


def test_all_matches_lazy_imports() -> None:
    """Ensure __all__ and _LAZY_IMPORTS stay in sync."""
    assert set(spatialdata_io.__all__) == set(spatialdata_io._LAZY_IMPORTS.keys())


def test_all_are_importable() -> None:
    """Every name in __all__ should be accessible on the module."""
    for name in spatialdata_io.__all__:
        assert hasattr(spatialdata_io, name), f"{name!r} listed in __all__ but not resolvable"


def test_all_are_in_dir() -> None:
    """dir(spatialdata_io) should expose everything in __all__."""
    module_dir = dir(spatialdata_io)
    for name in spatialdata_io.__all__:
        assert name in module_dir, f"{name!r} missing from dir()"
