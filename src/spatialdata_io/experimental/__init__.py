from spatialdata_io.converters.legacy_anndata import (
    from_legacy_anndata,
    to_legacy_anndata,
)
from spatialdata_io.readers.iss import iss

_readers_technologies = [
    "iss",
]
_readers_file_types: list[str] = [
    # add experimental readers for new file types here
]
_converters = [
    "from_legacy_anndata",
    "to_legacy_anndata",
]

__all__ = _readers_technologies + _readers_file_types + _converters
