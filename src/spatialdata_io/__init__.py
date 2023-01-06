from importlib.metadata import version

from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.xenium import xenium

__all__ = [
    "visium",
    "xenium",
    "cosmx",
]

__version__ = version("spatialdata-io")
