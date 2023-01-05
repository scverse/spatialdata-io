from importlib.metadata import version

from spatialdata_io.readers.cosmx import cosmx

# from spatialdata_io import pl, pp, tl
from spatialdata_io.readers.visium import read_visium
from spatialdata_io.readers.xenium import xenium

__all__ = [
    "read_visium",
    "xenium",
    "cosmx",
]

__version__ = version("spatialdata-io")
