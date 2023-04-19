from importlib.metadata import version

from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.mcmicro import mcmicro
from spatialdata_io.readers.merfish import merfish
from spatialdata_io.readers.steinbock import steinbock
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.xenium import xenium

__all__ = [
    "visium",
    "xenium",
    "cosmx",
    "mcmicro",
    "steinbock",
    "merfish",
]

__version__ = version("spatialdata-io")
