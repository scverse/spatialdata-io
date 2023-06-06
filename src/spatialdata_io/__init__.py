from importlib.metadata import version

from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.mcmicro import mcmicro
from spatialdata_io.readers.steinbock import steinbock
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.xenium import xenium
from spatialdata_io.utils.get_table import get_table

__all__ = ["curio", "visium", "xenium", "cosmx", "mcmicro", "steinbock", "get_table"]

__version__ = version("spatialdata-io")
