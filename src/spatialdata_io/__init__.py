from importlib.metadata import version

from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.mcmicro import mcmicro
from spatialdata_io.readers.merscope import merscope
from spatialdata_io.readers.steinbock import steinbock
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.xenium import xenium

__all__ = [
    "curio",
    "visium",
    "xenium",
    "codex",
    "cosmx",
    "mcmicro",
    "steinbock",
    "merscope",
]

__version__ = version("spatialdata-io")
