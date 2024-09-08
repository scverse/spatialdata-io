from importlib.metadata import version

from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.dbit import dbit
from spatialdata_io.readers.mcmicro import mcmicro
from spatialdata_io.readers.merscope import merscope
from spatialdata_io.readers.seqfish import seqfish
from spatialdata_io.readers.steinbock import steinbock
from spatialdata_io.readers.stereoseq import stereoseq
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.visium_hd import visium_hd
from spatialdata_io.readers.xenium import (
    xenium,
    xenium_aligned_image,
    xenium_explorer_selection,
)

__all__ = [
    "curio",
    "seqfish",
    "visium",
    "xenium",
    "codex",
    "cosmx",
    "mcmicro",
    "steinbock",
    "stereoseq",
    "merscope",
    "xenium_aligned_image",
    "xenium_explorer_selection",
    "dbit",
    "visium_hd",
]

__version__ = version("spatialdata-io")
