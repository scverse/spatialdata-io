from importlib.metadata import version

from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.dbit import dbit
from spatialdata_io.readers.macsima import macsima
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

_readers = [
    "codex",
    "cosmx",
    "curio",
    "dbit",
    "macsima",
    "mcmicro",
    "merscope",
    "seqfish",
    "steinbock",
    "stereoseq",
    "visium",
    "visium_hd",
    "xenium",
]

__all__ = [
    "xenium_aligned_image",
    "xenium_explorer_selection",
] + _readers

__version__ = version("spatialdata-io")
