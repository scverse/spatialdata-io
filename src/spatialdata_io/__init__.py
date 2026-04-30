from importlib import import_module
from importlib.metadata import version
from typing import Any, TYPE_CHECKING

__version__ = version("spatialdata-io")

_LAZY_IMPORTS: dict[str, str] = {
    # readers
    "codex": "spatialdata_io.readers.codex",
    "cosmx": "spatialdata_io.readers.cosmx",
    "curio": "spatialdata_io.readers.curio",
    "dbit": "spatialdata_io.readers.dbit",
    "macsima": "spatialdata_io.readers.macsima",
    "mcmicro": "spatialdata_io.readers.mcmicro",
    "merscope": "spatialdata_io.readers.merscope",
    "seqfish": "spatialdata_io.readers.seqfish",
    "steinbock": "spatialdata_io.readers.steinbock",
    "stereoseq": "spatialdata_io.readers.stereoseq",
    "visium": "spatialdata_io.readers.visium",
    "visium_hd": "spatialdata_io.readers.visium_hd",
    "xenium": "spatialdata_io.readers.xenium",
    "xenium_aligned_image": "spatialdata_io.readers.xenium",
    "xenium_explorer_selection": "spatialdata_io.readers.xenium",
    # readers file types
    "generic": "spatialdata_io.readers.generic",
    "geojson": "spatialdata_io.readers.generic",
    "image": "spatialdata_io.readers.generic",
    # converters
    "generic_to_zarr": "spatialdata_io.converters.generic_to_zarr",
}

__all__ = [
    # readers
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
    "xenium_aligned_image",
    "xenium_explorer_selection",
    # readers file types
    "generic",
    "geojson",
    "image",
    # converters
    "generic_to_zarr",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        mod = import_module(module_path)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    else:
        try:
            return globals()[name]
        except KeyError as e:
            raise AttributeError(f"Module 'spatialdata_io' has no attribute '{name}'") from e


def __dir__() -> list[str]:
    return __all__ + ["__version__"]


if TYPE_CHECKING:
    # readers
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

    # readers file types
    from spatialdata_io.readers.generic import generic, geojson, image

    # converters
    from spatialdata_io.converters.generic_to_zarr import generic_to_zarr
