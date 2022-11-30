from importlib.metadata import version

from spatialdata_io.readers.read_metaspace import read_metaspace

# from spatialdata_io import pl, pp, tl
from spatialdata_io.readers.read_visium import read_visium
from spatialdata_io.readers.read_xenium import read_xenium

__all__ = [
    "read_metaspace",
    "read_visium",
    "read_xenium",
]

__version__ = version("spatialdata-io")
