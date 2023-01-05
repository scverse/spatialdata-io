from importlib.metadata import version

from spatialdata_io.readers.metaspace import read_metaspace

# from spatialdata_io import pl, pp, tl
from spatialdata_io.readers.visium import read_visium
from spatialdata_io.readers.xenium import xenium

__all__ = [
    "read_metaspace",
    "read_visium",
    "convert_xenium_to_ngff",
]

__version__ = version("spatialdata-io")
