from importlib.metadata import version

# from spatialdata_io import pl, pp, tl
from spatialdata_io.readers import read_metaspace, read_visium, read_xenium
from spatialdata_io.constructors.images_labels import image_xarray_from_numpy, labels_xarray_from_numpy
from spatialdata_io.constructors.points import points_anndata_from_coordinates
from spatialdata_io.constructors.circles import circles_anndata_from_coordinates
from spatialdata_io.constructors.polygons import polygons_anndata_from_geojson
from spatialdata_io.constructors.table import table_update_anndata

__all__ = [
    "image_xarray_from_numpy",
    "labels_xarray_from_numpy",
    "points_anndata_from_coordinates",
    "circles_anndata_from_coordinates",
    "polygons_anndata_from_geojson",
    "table_update_anndata",
    "read_metaspace",
    "read_visium",
    "read_xenium",
]

__version__ = version("spatialdata-io")
