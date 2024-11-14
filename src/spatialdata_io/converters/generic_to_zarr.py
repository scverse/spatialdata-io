from pathlib import Path

import spatialdata as sd
from dask_image.imread import imread
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Identity


def read_generic(input, filetype, name, output, transformation, radius):
    if type == "shape":
        data = Path(input)
        if not name:
            name = data.stem
        sdata_path = Path(output)
        sdata = sd.read_zarr(sdata_path)
        if filetype == "shape":
            if data.suffix == ".geojson":
                if sdata_path.exists():
                    sdata.shapes[name] = ShapesModel.parse(
                        data, transformations={transformation: Identity()}, radius=radius
                    )
                else:
                    shapes = {}
                    shapes[name] = ShapesModel.parse(data)
                    sdata = sd.SpatialData(shapes=shapes)
            else:
                raise ValueError("Invalid file type for shape element. Must be .geojson")
        if filetype == "image":
            if data.suffix == ".tif" or data.suffix == ".tiff" or data.suffix == ".png":
                if sdata_path.exists():
                    sdata.images[name] = Image2DModel.parse(data)
                else:
                    image = imread(data)
                    images = {}
                    images[name] = Image2DModel.parse(image, dims=("c", "y", "x"))
                    sdata = sd.SpatialData(images=images)
            else:
                raise ValueError("Invalid file type for image element. Must be .tif, .tiff, or .png")
        # TODO: how to deal with geometries, transformations, indices?
        # TODO: really necessary to move code from spatialdata/models.py to here?
        # TODO: how to overwrite existing zarr store?

        return sdata
