from anndata import AnnData
from spatialdata._types import ArrayLike
from typing import Iterable, Any
import json
import numpy as np
import pandas as pd

__all__ = ["polygons_anndata_from_geojson"]


def polygons_anndata_from_geojson(path: str) -> AnnData:
    from spatialdata._core.elements import Polygons

    with open(path) as f:
        j = json.load(f)

    names = []
    coordinates = []
    assert "geometries" in j
    for region in j["geometries"]:
        if region["type"] == "Polygon":
            names.append(region["name"])
            vertices: ArrayLike = np.array(region["coordinates"])
            vertices = np.squeeze(vertices, 0)
            assert len(vertices.shape) == 2
            coordinates.append(vertices)
        else:
            print(f'ignoring "{region["type"]}" from geojson')

    string_coordinates = [Polygons.tensor_to_string(c) for c in coordinates]
    a = AnnData(shape=(len(names), 0), obs=pd.DataFrame({"name": names, "spatial": string_coordinates}))
    return a
