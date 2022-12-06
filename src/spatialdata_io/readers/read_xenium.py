# format specification https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html#polygon_vertices
import os

from geopandas import GeoDataFrame
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata._io.write import write_polygons
import numpy as np
import scanpy as sc
from spatialdata import Image2DModel, Scale, ShapesModel, SpatialData, TableModel, PolygonsModel, PointsModel
from typing import Optional, Dict, Any
import re
import json
import tifffile
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from itertools import chain
import time
import zarr
from ome_zarr.io import parse_url
import pyarrow.parquet as pq
from loguru import logger

DEBUG = True

def _identify_files(in_path: str, library_id: Optional[str] = None) -> Dict[str, Any]:
    files = os.listdir(in_path)
    xenium_files = [f for f in files if f.endswith(".xenium")]
    assert len(xenium_files) == 1
    xenium_file = xenium_files[0]
    with open(os.path.join(in_path, xenium_file)) as f:
        data = json.load(f)
    return data


def _build_polygons(indices, df):
    first_index = indices[0]
    master_process = first_index == 1
    polygons = []
    for i in tqdm(indices, disable=not master_process):
        vertices = df[df.cell_id == i][["vertex_x", "vertex_y"]].to_numpy()
        assert np.array_equal(vertices[0], vertices[-1])
        polygon = Polygon(vertices[:-1])
        polygons.append(polygon)
    # first_index is use to sort the polygons in the same order as in the file
    return polygons, first_index


def _get_zarr_group(out_path: str, element_type: str) -> zarr.Group:
    os.makedirs(out_path, exist_ok=True)
    store = parse_url(out_path, mode="r+").store
    root = zarr.group(store)
    elem_group = root.require_group(element_type)
    return elem_group


def _convert_polygons(in_path: str, data: Dict[str, Any], out_path: str, name: str) -> None:
    df = pd.read_csv(f"{in_path}/{data['run_name']}_{name}.csv.gz")
    if DEBUG:
        n_cells = 1000
        logger.info(f"DEBUG: only using {n_cells} cells")
    else:
        n_cells = df.cell_id.max() + 1
    pool = Pool()
    splits = np.array_split(range(1, n_cells), pool._processes)
    nested = pool.map(partial(_build_polygons, df=df), splits)
    start = time.time()
    nested_sorted = map(lambda x: x[0], sorted(nested, key=lambda x: x[1]))
    polygons = list(chain.from_iterable(nested_sorted))
    print(f"list flattening: {time.time() - start}")
    start = time.time()
    polygons = GeoDataFrame(geometry=polygons)
    print(f"GeoDataFrame instantiation: {time.time() - start}")
    len(polygons)
    parsed = PolygonsModel.parse(polygons)
    # TODO: put the login of the next two lines inside spatialdata._io.write and import from there
    group = _get_zarr_group(out_path, "polygons")
    write_polygons(polygons=parsed, group=group, name=name)

def _convert_points(in_path: str, data: Dict[str, Any], out_path: str) -> None:
    # using parquet is 10 times faster than reading from csv
    start = time.time()
    table = pq.read_table(f"{in_path}/{data['run_name']}_transcripts.parquet")
    xyz = table.select(('x_location', 'y_location', 'z_location')).to_pandas().to_numpy()
    # TODO: the construction of the sparse matrix is slow, optimize by converting to a categorical, the code in the
    #  parser needs to be adapted
    feature_name = table.select(('feature_name',)).to_pandas()['feature_name'].to_list()
    if DEBUG:
        n = 100000
        xyz = xyz[:n]
        feature_name = feature_name[:n]
        logger.info(f"DEBUG: only using {n} transcripts")
    print(f'parquet: {time.time() - start}')
    ##
    start = time.time()
    parsed = PointsModel.parse(coords=xyz, points_assignment=feature_name)
    print(f'parsing: {time.time() - start}')
    parsed
    # TODO: obs.index is a object, this is slow
    # TODO: (see above): feature_name needs to be a categorical
    ##
    print('')

def _convert_table(in_path: str, data: Dict[str, Any], out_path: str) -> None:
    df = pd.read_csv(f"{in_path}/{data['run_name']}_cells.csv.gz")
    # TODO: implement
    pass

def _convert_image(in_path: str, data: Dict[str, Any], out_path: str, name: str) -> None:
    # TODO: convert with bioformats2raw
    pass

def convert_xenium_to_ngff(path: str, out_path: str):
    data = _identify_files(path)
    # _convert_polygons(in_path=path, data=data, out_path=out_path, name="nucleus_boundaries")
    # _convert_polygons(in_path=path, data=data, out_path=out_path, name="cell_boundaries")
    _convert_points(in_path=path, data=data, out_path=out_path)
    _convert_table(in_path=path, data=data, out_path=out_path)
    _convert_image(in_path=path, data=data, out_path=out_path, name="morphology")
    _convert_image(in_path=path, data=data, out_path=out_path, name="morphology_mip")
    _convert_image(in_path=path, data=data, out_path=out_path, name="morphology_focus")
    pass

if __name__ == "__main__":
    convert_xenium_to_ngff(path="./data/", out_path="./data.zarr/")
