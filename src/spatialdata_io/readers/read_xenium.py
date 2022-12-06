# format specification https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html#polygon_vertices
import os

from geopandas import GeoDataFrame
from shapely import Polygon
from spatialdata import SpatialData
import numpy as np
import scanpy as sc
from spatialdata import Image2DModel, Scale, ShapesModel, SpatialData, TableModel, PolygonsModel
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


def _identify_files(path: str, library_id: Optional[str] = None) -> Dict[str, Any]:
    files = os.listdir(path)
    xenium_files = [f for f in files if f.endswith(".xenium")]
    assert len(xenium_files) == 1
    xenium_file = xenium_files[0]
    with open(os.path.join(path, xenium_file)) as f:
        data = json.load(f)
    return data


def _build_polygons(indices, nuclei):
    master_process = indices[0] == 1
    polygons = []
    for i in tqdm(indices, disable= not master_process):
        vertices = nuclei[nuclei.cell_id == i][["vertex_x", "vertex_y"]].to_numpy()
        assert np.array_equal(vertices[0], vertices[-1])
        polygon = Polygon(vertices[:-1])
        polygons.append(polygon)
    return polygons


def _convert_nuclei_polygons(path: str, data: Dict[str, Any], out_path: str) -> None:
    nuclei = pd.read_csv(f"{path}/{data['run_name']}_nucleus_boundaries.csv.gz")
    n_nuclei = 10000
    # n_nuclei = nuclei.cell_id.max() + 1
    pool = Pool()
    splits = np.array_split(range(1, n_nuclei), pool._processes)
    nested = pool.map(partial(_build_polygons, nuclei=nuclei), splits)
    start = time.time()
    polygons = list(chain.from_iterable(nested))
    print(f'list flattening: {time.time() - start}')
    start = time.time()
    polygons = GeoDataFrame(geometry=polygons)
    print(f'GeoDataFrame instantiation: {time.time() - start}')
    len(polygons)
    parsed = PolygonsModel.parse(polygons)
    # TODO: save to zarr


def convert_xenium_to_ngff(path: str, out_path: str):
    data = _identify_files(path)
    _convert_nuclei_polygons(path, data, out_path)
    pass

    # too slow, better to convert with bioformats2raw and then read the .zarr store
    # # images
    # image_names = ['morphology', 'morphology_mip', 'morphology_focus']
    # for image_name in image_names:
    #     f = os.path.join(path, f'{run_name}_{image_name}.ome.tif')
    #     assert os.path.exists(f)
    #     img = tifffile.imread(f)
    print("")
    # cell masks
    ##

    ##
    # single-molecule transcripts

    # transcripts

    # adata = sc.read_visium(path)
    # libraries = list(adata.uns["spatial"].keys())
    # assert len(libraries) == 1
    # lib = libraries[0]
    # if library_id is None:
    #     library_id = lib
    #
    # # expression table
    # expression = adata.copy()
    # del expression.uns
    # del expression.obsm
    # expression.obs_names_make_unique()
    # expression.var_names_make_unique()
    # expression = TableModel.parse(
    #     expression,
    #     region=f"/shapes/{library_id}",
    #     instance_key="visium_spot_id",
    #     instance_values=np.arange(len(adata)),
    # )
    #
    # # circles ("visium spots")
    # radius = adata.uns["spatial"][lib]["scalefactors"]["spot_diameter_fullres"] / 2
    # shapes = ShapesModel.parse(
    #     coords=adata.obsm["spatial"],
    #     shape_type="Circle",
    #     shape_size=radius,
    #     instance_key="visium_spot_id",
    #     instance_values=np.arange(len(adata)),
    # )
    # # transformation
    # scale_factors = np.array([1.0] + [1 / adata.uns["spatial"][lib]["scalefactors"]["tissue_hires_scalef"]] * 2)
    # transform = Scale(scale=scale_factors)
    #
    # # image
    # img = adata.uns["spatial"][lib]["images"]["hires"]
    # assert img.dtype == np.float32 and np.min(img) >= 0.0 and np.max(img) <= 1.0
    # img = (img * 255).astype(np.uint8)
    # img = Image2DModel.parse(img, transform=transform, dims=("y", "x", "c"))
    #
    # sdata = SpatialData(
    #     images={library_id: img},
    #     shapes={library_id: shapes},
    #     table=expression,
    # )
    # return sdata


if __name__ == "__main__":
    convert_xenium_to_ngff(path="./data/", out_path="./data.zarr/")
