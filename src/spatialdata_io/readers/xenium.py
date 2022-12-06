# format specification https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html#polygon_vertices
import os
import shutil
import subprocess
import psutil

from geopandas import GeoDataFrame
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata._io.write import write_polygons, write_points, write_table, write_shapes
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
from anndata import AnnData

DEBUG = True
# DEBUG = False

__all__ = ["convert_xenium_to_ngff"]


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


def _get_zarr_group(out_path: str, element_type: str, name: str) -> zarr.Group:
    if not os.path.isdir(out_path):
        store = parse_url(out_path, mode='w').store
    else:
        store = parse_url(out_path, mode="r+").store
    root = zarr.group(store)
    elem_group = root.require_group(element_type)
    inner_path = os.path.normpath(os.path.join(elem_group._store.path, elem_group.path, name))
    if os.path.isdir(inner_path):
        logger.info(f"Removing existing directory {inner_path}")
        shutil.rmtree(inner_path)
    return elem_group


def _convert_polygons(in_path: str, data: Dict[str, Any], out_path: str, name: str, num_workers: int) -> None:
    df = pd.read_csv(f"{in_path}/{data['run_name']}_{name}.csv.gz")
    if DEBUG:
        n_cells = 1000
        logger.info(f"DEBUG: considering only {n_cells} cells")
    else:
        n_cells = df.cell_id.max() + 1
    pool = Pool(num_workers)
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
    group = _get_zarr_group(out_path, "polygons", name)
    write_polygons(polygons=parsed, group=group, name=name)


def _convert_points(in_path: str, data: Dict[str, Any], out_path: str) -> None:
    # using parquet is 10 times faster than reading from csv
    start = time.time()
    name = "transcripts"
    table = pq.read_table(f"{in_path}/{data['run_name']}_{name}.parquet")
    xyz = table.select(("x_location", "y_location", "z_location")).to_pandas().to_numpy()
    # TODO: the construction of the sparse matrix is slow, optimize by converting to a categorical, the code in the
    #  parser needs to be adapted
    d = table.column("feature_name").cast("string").dictionary_encode()
    feature_name = d.to_pandas()
    if DEBUG:
        n = 100000
        xyz = xyz[:n]
        feature_name = feature_name[:n]
        logger.info(f"DEBUG: only using {n} transcripts")
    print(f"parquet: {time.time() - start}")
    ##
    start = time.time()
    # TODO: this is slow because of perfomance issues in anndata, maybe we will use geodataframes instead
    parsed = PointsModel.parse(coords=xyz, points_assignment=feature_name)
    print(f"parsing: {time.time() - start}")
    group = _get_zarr_group(out_path, "points", name)
    write_points(points=parsed, group=group, name=name)


def _convert_table_and_shapes(in_path: str, data: Dict[str, Any], out_path: str) -> None:
    name = "cells"
    df = pd.read_csv(f"{in_path}/{data['run_name']}_{name}.csv.gz")
    feature_matrix = sc.read_10x_h5(f"{in_path}/{data['run_name']}_cell_feature_matrix.h5")

    nuclei_radii = np.sqrt(df["nucleus_area"].to_numpy() / np.pi)
    cells_radii = np.sqrt(df["cell_area"].to_numpy() / np.pi)
    shapes_nuclei = ShapesModel.parse(coords=df[['x_centroid', 'y_centroid']].to_numpy(), shape_type='Circle',
                                      shape_size=nuclei_radii)
    shapes_cells = ShapesModel.parse(coords=df[['x_centroid', 'y_centroid']].to_numpy(), shape_type='Circle',
                                      shape_size=cells_radii)
    group = _get_zarr_group(out_path, "shapes", "nuclei")
    write_shapes(shapes=shapes_nuclei, group=group, name="nuclei")
    group = _get_zarr_group(out_path, "shapes", "cells")
    write_shapes(shapes=shapes_cells, group=group, name="cells")


    df.drop(columns=["x_centroid", "y_centroid"], inplace=True)
    adata = AnnData(X=feature_matrix.X, var=feature_matrix.var, obs=df)
    parsed = TableModel.parse(adata, region='/polygons/cell_boundaries', instance_key='cell_id')
    group = _get_zarr_group(out_path, "table", name)
    write_table(table=parsed, group=group, name=name)


def _ome_ngff_dims_workaround(zarr_path: str):
    # manually, very hackily, make the data cyx from tczyx. Replace with a better option by either supporting tczyx
    # images or with solution arising from the discussion in
    # https://github.com/glencoesoftware/bioformats2raw/issues/180

    # adjust .zattrs
    os.listdir(zarr_path)
    with open(os.path.join(zarr_path, ".zattrs")) as f:
        attrs = json.load(f)
    # removing t and z axes
    assert len(attrs['multiscales']) == 1
    axes = attrs['multiscales'][0]['axes']
    for ax in axes:
        if ax['name'] in ('t', 'z'):
            axes.remove(ax)
    # adjust the multiscale coordinateTransformations
    for multiscale in attrs['multiscales'][0]['datasets']:
        for transform in multiscale['coordinateTransformations']:
            assert transform['type'] == 'scale'
            scale = transform['scale']
            assert len(scale) == 5
            assert scale[0] == 1
            assert scale[2] == 1
            new_scale = np.array(scale)[np.array([1, 3, 4], dtype=int)].tolist()
            transform['scale'] = new_scale
    with open(os.path.join(zarr_path, ".zattrs"), 'w') as f:
        json.dump(attrs, f)

    # adjust .zarray
    for path in os.listdir(zarr_path):
        if os.path.isdir(os.path.join(zarr_path, path)):
            multiscale_path = os.path.join(zarr_path, path)
            with open(os.path.join(multiscale_path, ".zarray")) as f:
                zarray = json.load(f)
            chunks = zarray['chunks']
            shape = zarray['shape']
            assert len(chunks) == 5
            assert len(shape) == 5
            assert chunks[0] == 1
            assert chunks[2] == 1
            assert shape[0] == 1
            assert shape[2] == 1
            new_chunks = np.array(chunks)[np.array([1, 3, 4], dtype=int)].tolist()
            new_shape = np.array(shape)[np.array([1, 3, 4], dtype=int)].tolist()
            zarray['chunks'] = new_chunks
            zarray['shape'] = new_shape
            with open(os.path.join(multiscale_path, ".zarray"), 'w') as f:
                json.dump(zarray, f)

    # remove t dimension from raw storage
    for path in os.listdir(zarr_path):
        if os.path.isdir(os.path.join(zarr_path, path)):
            multiscale_path = os.path.join(zarr_path, path)
            t_dims = [os.path.join(multiscale_path, d) for d in os.listdir(multiscale_path)]
            t_dims = [d for d in t_dims if os.path.isdir(d)]
            assert len(t_dims) == 1
            t_path = t_dims[0]
            temp_path = os.path.join(multiscale_path, 'temp')
            shutil.move(t_path, temp_path)
            for f in os.listdir(temp_path):
                shutil.move(os.path.join(temp_path, f), os.path.join(multiscale_path, f))
            shutil.rmtree(temp_path)

    # remove z dimension from raw storage
    for path in os.listdir(zarr_path):
        if os.path.isdir(os.path.join(zarr_path, path)):
            multiscale_path = os.path.join(zarr_path, path)
            c_dims = [os.path.join(multiscale_path, d) for d in os.listdir(multiscale_path)]
            c_dims = [d for d in c_dims if os.path.isdir(d)]
            for c_path in c_dims:
                z_dims = [os.path.join(c_path, d) for d in os.listdir(c_path)]
                z_dims = [d for d in z_dims if os.path.isdir(d)]
                assert len(z_dims) == 1
                z_path = z_dims[0]
                temp_path = os.path.join(c_path, 'temp')
                shutil.move(z_path, temp_path)
                for f in os.listdir(temp_path):
                    shutil.move(os.path.join(temp_path, f),
                                os.path.join(c_path, f))
                shutil.rmtree(temp_path)

def _convert_image(in_path: str, data: Dict[str, Any], out_path: str, name: str, num_workers: int) -> None:
    image = f"{in_path}/{data['run_name']}_{name}.ome.tif"
    assert os.path.isfile(image)
    _ = _get_zarr_group(out_path, "images", name)
    full_out_path = os.path.normpath(os.path.join(out_path, "images", name))
    try:
        # TODO: here we discard the ome xml metadata, check if keeping it. It can contain for instance the channel names
        # the option --scale-format-string '%2$d/' supresses the creating of a outer directory named "0" (one
        # directory per series, here we one series so we can omit). But it doesn't work https://github.com/glencoesoftware/bioformats2raw/issues/179
        # so we are moving the directory manually
        subprocess.check_output(
            [
                "bioformats2raw",
                image,
                full_out_path,
                "-p",
                "--max_workers",
                str(num_workers),
                "--no-root-group",
                "--no-ome-meta-export",
            ]
        )
        # workaround to remove the "0" directory (series directory)
        shutil.move(os.path.join(full_out_path, "0"), os.path.join(full_out_path, "temp"))
        for f in os.listdir(os.path.join(full_out_path, "temp")):
            shutil.move(os.path.join(full_out_path, "temp", f), os.path.join(full_out_path, f))
        shutil.rmtree(os.path.join(full_out_path, "temp"))
        _ome_ngff_dims_workaround(full_out_path)
    except FileNotFoundError as e:
        ##
        raise FileNotFoundError(
            "bioformats2raw not found, please check https://github.com/glencoesoftware/bioformats2raw for the "
            "installation instructions.\nIf you use conda/mamba, you can install it with `mamba install -c "
            "ome bioformats2raw`, as described in https://github.com/ome/conda-bioformats2raw"
        )
        ##


def convert_xenium_to_ngff(path: str, out_path: str, num_workers: int = -1) -> None:
    if num_workers == -1:
        MAX_WORKERS = psutil.cpu_count()
        logger.info(
            f"Using {MAX_WORKERS} workers to speed up the conversion of polygons and images. A note on the "
            "conversion of images to NGFF: as explained in the bioformats2raw readme, a large number of workers "
            "doesn't always imply better performance in systems with systems with significant I/O bandwidth. See "
            "https://github.com/glencoesoftware/bioformats2raw for more details https://github.com/glencoesoftware/bioformats2raw"
        )
        num_workers = MAX_WORKERS
    data = _identify_files(path)
    # _convert_polygons(in_path=path, data=data, out_path=out_path, name="nucleus_boundaries", num_workers=num_workers)
    # _convert_polygons(in_path=path, data=data, out_path=out_path, name="cell_boundaries", num_workers=num_workers)
    _convert_points(in_path=path, data=data, out_path=out_path)
    # _convert_table_and_shapes(in_path=path, data=data, out_path=out_path)
    # TODO: can't convert morphology since there is a t dimension that is not currently supported (TODO: check the
    #  raw data with FIJI). The other images are fine
    # # _convert_image(in_path=path, data=data, out_path=out_path, name="morphology", num_workers=num_workers)
    # _convert_image(in_path=path, data=data, out_path=out_path, name="morphology_mip", num_workers=num_workers)
    # _convert_image(in_path=path, data=data, out_path=out_path, name="morphology_focus", num_workers=num_workers)
    # TODO: decide if to save the extra metadata present in `data`
    pass


if __name__ == "__main__":
    convert_xenium_to_ngff(path="./data/", out_path="./data.zarr/")
    sdata = SpatialData.read("./data.zarr/")
    print(sdata)
