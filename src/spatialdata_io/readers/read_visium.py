from typing import Optional

import numpy as np
import scanpy as sc
from spatialdata import Image2DModel, Scale, ShapesModel, SpatialData, TableModel


def read_visium(path: str, coordinate_system_name: Optional[str] = None) -> SpatialData:
    """
    Read Visium data from a directory containing the output of the spaceranger pipeline.

    Parameters
    ----------
    path : str
        Path to the directory containing the output of the spaceranger pipeline.
    coordinate_system_name : str, optional
        Name of the coordinate system to use. If not provided, it's the name of the library found in the .h5 matrix

    Returns
    -------
    SpatialData
        SpatialData object containing the data from the Visium experiment.
    """
    adata = sc.read_visium(path)
    libraries = list(adata.uns["spatial"].keys())
    assert len(libraries) == 1
    lib = libraries[0]
    if coordinate_system_name is None:
        coordinate_system_name = lib
    csn = coordinate_system_name

    # expression table
    expression = adata.copy()
    del expression.uns
    del expression.obsm
    expression.obs_names_make_unique()
    expression.var_names_make_unique()
    expression = TableModel.parse(
        expression,
        region=f"/shapes/{csn}",
        instance_key="visium_spot_id",
        instance_values=np.arange(len(adata)),
    )

    # circles ("visium spots")
    radius = adata.uns["spatial"][lib]["scalefactors"]["spot_diameter_fullres"] / 2
    shapes = ShapesModel.parse(
        coords=adata.obsm["spatial"],
        shape_type="Circle",
        shape_size=radius,
        instance_key="visium_spot_id",
        instance_values=np.arange(len(adata)),
    )
    # transformation
    scale_factors = np.array([1.0] + [1 / adata.uns["spatial"][lib]["scalefactors"]["tissue_hires_scalef"]] * 2)
    transform = Scale(scale=scale_factors)

    # image
    img = adata.uns["spatial"][lib]["images"]["hires"]
    assert img.dtype == np.float32 and np.min(img) >= 0.0 and np.max(img) <= 1.0
    img = (img * 255).astype(np.uint8)
    img = Image2DModel.parse(img, transform=transform, dims=("y", "x", "c"))

    sdata = SpatialData(
        images={csn: img},
        shapes={csn: shapes},
        table=expression,
    )
    return sdata
