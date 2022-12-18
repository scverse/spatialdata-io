import os
import re
from pathlib import Path
from typing import Union  # noqa: F401

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy.sparse import csr_matrix

from spatialdata_io._utils import _load_image

__all__ = ["cosmx"]


def cosmx(
    path: str | Path,
    *,
    counts_file: str,
    meta_file: str,
    fov_file: str | None = None,
) -> AnnData:
    """
    Read *Nanostring* formatted dataset.

    In addition to reading the regular *Nanostring* output, it loads the metadata file, *CellComposite* and *CellLabels*
    directories containing the images and optionally the field of view file.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    counts_file
        File containing the counts. Typically ends with *_exprMat_file.csv*.
    meta_file
        File containing the spatial coordinates and additional cell-level metadata.
        Typically ends with *_metadata_file.csv*.
    fov_file
        File containing the coordinates of all the fields of view.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` -  local coordinates of the centers of cells.
        - :attr:`anndata.AnnData.obsm` ``['spatial_fov']`` - global coordinates of the centers of cells in the
          field of view.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['images']`` - *hires* and *segmentation* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['metadata']]['{x,y}_global_px']`` - coordinates of the field of view.
          Only present if ``fov_file != None``.
    """
    path, fov_key = Path(path), "fov"
    cell_id_key = "cell_ID"
    counts = pd.read_csv(path / counts_file, header=0, index_col=cell_id_key)
    counts.index = counts.index.astype(str).str.cat(counts.pop(fov_key).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=cell_id_key)
    obs[fov_key] = pd.Categorical(obs[fov_key].astype(str))
    obs[cell_id_key] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[fov_key].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
        uns={"spatial": {}},
    )
    adata.var_names = counts.columns

    adata.obsm["spatial"] = adata.obs[["CenterX_local_px", "CenterY_local_px"]].values
    adata.obsm["spatial_fov"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    adata.obs.drop(columns=["CenterX_local_px", "CenterY_local_px"], inplace=True)

    for fov in adata.obs[fov_key].cat.categories:
        adata.uns["spatial"][fov] = {
            "images": {},
            "scalefactors": {"tissue_hires_scalef": 1, "spot_diameter_fullres": 1},
        }

    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")

    pat = re.compile(r".*_F(\d+)")
    for subdir in ["CellComposite", "CellLabels"]:
        kind = "hires" if subdir == "CellComposite" else "segmentation"
        for fname in os.listdir(path / subdir):
            if fname.endswith(file_extensions):
                fov = str(int(pat.findall(fname)[0]))
                adata.uns["spatial"][fov]["images"][kind] = _load_image(path / subdir / fname)

    if fov_file is not None:
        fov_positions = pd.read_csv(path / fov_file, header=0, index_col=fov_key)
        for fov, row in fov_positions.iterrows():
            try:
                adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
            except KeyError:
                logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
                continue

    return adata
