# BSD 3-Clause License

# Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# code below taken from https://github.com/scverse/scanpy/blob/master/scanpy/readwrite.py

from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
from anndata import AnnData
from spatialdata._logging import logger


def _read_10x_h5(
    filename: Union[str, Path],
    genome: Optional[str] = None,
    gex_only: bool = True,
) -> AnnData:
    """
    Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to a 10x hdf5 file.
    genome
        Filter expression to genes within this genome. For legacy 10x h5
        files, this must be provided if the data contains more than one genome.
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name.
    Stores the following information:

        - `~anndata.AnnData.X`: The data matrix is stored
        - `~anndata.AnnData.obs_names`: Cell names
        - `~anndata.AnnData.var_names`: Gene names
        - `['gene_ids']`: Gene IDs
        - `['feature_types']`: Feature types
    """
    start = logger.info(f"reading {filename}")
    filename = Path(filename) if isinstance(filename, str) else filename
    is_present = filename.is_file()
    if not is_present:
        logger.debug(f"... did not find original file {filename}")
    with h5py.File(str(filename), "r") as f:
        v3 = "/matrix" in f

    if v3:
        adata = _read_v3_10x_h5(filename, start=start)
        if genome:
            if genome not in adata.var["genome"].values:
                raise ValueError(
                    f"Could not find data corresponding to genome `{genome}` in `{filename}`. "
                    f'Available genomes are: {list(adata.var["genome"].unique())}.'
                )
            adata = adata[:, adata.var["genome"] == genome]
        if gex_only:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"]
        if adata.is_view:
            adata = adata.copy()
    else:
        raise ValueError("Versions older than V3 are not supported.")
    return adata


def _read_v3_10x_h5(filename: Union[str, Path], *, start: Optional[Any] = None) -> AnnData:
    """Read hdf5 file from Cell Ranger v3 or later versions."""
    with h5py.File(str(filename), "r") as f:
        try:
            dsets: dict[str, Any] = {}
            _collect_datasets(dsets, f["matrix"])

            from scipy.sparse import csr_matrix

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == np.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )
            adata = AnnData(
                matrix,
                obs={"obs_names": dsets["barcodes"].astype(str)},
                var={
                    "var_names": dsets["name"].astype(str),
                    "gene_ids": dsets["id"].astype(str),
                    "feature_types": dsets["feature_type"].astype(str),
                    "genome": dsets["genome"].astype(str),
                },
            )
            return adata
        except KeyError:
            raise Exception("File is missing one or more required datasets.")


def _collect_datasets(dsets: dict[str, Any], group: h5py.Group) -> None:
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[:]
        else:
            _collect_datasets(dsets, v)
