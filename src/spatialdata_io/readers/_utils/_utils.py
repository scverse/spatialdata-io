from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from anndata import AnnData, read_text
from h5py import File

from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

PathLike = Union[os.PathLike, str]

try:
    from numpy.typing import NDArray

    NDArrayA = NDArray[Any]
except (ImportError, TypeError):
    NDArray = np.ndarray  # type: ignore[misc]
    NDArrayA = np.ndarray  # type: ignore[misc]


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: Optional[str] = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if counts_file.endswith(".h5"):
        print(counts_file)
        adata: AnnData = _read_10x_h5(path / counts_file, **kwargs)
        with File(path / counts_file, mode="r") as f:
            attrs = dict(f.attrs)
            if library_id is None:
                try:
                    lid = attrs.pop("library_ids")[0]
                    library_id = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
                except ValueError:
                    raise KeyError(
                        "Unable to extract library id from attributes. Please specify one explicitly."
                    ) from None

            adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id]["metadata"][key] = metadata

        return adata, library_id

    if library_id is None:
        raise ValueError("Please explicitly specify library id.")

    if counts_file.endswith((".csv", ".txt")):
        adata = read_text(path / counts_file, **kwargs)
    elif counts_file.endswith(".mtx.gz"):
        try:
            from scanpy.readwrite import read_10x_mtx
        except ImportError:
            raise ImportError("Please install scanpy to read 10x mtx files, `pip install scanpy`.")
        prefix = counts_file.replace("matrix.mtx.gz", "")
        adata = read_10x_mtx(path, prefix=prefix, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns["spatial"] = {library_id: {"metadata": {}}}  # can overwrite
    return adata, library_id
