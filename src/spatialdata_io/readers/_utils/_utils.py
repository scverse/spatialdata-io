from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from anndata import AnnData, read_mtx, read_text
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
    count_file: str,
    library_id: Optional[str] = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    library_id_: Optional[str] = None
    if count_file.endswith(".h5"):
        adata: AnnData = _read_10x_h5(path / count_file, **kwargs)
        with File(path / count_file, mode="r") as f:
            attrs = dict(f.attrs)
            try:
                lid = attrs.pop("library_ids")[0]
                library_id_ = lid.decode("utf-8") if isinstance(lid, bytes) else str(lid)
            except ValueError:
                raise KeyError("Unable to extract library id from attributes. Please specify one explicitly.") from None
            if library_id is not None:
                if library_id != library_id_:
                    raise ValueError(
                        f"library_id {library_id} does not match library_id {library_id_} in the file. Check the output file."
                    )

            adata.uns["spatial"] = {library_id_: {"metadata": {}}}  # can overwrite
            for key in ["chemistry_description", "software_version"]:
                if key not in attrs:
                    continue
                metadata = attrs[key].decode("utf-8") if isinstance(attrs[key], bytes) else attrs[key]
                adata.uns["spatial"][library_id_]["metadata"][key] = metadata

        return adata, library_id_

    if library_id_ is None:
        raise ValueError("Please explicitly specify library id.")

    if count_file.endswith((".csv", ".txt")):
        adata = read_text(path / count_file, **kwargs)
    elif count_file.endswith(".mtx"):
        adata = read_mtx(path / count_file, **kwargs)
    else:
        raise NotImplementedError("TODO")

    adata.uns["spatial"] = {library_id_: {"metadata": {}}}  # can overwrite
    return adata, library_id_
