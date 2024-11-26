from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from anndata import AnnData, read_text
from h5py import File

from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

PathLike = Union[os.PathLike, str]  # type:ignore[type-arg]


def _read_counts(
    path: str | Path,
    counts_file: str,
    library_id: str | None = None,
    **kwargs: Any,
) -> tuple[AnnData, str]:
    path = Path(path)
    if counts_file.endswith(".h5"):
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
        raise ValueError("Please explicitly specify `library id`.")

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


def _initialize_raster_models_kwargs(
    image_models_kwargs: Mapping[str, Any], labels_models_kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_models_kwargs = dict(image_models_kwargs)
    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    labels_models_kwargs = dict(labels_models_kwargs)
    if "chunks" not in labels_models_kwargs:
        labels_models_kwargs["chunks"] = (4096, 4096)
    if "scale_factors" not in labels_models_kwargs:
        labels_models_kwargs["scale_factors"] = [2, 2, 2, 2]
    return image_models_kwargs, labels_models_kwargs
