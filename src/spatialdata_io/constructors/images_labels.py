from spatialdata._types import ArrayLike
from typing import Iterable, Any
from xarray import DataArray
import numpy as np

__all__ = ["image_xarray_from_numpy", "labels_xarray_from_numpy"]


def image_xarray_from_numpy(x: np.ndarray, axes: Iterable[str]) -> DataArray:
    pass


def labels_xarray_from_numpy(x: np.ndarray, axes: Iterable[str]) -> DataArray:
    pass
