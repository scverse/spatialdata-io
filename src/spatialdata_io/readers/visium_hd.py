from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from dask_image.imread import imread
from geopandas import GeoDataFrame
from imageio import imread as imread2
from numpy.random import default_rng
from skimage.transform import ProjectiveTransform, warp
from spatialdata import (
    SpatialData,
    get_extent,
    rasterize_bins,
    rasterize_bins_link_table_to_labels,
)
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity, Scale, set_transformation
from xarray import DataArray

from spatialdata_io._constants._constants import VisiumHDKeys
from spatialdata_io._docs import inject_docs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from multiscale_spatial_image import MultiscaleSpatialImage
    from spatial_image import SpatialImage
    from spatialdata._types import ArrayLike

RNG = default_rng(0)


@inject_docs(vx=VisiumHDKeys)
def visium_hd(
    path: str | Path,
    dataset_id: str | None = None,
    filtered_counts_file: bool = True,
    bin_size: int | list[int] | None = None,
    bins_as_squares: bool = True,
    annotate_table_by_labels: bool = False,
    fullres_image_file: str | Path | None = None,
    load_all_images: bool = False,
    var_names_make_unique: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    anndata_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *10x Genomics* Visium HD formatted dataset.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to directory containing the *10x Genomics* Visium HD output.
    dataset_id
        Unique identifier of the dataset, used to name the elements of the `SpatialData` object. If `None`, it tries to
         infer it from the file name of the feature slice file.
    filtered_counts_file
        It sets the value of `counts_file` to ``{vx.FILTERED_COUNTS_FILE!r}`` (when `True`) or to
        ``{vx.RAW_COUNTS_FILE!r}`` (when `False`).
    bin_size
        When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the
        available bin sizes.
    bins_as_squares
        If `True`, the bins are represented as squares. If `False`, the bins are represented as circles. For a correct
        visualization one should use squares.
    annotate_table_by_labels
        If `True`, the tables will annotate labels layers representing the bins, if `False`, the tables will annotate
        shapes layer.
    fullres_image_file
        Path to the full-resolution image. By default the image is searched in the ``{vx.MICROSCOPE_IMAGE!r}``
        directory.
    load_all_images
        If `False`, load only the full resolution, high resolution and low resolution images. If `True`, also the
        following images: ``{vx.IMAGE_CYTASSIST!r}``.
    var_names_make_unique
        If `True`, call `.var_names_make_unique()` on each `AnnData` table.
    imread_kwargs
        Keyword arguments for :func:`imageio.imread`.
    image_models_kwargs
        Keyword arguments for :class:`spatialdata.models.Image2DModel`.
    anndata_kwargs
        Keyword arguments for :func:`anndata.io.read_h5ad`.

    Returns
    -------
    SpatialData object for the Visium HD data.
    """
    path = Path(path)
    all_files = [file for file in path.rglob("*") if file.is_file()]
    tables = {}
    shapes = {}
    images: dict[str, Any] = {}
    labels: dict[str, Any] = {}

    if dataset_id is None:
        dataset_id = _infer_dataset_id(path)

    filename_prefix = _get_filename_prefix(path, dataset_id)

    def load_image(path: Path, suffix: str, scale_factors: list[int] | None = None) -> None:
        _load_image(
            path=path,
            images=images,
            suffix=suffix,
            dataset_id=dataset_id,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
            scale_factors=scale_factors,
        )

    metadata, hd_layout = _parse_metadata(path, filename_prefix)
    file_format = hd_layout[VisiumHDKeys.FILE_FORMAT]
    if file_format != "1.0":
        warnings.warn(
            f"File format {file_format} is not supported. A more recent file format may be supported in a newer version"
            f"of the spatialdata-io package.",
            UserWarning,
            stacklevel=2,
        )

    def _get_bins(path_bins: Path) -> list[str]:
        return sorted(
            [
                bin_size.name
                for bin_size in path_bins.iterdir()
                if bin_size.is_dir() and bin_size.name.startswith(VisiumHDKeys.BIN_PREFIX)
            ]
        )

    all_path_bins = [path_bin for path_bin in all_files if VisiumHDKeys.BINNED_OUTPUTS in str(path_bin)]
    if len(all_path_bins) != 0:
        path_bins_parts = all_path_bins[
            -1
        ].parts  # just choosing last one here as users might have tar file which would be first
        path_bins = Path(*path_bins_parts[: path_bins_parts.index(VisiumHDKeys.BINNED_OUTPUTS) + 1])
    else:
        path_bins = path
    all_bin_sizes = _get_bins(path_bins)

    bin_sizes = []
    if bin_size is not None:
        if not isinstance(bin_size, list):
            bin_size = [bin_size]
        bin_sizes = [f"square_{bs:03}um" for bs in bin_size if f"square_{bs:03}um" in all_bin_sizes]
        if len(bin_sizes) < len(bin_size):
            warnings.warn(
                f"Requested bin size {bin_size} (available {all_bin_sizes}); ignoring the bin sizes that are not "
                "found.",
                UserWarning,
                stacklevel=2,
            )
    if bin_size is None or bin_sizes == []:
        bin_sizes = all_bin_sizes

    # iterate over the given bins and load the data
    for bin_size_str in bin_sizes:
        path_bin = path_bins / bin_size_str
        counts_file = VisiumHDKeys.FILTERED_COUNTS_FILE if filtered_counts_file else VisiumHDKeys.RAW_COUNTS_FILE
        adata = sc.read_10x_h5(
            path_bin / counts_file,
            gex_only=False,
            **anndata_kwargs,
        )

        path_bin_spatial = path_bin / VisiumHDKeys.SPATIAL

        with open(path_bin_spatial / VisiumHDKeys.SCALEFACTORS_FILE) as file:
            scalefactors = json.load(file)

        # consistency check
        found_bin_size = re.search(r"\d{3}", bin_size_str)
        assert found_bin_size is not None
        assert float(found_bin_size.group()) == scalefactors[VisiumHDKeys.SCALEFACTORS_BIN_SIZE_UM]
        assert np.isclose(
            scalefactors[VisiumHDKeys.SCALEFACTORS_BIN_SIZE_UM]
            / scalefactors[VisiumHDKeys.SCALEFACTORS_SPOT_DIAMETER_FULLRES],
            scalefactors[VisiumHDKeys.SCALEFACTORS_MICRONS_PER_PIXEL],
        )

        tissue_positions_file = path_bin_spatial / VisiumHDKeys.TISSUE_POSITIONS_FILE

        # read coordinates and set up adata.obs and adata.obsm
        coords = pd.read_parquet(tissue_positions_file)
        assert all(
            coords.columns.values
            == [
                VisiumHDKeys.BARCODE,
                VisiumHDKeys.IN_TISSUE,
                VisiumHDKeys.ARRAY_ROW,
                VisiumHDKeys.ARRAY_COL,
                VisiumHDKeys.LOCATIONS_Y,
                VisiumHDKeys.LOCATIONS_X,
            ]
        )
        coords.set_index(VisiumHDKeys.BARCODE, inplace=True, drop=True)
        coords_filtered = coords.loc[adata.obs.index]
        adata.obs = pd.merge(adata.obs, coords_filtered, how="left", left_index=True, right_index=True)
        # compatibility to legacy squidpy
        adata.obsm["spatial"] = adata.obs[[VisiumHDKeys.LOCATIONS_X, VisiumHDKeys.LOCATIONS_Y]].values
        # dropping the spatial coordinates (will be stored in shapes)
        adata.obs.drop(
            columns=[
                VisiumHDKeys.LOCATIONS_X,
                VisiumHDKeys.LOCATIONS_Y,
            ],
            inplace=True,
        )
        adata.obs[VisiumHDKeys.INSTANCE_KEY] = np.arange(len(adata))

        # scaling
        transform_original = Identity()
        transform_lowres = Scale(
            np.array(
                [
                    scalefactors[VisiumHDKeys.SCALEFACTORS_LOWRES],
                    scalefactors[VisiumHDKeys.SCALEFACTORS_LOWRES],
                ]
            ),
            axes=("x", "y"),
        )
        transform_hires = Scale(
            np.array(
                [
                    scalefactors[VisiumHDKeys.SCALEFACTORS_HIRES],
                    scalefactors[VisiumHDKeys.SCALEFACTORS_HIRES],
                ]
            ),
            axes=("x", "y"),
        )
        # parse shapes
        shapes_name = dataset_id + "_" + bin_size_str
        radius = scalefactors[VisiumHDKeys.SCALEFACTORS_SPOT_DIAMETER_FULLRES] / 2.0
        transformations = {
            dataset_id: transform_original,
            f"{dataset_id}_downscaled_hires": transform_hires,
            f"{dataset_id}_downscaled_lowres": transform_lowres,
        }
        circles = ShapesModel.parse(
            adata.obsm["spatial"],
            geometry=0,
            radius=radius,
            index=adata.obs[VisiumHDKeys.INSTANCE_KEY].copy(),
            transformations=transformations,
        )
        if not bins_as_squares:
            shapes[shapes_name] = circles
        else:
            squares_series = circles.buffer(radius, cap_style=3)
            shapes[shapes_name] = ShapesModel.parse(
                GeoDataFrame(geometry=squares_series), transformations=transformations
            )

        # parse table
        adata.obs[VisiumHDKeys.REGION_KEY] = shapes_name
        adata.obs[VisiumHDKeys.REGION_KEY] = adata.obs[VisiumHDKeys.REGION_KEY].astype("category")

        tables[bin_size_str] = TableModel.parse(
            adata,
            region=shapes_name,
            region_key=str(VisiumHDKeys.REGION_KEY),
            instance_key=str(VisiumHDKeys.INSTANCE_KEY),
        )
        if var_names_make_unique:
            tables[bin_size_str].var_names_make_unique()

    # read full resolution image
    if fullres_image_file is not None:
        fullres_image_file = Path(fullres_image_file)
    else:
        path_fullres = path / VisiumHDKeys.MICROSCOPE_IMAGE
        if path_fullres.exists():
            fullres_image_paths = [file for file in path_fullres.iterdir() if file.is_file()]
        elif list((path_fullres := (path / f"{filename_prefix}tissue_image")).parent.glob(f"{path_fullres.name}.*")):
            fullres_image_paths = list(path_fullres.parent.glob(f"{path_fullres.name}.*"))
        else:
            fullres_image_paths = []
        if len(fullres_image_paths) > 1:
            warnings.warn(
                f"Multiple files found in {path_fullres}, using the first one: {fullres_image_paths[0].stem}. Please"
                " specify the path to the full resolution image manually using the `fullres_image_file` argument.",
                UserWarning,
                stacklevel=2,
            )
        if len(fullres_image_paths) == 0:
            warnings.warn(
                "No full resolution image found. If incorrect, please specify the path in the "
                "`fullres_image_file` parameter when calling the `visium_hd` reader function.",
                UserWarning,
                stacklevel=2,
            )
        fullres_image_file = fullres_image_paths[0] if len(fullres_image_paths) > 0 else None

    if fullres_image_file is not None:
        load_image(
            path=fullres_image_file,
            suffix="_full_image",
            scale_factors=[2, 2, 2, 2],
        )

    # hires image
    hires_image_path = [path for path in all_files if VisiumHDKeys.IMAGE_HIRES_FILE in str(path)]
    if len(hires_image_path) == 0:
        warnings.warn(
            f"No image path found containing the hires image: {VisiumHDKeys.IMAGE_HIRES_FILE}",
            UserWarning,
            stacklevel=2,
        )
    load_image(
        path=hires_image_path[0],
        suffix="_hires_image",
    )
    set_transformation(
        images[dataset_id + "_hires_image"],
        {
            f"{dataset_id}_downscaled_hires": Identity(),
            dataset_id: transform_hires.inverse(),
        },
        set_all=True,
    )

    # lowres image
    lowres_image_path = [path for path in all_files if VisiumHDKeys.IMAGE_LOWRES_FILE in str(path)]
    if len(lowres_image_path) == 0:
        warnings.warn(
            f"No image path found containing the lowres image: {VisiumHDKeys.IMAGE_LOWRES_FILE}",
            UserWarning,
            stacklevel=2,
        )
    load_image(
        path=lowres_image_path[0],
        suffix="_lowres_image",
    )
    set_transformation(
        images[dataset_id + "_lowres_image"],
        {
            f"{dataset_id}_downscaled_lowres": Identity(),
            dataset_id: transform_lowres.inverse(),
        },
        set_all=True,
    )

    # cytassist image
    cytassist_path = [path for path in all_files if VisiumHDKeys.IMAGE_CYTASSIST in str(path)]
    if len(cytassist_path) == 0:
        warnings.warn(
            f"No image path found containing the cytassist image: {VisiumHDKeys.IMAGE_CYTASSIST}",
            UserWarning,
            stacklevel=2,
        )
    if load_all_images:
        load_image(
            path=cytassist_path[0],
            suffix="_cytassist_image",
        )
        image = images[dataset_id + "_cytassist_image"]
        transform_matrices = _get_transform_matrices(metadata, hd_layout)
        projective0 = transform_matrices["cytassist_colrow_to_spot_colrow"]
        projective1 = transform_matrices["spot_colrow_to_microscope_colrow"]
        projective = projective1 @ projective0
        projective /= projective[2, 2]
        if _projective_matrix_is_affine(projective):
            affine = Affine(projective, input_axes=("x", "y"), output_axes=("x", "y"))
            set_transformation(image, affine, dataset_id)
        else:
            # the projective matrix is not affine, we will separate the affine part and the projective shift, and apply
            # the projective shift to the image
            affine_matrix, projective_shift = _decompose_projective_matrix(projective)
            affine = Affine(affine_matrix, input_axes=("x", "y"), output_axes=("x", "y"))

            # determine the size of the transformed image
            bounding_box = get_extent(image, coordinate_system=dataset_id)
            x0, x1 = bounding_box["x"]
            y0, y1 = bounding_box["y"]
            x1 -= 1
            y1 -= 1
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

            transformed_corners = []
            for x, y in corners:
                px, py = _projective_matrix_transform_point(projective_shift, x, y)
                transformed_corners.append((px, py))
            transformed_corners_array = np.array(transformed_corners)
            transformed_bounds = (
                np.min(transformed_corners_array[:, 0]),
                np.min(transformed_corners_array[:, 1]),
                np.max(transformed_corners_array[:, 0]),
                np.max(transformed_corners_array[:, 1]),
            )
            # the first two components are <= 0, we just discard them since the cytassist image has a lot of padding
            # and therefore we can safely discard pixels with negative coordinates
            transformed_shape = (np.ceil(transformed_bounds[2]), np.ceil(transformed_bounds[3]))

            # flip xy
            transformed_shape = (transformed_shape[1], transformed_shape[0])

            # the cytassist image is a small, single-scale image, so we can compute it in memory
            numpy_data = image.transpose("y", "x", "c").data.compute()
            warped = warp(
                numpy_data, ProjectiveTransform(projective_shift).inverse, output_shape=transformed_shape, order=1
            )
            warped = np.round(warped * 255).astype(np.uint8)
            warped = Image2DModel.parse(warped, dims=("y", "x", "c"), transformations={dataset_id: affine}, rgb=True)

            # we replace the cytassist image with the warped image
            images[dataset_id + "_cytassist_image"] = warped

    sdata = SpatialData(tables=tables, images=images, shapes=shapes, labels=labels)

    if annotate_table_by_labels:
        for bin_size_str in bin_sizes:
            shapes_name = dataset_id + "_" + bin_size_str

            # add labels layer (rasterized bins).
            labels_name = f"{dataset_id}_{bin_size_str}_labels"

            labels_element = rasterize_bins(
                sdata,
                bins=shapes_name,
                table_name=bin_size_str,
                row_key=VisiumHDKeys.ARRAY_ROW,
                col_key=VisiumHDKeys.ARRAY_COL,
                value_key=None,
                return_region_as_labels=True,
            )

            sdata[labels_name] = labels_element
            rasterize_bins_link_table_to_labels(
                sdata=sdata, table_name=bin_size_str, rasterized_labels_name=labels_name
            )

    return sdata


def _infer_dataset_id(path: Path) -> str:
    suffix = f"_{VisiumHDKeys.FEATURE_SLICE_FILE.value}"
    files = [file.name for file in path.iterdir() if file.is_file() and file.name.endswith(suffix)]
    if len(files) == 0 or len(files) > 1:
        raise ValueError(
            f"Cannot infer `dataset_id` from the feature slice file in {path}, please pass `dataset_id` as an "
            f"argument. The `dataset_id` value will be used to name the elements in the `SpatialData` object."
        )
    return files[0].replace(suffix, "")


def _load_image(
    path: Path,
    images: dict[str, SpatialImage | MultiscaleSpatialImage],
    suffix: str,
    dataset_id: str,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any],
    scale_factors: list[int] | None,
) -> None:
    if path.exists():
        if path.suffix != ".btf":
            data = imread(path)
            if len(data.shape) == 4:
                # this happens for the cytassist, hires and lowres images; the umi image doesn't need processing
                data = data.squeeze()
        else:
            if "MAX_IMAGE_PIXELS" in imread_kwargs:
                from PIL import Image as ImagePIL

                ImagePIL.MAX_IMAGE_PIXELS = dict(imread_kwargs).pop("MAX_IMAGE_PIXELS")
            # dask_image doesn't recognize .btf automatically and imageio v3 throws error due to pixel limit -> use imageio v2
            data = imread2(path, **imread_kwargs).squeeze()

        if data.shape[-1] == 3:  # HE image in RGB format
            data = data.transpose(2, 0, 1)
        else:
            assert data.shape[0] == min(data.shape), (
                "When the image is not in RGB, the first dimension should be the number of channels."
            )

        image = DataArray(data, dims=("c", "y", "x"))
        parsed = Image2DModel.parse(
            image,
            scale_factors=scale_factors,
            rgb=None,
            transformations={dataset_id: Identity()},
            **image_models_kwargs,
        )
        images[dataset_id + suffix] = parsed
    else:
        warnings.warn(f"File {path} does not exist, skipping it.", UserWarning, stacklevel=2)
    return None


def _projective_matrix_transform_point(projective_shift: ArrayLike, x: float, y: float) -> tuple[float, float]:
    v = np.array([x, y, 1])
    v = projective_shift @ v
    v /= v[2]
    return v[0], v[1]


def _projective_matrix_is_affine(projective_matrix: ArrayLike) -> bool:
    assert np.allclose(projective_matrix[2, 2], 1), "A projective matrix should have a 1 in the bottom right corner."
    return np.allclose(projective_matrix[2, :2], [0, 0])


def _decompose_projective_matrix(projective_matrix: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Decompose a projective transformation matrix into an affine transformation and a projective shift.

    Parameters
    ----------
    projective_matrix
        Projective transformation matrix.

    Returns
    -------
    A tuple where the first element is the affine matrix and the second element is the projective shift.

    Let P be the initial projective matrix and A the affine matrix. The projective shift S is defined as: S = A^-1 @ P.
    """
    assert np.allclose(projective_matrix[2, 2], 1), "A projective matrix should have a 1 in the bottom right corner."
    affine_matrix = projective_matrix.copy()
    affine_matrix[2] = [0, 0, 1]
    # equivalent to np.linalg.inv(affine_matrix) @ projective_matrix, but more numerically stable
    projective_shift = np.linalg.solve(affine_matrix, projective_matrix)
    projective_shift /= projective_shift[2, 2]
    return affine_matrix, projective_shift


def _get_filename_prefix(path: Path, dataset_id: str) -> str:
    if (path / f"{dataset_id}_{VisiumHDKeys.FEATURE_SLICE_FILE.value}").exists():
        return f"{dataset_id}_"
    assert (path / VisiumHDKeys.FEATURE_SLICE_FILE.value).exists(), (
        f"Cannot locate the feature slice file, please ensure the file is present in the {path} directory and/or adjust"
        "the `dataset_id` parameter"
    )
    return ""


def _parse_metadata(path: Path, filename_prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(path / f"{filename_prefix}{VisiumHDKeys.FEATURE_SLICE_FILE.value}", "r") as f5:
        metadata = json.loads(dict(f5.attrs)[VisiumHDKeys.METADATA_JSON])
        hd_layout = json.loads(metadata[VisiumHDKeys.HD_LAYOUT_JSON])
    return metadata, hd_layout


def _get_transform_matrices(metadata: dict[str, Any], hd_layout: dict[str, Any]) -> dict[str, ArrayLike]:
    """Gets 4 projective transformation matrices, describing how to align the CytAssist, spots and microscope coordinates.

    Parameters
    ----------
    metadata
        Metadata of the Visium HD dataset parsed using `_parse_metadata()` from the feature slice file.
    hd_layout
        Layout of the Visium HD dataset parsed using `_parse_metadata()` from the feature slice file.

    Returns
    -------
    A dictionary containing four projective transformation matrices:
    - CytAssist col/row to Spot col/row
    - Spot col/row to CytAssist col/row
    - Microscope col/row to Spot col/row
    - Spot col/row to Microscope col/row
    """
    transform_matrices = {}

    # this transformation is parsed but not used in the current implementation
    transform_matrices["hd_layout_transform"] = np.array(hd_layout[VisiumHDKeys.TRANSFORM]).reshape(3, 3)

    for key in [
        VisiumHDKeys.CYTASSIST_COLROW_TO_SPOT_COLROW,
        VisiumHDKeys.SPOT_COLROW_TO_CYTASSIST_COLROW,
        VisiumHDKeys.MICROSCOPE_COLROW_TO_SPOT_COLROW,
        VisiumHDKeys.SPOT_COLROW_TO_MICROSCOPE_COLROW,
    ]:
        coefficients = metadata[VisiumHDKeys.TRANSFORM_MATRICES][key]
        transform_matrices[key] = np.array(coefficients).reshape(3, 3)

    return transform_matrices
