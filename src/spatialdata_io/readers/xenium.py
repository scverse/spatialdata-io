from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anndata import AnnData
from dask.dataframe import read_parquet
from dask_image.imread import imread
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pyarrow import Table
from shapely import Polygon
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Affine, Identity, Scale

from spatialdata_io._constants._constants import XeniumKeys
from spatialdata_io._docs import inject_docs
from spatialdata_io._utils import deprecation_alias
from spatialdata_io.readers._utils._read_10x_h5 import _read_10x_h5

__all__ = ["xenium"]


@deprecation_alias(cells_as_shapes="cells_as_circles")
@inject_docs(xx=XeniumKeys)
def xenium(
    path: str | Path,
    n_jobs: int = 1,
    cells_as_circles: bool = True,
    nucleus_boundaries: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read a *10X Genomics Xenium* dataset into a SpatialData object.

    This function reads the following files:

        - ``{xx.XENIUM_SPECS!r}``: File containing specifications.
        - ``{xx.NUCLEUS_BOUNDARIES_FILE!r}``: Polygons of nucleus boundaries.
        - ``{xx.CELL_BOUNDARIES_FILE!r}``: Polygons of cell boundaries.
        - ``{xx.TRANSCRIPTS_FILE!r}``: File containing transcripts.
        - ``{xx.CELL_FEATURE_MATRIX_FILE!r}``: File containing cell feature matrix.
        - ``{xx.CELL_METADATA_FILE!r}``: File containing cell metadata.
        - ``{xx.MORPHOLOGY_MIP_FILE!r}``: File containing morphology mip.
        - ``{xx.MORPHOLOGY_FOCUS_FILE!r}``: File containing morphology focus.

    .. seealso::

        - `10X Genomics Xenium file format  <https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html>`_.

    Parameters
    ----------
    path
        Path to the dataset.
    n_jobs
        Number of jobs to use for parallel processing.
    cells_as_circles
        Whether to read cells also as circles. Useful for performant visualization.
    nucleus_boundaries
        Whether to read nucleus boundaries.
    transcripts
        Whether to read transcripts.
    morphology_mip
        Whether to read morphology mip.
    morphology_focus
        Whether to read morphology focus.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    path = Path(path)
    with open(path / XeniumKeys.XENIUM_SPECS) as f:
        specs = json.load(f)

    specs["region"] = "cell_circles" if cells_as_circles else "cell_boundaries"
    return_values = _get_tables_and_circles(path, cells_as_circles, specs)
    if cells_as_circles:
        table, circles = return_values
    else:
        table = return_values
    polygons = {}

    if nucleus_boundaries:
        polygons["nucleus_boundaries"] = _get_polygons(
            path,
            XeniumKeys.NUCLEUS_BOUNDARIES_FILE,
            specs,
            n_jobs,
            idx=table.obs[str(XeniumKeys.CELL_ID)].copy(),
        )

    polygons["cell_boundaries"] = _get_polygons(
        path,
        XeniumKeys.CELL_BOUNDARIES_FILE,
        specs,
        n_jobs,
        idx=table.obs[str(XeniumKeys.CELL_ID)].copy(),
    )

    points = {}
    if transcripts:
        points["transcripts"] = _get_points(path, specs)

    images = {}
    if morphology_mip:
        images["morphology_mip"] = _get_images(
            path,
            XeniumKeys.MORPHOLOGY_MIP_FILE,
            imread_kwargs,
            image_models_kwargs,
        )
    if morphology_focus:
        images["morphology_focus"] = _get_images(
            path,
            XeniumKeys.MORPHOLOGY_FOCUS_FILE,
            imread_kwargs,
            image_models_kwargs,
        )
    if cells_as_circles:
        sdata = SpatialData(images=images, shapes=polygons | {specs["region"]: circles}, points=points, table=table)
    else:
        sdata = SpatialData(images=images, shapes=polygons, points=points, table=table)

    # find and add additional aligned images
    aligned_images = _add_aligned_images(path, imread_kwargs, image_models_kwargs)
    for key, value in aligned_images.items():
        sdata.images[key] = value

    return sdata


def _get_polygons(
    path: Path, file: str, specs: dict[str, Any], n_jobs: int, idx: Optional[ArrayLike] = None
) -> GeoDataFrame:
    def _poly(arr: ArrayLike) -> Polygon:
        return Polygon(arr[:-1])

    # seems to be faster than pd.read_parquet
    df = pq.read_table(path / file).to_pandas()

    out = Parallel(n_jobs=n_jobs)(
        delayed(_poly)(i.to_numpy())
        for _, i in df.groupby(XeniumKeys.CELL_ID)[[XeniumKeys.BOUNDARIES_VERTEX_X, XeniumKeys.BOUNDARIES_VERTEX_Y]]
    )
    geo_df = GeoDataFrame({"geometry": out})
    if idx is not None:
        geo_df.index = idx
    scale = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    return ShapesModel.parse(geo_df, transformations={"global": scale})


def _get_points(path: Path, specs: dict[str, Any]) -> Table:
    table = read_parquet(path / XeniumKeys.TRANSCRIPTS_FILE)
    table["feature_name"] = table["feature_name"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x), meta=("feature_name", "object")
    )

    transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
    points = PointsModel.parse(
        table,
        coordinates={"x": XeniumKeys.TRANSCRIPTS_X, "y": XeniumKeys.TRANSCRIPTS_Y, "z": XeniumKeys.TRANSCRIPTS_Z},
        feature_key=XeniumKeys.FEATURE_NAME,
        instance_key=XeniumKeys.CELL_ID,
        transformations={"global": transform},
    )
    return points


def _get_tables_and_circles(
    path: Path, cells_as_circles: bool, specs: dict[str, Any]
) -> AnnData | tuple[AnnData, AnnData]:
    adata = _read_10x_h5(path / XeniumKeys.CELL_FEATURE_MATRIX_FILE)
    metadata = pd.read_parquet(path / XeniumKeys.CELL_METADATA_FILE)
    np.testing.assert_array_equal(metadata.cell_id.astype(str), adata.obs_names.values)
    circ = metadata[[XeniumKeys.CELL_X, XeniumKeys.CELL_Y]].to_numpy()
    adata.obsm["spatial"] = circ
    metadata.drop([XeniumKeys.CELL_X, XeniumKeys.CELL_Y], axis=1, inplace=True)
    adata.obs = metadata
    adata.obs["region"] = specs["region"]
    adata.obs["region"] = adata.obs["region"].astype("category")
    if isinstance(adata.obs[XeniumKeys.CELL_ID].iloc[0], bytes):
        adata.obs[XeniumKeys.CELL_ID] = adata.obs[XeniumKeys.CELL_ID].apply(lambda x: x.decode("utf-8"))
    table = TableModel.parse(adata, region=specs["region"], region_key="region", instance_key=str(XeniumKeys.CELL_ID))
    if cells_as_circles:
        transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
        radii = np.sqrt(adata.obs[XeniumKeys.CELL_NUCLEUS_AREA].to_numpy() / np.pi)
        circles = ShapesModel.parse(
            circ,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs[XeniumKeys.CELL_ID].copy(),
        )
        return table, circles
    return table


def _get_images(
    path: Path,
    file: str,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialImage | MultiscaleSpatialImage:
    image = imread(path / file, **imread_kwargs)
    return Image2DModel.parse(
        image, transformations={"global": Identity()}, dims=("c", "y", "x"), **image_models_kwargs
    )


def _add_aligned_images(
    path: Path,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> dict[str, MultiscaleSpatialImage]:
    """Discover and parse aligned images."""
    images = {}
    ome_tif_files = list(path.glob("*.ome.tif"))
    csv_files = list(path.glob("*.csv"))
    for file in ome_tif_files:
        element_name = None
        for suffix in [XeniumKeys.ALIGNED_HE_IMAGE_SUFFIX, XeniumKeys.ALIGNED_IF_IMAGE_SUFFIX]:
            if file.name.endswith(suffix):
                element_name = suffix.replace(XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_REMOVE, "")
                break
        if element_name is not None:
            # check if an alignment file exists
            expected_filename = file.name.replace(
                XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_REMOVE, XeniumKeys.ALIGNMENT_FILE_SUFFIX_TO_ADD
            )
            alignment_files = [f for f in csv_files if f.name == expected_filename]
            assert len(alignment_files) <= 1, f"Found more than one alignment file for {file.name}."
            alignment_file = alignment_files[0] if alignment_files else None

            # parse the image
            image = xenium_aligned_image(file, alignment_file, imread_kwargs, image_models_kwargs)
            images[element_name] = image
    return images


def xenium_aligned_image(
    image_path: str | Path,
    alignment_file: str | Path | None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> MultiscaleSpatialImage:
    """
    Read an image aligned to a Xenium dataset, with an optional alignment file.

    Parameters
    ----------
    image_path
        Path to the image.
    alignment_file
        Path to the alignment file, if not passed it is assumed that the image is aligned.
    image_models_kwargs
        Keyword arguments to pass to the image models.

    Returns
    -------
    The single-scale or multi-scale aligned image element.
    """
    image_path = Path(image_path)
    assert image_path.exists(), f"File {image_path} does not exist."
    image = imread(image_path, **imread_kwargs)

    # Depending on the version of pipeline that was used, some images have shape (1, y, x, 3) and others (3, y, x)
    # since y and x are always different from 1, let's differentiate from the two cases here, independently of the
    # pipeline version.
    # Note that a more robust approach is to look at the xml metadata in the ome.tif; we could use this in a future PR.
    print(image.shape)
    if len(image.shape) == 4:
        assert image.shape[0] == 1
        assert image.shape[-1] == 3
        image = image.squeeze(0)
        dims = ("y", "x", "c")
    else:
        assert len(image.shape) == 3
        assert image.shape[0] == 3
        dims = ("c", "y", "x")

    if alignment_file is None:
        transformation = Identity()
    else:
        alignment_file = Path(alignment_file)
        assert alignment_file.exists(), f"File {alignment_file} does not exist."
        alignment = pd.read_csv(alignment_file, header=None).values
        transformation = Affine(alignment, input_axes=("x", "y"), output_axes=("x", "y"))

    return Image2DModel.parse(
        image,
        dims=dims,
        transformations={"global": transformation},
        **image_models_kwargs,
    )
