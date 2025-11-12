from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from anndata import AnnData
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Scale

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

if TYPE_CHECKING:
    from collections.abc import Mapping

COSMX_PIXEL_SIZE = 0.120280945

__all__ = ["cosmx"]


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: str | None = None,
    fov: int | None = None,
    read_image: bool = True,
    read_proteins: bool = False,
    cells_labels: bool = False,
    cells_table: bool = False,
    cells_polygons: bool = False,
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    flip_image: bool = False,
    fov_shift: bool | None = None,
) -> SpatialData:
    """Read *CosMx Nanostring / Bruker* data. The fields of view are stitched together, except if `fov` is provided.

    This function reads the following files:
        - `*_fov_positions_file.csv` or `*_fov_positions_file.csv.gz`: FOV locations
        - `Morphology2D` directory: all the FOVs morphology images
        - `*_tx_file.csv.gz` or `*_tx_file.csv`: Transcripts location and names
        - If `read_proteins` is `True`, all the images under the nested `ProteinImages` directories will be read

        These files must be exported as flat files in AtomX. That is: within a study, click on "Export" and then select files from the "Flat CSV Files" section (transcripts flat and FOV position flat).

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    dataset_id
        Optional name of the dataset (needs to be provided if not inferred).
    fov
        Number of one single field of view to be read. If not provided, reads all FOVs and create a stitched image.
    read_image
        Whether to read the images or not.
    read_proteins
        Whether to read the proteins or the transcripts.
    cells_labels
        Whether to read the cell labels or not.
    cells_table
        Whether to read the cell table or not.
    cells_polygons
        Whether to read the cell polygons or not.
    labels_models_kwargs
        Keyword arguments passed to `spatialdata.models.Labels2DModel`.
    image_models_kwargs
        Keyword arguments passed to `spatialdata.models.Image2DModel`.
    imread_kwargs
        Keyword arguments passed to `dask_image.imread.imread`.
    flip_image
        For some buggy exports of AtomX 1.3.2, `flip_image=True` has to be used for stitching. See [this](https://github.com/gustaveroussy/sopa/issues/231) issue.
    fov_shift
        Whether to apply FOV shift correction. For some datasets, there is a one-FOV shift in the y direction between the image and the polygons/transcripts. If `None`, it will be inferred automatically based on the FOV positions file and the polygons files.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    assert read_image or cells_labels, "At least one of `read_image` or `cells_labels` must be True."

    labels_models_kwargs = {"chunks": (4256, 4256)} if labels_models_kwargs is None else labels_models_kwargs

    dataset_id = _infer_dataset_id(path, dataset_id)

    _reader = _CosMXReader(path, dataset_id, fov, flip_image, fov_shift)

    ### Read elements
    region = _reader.shapes_key if cells_polygons else (_reader.labels_key if cells_labels else None)
    tables = _reader.read_tables(region) if cells_table else {}

    images = {}
    if read_image:
        images = _reader.read_images(
            read_proteins=read_proteins,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
        )

    labels = (
        _reader.read_labels(imread_kwargs=imread_kwargs, labels_models_kwargs=labels_models_kwargs)
        if cells_labels
        else {}
    )
    shapes = _reader.read_shapes() if cells_polygons else {}

    points = _reader.read_transcripts() if not read_proteins else {}

    return SpatialData(images=images, labels=labels, points=points, tables=tables, shapes=shapes)


class _CosMXReader:
    max_cell_id: int

    def __init__(self, path: Path, dataset_id: str, fov: int | None, flip_image: bool, fov_shift: bool | None):
        self.path = path
        self.dataset_id = dataset_id
        self.fov = fov
        self.flip_image = flip_image

        self.fov_locs = self._read_fov_locs()

        if self.fov is not None:
            fov_shift = False
            logger.info(f"Reading single FOV ({self.fov}), the image will not be stitched")
        elif fov_shift is None:
            fov_shift = self._infer_fov_shift()
            logger.info(
                f"FOV shift correction is {'enabled' if fov_shift else 'disabled'} (if this is not correct, please set `fov_shift` manually)"
            )
        self.fov_shift = fov_shift

    def read_transcripts(self) -> dict[str, PointsModel]:
        transcripts_file = self.path / f"{self.dataset_id}_tx_file.csv.gz"

        if transcripts_file.exists():
            df = pd.read_csv(transcripts_file, compression="gzip")
        else:
            transcripts_file = self.path / f"{self.dataset_id}_tx_file.csv"
            assert transcripts_file.exists(), f"Transcript file {transcripts_file} not found."
            df = pd.read_csv(transcripts_file)

        TRANSCRIPT_COLUMNS = ["x_global_px", "y_global_px", "target"]
        assert np.isin(TRANSCRIPT_COLUMNS, df.columns).all(), (
            f"The file {transcripts_file} must contain the following columns: {', '.join(TRANSCRIPT_COLUMNS)}. Consider using a different export module."
        )

        df["global_cell_id"] = self._get_global_cell_id(df)

        if self.fov is None:
            df["x"] = (df["x_global_px"] - self.fov_locs["xmin"].min()) * COSMX_PIXEL_SIZE
            df["y"] = (df["y_global_px"] - self.fov_locs["ymin"].min()) * COSMX_PIXEL_SIZE
            points_name = "points"
        else:
            df = df[df["fov"] == self.fov]
            df["x"] = df["x_local_px"] * COSMX_PIXEL_SIZE
            df["y"] = df["y_local_px"] * COSMX_PIXEL_SIZE
            points_name = f"F{self.fov:0>5}_points"

        from spatialdata_io._constants._constants import CosmxKeys

        transcripts = PointsModel.parse(
            df,
            feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
            transformations={"global": Scale([1 / COSMX_PIXEL_SIZE, 1 / COSMX_PIXEL_SIZE], axes=("x", "y"))},
        )

        return {points_name: transcripts}

    def _read_cell_metadata(self) -> pd.DataFrame:
        from spatialdata_io._constants._constants import CosmxKeys

        metadata = self._read_csv_gz(f"{self.dataset_id}_{CosmxKeys.METADATA_SUFFIX}")
        metadata.index = self._get_global_cell_id(metadata)

        if "cell_id" in metadata.columns:
            del metadata["cell_id"]

        return metadata

    def read_tables(self, region: str | None) -> dict[str, AnnData]:
        from spatialdata_io._constants._constants import CosmxKeys

        counts = self._read_csv_gz(f"{self.dataset_id}_{CosmxKeys.COUNTS_SUFFIX}")
        counts = counts[counts["cell_ID"] != 0]  # remove background
        counts.index = self._get_global_cell_id(counts)
        counts.drop(columns=["fov", "cell_ID"], inplace=True)

        obs = self._read_cell_metadata()

        assert (obs.index == counts.index).all(), "The cell IDs in the metadata and counts files do not match."

        obs[CosmxKeys.FOV] = pd.Categorical(obs[CosmxKeys.FOV].astype(str))
        if region is not None:
            obs["region_key"] = pd.Series(region, index=obs.index, dtype="category")
            obs["global_cell_id"] = obs.index
        obs.index = obs.index.astype(str)

        adata = AnnData(csr_matrix(counts.values), obs=obs, var=pd.DataFrame(index=counts.columns))

        table = TableModel.parse(
            adata,
            region=region,
            region_key="region_key" if region is not None else None,
            instance_key="global_cell_id" if region is not None else None,
        )

        table.obsm["spatial"] = table.obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].to_numpy()
        table.obs.drop(columns=[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL], inplace=True)

        return {"table": table}

    def _infer_fov_shift(self) -> bool:
        if self.flip_image:
            return True

        try:
            df_poly = self.df_poly
        except FileNotFoundError:
            logger.warning("Polygons file not found, cannot infer FOV shift.")
            return False

        fov_y_min = df_poly.groupby("fov")["y_global_px"].min()

        return float((fov_y_min < self.fov_locs.loc[fov_y_min.index, "y"]).mean()) > 0.1

    @property
    def df_poly(self) -> pd.DataFrame:
        if not hasattr(self, "_df_poly"):
            self._df_poly = self._read_csv_gz(f"{self.dataset_id}-polygons.csv")
        return self._df_poly

    def read_shapes(self) -> dict[str, gpd.GeoDataFrame]:
        df_poly = self.df_poly
        df_poly.rename(columns={"cellID": "cell_ID"}, inplace=True)

        if self.flip_image:
            x_origin = self.fov_locs.loc[df_poly["fov"], "xmin"].values
            y_origin = self.fov_locs.loc[df_poly["fov"], "ymin"].values
            df_poly["x_global_px"] = x_origin + (x_origin - df_poly["x_global_px"])
            df_poly["y_global_px"] = y_origin + (y_origin - df_poly["y_global_px"])

        if self.fov is None:
            df_poly.index = self._get_global_cell_id(df_poly)
            xy_keys = ["x_global_px", "y_global_px"]
        else:
            df_poly = df_poly[df_poly["fov"] == self.fov]
            df_poly.index = df_poly["cell_ID"]
            xy_keys = ["x_local_px", "y_local_px"]

        geometry = df_poly.groupby(level=0).apply(lambda sub_df: Polygon(sub_df[xy_keys]))
        gdf = gpd.GeoDataFrame(df_poly.groupby(level=0)[["fov"]].first(), geometry=geometry)

        if self.fov is None:
            gdf.geometry = gdf.geometry.translate(-self.fov_locs["xmin"].min(), -self.fov_locs["ymin"].min())

        return {self.shapes_key: ShapesModel.parse(gdf)}

    @property
    def shapes_key(self) -> str:
        return f"F{self.fov:0>5}_cells_polygons" if self.fov is not None else "cells_polygons"

    @property
    def labels_key(self) -> str:
        return f"F{self.fov:0>5}_labels" if self.fov is not None else "stitched_labels"

    def read_images(
        self, read_proteins: bool, imread_kwargs: Mapping[str, Any], image_models_kwargs: Mapping[str, Any]
    ) -> dict[str, xr.DataArray | xr.DataTree]:
        images_dir = _find_dir(self.path, "Morphology2D")
        morphology_coords = _get_morphology_coords(images_dir)

        protein_dir_dict = {}
        if read_proteins:
            protein_dir_dict = {
                int(protein_dir.parent.name[3:]): protein_dir
                for protein_dir in list(self.path.rglob("**/FOV*/ProteinImages"))
            }
            assert len(protein_dir_dict), f"No directory called 'ProteinImages' was found under {self.path}"

        if self.fov is None:
            return {
                "stitched_image": self._stitch_tifffiles(
                    images_dir,
                    protein_dir_dict=protein_dir_dict,
                    morphology_coords=morphology_coords,
                    imread_kwargs=imread_kwargs,
                    image_models_kwargs=image_models_kwargs,
                )
            }

        fov_file = _find_matching_fov_file(images_dir, self.fov)
        image, c_coords = _read_fov_image(fov_file, protein_dir_dict.get(self.fov), morphology_coords, **imread_kwargs)
        image = Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=c_coords, **image_models_kwargs)

        return {f"F{self.fov:0>5}_image": image}

    def read_labels(
        self,
        imread_kwargs: Mapping[str, Any],
        labels_models_kwargs: Mapping[str, Any],
    ) -> dict[str, xr.DataArray | xr.DataTree]:
        labels_dir = _find_dir(self.path, "CellLabels")

        if self.fov is None:
            return {
                self.labels_key: self._stitch_tifffiles(
                    labels_dir,
                    labels=True,
                    imread_kwargs=imread_kwargs,
                    image_models_kwargs=labels_models_kwargs,
                )
            }

        fov_file = _find_matching_fov_file(labels_dir, self.fov)
        labels, _ = _read_fov_image(fov_file, None, [], **imread_kwargs)
        labels = Labels2DModel.parse(labels[0], dims=("y", "x"), **labels_models_kwargs)

        return {self.labels_key: labels}

    def _stitch_tifffiles(
        self,
        images_dir: Path,
        imread_kwargs: Mapping[str, Any],
        image_models_kwargs: Mapping[str, Any],
        protein_dir_dict: dict[int, Path] | None = None,
        morphology_coords: list[str] | None = None,
        labels: bool = False,
    ) -> xr.DataArray | xr.DataTree:
        images_paths = list(images_dir.glob("*.[Tt][Ii][Ff]"))
        shape = imread(images_paths[0]).shape[1:]

        self._set_fov_locs_bounding_boxes(shape)

        fov_images, c_coords_dict = {}, {}
        pattern = re.compile(r".*_F(\d+)")

        protein_dir_dict = protein_dir_dict or {}
        morphology_coords = morphology_coords or []

        if labels:
            obs = self._read_cell_metadata()

        for image_path in images_paths:
            fov = int(pattern.findall(image_path.name)[0])

            image, c_coords = _read_fov_image(image_path, protein_dir_dict.get(fov), morphology_coords, **imread_kwargs)
            assert image.shape[1:] == shape, (
                f"Expected all images to have the same shape {shape}, but found {image.shape[1:]} for FOV {fov}."
            )

            if labels and (max_label := image.max().compute()) > 0:
                fov_obs = obs[obs["fov"] == fov]
                local_ids, global_ids = fov_obs["cell_ID"], fov_obs.index

                mapping = np.zeros(max(max_label, local_ids.max()) + 1, dtype=int)
                mapping[local_ids] = global_ids

                image = da.map_blocks(mapping.__getitem__, image, dtype=int)

            fov_images[fov] = da.flip(image, axis=1)
            c_coords_dict[fov] = c_coords

        height, width = self.fov_locs["y1"].max(), self.fov_locs["x1"].max()

        if labels:
            stitched_image = da.zeros(shape=(height, width), dtype=image.dtype)
            stitched_image = xr.DataArray(stitched_image, dims=("y", "x"))
        else:
            c_coords = list(set.union(*[set(names) for names in c_coords_dict.values()]))

            stitched_image = da.zeros(shape=(len(c_coords), height, width), dtype=image.dtype)
            stitched_image = xr.DataArray(stitched_image, dims=("c", "y", "x"), coords={"c": c_coords})

        for fov, im in fov_images.items():
            xmin, xmax = self.fov_locs.loc[fov, "x0"], self.fov_locs.loc[fov, "x1"]
            ymin, ymax = self.fov_locs.loc[fov, "y0"], self.fov_locs.loc[fov, "y1"]

            if self.flip_image:
                y_slice, x_slice = slice(height - ymax, height - ymin), slice(width - xmax, width - xmin)
            else:
                y_slice, x_slice = slice(ymin, ymax), slice(xmin, xmax)

            if labels:
                stitched_image[y_slice, x_slice] = im[0]
            else:
                stitched_image.loc[{"c": c_coords_dict[fov], "y": y_slice, "x": x_slice}] = im

                if len(c_coords_dict[fov]) < len(c_coords):
                    logger.warning(f"Missing channels ({len(c_coords) - len(c_coords_dict[fov])}) for FOV {fov}")

        if labels:
            return Labels2DModel.parse(stitched_image, **image_models_kwargs)
        else:
            return Image2DModel.parse(stitched_image, c_coords=c_coords, **image_models_kwargs)

    def _set_fov_locs_bounding_boxes(self, shape: tuple[int, int]) -> None:
        self.fov_locs["xmin"] = self.fov_locs["x"]
        self.fov_locs["xmax"] = self.fov_locs["x"] + shape[1]

        self.fov_locs["ymin"] = self.fov_locs["y"] - shape[0] * self.fov_shift
        self.fov_locs["ymax"] = self.fov_locs["y"] + shape[0] * (1 - self.fov_shift)

        for dim in ["x", "y"]:
            origin = self.fov_locs[f"{dim}min"].min()
            self.fov_locs[f"{dim}0"] = (self.fov_locs[f"{dim}min"] - origin).round().astype(int)
            self.fov_locs[f"{dim}1"] = (self.fov_locs[f"{dim}max"] - origin).round().astype(int)

    def _read_fov_locs(self) -> pd.DataFrame:
        fov_file = self.path / f"{self.dataset_id}_fov_positions_file.csv"

        if not fov_file.exists():
            fov_file = self.path / f"{self.dataset_id}_fov_positions_file.csv.gz"

        assert fov_file.exists(), f"Missing field of view file: {fov_file}"

        fov_locs = pd.read_csv(fov_file)
        fov_locs.columns = fov_locs.columns.str.lower()

        valid_keys = [
            ["fov", "x_global_px", "y_global_px"],
            ["fov", "x_mm", "y_mm"],
            ["fov", "x_global_mm", "y_global_mm"],
        ]
        mm_to_pixels = 1e3 / COSMX_PIXEL_SIZE  # conversion factor from mm to pixels for CosMx

        for (fov_key, x_key, y_key), scale_factor in zip(valid_keys, [1, mm_to_pixels, mm_to_pixels], strict=False):
            if not np.isin([fov_key, x_key, y_key], fov_locs.columns).all():  # try different column names
                continue

            fov_locs.index = fov_locs[fov_key]
            fov_locs["x"] = fov_locs[x_key] * scale_factor
            fov_locs["y"] = fov_locs[y_key] * scale_factor

            return fov_locs

        raise ValueError(
            f"The FOV positions file must contain one of the following sets of columns: {', or '.join(list(map(str, valid_keys)))}"
        )

    def _get_global_cell_id(self, df: pd.DataFrame) -> pd.Series:
        max_cell_id: int = df["cell_ID"].max()

        if hasattr(self, "max_cell_id"):
            assert max_cell_id == self.max_cell_id, (
                f"Expected max cell ID to be {self.max_cell_id}, but got {max_cell_id}."
            )
        self.max_cell_id = max_cell_id

        return df["fov"] * (self.max_cell_id + 1) * (df["cell_ID"] > 0) + df["cell_ID"]

    def _read_csv_gz(self, name: str) -> pd.DataFrame:
        for extension in [".gz", ""]:
            file = self.path / f"{name}{extension}"
            if file.exists():
                return pd.read_csv(file)
        raise FileNotFoundError(f"Input file not found: {self.path / f'{name}.gz'}")


def _infer_dataset_id(path: Path, dataset_id: str | None) -> str:
    if isinstance(dataset_id, str):
        return dataset_id

    for suffix in [".csv", ".csv.gz"]:
        counts_files = list(path.rglob(f"[!\\.]*_fov_positions_file{suffix}"))

        if len(counts_files) == 1:
            found = re.match(rf"(.*)_fov_positions_file{suffix}", counts_files[0].name)
            if found:
                return found.group(1)

    raise ValueError("Could not infer `dataset_id` from the name of the transcript file. Please specify it manually.")


def _read_fov_image(
    morphology_path: Path,
    protein_path: Path | None,
    morphology_coords: list[str],
    **imread_kwargs: int,
) -> tuple[da.Array, list[str]]:
    image: da.Array = imread(morphology_path, **imread_kwargs)

    protein_names: list[str] = []
    if protein_path is not None:
        protein_image, protein_names = _read_protein_fov(protein_path)
        image: da.Array = da.concatenate([image, protein_image], axis=0)  # type: ignore[no-redef]

    return image, _deduplicate_names(morphology_coords + protein_names)


def _read_protein_fov(protein_dir: Path) -> tuple[da.Array, list[str]]:
    images_paths = list(protein_dir.rglob("*.[Tt][Ii][Ff]"))
    protein_image = da.concatenate([imread(image_path) for image_path in images_paths], axis=0)
    channel_names = [_get_protein_name(image_path) for image_path in images_paths]

    return protein_image, channel_names


def _find_matching_fov_file(images_dir: Path, fov: str | int) -> Path:
    assert isinstance(fov, int), "Expected `fov` to be an integer"

    pattern = re.compile(rf".*_F0*{fov}\.[Tt][Ii][Ff]")
    fov_files = [file for file in images_dir.rglob("*") if pattern.match(file.name)]

    assert len(fov_files), f"No file matches the pattern {pattern} inside {images_dir}"
    assert len(fov_files) == 1, f"Multiple files match the pattern {pattern}: {', '.join(map(str, fov_files))}"

    return fov_files[0]


def _find_dir(path: Path, name: str) -> Path:
    if (path / name).is_dir():
        return path / name

    paths = list(path.rglob(f"**/{name}"))
    assert len(paths) == 1, f"Found {len(paths)} path(s) with name {name} inside {path}"

    return paths[0]


def _deduplicate_names(names: pd.Series | np.ndarray | list[str]) -> list[str]:
    if not isinstance(names, pd.Series):
        names = pd.Series(names)
    names = names.astype(str)

    duplicates = names.duplicated()
    names[duplicates] += " (" + names.groupby(by=names).cumcount().astype(str)[duplicates] + ")"

    return names.values.tolist()  # type: ignore[no-any-return]


def _get_morphology_coords(images_dir: Path) -> list[str]:
    images_paths = list(images_dir.glob("*.[Tt][Ii][Ff]"))
    assert len(images_paths) > 0, f"Expected to find images inside {images_dir}"

    with tifffile.TiffFile(images_paths[0]) as tif:
        description = tif.pages[0].description

        substrings = re.findall(r'"BiologicalTarget": "(.*?)",', description)
        channels = re.findall(r'"ChannelId": "(.*?)",', description)
        channel_order = list(re.findall(r'"ChannelOrder": "(.*?)",', description)[0])

        channels: list[str] = [substrings[channels.index(x)] if x in channels else x for x in channel_order]  # type: ignore[no-redef]
        return [channel.replace("/", ".") for channel in channels]


def _get_protein_name(image_path: Path) -> str:
    with tifffile.TiffFile(image_path) as tif:
        description = tif.pages[0].description
        substring: str = re.findall(r'"DisplayName": "(.*?)",', description)[0]
        return substring.replace("/", ".")
