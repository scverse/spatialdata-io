from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import dask.dataframe as dd
import numpy as np
from anndata.io import read_h5ad
from dask_image.imread import imread
from spatialdata import SpatialData, to_polygons
from spatialdata._logging import logger
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    TableModel,
)
from tqdm.auto import tqdm

from spatialdata_io._constants._constants import G4XKeys
from spatialdata_io._docs import inject_docs

__all__ = ["g4x"]


@inject_docs(xx=G4XKeys)
def g4x(
    input_path: str | Path,
    output_path: str | Path | None = None,
    include_he: bool = True,
    include_segmentation: bool = True,
    include_protein: bool = True,
    include_transcripts: bool = True,
    include_tables: bool = True,
    mode: str = "append",
):
    """
    Create SpatialData objects for each sample in a run directory or a single sample directory.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to input directory containing run data or a single sample directory.
        If a run directory, assumes each subdirectory contains a sample. e.g. `input_path/A01`, `input_path/B01`, etc.
    output_path : Union[str, Path]
        Path to directory where SpatialData zarr stores will be written. If None, zarr stores will be written to each sample directory found in `input_path`.
    include_he : bool
        Include H&E image if available.
    include_segmentation : bool
        Include segmentation if available.
    include_protein : bool
        Include protein images if available.
    include_transcripts : bool
        Include transcript data if available.
    include_tables : bool
        Include tables if available.
    mode : str
        Mode for handling existing elements. Options:
        - "append": Skip existing elements (default)
        - "overwrite": Replace existing elements
    Returns
    -------
    sdatas : Union[SpatialData, list[SpatialData]]
        A single SpatialData object if processing a single sample directory, otherwise a list of SpatialData objects.
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Determine if input_path is a run directory or a single sample directory
    if any(p.is_dir() and re.match(r"[A-Z][0-9]{2}", p.name) for p in input_path.iterdir()):
        # Run directory with multiple samples
        sample_input_paths = [p for p in input_path.iterdir() if p.is_dir() and re.match(r"[A-Z][0-9]{2}", p.name)]
        logger.debug(f"Found {len(sample_input_paths)} samples.")

        if output_path is None:
            sample_output_paths = [input_path / p.name / f"{p.name}.zarr" for p in sample_input_paths]
        else:
            sample_output_paths = [output_path / f"{p.name}.zarr" for p in sample_input_paths]

        sdatas = []
        for sample_input_path, sample_output_path in tqdm(
            zip(sample_input_paths, sample_output_paths),
            total=len(sample_input_paths),
            desc="Processing samples",
        ):
            sdata = g4x_sample(
                input_path=sample_input_path,
                output_zarr_path=sample_output_path,
                include_he=include_he,
                include_segmentation=include_segmentation,
                include_protein=include_protein,
                include_transcripts=include_transcripts,
                include_tables=include_tables,
                mode=mode,
            )
            sdatas.append(sdata)
        return sdatas
    else:
        # Single sample directory
        logger.debug("Processing single sample directory.")
        if output_path is None:
            output_path = input_path / f"{input_path.name}.zarr"

        sdata = g4x_sample(
            input_path=input_path,
            output_zarr_path=output_path,
            include_he=include_he,
            include_segmentation=include_segmentation,
            include_protein=include_protein,
            include_transcripts=include_transcripts,
            include_tables=include_tables,
            mode=mode,
        )
        return sdata


def g4x_sample(
    input_path: str | Path,
    output_zarr_path: str | Path,
    include_he: bool = True,
    include_segmentation: bool = True,
    include_protein: bool = True,
    include_transcripts: bool = True,
    include_tables: bool = True,
    mode: str = "append",
) -> SpatialData:
    """
    Create a SpatialData object from a G4X sample dataset.

    This function looks for the following files:

        - ``{xx.HE_DIR!r}/{xx.HE_PATTERN!r}``: H&E images.
        - ``{xx.NUCLEI_DIR!r}/{xx.NUCLEI_PATTERN!r}``: Segmentation files.
        - ``{xx.PROTEIN_DIR!r}/{xx.PROTEIN_PATTERN!r}``: Protein images.
        - ``{xx.TRANSCRIPTS_DIR!r}/{xx.TRANSCRIPTS_PATTERN!r}``: Transcript tables.
        - ``{xx.TABLES_DIR!r}/{xx.TABLE_PATTERN!r}``: Table file.

    Parameters
    ----------
    input_path : str
        Path to input directory containing G4X data
    output_path : str
        Writes/appends to a SpatialData zarr store at this path
    include_he : bool
        Include H&E image if available.
    include_segmentation : bool
        Include segmentation if available.
    include_protein : bool
        Include protein images if available.
    include_transcripts : bool
        Include transcript data if available.
    include_tables : bool
        Include tables if available.
    mode : str
        Mode for creating SpatialData object ('new' or 'append')

    Returns
    -------
    SpatialData
        SpatialData object containing requested data elements
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_zarr_path, str):
        output_zarr_path = Path(output_zarr_path)
        if output_zarr_path.suffix != ".zarr":
            logger.error(f"Output path must end with '.zarr'. Got {output_zarr_path}")
            raise ValueError(f"Output path must end with '.zarr'. Got {output_zarr_path}")

    if mode not in ["append", "overwrite"]:
        msg = f"Invalid mode '{mode}'. Must be one of: 'append', 'overwrite'"
        logger.error(msg)
        raise ValueError(msg)

    if output_zarr_path.exists():
        logger.debug(f"Found existing {output_zarr_path}")
        sdata = SpatialData.read(output_zarr_path)
    else:
        logger.debug(f"Creating new SpatialData object at {output_zarr_path}")
        sdata = SpatialData()
        sdata.write(output_zarr_path)

    # Create progress bar for main steps
    steps = []
    steps.append("H&E") if include_he else None
    steps.append("Segmentation") if include_segmentation else None
    steps.append("Protein Images") if include_protein else None
    steps.append("Transcripts") if include_transcripts else None
    steps.append("Tables") if include_tables else None
    with tqdm(total=len(steps)) as pbar:
        if include_he:
            pbar.set_description(steps[pbar.n])
            _write_he(
                sdata,
                he_dir=G4XKeys.HE_DIR,
                pattern=G4XKeys.HE_PATTERN,
                mode=mode,
                **G4XKeys.HE_IMG2DMODEL_KWARGS,
            )
            pbar.update(1)

        if include_segmentation:
            pbar.set_description(steps[pbar.n])
            _write_segmentation(
                sdata,
                nuclei_dir=G4XKeys.SEGMENTATION_DIR,
                pattern=G4XKeys.SEGMENTATION_PATTERN,
                nuclei_key=G4XKeys.NUCLEI_BOUNDARIES_KEY,
                nuclei_exp_key=G4XKeys.CELL_BOUNDARIES_KEY,
                mode=mode,
                **G4XKeys.SEG_IMG2DMODEL_KWARGS,
            )
            pbar.update(1)

        if include_protein:
            pbar.set_description(steps[pbar.n])
            _write_protein_images(
                sdata,
                protein_dir=G4XKeys.PROTEIN_DIR,
                pattern=G4XKeys.PROTEIN_PATTERN,
                mode=mode,
                **G4XKeys.PROTEIN_IMG2DMODEL_KWARGS,
            )
            pbar.update(1)

        if include_transcripts:
            pbar.set_description(steps[pbar.n])
            _write_transcripts(
                sdata,
                transcripts_dir=G4XKeys.TRANSCRIPTS_DIR,
                pattern=G4XKeys.TRANSCRIPTS_PATTERN,
                coordinates=G4XKeys.TRANSCRIPTS_COORDS,
                feature_key=G4XKeys.TRANSCRIPTS_FEATURE_KEY,
                swap_xy=G4XKeys.TRANSCRIPTS_SWAP_XY,
                mode=mode,
            )
            pbar.update(1)

        if include_tables:
            pbar.set_description(steps[pbar.n])
            _write_table(
                sdata,
                table_path=G4XKeys.TABLE_PATTERN,
                mode=mode,
            )
            pbar.update(1)

    logger.debug("Done!")

    # Read back to enable lazy loading
    sdata = SpatialData.read(output_zarr_path)
    return sdata


def _write_he(
    sdata: SpatialData,
    he_dir: str | None,
    pattern: str,
    mode: str = "append",
    **kwargs,
):
    """
    Write H&E images to SpatialData object. Each H&E image is stored as a separate object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to write to
    he_dir : Union[str, None]
        Path to directory containing H&E images. If None, this step will be skipped.
    pattern : str
        Glob pattern for selecting H&E images.
    mode : str, optional
        Mode for handling existing elements. Options:
        - "append": Skip if element exists (default)
        - "overwrite": Replace if element exists
    kwargs : dict
        Additional arguments passed to Image2DModel.parse()

    Modifies
    -------
    sdata : SpatialData
        SpatialData object with H&E images stored in sdata["{img_name}"] e.g. "h_and_e"
    """
    if he_dir is None:
        logger.debug("H&E skipped...")
        return

    # Get list of H&E images
    he_dir = Path(he_dir)
    if he_dir.is_file():
        he_files = [he_dir]
    else:
        he_files = list(Path(he_dir).glob(pattern))
        if not he_files:
            logger.warning(f"No H&E images found in {he_dir}")
            return
        he_files.sort()

    logger.debug(f"Found {len(he_files)} H&E images")

    # Process each H&E image
    for he_file in tqdm(he_files, desc="Processing H&E images", leave=False):
        # Extract sample ID from filename (e.g., "C02" from "C02_digital_he.jp2")
        logger.debug(f"Processing {he_file}")
        img_key = he_file.stem

        # Check if element exists
        if f"images/{img_key}" in sdata.elements_paths_on_disk():
            if mode == "append":
                logger.debug(f"H&E image '{img_key}' already exists. Skipping...")
                continue
            elif mode == "overwrite":
                logger.debug(f"Deleting existing H&E image '{img_key}'")
                if img_key in sdata:
                    del sdata[img_key]
                sdata.delete_element_from_disk(img_key)

        # Load and process image
        logger.debug(f"Loading H&E image from {he_file}")
        img = imread(str(he_file))
        if len(img.shape) == 4:
            img = img[0]  # [0] to remove extra dimension
        elif len(img.shape) == 3:
            img = img.transpose(1, 2, 0)  # move first dimension to last
        logger.debug(f"H&E image shape: {img.shape}")
        logger.debug(f"H&E image dtype: {img.dtype}")

        # Create Image2DModel and write
        logger.debug(f"Creating Image2DModel for {img_key}")
        sdata[img_key] = Image2DModel.parse(img, **kwargs)
        logger.debug(f"Writing Image2DModel for {img_key}")
        sdata.write_element(img_key)


def _write_segmentation(
    sdata: SpatialData,
    nuclei_dir: str | None,
    pattern: str,
    nuclei_key: str,
    nuclei_exp_key: str,
    mode: str = "append",
    **kwargs,
):
    """
    Write segmentation labels to SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to write to
    nuclei_dir : Union[str, None]
        Path to directory containing nuclei segmentation files.
        If None, this step will be skipped.
    pattern : str
        Glob pattern for selecting nuclei segmentation files.
    nuclei_key : str
        Key for nuclei segmentation array in the NPZ file
    nuclei_exp_key : str
        Key for expanded nuclei segmentation array in the NPZ file
    mode : str, optional
        Mode for handling existing elements. Options:
        - "append": Skip if elements exist (default)
        - "overwrite": Replace if elements exist
    kwargs : dict
        Additional arguments passed to Labels2DModel.parse()

    Modifies
    --------
    sdata : SpatialData
        Adds the following elements:
        - {nuclei_key}: Labels2DModel of nuclei segmentation
        - {nuclei_exp_key}: Labels2DModel of expanded nuclei segmentation
        - {nuclei_key}_shapes: Polygon shapes derived from nuclei segmentation
        - {nuclei_exp_key}_shapes: Polygon shapes derived from expanded segmentation
    """
    if nuclei_dir is None:
        logger.debug("Segmentation skipped...")
        return

    # Get list of nuclei files
    nuclei_dir = Path(nuclei_dir)
    nuclei_file = nuclei_dir / pattern
    if not nuclei_file.exists():
        logger.warning(f"No segmentation files matching {pattern} in {nuclei_dir}")
        return

    # Process each nuclei file
    shapes_seg_key = f"{nuclei_key}_shapes"
    shapes_exp_key = f"{nuclei_exp_key}_shapes"

    # Check if elements exist
    elements = [nuclei_key, nuclei_exp_key, shapes_seg_key, shapes_exp_key]
    elements_paths = [
        f"labels/{nuclei_key}",
        f"labels/{nuclei_exp_key}",
        f"shapes/{shapes_seg_key}",
        f"shapes/{shapes_exp_key}",
    ]

    if mode == "append" and any(p in sdata.elements_paths_on_disk() for p in elements_paths):
        logger.debug("Segmentation already exist. Skipping...")
        return
    elif mode == "overwrite":
        logger.debug("Deleting existing segmentation elements")
        for el in elements:
            if el in sdata:
                del sdata[el]
            if f"labels/{el}" in sdata.elements_paths_on_disk() or f"shapes/{el}" in sdata.elements_paths_on_disk():
                sdata.delete_element_from_disk(el)

    # Load and process segmentation data
    logger.debug(f"Loading segmentation data from {nuclei_file}")
    nuclei_dict = np.load(nuclei_file)
    nuclei_raw = nuclei_dict[nuclei_key]
    nuclei_exp = nuclei_dict[nuclei_exp_key]
    logger.debug(f"Nuclei masks shape: {nuclei_raw.shape}")
    logger.debug(f"Cell masks shape: {nuclei_exp.shape}")

    # Create progress bar for nuclei processing steps
    logger.debug("Converting to Labels2DModel")
    sdata[nuclei_key] = Labels2DModel.parse(nuclei_raw, **kwargs)
    sdata[nuclei_exp_key] = Labels2DModel.parse(nuclei_exp, **kwargs)
    logger.debug("Converting to polygons")
    sdata[shapes_seg_key] = to_polygons(sdata[nuclei_key]).reset_index(drop=True)
    sdata[shapes_exp_key] = to_polygons(sdata[nuclei_exp_key]).reset_index(drop=True)
    logger.debug("Writing elements")
    for element in elements:
        sdata.write_element(element)


def _write_protein_images(
    sdata: SpatialData,
    protein_dir: str | None,
    pattern: str,
    mode: str = "append",
    **kwargs,
):
    """
    Write protein images to SpatialData object. Proteins are stored as channels in a single Image2DModel object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to write to
    protein_dir : Union[str, None]
        Path to directory containing protein images.
        If None, this step will be skipped.
    pattern : str
        Glob pattern for selecting protein images.
    mode : str, optional
        Mode for handling existing elements. Options:
        - "append": Skip if element exists (default)
        - "overwrite": Replace if element exists
    kwargs : dict
        Additional arguments passed to Image2DModel.parse()
    """
    if protein_dir is None:
        logger.debug("Protein skipped...")
        return

    protein_dir = Path(protein_dir)

    # Get list of protein images for this sample
    img_list = list(protein_dir.glob(pattern))
    img_list.sort()

    if not img_list:
        logger.warning(f"No protein images found matching pattern '{pattern}' in {protein_dir}")
        return
    logger.debug(f"Found {len(img_list)} protein images")

    # Check if element exists
    if "images/protein" in sdata.elements_paths_on_disk():
        if mode == "append":
            logger.debug("Protein images already exist. Skipping...")
            return
        elif mode == "overwrite":
            logger.debug("Deleting existing protein images")
            if G4XKeys.PROTEIN_KEY in sdata:
                del sdata[G4XKeys.PROTEIN_KEY]
            sdata.delete_element_from_disk(G4XKeys.PROTEIN_KEY)
    img_list.sort()

    # Get channel names from filenames
    channel_names = [img_file.stem.split("_")[0] for img_file in img_list]

    # Load all images at once with dask imread
    logger.debug("Loading protein images")
    protein_stack = imread(str(protein_dir / pattern))
    logger.debug(f"Images shape: {protein_stack.shape}")

    # Create Image2DModel and write
    logger.debug("Converting to Image2DModel")
    sdata[G4XKeys.PROTEIN_KEY] = Image2DModel.parse(protein_stack, c_coords=channel_names, **kwargs)

    logger.debug("Writing protein images")
    sdata.write_element(G4XKeys.PROTEIN_KEY)


def _write_transcripts(
    sdata: SpatialData,
    transcripts_dir: str | None,
    pattern: str,
    coordinates: dict,
    feature_key: str,
    swap_xy: bool,
    mode: str = "append",
):
    """
    Write transcripts to SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to write to
    transcripts_dir : Union[str, None]
        Path to directory containing transcript tables.
    pattern : str
        Glob pattern for selecting transcript tables.
    coordinates : dict
        Dictionary mapping coordinate column names to standard x,y coordinates
    feature_key : str
        Column name containing transcript feature identifiers
    swap_xy : bool
        Whether to swap the x and y coordinates
    mode : str, optional
        Mode for handling existing element. Options:
        - "append": Skip if element exists (default)
        - "overwrite": Replace if element exists

    Modifies
    --------
    sdata : SpatialData
        Adds a "transcripts" PointsModel containing transcript locations and features
    """
    if transcripts_dir is None:
        logger.debug("Transcripts skipped...")
        return

    if f"points/{G4XKeys.TRANSCRIPTS_KEY}" in sdata.elements_paths_on_disk():
        if mode == "append":
            logger.debug("Transcripts already exist. Skipping...")
            return
        elif mode == "overwrite":
            logger.debug("Deleting existing transcripts")
            if G4XKeys.TRANSCRIPTS_KEY in sdata:
                del sdata[G4XKeys.TRANSCRIPTS_KEY]
            sdata.delete_element_from_disk(G4XKeys.TRANSCRIPTS_KEY)

    transcript_dir = Path(transcripts_dir)
    with tqdm(total=3, desc="Processing transcripts", leave=False) as pbar:
        pbar.set_description("Loading transcripts")

        if pattern.endswith(".csv") or pattern.endswith(".csv.gz"):
            # list files found in transcript_dir
            transcript_files = list(transcript_dir.glob(pattern))
            transcript_files.sort()
            logger.debug(f"Found {len(transcript_files)} transcript files")
            transcripts = dd.read_csv(transcript_files).compute().reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported file type: {transcript_dir / pattern}")
        pbar.update(1)

        if swap_xy:
            transcripts[[coordinates["x"], coordinates["y"]]] = transcripts[[coordinates["y"], coordinates["x"]]]

        pbar.set_description("Converting to PointsModel")
        sdata[G4XKeys.TRANSCRIPTS_KEY] = PointsModel.parse(
            transcripts,
            coordinates=coordinates,
            feature_key=feature_key,
        )
        pbar.update(1)

        pbar.set_description("Writing to disk")
        sdata.write_element(G4XKeys.TRANSCRIPTS_KEY)
        pbar.update(1)


def _write_table(
    sdata: SpatialData,
    table_path: str | None,
    mode: str = "append",
):
    """
    Write tables to SpatialData object.
    """
    if table_path is None:
        logger.debug("Table skipped...")
        return

    adata = read_h5ad(table_path)
    sdata[G4XKeys.TABLE_KEY] = TableModel.parse(adata)

    logger.debug("Writing table to disk")
    sdata.write_element(G4XKeys.TABLE_KEY)


def _deep_update(base_dict, update_dict):
    """
    Recursively update a dictionary with another dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
