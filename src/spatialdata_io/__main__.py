import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import click

# dynamically import all readers and converters (also the experimental ones)
from spatialdata_io import _converters, _readers_file_types, _readers_technologies
from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr
from spatialdata_io.experimental import _converters as _experimental_converters
from spatialdata_io.experimental import (
    _readers_file_types as _experimental_readers_file_types,
)
from spatialdata_io.experimental import (
    _readers_technologies as _experimental_readers_technologies,
)
from spatialdata_io.readers.generic import VALID_IMAGE_TYPES, VALID_SHAPE_TYPES

for func in _readers_technologies + _readers_file_types + _converters:
    module = importlib.import_module("spatialdata_io")
    globals()[func] = getattr(module, func)

for func in _experimental_readers_technologies + _experimental_readers_file_types + _experimental_converters:
    module = importlib.import_module("spatialdata_io.experimental")
    globals()[func] = getattr(module, func)


@click.group()
def cli() -> None:
    """Convert standard technology data formats to SpatialData object.

    Usage:

    python -m spatialdata_io <Command> -i <input> -o <output>

    For help on how to use a specific command, run:

    python -m spatialdata_io <Command> --help
    """


def _input_output_click_options(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to add input and output options to a Click command."""
    func = click.option(
        "--input",
        "-i",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        help="Path to the input file.",
        required=True,
    )(func)
    func = click.option(
        "--output", "-o", type=click.Path(exists=False), help="Path to the output file.", required=True
    )(func)
    return func


@cli.command(name="codex")
@_input_output_click_options
@click.option(
    "--fcs",
    type=bool,
    default=True,
    help="Whether the .fcs file is provided if False a .csv file is expected. [default: True]",
)
def codex_wrapper(input: str, output: str, fcs: bool = True) -> None:
    """Codex conversion to SpatialData."""
    sdata = codex(input, fcs=fcs)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="cosmx")
@_input_output_click_options
@click.option("--dataset-id", type=str, default=None, help="Name of the dataset [default: None]")
@click.option("--transcripts", type=bool, default=True, help="Whether to load transcript information. [default: True]")
def cosmx_wrapper(input: str, output: str, dataset_id: str | None = None, transcripts: bool = True) -> None:
    """Cosmic conversion to SpatialData."""
    sdata = cosmx(input, dataset_id=dataset_id, transcripts=transcripts)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="curio")
@_input_output_click_options
def curio_wrapper(input: str, output: str) -> None:
    """Curio conversion to SpatialData."""
    sdata = curio(input)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="dbit")
@_input_output_click_options
@click.option(
    "--anndata-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to the counts and metadata file. [default: None]",
)
@click.option(
    "--barcode-position",
    type=click.Path(exists=True),
    default=None,
    help="Path to the barcode coordinates file. [default: None]",
)
@click.option("--image-path", type=str, default=None, help="Path to the low resolution image file. [default: None]")
@click.option("--dataset-id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option("--border", type=bool, default=True, help="Value pass internally to _xy2edges. [default: True]")
@click.option("--border-scale", type=float, default=1, help="The factor by which the border is scaled. [default: 1]")
def dbit_wrapper(
    input: str,
    output: str,
    anndata_path: str | None = None,
    barcode_position: str | None = None,
    image_path: str | None = None,
    dataset_id: str | None = None,
    border: bool = True,
    border_scale: float = 1,
) -> None:
    """Conversion of DBit-seq to SpatialData."""
    sdata = dbit(  # type: ignore[name-defined] # noqa: F821
        input,
        anndata_path=anndata_path,
        barcode_position=barcode_position,
        image_path=image_path,
        dataset_id=dataset_id,
        border=border,
        border_scale=border_scale,
    )
    sdata.write(output)


@cli.command(name="iss")
@_input_output_click_options
@click.option(
    "--raw-relative-path", type=click.Path(exists=True), required=True, help="Relative path to raw raster image file."
)
@click.option(
    "--labels-relative-path", type=click.Path(exists=True), required=True, help="Relative path to label image file."
)
@click.option(
    "--h5ad-relative-path",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to counts and metadata file.",
)
@click.option(
    "--instance-key",
    type=str,
    default=None,
    help="Which column of the AnnData table contains the CellID. [default: None]",
)
@click.option("--dataset-id", type=str, default="region", help="Dataset ID [default: region]")
@click.option(
    "--multiscale-image",
    type=bool,
    default=True,
    help="Whether to process the image into a multiscale image [default: True]",
)
@click.option(
    "--multiscale-labels",
    type=bool,
    default=True,
    help="Whether to process the label image into a multiscale image [default: True]",
)
def iss_wrapper(
    input: str,
    output: str,
    raw_relative_path: Path,
    labels_relative_path: Path,
    h5ad_relative_path: Path,
    instance_key: str | None = None,
    dataset_id: str = "region",
    multiscale_image: bool = True,
    multiscale_labels: bool = True,
) -> None:
    """ISS conversion to SpatialData."""
    sdata = iss(  # type: ignore[name-defined] # noqa: F821
        input,
        raw_relative_path,
        labels_relative_path,
        h5ad_relative_path,
        instance_key=instance_key,
        dataset_id=dataset_id,
        multiscale_image=multiscale_image,
        multiscale_labels=multiscale_labels,
    )
    sdata.write(output)


@cli.command(name="mcmicro")
@click.option(
    "--input", "-i", type=click.Path(exists=True), help="Path to the mcmicro project directory.", required=True
)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
def mcmicro_wrapper(input: str, output: str) -> None:
    """Conversion of MCMicro to SpatialData."""
    sdata = mcmicro(input)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="merscope")
@_input_output_click_options
@click.option(
    "--vpt-outputs",
    type=click.Path(exists=True),
    default=None,
    help="Optional argument to specify the path to the Vizgen postprocessing tool. [default: None]",
)
@click.option("--z-layers", type=int, default=3, help="Indices of the z-layers to consider. [default: 3]")
@click.option("--region-name", type=str, default=None, help="Name of the ROI. [default: None]")
@click.option("--slide-name", type=str, default=None, help="Name of the slide/run [default: None]")
@click.option(
    "--backend",
    type=click.Choice(["dask_image", "rioxarray"]),
    default=None,
    help="Either 'dask_image' or 'rioxarray'. [default: None]",
)
@click.option("--transcripts", type=bool, default=True, help="Whether to read transcripts.  [default: True]")
@click.option("--cells-boundaries", type=bool, default=True, help="Whether to read cells boundaries. [default: True]")
@click.option("--cells-table", type=bool, default=True, help="Whether to read cells table.  [default: True]")
@click.option("--mosaic-images", type=bool, default=True, help="Whether to read the mosaic images.  [default: True]")
def merscope_wrapper(
    input: str,
    output: str,
    vpt_outputs: Path | str | dict[str, Any] | None = None,
    z_layers: int | list[int] | None = 3,
    region_name: str | None = None,
    slide_name: str | None = None,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    cells_boundaries: bool = True,
    cells_table: bool = True,
    mosaic_images: bool = True,
) -> None:
    """Merscope conversion to SpatialData."""
    sdata = merscope(  # type: ignore[name-defined] # noqa: F821
        input,
        vpt_outputs=vpt_outputs,
        z_layers=z_layers,
        region_name=region_name,
        slide_name=slide_name,
        backend=backend,
        transcripts=transcripts,
        cells_boundaries=cells_boundaries,
        cells_table=cells_table,
        mosaic_images=mosaic_images,
    )
    sdata.write(output)


@cli.command(name="seqfish")
@_input_output_click_options
@click.option("--load-images", type=bool, default=True, help="Whether to load images. [default: True]")
@click.option("--load-labels", type=bool, default=True, help="Whether to load labels. [default: True]")
@click.option("--load-points", type=bool, default=True, help="Whether to load points. [default: True]")
@click.option("--load-shapes", type=bool, default=True, help="Whether to load shapes. [default: True]")
@click.option("--cells-as-circles", type=bool, default=False, help="Whether to read cells as circles. [default: False]")
@click.option(
    "--rois",
    type=click.IntRange(min=0),
    multiple=True,
    default=None,
    help="Which sections to load. Provide one or more section indices. [default: All sections are loaded]",
)
def seqfish_wrapper(
    input: str,
    output: str,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    load_shapes: bool = True,
    cells_as_circles: bool = False,
    rois: list[int] | None = None,
) -> None:
    """Seqfish conversion to SpatialData."""
    rois = list(rois) if rois else None
    sdata = seqfish(  # type: ignore[name-defined] # noqa: F821
        input,
        load_images=load_images,
        load_labels=load_labels,
        load_points=load_points,
        load_shapes=load_shapes,
        cells_as_circles=cells_as_circles,
        rois=rois,
    )
    sdata.write(output)


@cli.command(name="steinbock")
@_input_output_click_options
@click.option(
    "--labels-kind",
    type=click.Choice(["deepcell", "ilastik"]),
    default="deepcell",
    help="What kind of labels to use. [default: 'deepcell']",
)
def steinbock_wrapper(input: str, output: str, labels_kind: Literal["deepcell", "ilastik"] = "deepcell") -> None:
    """Steinbock conversion to SpatialData."""
    sdata = steinbock(input, labels_kind=labels_kind)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="stereoseq")
@_input_output_click_options
@click.option("--dataset-id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option(
    "--read-square-bin",
    type=bool,
    default=True,
    help="If True, will read the square bin ``{xx.GEF_FILE!r}`` file and build corresponding points element. [default: True]",
)
@click.option(
    "--optional-tif", type=bool, default=False, help="If True, will read ``{xx.TISSUE_TIF!r}`` files. [default: False]"
)
def stereoseq_wrapper(
    input: str,
    output: str,
    dataset_id: str | None = None,
    read_square_bin: bool = True,
    optional_tif: bool = False,
) -> None:
    """Stereoseq conversion to SpatialData."""
    sdata = stereoseq(input, dataset_id=dataset_id, read_square_bin=read_square_bin, optional_tif=optional_tif)  # type: ignore[name-defined] # noqa: F821
    sdata.write(output)


@cli.command(name="visium")
@_input_output_click_options
@click.option("--dataset-id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option(
    "--counts-file",
    type=str,
    default=VisiumKeys.FILTERED_COUNTS_FILE,
    help="Name of the counts file, defaults to ``{vx.FILTERED_COUNTS_FILE!r}``. [default: None]",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the full resolution image. [default: None]",
)
@click.option(
    "--tissue-positions-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the tissue positions file. [default: None]",
)
@click.option(
    "--scalefactors-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the scalefactors file. [default: None]",
)
def visium_wrapper(
    input: str,
    output: str,
    dataset_id: str | None = None,
    counts_file: str = VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file: str | Path | None = None,
    tissue_positions_file: str | Path | None = None,
    scalefactors_file: str | Path | None = None,
) -> None:
    """Visium conversion to SpatialData."""
    sdata = visium(  # type: ignore[name-defined] # noqa: F821
        input,
        dataset_id=dataset_id,
        counts_file=counts_file,
        fullres_image_file=fullres_image_file,
        tissue_positions_file=tissue_positions_file,
        scalefactors_file=scalefactors_file,
    )
    sdata.write(output)


@cli.command(name="visium-hd")
@_input_output_click_options
@click.option("--dataset-id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option(
    "--filtered-counts-file",
    type=bool,
    default=True,
    help="It sets the value of `counts_file` to ``{vx.FILTERED_COUNTS_FILE!r}`` (when `True`) or to``{vx.RAW_COUNTS_FILE!r}`` (when `False`). [default: True]",
)
@click.option(
    "--bin-size",
    type=int,
    multiple=True,
    default=None,
    help="When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the available bin sizes. [default: None]",
)
@click.option(
    "--bins-as-squares",
    type=bool,
    default=True,
    help="If true, bins are represented as squares otherwise as circles. [default: True]",
)
@click.option(
    "--fullres-image-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the full resolution image. [default: None]",
)
@click.option(
    "--load-all-images",
    type=bool,
    default=False,
    help="If `False`, load only the full resolution, high resolution, and low resolution images. If `True`, also the following images: ``{vx.IMAGE_CYTASSIST!r}``. [default: False]",
)
@click.option(
    "--annotate-table-by-labels",
    type=bool,
    default=False,
    help="If true, annotates the table by labels. [default: False]",
)
def visium_hd_wrapper(
    input: str,
    output: str,
    dataset_id: str | None = None,
    filtered_counts_file: bool = True,
    bin_size: int | list[int] | None = None,
    bins_as_squares: bool = True,
    fullres_image_file: str | Path | None = None,
    load_all_images: bool = False,
    annotate_table_by_labels: bool = False,
) -> None:
    """Visium HD conversion to SpatialData."""
    sdata = visium_hd(  # type: ignore[name-defined] # noqa: F821
        path=input,
        dataset_id=dataset_id,
        filtered_counts_file=filtered_counts_file,
        bin_size=bin_size,
        bins_as_squares=bins_as_squares,
        fullres_image_file=fullres_image_file,
        load_all_images=load_all_images,
        annotate_table_by_labels=annotate_table_by_labels,
    )
    sdata.write(output)


@cli.command(name="xenium")
@_input_output_click_options
@click.option("--cells-boundaries", type=bool, default=True, help="Whether to read cells boundaries. [default: True]")
@click.option(
    "--nucleus-boundaries", type=bool, default=True, help="Whether to read Nucleus boundaries. [default: True]"
)
@click.option("--cells-as-circles", type=bool, default=None, help="Whether to read cells as circles. [default: None]")
@click.option("--cells-labels", type=bool, default=True, help="Whether to read cells labels (raster). [default: True]")
@click.option(
    "--nucleus-labels", type=bool, default=True, help="Whether to read nucleus labels (raster). [default: True]"
)
@click.option("--transcripts", type=bool, default=True, help="Whether to read transcripts. [default: True]")
@click.option("--morphology-mip", type=bool, default=True, help="Whether to read morphology mip image. [default: True]")
@click.option(
    "--morphology-focus", type=bool, default=True, help="Whether to read morphology focus image. [default: True]"
)
@click.option(
    "--aligned-images",
    type=bool,
    default=True,
    help="Whether to parse additional H&E or IF aligned images. [default: True]",
)
@click.option(
    "--cells-table",
    type=bool,
    default=True,
    help="Whether to read cells annotations in the AnnData table. [default: True]",
)
@click.option("--n-jobs", type=int, default=1, help="Number of jobs. [default: 1]")
def xenium_wrapper(
    input: str,
    output: str,
    *,
    cells_boundaries: bool = True,
    nucleus_boundaries: bool = True,
    cells_as_circles: bool | None = None,
    cells_labels: bool = True,
    nucleus_labels: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    aligned_images: bool = True,
    cells_table: bool = True,
    n_jobs: int = 1,
) -> None:
    """Xenium conversion to SpatialData."""
    sdata = xenium(  # type: ignore[name-defined] # noqa: F821
        input,
        cells_boundaries=cells_boundaries,
        nucleus_boundaries=nucleus_boundaries,
        cells_as_circles=cells_as_circles,
        cells_labels=cells_labels,
        nucleus_labels=nucleus_labels,
        transcripts=transcripts,
        morphology_mip=morphology_mip,
        morphology_focus=morphology_focus,
        aligned_images=aligned_images,
        cells_table=cells_table,
        n_jobs=n_jobs,
    )
    sdata.write(output)


@cli.command(name="macsima")
@_input_output_click_options
@click.option(
    "--filter-folder-names",
    type=str,
    multiple=True,
    default=None,
    help="List of folder names to filter out when parsing multiple folders. [default: None]",
)
@click.option(
    "--subset",
    type=int,
    default=None,
    help="Subset the image to the first 'subset' pixels in x and y dimensions. [default: None]",
)
@click.option(
    "--c-subset", type=int, default=None, help="Subset the image to the first 'c-subset' channels. [default: None]"
)
@click.option(
    "--max-chunk-size", type=int, default=1024, help="Maximum chunk size for x and y dimensions. [default: 1024]"
)
@click.option("--c-chunks-size", type=int, default=1, help="Chunk size for c dimension. [default: 1]")
@click.option("--multiscale", type=bool, default=True, help="Whether to create a multiscale image. [default: True]")
@click.option(
    "--transformations",
    type=bool,
    default=True,
    help="Whether to add a transformation from pixels to microns to the image. [default: True]",
)
@click.option(
    "--scale-factors",
    type=int,
    multiple=True,
    default=None,
    help="Scale factors to use for downsampling. If None, scale factors are calculated based on image size. [default: None]",
)
@click.option(
    "--default-scale-factor", type=int, default=2, help="Default scale factor to use for downsampling. [default: 2]"
)
@click.option(
    "--nuclei-channel-name",
    type=str,
    default="DAPI",
    help="Common string of the nuclei channel to separate nuclei from other channels. [default: 'DAPI']",
)
@click.option(
    "--split-threshold-nuclei-channel",
    type=int,
    default=2,
    help="Threshold for splitting nuclei channels. [default: 2]",
)
@click.option(
    "--skip-rounds",
    type=int,
    multiple=True,
    default=None,
    help="List of round numbers to skip when parsing the data. [default: None]",
)
@click.option(
    "--include-cycle-in-channel-name",
    type=bool,
    default=False,
    help="Whether to include the cycle number in the channel name. [default: False]",
)
def macsima_wrapper(
    input: str,
    output: str,
    *,
    filter_folder_names: list[str] | None = None,
    subset: int | None = None,
    c_subset: int | None = None,
    max_chunk_size: int = 1024,
    c_chunks_size: int = 1,
    multiscale: bool = True,
    transformations: bool = True,
    scale_factors: list[int] | None = None,
    default_scale_factor: int = 2,
    nuclei_channel_name: str = "DAPI",
    split_threshold_nuclei_channel: int | None = 2,
    skip_rounds: list[int] | None = None,
    include_cycle_in_channel_name: bool = False,
) -> None:
    """Read MACSima formatted dataset and convert to SpatialData."""
    sdata = macsima(  # type: ignore[name-defined] # noqa: F821
        path=input,
        filter_folder_names=filter_folder_names,
        subset=subset,
        c_subset=c_subset,
        max_chunk_size=max_chunk_size,
        c_chunks_size=c_chunks_size,
        multiscale=multiscale,
        transformations=transformations,
        scale_factors=scale_factors,
        default_scale_factor=default_scale_factor,
        nuclei_channel_name=nuclei_channel_name,
        split_threshold_nuclei_channel=split_threshold_nuclei_channel,
        skip_rounds=skip_rounds,
        include_cycle_in_channel_name=include_cycle_in_channel_name,
    )
    sdata.write(output)


@cli.command(name="generic")
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help=f"Path to the image/shapes input file. Supported extensions: {VALID_IMAGE_TYPES + VALID_SHAPE_TYPES}",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False),
    required=True,
    help="Path to zarr store to write to. If it does not exist yet, create new zarr store from input",
)
@click.option("--name", "-n", type=str, help="name of the element to be stored")
@click.option(
    "--data-axes",
    type=str,
    help="Axes of the data for image files. Valid values are permutations of 'cyx' and 'czyx'.",
)
@click.option(
    "--coordinate-system",
    "-c",
    type=str,
    help="Coordinate system in spatialdata object to which an element should belong",
)
def read_generic_wrapper(
    input: str,
    output: str,
    name: str | None = None,
    data_axes: str | None = None,
    coordinate_system: str | None = None,
) -> None:
    """Read generic data to SpatialData."""
    if data_axes is not None and "".join(sorted(data_axes)) not in ["cxy", "cxyz"]:
        raise ValueError("data_axes must be a permutation of 'cyx' or 'czyx'.")
    generic_to_zarr(input=input, output=output, name=name, data_axes=data_axes, coordinate_system=coordinate_system)


if __name__ == "__main__":
    cli()
