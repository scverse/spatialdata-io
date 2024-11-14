from os.path import exists
import click
from pathlib import Path
from typing import Any, Union, Literal, Optional

from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr
from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.cosmx import cosmx
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.dbit import dbit
from spatialdata_io.readers.iss import iss
from spatialdata_io.readers.mcmicro import mcmicro
from spatialdata_io.readers.merscope import merscope
from spatialdata_io.readers.seqfish import seqfish
from spatialdata_io.readers.steinbock import steinbock
from spatialdata_io.readers.stereoseq import stereoseq
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.visium_hd import visium_hd
from spatialdata_io.readers.xenium import xenium

@click.group()
def cli():
    """
    Convert standard technology data formats to SpatialData object.

    Usage:

    python -m spatialdata-io <Command> -i <input> -o <output>

    For help on how to use a specific command, run:

    python -m spatialdata-io <Command> --help
    """


@cli.command(name="codex")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--fcs", type=bool, default=True,
              help="Whether the .fcs file is provided if False a .csv file is expected. [default: True]")
def codex_wrapper(
    input: str | Path,
    output: str | Path,
    fcs=True) -> None:
    """Codex conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = codex(input,
                  fcs=fcs)
    sdata.write(output)


@cli.command(name="cosmx")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--dataset_id", type=str, default=None, help="Name of the dataset [default: None]")
@click.option("--transcripts", type=bool, default=True, help="Whether to load transcript information. [default: True]")
def cosmx_wrapper(
    input: str | Path,
    output: str | Path,
    dataset_id: Optional[str] = None,
    transcripts: bool = True) -> None:
    """Cosmic conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = cosmx(input,
                  dataset_id=dataset_id,
                  transcripts=transcripts)
    sdata.write(output)


@cli.command(name="curio")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
def curio_wrapper(
    input: str | Path,
    output: str | Path) -> None:
    """Curio conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = curio(input)
    sdata.write(output)


@cli.command(name="dbit")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--anndata_path", type=click.Path(exists=True), default=None, help="Path to the counts and metadata file. [default: None]")
@click.option("--barcode_position", type=click.Path(exists=True), default=None, help="Path to the barcode coordinates file. [default: None]")
@click.option("--image_path", type=str, default=None, help="Path to the low resolution image file. [default: None]")
@click.option("--dataset_id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option("--border", type=bool, default=True, help="Value pass internally to _xy2edges. [default: True]")
@click.option("--border_scale", type=float, default=1, help="The factor by which the border is scaled. [default: 1]")
def dbit_wrapper(
    input: Optional[str | Path] = None,
    output: Optional[str | Path] = None,
    anndata_path: Optional[str] = None,
    barcode_position: Optional[str] = None,
    image_path: Optional[str] = None,
    dataset_id: Optional[str] = None,
    border: bool = True,
    border_scale: float = 1) -> None:
    """DBiT conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = dbit(input,
                 anndata_path=anndata_path,
                 barcode_position=barcode_position,
                 image_path=image_path,
                 dataset_id=dataset_id,
                 border=border,
                 border_scale=border_scale)
    sdata.write(output)


@cli.command(name="iss")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--raw_relative_path", type=click.Path(exists=True), required=True,
              help="Relative path to raw raster image file.")
@click.option("--labels_relative_path", type=click.Path(exists=True), required=True,
              help="Relative path to label image file.")
@click.option("--h5ad_relative_path", type=click.Path(exists=True), required=True,
              help="Relative path to counts and metadata file.")
@click.option("--instance_key", type=str, default=None,
              help="Which column of the AnnData table contains the CellID. [default: None]")
@click.option("--dataset_id", type=str, default="region", help="Dataset ID [default: region]")
@click.option("--multiscale_image", type=bool, default=True,
              help="Whether to process the image into a multiscale image [default: True]")
@click.option("--multiscale_labels", type=bool, default=True,
              help="Whether to process the label image into a multiscale image [default: True]")
def iss_wrapper(
    input: str | Path,
    output: str | Path,
    raw_relative_path: str | Path,
    labels_relative_path: str | Path,
    h5ad_relative_path: str | Path,
    instance_key: str | None = None,
    dataset_id: str = "region",
    multiscale_image: bool = True,
    multiscale_labels: bool = True) -> None:
    """ISS conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = iss(
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
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the mcmicro project directory.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
def mcmicro_wrapper(input: str | Path,
                    output: str | Path) -> None:
    """MCMicro conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = mcmicro(input)
    sdata.write(output)


@cli.command(name="merscope")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--vpt_outputs", type=click.Path(exists=True), default=None,
              help="Optional argument to specify the path to the Vizgen postprocessing tool. [default: None]")
@click.option("--z_layers", type=int, default=3, help="Indices of the z-layers to consider. [default: 3]")
@click.option("--region_name", type=str, default=None, help="Name of the ROI. [default: None]")
@click.option("--slide_name", type=str, default=None, help="Name of the slide/run [default: None]")
@click.option("--backend", type=click.Choice(['dask_image', 'rioxarray']), default=None,
              help="Either 'dask_image' or 'rioxarray'. [default: None]")
@click.option("--transcripts", type=bool, default=True, help="Whether to read transcripts.  [default: True]")
@click.option("--cells_boundaries", type=bool, default=True, help="Whether to read cells boundaries. [default: True]")
@click.option("--cells_table", type=bool, default=True, help="Whether to read cells table.  [default: True]")
@click.option("--mosaic_images", type=bool, default=True, help="Whether to read the mosaic images.  [default: True]")
def merscope_wrapper(
    input: str | Path,
    output: str | Path,
    vpt_outputs: Path | str | dict[str, Any] | None = None,
    z_layers: int | list[int] | None = 3,
    region_name: str | None = None,
    slide_name: str | None = None,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    cells_boundaries: bool = True,
    cells_table: bool = True,
    mosaic_images: bool = True) -> None:
    """Merscope conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = merscope(
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
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--load_images", type=bool, default=True, help="Whether to load images. [default: True]")
@click.option("--load_labels", type=bool, default=True, help="Whether to load labels. [default: True]")
@click.option("--load_points", type=bool, default=True, help="Whether to load points. [default: True]")
@click.option("--sections", type=int, default=None,
              help="Which sections to load. [default: 'All of the sections are loaded']")
def seqfish_wrapper(
    input: str | Path,
    output: str | Path,
    load_images: bool = True,
    load_labels: bool = True,
    load_points: bool = True,
    sections: list[int] | None = None) -> None:
    """Seqfish conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = seqfish(input,
                    load_images=load_images,
                    load_labels=load_labels,
                    load_points=load_points,
                    sections=sections)
    sdata.write(output)


@cli.command(name="steinbock")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--labels_kind", type=click.Choice(['deepcell', 'ilastik']), default="deepcell",
              help="What kind of labels to use. [default: 'deepcell']")
def steinbock_wrapper(
    input: str | Path,
    output: str | Path,
    labels_kind: Literal["deepcell", "ilastik"] = "deepcell") -> None:
    """Steinbock conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = steinbock(input,
                      labels_kind=labels_kind)
    sdata.write(output)


@cli.command(name="stereoseq")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--dataset_id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option("--read_square_bin", type=bool, default=True,
              help="f True, will read the square bin ``{xx.GEF_FILE!r}`` file and build corresponding points element. [default: True]")
@click.option("--optional_tif", type=bool, default=False,
              help="If True, will read ``{xx.TISSUE_TIF!r}`` files. [default: False]")
def stereoseq_wrapper(
    input: str | Path,
    output: str | Path,
    dataset_id: Union[str, None] = None,
    read_square_bin: bool = True,
    optional_tif: bool = False) -> None:
    """Stereoseq conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = stereoseq(input,
                      dataset_id=dataset_id,
                      read_square_bin=read_square_bin,
                      optional_tif=optional_tif)
    sdata.write(output)


@cli.command(name="visium")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--dataset_id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option("--counts_file", type=str, default=VisiumKeys.FILTERED_COUNTS_FILE,
              help="Name of the counts file, defaults to ``{vx.FILTERED_COUNTS_FILE!r}``. [default: None]")
@click.option("--fullres_image_file", type=click.Path(exists=True), default=None,
              help="Path to the full resolution image. [default: None]")
@click.option("--tissue_positions_file", type=click.Path(exists=True), default=None,
              help="Path to the tissue positions file. [default: None]")
@click.option("--scalefactors_file", type=click.Path(exists=True), default=None,
              help="Path to the scalefactors file. [default: None]")
def visium_wrapper(
    input: str | Path,
    output: str | Path,
    dataset_id: str | None = None,
    counts_file: str = VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file: str | Path | None = None,
    tissue_positions_file: str | Path | None = None,
    scalefactors_file: str | Path | None = None) -> None:
    """Visium conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = visium(
        input,
        dataset_id=dataset_id,
        counts_file=counts_file,
        fullres_image_file=fullres_image_file,
        tissue_positions_file=tissue_positions_file,
        scalefactors_file=scalefactors_file,
    )
    sdata.write(output)


@cli.command(name="visium_hd")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--dataset_id", type=str, default=None, help="Dataset ID. [default: None]")
@click.option("--filtercounts_file", type=bool, default=True,
              help="It sets the value of `counts_file` to ``{vx.FILTERED_COUNTS_FILE!r}`` (when `True`) or to``{vx.RAW_COUNTS_FILE!r}`` (when `False`). [default: True]")
@click.option("--bin_size", type=int, default=None,
              help="When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the available bin sizes. [default: None]")
@click.option("--bins_s_squares", type=bool, default=True,
              help="If true bins are represented as squares otherwise as circles. [default: True]")
@click.option("--fullres_image_file", type=click.Path(exists=True), default=None,
              help="Path to the full resolution image. [default: None]")
@click.option("--load_all_images", type=bool, default=False,
              help="If `False`, load only the full resolution, high resolution and low resolution images. If `True`, also the following images: ``{vx.IMAGE_CYTASSIST!r}``. [default: False]")
def visium_hd_wrapper(
    input: str | Path,
    output: str | Path,
    dataset_id: str | None = None,
    filtered_counts_file: bool = True,
    bin_size: int | list[int] | None = None,
    bins_as_squares: bool = True,
    fullres_image_file: str | Path | None = None,
    load_all_images: bool = False) -> None:
    """Visium HD conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = visium_hd(
        input,
        dataset_id=dataset_id,
        filtered_counts_file=filtered_counts_file,
        bin_size=bin_size,
        bins_s_squares=bins_as_squares,
        fullres_image_file=fullres_image_file,
        load_all_images=load_all_images,
    )
    sdata.write(output)


@cli.command(name="xenium")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--cells_boundaries", type=bool, default=True, help="Whether to read cells boundaries. [default: True]")
@click.option("--nucleus_boundaries", type=bool, default=True,
              help="Whether to read Nucleus boundaries. [default: True]")
@click.option("--cells_as_circles", type=bool, default=None, help="Whether to read cells as circles. [default: None]")
@click.option("--cells_labels", type=bool, default=True, help="Whether to read cells labels (raster). [default: True]")
@click.option("--nucleus_labels", type=bool, default=True,
              help="Whether to read nucleus labels (raster). [default: True]")
@click.option("--transcripts", type=bool, default=True, help="Whether to read transcripts. [default: True]")
@click.option("--morphology_mip", type=bool, default=True, help="Whether to read ,orphology mip image. [default: True]")
@click.option("--morphology_focus", type=bool, default=True,
              help="Whether to read morphology focus image. [default: True]")
@click.option("--aligned_images", type=bool, default=True,
              help="Whether to parse additional H&E or IF aligned images. [default: True]")
@click.option("--cells_table", type=bool, default=True,
              help="Whether to read cells annotations in the AnnData table. [default: True]")
@click.option("--n_jobs", type=int, default=1, help="Number of jobs. [default: 1]")
def xenium_wrapper(
    input: str | Path,
    output: str | Path,
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
    n_jobs: int = 1) -> None:
    """Xenium conversion to SpatialData"""
    # Make sure output path is .zarr file
    if not output.endswith(".zarr"):
        raise ValueError("Output path must be a .zarr file.")
    sdata = xenium(
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


@cli.command(name="ReadGeneric")
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Path to the input file.")
@click.option("--filetype", "-t", type=click.Choice(["shape", "image"]), required=True,
              help='Type of the element to store. Can be "shape" or "image". If shape, input must be .geojson')
@click.option("--name", "-n", type=str, help="name of the element to be stored")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Path to zarr store to write to. If it does not exist yet, create new zarr store from input")
@click.option("--coordinate_system", "-c", type=str,
              help="Coordinate system in spatialdata object to which an element should belong")
@click.option("--geometry", "-g", type=click.Choice([0, 3, 6]),
              help="Geometry of shapes element. 0: Circles, 3: Polygon, 6: MultiPolygon")
@click.option("--radius", "-r", type=int, help="Radius of shapes element if geometry is circle.")
def read_generic_wrapper(
    input: str | Path,
    filetype: Literal["shape", "image"],
    name: str,
    output: str | Path,
    coordinate_system: Optional[str] = None,
    geometry: Literal[0, 3, 6] = 0,
    radius: int | None = None) -> None:
    """Read generic data to SpatialData"""
    if not coordinate_system:
        coordinate_system = "global"
    if geometry == 0:
        if not radius:
            raise ValueError("Radius must be provided if geometry is circle")
    else:
        radius = None

    sdata = generic_to_zarr(input,
                            filetype,
                            name,
                            output,
                            coordinate_system=coordinate_system,
                            geometry=geometry,
                            radius=radius)
    sdata.write(output)


if __name__ == "__main__":
    cli()

