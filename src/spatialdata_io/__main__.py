from os.path import exists

import click

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
def codex_wrapper(input, output, fcs=True):
    """Codex conversion to SpatialData"""
    sdata = codex(input, fcs)
    sdata.write(output)


@cli.command(name="cosmx")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--dataset_id", type=str, default=None, help="Name of the dataset [default: None]")
@click.option("--transcripts", type=bool, default=True, help="Whether to load transcript information. [default: True]")
def cosmx_wrapper(input, output, dataset_id=None, transcripts=True):
    """Cosmic conversion to SpatialData"""
    sdata = cosmx(input, dataset_id, transcripts)
    sdata.write(output)


@cli.command(name="curio")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
def curio_wrapper(input, output):
    """Curio conversion to SpatialData"""
    sdata = curio(input)
    sdata.write(output)


@cli.command(name="dbit")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--pattern", type=str, default=None, help="Regex pattern to match files. [default: None]")
@click.option("--key", type=str, default=None,
              help="String to match in the path or path-specific path. [default: None]")
@click.option("--path_specific", type=click.Path(exists=True), default=None,
              help="Path to the file if it is not in the main directory. [default: None]")
@click.option("--optional_arg", type=bool, default=False,
              help="User specify if file is mandatory or optional. [default: False]")
def dbit_wrapper(input, output, pattern, key, path_specific=None, optional_arg=False):
    """DBiT conversion to SpatialData"""
    sdata = dbit(input, pattern, key, path_specific, optional_arg)
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
    input,
    output,
    raw_relative_path,
    labels_relative_path,
    h5ad_relative_path,
    instance_key=None,
    dataset_id="region",
    multiscale_image=True,
    multiscale_labels=True,
):
    """ISS conversion to SpatialData"""
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
def mcmicro_wrapper(input, output):
    """MCMicro conversion to SpatialData"""
    sdata = mcmicro(input)
    sdata.write(output)


@cli.command(name="merscope")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--vpt_output", type=click.Path(exists=True), default=None,
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
    input,
    output,
    vpt_output=None,
    z_layers=3,
    region_name=None,
    slide_name=None,
    backend=None,
    transcripts=True,
    cells_boundaries=True,
    cells_table=True,
    mosaic_images=True,
):
    """Merscope conversion to SpatialData"""
    sdata = merscope(
        input,
        vpt_output=vpt_output,
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
def seqfish_wrapper(input, output, load_images=True, load_labels=True, load_points=True, sections=None):
    """Seqfish conversion to SpatialData"""
    sdata = seqfish(input, load_images=load_images, load_labels=load_labels, load_points=load_points, sections=sections)
    sdata.write(output)


@cli.command(name="steinbock")
@click.option("--input", "-i", type=click.Path(exists=True), help="Path to the directory containing the data.",
              required=True)
@click.option("--output", "-o", type=click.Path(), help="Path to the output.zarr file.", required=True)
@click.option("--labels_kind", type=click.Choice(['deepcell', 'ilastik']), default="deepcell",
              help="What kind of labels to use. [default: 'deepcell']")
def steinbock_wrapper(input, output, labels_kind="deepcell"):
    """Steinbock conversion to SpatialData"""
    sdata = steinbock(input, labels_kind=labels_kind)
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
def stereoseq_wrapper(input, output, dataset_id=None, read_square_bin=True, optional_tif=False):
    """Stereoseq conversion to SpatialData"""
    sdata = stereoseq(input, dataset_id=dataset_id, read_square_bin=read_square_bin, optional_tif=optional_tif)
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
    input,
    output,
    dataset_id=None,
    counts_file=VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file=None,
    tissue_positions_file=None,
    scalefactors_file=None,
):
    """Visium conversion to SpatialData"""
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
    input,
    output,
    dataset_id=None,
    filtercounts_file=True,
    bin_size=None,
    bins_s_squares=True,
    fullres_image_file=None,
    load_all_images=False,
):
    """Visium HD conversion to SpatialData"""
    sdata = visium_hd(
        input,
        dataset_id=dataset_id,
        filtercounts_file=filtercounts_file,
        bin_size=bin_size,
        bins_s_squares=bins_s_squares,
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
    input,
    output,
    cells_boundaries=True,
    nucleus_boundaries=True,
    cells_as_circles=None,
    cells_labels=True,
    nucleus_labels=True,
    transcripts=True,
    morphology_mip=True,
    morphology_focus=True,
    aligned_images=True,
    cells_table=True,
    n_jobs=1,
):
    """Xenium conversion to SpatialData"""
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


if __name__ == "__main__":
    cli()


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
def read_generic_wrapper(input, filetype, name, output, coordinate_system, geometry, radius):
    """Read generic data to SpatialData"""
    if not coordinate_system:
        coordinate_system = "global"
    if geometry == 0:
        if not radius:
            raise ValueError("Radius must be provided if geometry is circle")
    else:
        radius = None
    sdata = generic_to_zarr(input, filetype, name, output, coordinate_system, geometry, radius)
    sdata.write(output)
