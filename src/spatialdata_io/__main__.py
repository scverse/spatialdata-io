from os.path import exists
from random import choice
import click
import os
from pathlib import Path
from pkg_resources import require
from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io import __readers__
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
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr


@click.group()
def cli():
    """
    Convert standard technology data formats to SpatialData object.

    Usage:

    python -m spatialdata-io <Command> -i <input> -o <output>

    For help on how to use a specific command, run:

    python -m spatialdata-io <Command> --help
    """
    pass


@cli.command(name="codex")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--fcs', type=bool, default=True, help='FCS')
def codex_wrapper(input,
                  output,
                  fcs=True):
    """Codex conversion to SpatialData"""
    sdata = codex(input, fcs)
    sdata.write(output)


@cli.command(name="cosmx")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--dataset_id', type=str, default=None, help='Dataset ID')
@click.option('--transcripts', type=bool, default=True, help='Transcripts')
def cosmx_wrapper(input,
                  output,
                  dataset_id=None,
                  transcripts=True):
    """Cosmic conversion to SpatialData"""
    sdata = cosmx(input,
                  dataset_id,
                  transcripts)
    sdata.write(output)


@cli.command(name="curio")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
def curio_wrapper(input,
                  output):
    """Curio conversion to SpatialData"""
    sdata = curio(input)
    sdata.write(output)


@cli.command(name="dbit")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--pattern', type=str, default=None, help='Pattern')
@click.option('--key', type=str, default=None, help='Key')
@click.option('--path_specific', type=bool, default=None, help='Path specific')
@click.option('--optional_arg', type=bool, default=False, help='Optional argument')
def dbit_wrapper(input,
                 output,
                 pattern,
                 key,
                 path_specific=None,
                 optional_arg=False):
    """DBiT conversion to SpatialData"""
    sdata = dbit(input,
                 pattern,
                 key,
                 path_specific,
                 optional_arg)
    sdata.write(output)


@cli.command(name="iss")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--instance_key', type=str, default=None, help='Instance key')
@click.option('--dataset_id', type=str, default='region', help='Dataset ID')
@click.option('--multiscale_image', type=bool, default=True, help='Multiscale image')
@click.option('--multiscale_labels', type=bool, default=True, help='Multiscale labels')
def iss_wrapper(input,
                output,
                raw_relative_path,
                labels_relative_path,
                h5ad_relative_path,
                instance_key=None,
                dataset_id="region",
                multiscale_image=True,
                multiscale_labels=True):
    """ISS conversion to SpatialData"""
    sdata = iss(input,
                raw_relative_path,
                labels_relative_path,
                h5ad_relative_path,
                instance_key=instance_key,
                dataset_id=dataset_id,
                multiscale_image=multiscale_image,
                multiscale_labels=multiscale_labels)
    sdata.write(output)


@cli.command(name="mcmicro")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
def mcmicro_wrapper(input, output):
    """MCMicro conversion to SpatialData"""
    sdata = mcmicro(input)
    sdata.write(output)


@cli.command(name="merscope")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--vpt_output', type=str, default=None, help='VPT output')
@click.option('--z_layers', type=int, default=3, help='Z layers')
@click.option('--region_name', type=str, default=None, help='Region name')
@click.option('--slide_name', type=str, default=None, help='Slide name')
@click.option('--backend', type=str, default=None, help='Backend')
@click.option('--transcripts', type=bool, default=True, help='Transcripts')
@click.option('--cells_boundaries', type=bool, default=True, help='Cells boundaries')
@click.option('--cells_table', type=bool, default=True, help='Cells table')
@click.option('--mosaic_images', type=bool, default=True, help='Mosaic images')
def merscope_wrapper(input,
                     output,
                     vpt_output=None,
                     z_layers=3,
                     region_name=None,
                     slide_name=None,
                     backend=None,
                     transcripts=True,
                     cells_boundaries=True,
                     cells_table=True,
                     mosaic_images=True):
    """Merscope conversion to SpatialData"""
    sdata = merscope(input,
                     vpt_output=vpt_output,
                     z_layers=z_layers,
                     region_name=region_name,
                     slide_name=slide_name,
                     backend=backend,
                     transcripts=transcripts,
                     cells_boundaries=cells_boundaries,
                     cells_table=cells_table,
                     mosaic_images=mosaic_images)
    sdata.write(output)


@cli.command(name="seqfish")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--load_images', type=bool, default=True, help='Load images')
@click.option('--load_labels', type=bool, default=True, help='Load labels')
@click.option('--load_points', type=bool, default=True, help='Load points')
@click.option('--sections', type=int, default=None, help='Sections')
def seqfish_wrapper(input,
                    output,
                    load_images=True,
                    load_labels=True,
                    load_points=True,
                    sections=None):
    """Seqfish conversion to SpatialData"""
    sdata = seqfish(input,
                    load_images=load_images,
                    load_labels=load_labels,
                    load_points=load_points,
                    sections=sections)
    sdata.write(output)


@cli.command(name="steinbock")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--labels_kind', type=str, default='deepcell', help='Labels kind')
def steinbock_wrapper(input,
                      output,
                      labels_kind="deepcell"):
    """Steinbock conversion to SpatialData"""
    sdata = steinbock(input,
                      labels_kind=labels_kind)
    sdata.write(output)


@cli.command(name="stereoseq")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--dataset_id', type=str, default=None, help='Dataset ID')
@click.option('--read_square_bin', type=bool, default=True, help='Read square bin')
@click.option('--optional_tif', type=bool, default=False, help='Optional TIF')
def stereoseq_wrapper(input,
                      output,
                      dataset_id=None,
                      read_square_bin=True,
                      optional_tif=False):
    """Stereoseq conversion to SpatialData"""
    sdata = stereoseq(input,
                      dataset_id=dataset_id,
                      read_square_bin=read_square_bin,
                      optional_tif=optional_tif)
    sdata.write(output)


@cli.command(name="visium")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--dataset_id', type=str, default=None, help='Dataset ID')
@click.option('--counts_file', type=str, default=VisiumKeys.FILTERED_COUNTS_FILE, help='Counts file')
@click.option('--fullres_image_file', type=str, default=None, help='Fullres image file')
@click.option('--tissue_positions_file', type=str, default=None, help='Tissue positions file')
@click.option('--scalefactors_file', type=str, default=None, help='Scalefactors file')
def visium_wrapper(input,
                   output,
                   dataset_id=None,
                   counts_file=VisiumKeys.FILTERED_COUNTS_FILE,
                   fullres_image_file=None,
                   tissue_positions_file=None,
                   scalefactors_file=None):
    """Visium conversion to SpatialData"""
    sdata = visium(input,
                   dataset_id=dataset_id,
                   counts_file=counts_file,
                   fullres_image_file=fullres_image_file,
                   tissue_positions_file=tissue_positions_file,
                   scalefactors_file=scalefactors_file)
    sdata.write(output)


@cli.command(name="visium_hd")
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--dataset_id', type=str, default=None, help='Dataset ID')
@click.option('--filtercounts_file', type=bool, default=True, help='Filtercounts file')
@click.option('--bin_size', type=int, default=None, help='Bin size')
@click.option('--bins_s_squares', type=bool, default=True, help='Bins s squares')
@click.option('--fullres_image_file', type=str, default=None, help='Fullres image file')
@click.option('--load_all_images', type=bool, default=False, help='Load all images')
def visium_hd_wrapper(input,
                      output,
                      dataset_id=None,
                      filtercounts_file=True,
                      bin_size=None,
                      bins_s_squares=True,
                      fullres_image_file=None,
                      load_all_images=False):
    """Visium HD conversion to SpatialData"""
    sdata = visium_hd(input,
                      dataset_id=dataset_id,
                      filtercounts_file=filtercounts_file,
                      bin_size=bin_size,
                      bins_s_squares=bins_s_squares,
                      fullres_image_file=fullres_image_file,
                      load_all_images=load_all_images)
    sdata.write(output)


@cli.command(name='xenium')
@click.option('--input', '-i', type=click.Path(exists=True), help="Path to the input file.", required=True)
@click.option('--output', '-o', type=click.Path(), help="Path to the input file.", required=True)
@click.option('--cells_boundaries', type=bool, default=True, help='Cells boundaries')
@click.option('--nucleus_boundaries', type=bool, default=True, help='Nucleus boundaries')
@click.option('--cells_as_circles', type=bool, default=None, help='Cells as circles')
@click.option('--cells_labels', type=bool, default=True, help='Cells labels')
@click.option('--nucleus_labels', type=bool, default=True, help='Nucleus labels')
@click.option('--transcripts', type=bool, default=True, help='Transcripts')
@click.option('--morphology_mip', type=bool, default=True, help='Morphology MIP')
@click.option('--morphology_focus', type=bool, default=True, help='Morphology focus')
@click.option('--aligned_images', type=bool, default=True, help='Aligned images')
@click.option('--cells_table', type=bool, default=True, help='Cells table')
@click.option('--n_jobs', type=int, default=1, help='Number of jobs')
def xenium_wrapper(input,
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
                   n_jobs=1):
    """Xenium conversion to SpatialData"""
    sdata = xenium(input,
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
                   n_jobs=n_jobs)
    sdata.write(output)

@click.option('--input', '-i', type=click.Path(exists=True), help='path to the input file', required=True)
@click.option('--filetype', '-t', type=click.Choice(['shape','image']), help='type of the element to store. Can be "shape" or "image". If shape, input must be .geojson', required=True)
@click.option('--name', '-n', type=click.str, help='name of the element to be stored')
@click.option('--output', '-o', type=click.Path, help='path to zarr store to write to. If it does not exist yet, create new zarr store from input', required=True)
@click.option('--coordinate_system', '-c', type=click.str, help='coordinate system in spatialdata object to which an element should belong')
@click.option('--geometry', '-g', type=click.Choice([0,3,6]), help='geometry of shapes element. 0: Circles, 3: Polygon, 6: MultiPolygon')
@click.option('--radius', '-r', type=click.int, help='radius of shapes element if geometry is circle')
def read_generic_wrapper(input, filetype, name, output, coordinate_system, geometry, radius):
    if not coordinate_system:
        coordinate_system = "global"
    if geometry == 0:
        if not radius:
            raise ValueError('Radius must be provided if geometry is circle')
    else:
        radius = None
    sdata = generic_to_zarr(input, filetype, name, output, coordinate_system, geometry, radius)
    sdata.write(output)



if __name__ == '__main__':
    cli()
