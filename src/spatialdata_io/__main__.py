from random import choice
import click
import os
from pathlib import Path
# import spatialdata_io

def codex_wrapper(input, output):
    from spatialdata_io.readers.codex import codex
    sdata = codex(input, output)
    sdata.write(output)

def cosmx_wrapper(input, output):
    from spatialdata_io.readers.cosmx import cosmx
    sdata = cosmx(input, output)
    sdata.write(output)

def curio_wrapper(input, output):
    from spatialdata_io.readers.curio import curio
    sdata = curio(input, output)
    sdata.write(output)

def dbit_wrapper(input, output):
    from spatialdata_io.readers.dbit import dbit
    sdata = dbit(input, output)
    sdata.write(output)

def iss_wrapper(input, output):
    from spatialdata_io.readers.iss import iss
    sdata = iss(input, output)
    sdata.write(output)

def mcmicro_wrapper(input, output):
    from spatialdata_io.readers.mcmicro import mcmicro
    sdata = mcmicro(input, output)
    sdata.write(output)

def merscope_wrapper(input, output):
    from spatialdata_io.readers.merscope import merscope
    sdata = merscope(input, output)
    sdata.write(output)

def seqfish_wrapper(input, output):
    from spatialdata_io.readers.seqfish import seqfish
    sdata = seqfish(input, output)
    sdata.write(output)

def steinbock_wrapper(input, output):
    from spatialdata_io.readers.steinbock import steinbock
    sdata = steinbock(input, output)
    sdata.write(output)

def stereoseq_wrapper(input, output):
    from spatialdata_io.readers.stereoseq import stereoseq
    sdata = stereoseq(input, output)
    sdata.write(output)

def visium_wrapper(input, output):
    from spatialdata_io.readers.visium import visium
    sdata = visium(input, output)
    sdata.write(output)

def visium_hd_wrapper(input, output):
    from spatialdata_io.readers.visium_hd import visium_hd
    sdata = visium_hd(input, output)
    sdata.write(output)

def xenium_wrapper(input, output):
    from spatialdata_io.readers.xenium import xenium
    sdata = xenium(input, output)
    sdata.write(output)

@click.command()
@click.option('--input', '-i', type=click.Path(), help='Input file path')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--reader_name', '-r', type=click.Choice(['codex', 'cosmx', 'curio', 'dbit', 'iss', 'mcmicro', 'merscope', 'seqfish', 'steinbock', 'stereoseq', 'visium', 'visum_hd', 'xenium']), help='Reader name to use for conversion')
def main(input, output, reader_name):
    # For Codex
    if reader_name == 'codex':
        codex_wrapper(input, output)
    elif reader_name == 'cosmx':
        cosmx_wrapper(input, output)
    elif reader_name == 'curio':
        curio_wrapper(input, output)
    elif reader_name == 'dbit':
        dbit_wrapper(input, output)
    elif reader_name == 'iss':
        iss_wrapper(input, output)
    elif reader_name == 'mcmicro':
        mcmicro_wrapper(input, output)
    elif reader_name == 'merscope':
        merscope_wrapper(input, output)
    elif reader_name == 'seqfish':
        seqfish_wrapper(input, output)
    elif reader_name == 'steinbock':
        steinbock_wrapper(input, output)
    elif reader_name == 'stereoseq':
        stereoseq_wrapper(input, output)
    elif reader_name == 'visium':
        visium_wrapper(input, output)
    elif reader_name == 'visium_hd':
        visium_hd_wrapper(input, output)
    elif reader_name == 'xenium':
        xenium_wrapper(input, output)
    else:
        raise ValueError(f'Invalid reader name, choose from {["codex", "cosmx", "curio", "dbit", "iss", "mcmicro", "merscope", "seqfish", "steinbock", "stereoseq", "visium", "visum_hd", "xenium"]}')


if __name__ == '__main__':
    main()
