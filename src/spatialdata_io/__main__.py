import os

import click

import spatialdata_io


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--reader_name",
    type=click.Choice(
        ["codex", "cosmx", "curio", "mcmicro", "merscope", "metaspace", "resolve", "steinbock", "visium", "xenium"]
    ),
    help="name of the reader to use",
    required=True,
)
def main(input_path, output_path, reader_name):
    if os.path.exists(output_path):
        print("Spatialdata object already exists! If you want to recompute it, please delete the existing Zarr store.")
    else:
        reader_func = getattr(spatialdata_io, reader_name)
        sdata = reader_func(input_path)  ## Can only handle mcmicro at the moment. Add logic for other readers!!!
        sdata.write(output_path)


if __name__ == "__main__":
    main()
