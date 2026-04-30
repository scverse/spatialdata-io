from pathlib import Path

from spatialdata import SpatialData
from spatialdata._docs import docstring_parameter
from spatialdata.models.models import DEFAULT_COORDINATE_SYSTEM

from spatialdata_io.readers.generic import VALID_IMAGE_TYPES, VALID_SHAPE_TYPES

__all__ = ["generic_to_zarr"]


@docstring_parameter(
    valid_image_types=", ".join(VALID_IMAGE_TYPES),
    valid_shape_types=", ".join(VALID_SHAPE_TYPES),
    default_coordinate_system=DEFAULT_COORDINATE_SYSTEM,
)
def generic_to_zarr(
    input: str | Path,
    output: str | Path,
    name: str | None = None,
    data_axes: str | None = None,
    coordinate_system: str | None = None,
) -> None:
    """Read generic data from an input file and save it as a SpatialData zarr store.

    Parameters
    ----------
    input
        Path to the image/shapes input file. The file must exist and have a supported extension.
        Supported image extensions: {valid_image_types}.
        Supported shapes extensions: {valid_shape_types}.
    output
        Path to the zarr store to write to. If the zarr store does not exist, it will be created from the input.
    name
        Name of the element to be stored. If not provided, the name will default to the stem of the input file.
    data_axes
        Axes of the data for image files. Valid values are 'cyx' and 'czyx'. If not provided, it defaults to None.
    coordinate_system
        Coordinate system in the spatialdata object to which an element should belong. If not provided, it defaults
        to {default_coordinate_system}.

    Raises
    ------
    ValueError
        If the name already exists in the output zarr store, a ValueError is raised, prompting the user to provide
        a different name or delete the existing element.

    Notes
    -----
    This function reads data using the `generic()` method from `spatialdata_io` and writes it to a zarr store
    using the `SpatialData` class. It handles both existing and new zarr stores, ensuring that data is appropriately
    appended or initialized.
    """
    from spatialdata_io.readers.generic import generic

    input = Path(input)
    output = Path(output)

    if name is None:
        name = input.stem
    if not data_axes:
        data_axes = None
    if not coordinate_system:
        coordinate_system = "global"

    element = generic(
        input=input, data_axes=list(data_axes) if data_axes is not None else None, coordinate_system=coordinate_system
    )

    if output.exists():
        sdata = SpatialData.read(output)
        if name in sdata:
            raise ValueError(
                f"Name {name} already exists in {output}; please provide a different name or delete the "
                f"existing element."
            )
        sdata[name] = element
        sdata.write_element(element_name=name)
        print(f"Element {name} written to {output}")
    else:
        sdata = SpatialData.init_from_elements(elements={name: element})
        sdata.write(output)
        print(f"Data written to {output}")
