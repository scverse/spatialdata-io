from anndata import AnnData
from spatialdata import SpatialData, match_sdata_to_table

from spatialdata_io.converters.legacy_anndata import to_legacy_anndata


def test_deconcatenate(
    full_sdata: SpatialData,
    by: str,
    target_coordinate_system: str,
    sdata_table_name: str = "table",
    region_key: str = "region",
    join: str = "right",
) -> AnnData:
    """From a `SpatialData` object containing multiple regions, subset to a single region and return as an AnnData object.

    Parameters
    ----------
    full_sdata : SpatialData
        `SpatialData` object containing regions to deconcatenate.
    by : str
        Value in `region_key` to use for subsetting.
    target_coordinate_system : str
        Coordinate system to use to populate `AnnData` object.
    sdata_table_name : str, optional
        Name of the table in `full_sdata` to subset. Default is "table".
    region_key : str, optional
        Name of obs column to filter `full_sdata` on. Default is "region".
    join : str, optional
        Join method to use in `match_sdata_to_table`. Default is "right" to keep all rows in the table.

    Returns
    -------
    AnnData
        An `AnnData` object containing a subset of `full_sdata` filtered according to `region_key == by`.
    """
    sdata_table = full_sdata[sdata_table_name]

    # maybe add "table" parameter coupled with "table_name" to follow match_sdata_to_table structure?
    sdata_deconcat = match_sdata_to_table(
        full_sdata, table=sdata_table[sdata_table.obs[region_key] == by], table_name="table_name_test", how=join
    )
    adata = to_legacy_anndata(
        sdata_deconcat, coordinate_system=target_coordinate_system, table_name="table_name_test", include_images=False
    )
    # TODO: support for adding images to anndata object?

    return adata
