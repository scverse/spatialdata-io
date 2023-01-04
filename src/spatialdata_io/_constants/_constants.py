from enum import unique

from spatialdata_io._constants._enum import ModeEnum


@unique
class CosmxKeys(ModeEnum):
    """Keys for *Nanostring Cosmx* formatted dataset."""

    # files and directories
    COUNTS_SUFFIX = "exprMat_file.csv"
    METADATA_SUFFIX = "metadata_file.csv"
    FOV_SUFFIX = "fov_positions_file.csv"
    IMAGES_DIR = "CellComposite"
    LABELS_DIR = "CellLabels"
    # metadata
    REGION_KEY = "fov"
    INSTANCE_KEY = "cell_ID"
    X_GLOBAL = "CenterX_global_px"
    Y_GLOBAL = "CenterY_global_px"
    X_LOCAL = "CenterX_local_px"
    Y_LOCAL = "CenterY_local_px"
