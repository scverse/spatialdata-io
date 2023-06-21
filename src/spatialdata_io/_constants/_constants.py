from enum import unique

from spatialdata_io._constants._enum import ModeEnum


@unique
class CodexKeys(ModeEnum):
    """Keys for *CODEX* formatted dataset."""

    # files and directories
    FCS_FILE = ".fcs"
    FCS_FILE_CSV = ".csv"
    # metadata
    REGION_KEY = "region"
    INSTANCE_KEY = "cell_id"
    SPATIAL_KEY = "spatial"
    # images
    IMAGE_TIF = ".tif"


class CurioKeys(ModeEnum):
    """Keys for *Curio* formatted dataset."""

    # files and directories
    ANNDATA_FILE = "anndata.h5ad"
    CLUSTER_ASSIGNMENT = "cluster_assignment.txt"
    METRICS_FILE = "Metrics.csv"
    VAR_FEATURES_CLUSTERS = "variable_features_clusters.txt"
    VAR_FEATURES_MORANSI = "variable_features_spatial_moransi.txt"
    # metadata
    CATEGORY = "Category"
    REGION = "cells"
    REGION_KEY = "region"
    INSTANCE_KEY = "instance_id"
    TOP_CLUSTER_DEFINING_FEATURES = "Top_cluster_defining_features"


@unique
class CosmxKeys(ModeEnum):
    """Keys for *Nanostring Cosmx* formatted dataset."""

    # files and directories
    COUNTS_SUFFIX = "exprMat_file.csv"
    TRANSCRIPTS_SUFFIX = "tx_file.csv"
    METADATA_SUFFIX = "metadata_file.csv"
    FOV_SUFFIX = "fov_positions_file.csv"
    IMAGES_DIR = "CellComposite"
    LABELS_DIR = "CellLabels"
    # metadata
    FOV = "fov"
    REGION_KEY = "fov_labels"
    INSTANCE_KEY = "cell_ID"
    X_GLOBAL_CELL = "CenterX_global_px"
    Y_GLOBAL_CELL = "CenterY_global_px"
    X_LOCAL_CELL = "CenterX_local_px"
    Y_LOCAL_CELL = "CenterY_local_px"
    X_LOCAL_TRANSCRIPT = "x_local_px"
    Y_LOCAL_TRANSCRIPT = "y_local_px"
    TARGET_OF_TRANSCRIPT = "target"


@unique
class XeniumKeys(ModeEnum):
    """Keys for *10X Genomics Xenium* formatted dataset."""

    # specifications
    XENIUM_SPECS = "experiment.xenium"

    # cell identifiers
    CELL_ID = "cell_id"

    # nucleus and cell boundaries
    NUCLEUS_BOUNDARIES_FILE = "nucleus_boundaries.parquet"
    CELL_BOUNDARIES_FILE = "cell_boundaries.parquet"
    BOUNDARIES_VERTEX_X = "vertex_x"
    BOUNDARIES_VERTEX_Y = "vertex_y"

    # transcripts
    TRANSCRIPTS_FILE = "transcripts.parquet"
    TRANSCRIPTS_X = "x_location"
    TRANSCRIPTS_Y = "y_location"
    TRANSCRIPTS_Z = "z_location"
    QUALITY_VALUE = "qv"
    OVERLAPS_NUCLEUS = "overlaps_nucleus"
    FEATURE_NAME = "feature_name"

    # cell features matrix
    CELL_FEATURE_MATRIX_FILE = "cell_feature_matrix.h5"
    CELL_METADATA_FILE = "cells.parquet"
    CELL_X = "x_centroid"
    CELL_Y = "y_centroid"
    CELL_AREA = "cell_area"
    CELL_NUCLEUS_AREA = "nucleus_area"

    # morphology iamges
    MORPHOLOGY_MIP_FILE = "morphology_mip.ome.tif"
    MORPHOLOGY_FOCUS_FILE = "morphology_focus.ome.tif"


@unique
class VisiumKeys(ModeEnum):
    """Keys for *10X Genomics Visium* formatted dataset."""

    # files and directories
    COUNTS_FILE = "filtered_feature_bc_matrix.h5"

    # images
    IMAGE_HIRES_FILE = "spatial/tissue_hires_image.png"
    IMAGE_LOWRES_FILE = "spatial/tissue_lowres_image.png"

    # scalefactors
    SCALEFACTORS_FILE = "scalefactors_json.json"
    SCALEFACTORS_HIRES = "tissue_hires_scalef"
    SCALEFACTORS_LOWRES = "tissue_lowres_scalef"

    # spots
    SPOTS_FILE_1 = "tissue_positions_list.csv"
    SPOTS_FILE_2 = "tissue_positions.csv"
    SPOTS_X = "pxl_row_in_fullres"
    SPOTS_Y = "pxl_col_in_fullres"


@unique
class SteinbockKeys(ModeEnum):
    """Keys for *Steinbock* formatted dataset."""

    # files and directories
    CELLS_FILE = "cells.h5ad"
    DEEPCELL_MASKS_DIR = "masks_deepcell"
    ILASTIK_MASKS_DIR = "masks_ilastik"
    IMAGES_DIR = "ome"

    # suffixes for images and labels
    IMAGE_SUFFIX = ".ome.tiff"
    LABEL_SUFFIX = ".tiff"


@unique
class McmicroKeys(ModeEnum):
    """Keys for *Mcmicro* formatted dataset."""

    # files and directories
    CELL_FEATURES_SUFFIX = "--unmicst_cell.csv"
    QUANTIFICATION_DIR = "quantification"
    MARKERS_FILE = "markers.csv"
    IMAGES_DIR = "registration"
    IMAGE_SUFFIX = ".ome.tif"
    LABELS_DIR = "segmentation"
    LABELS_PREFIX = "unmicst-"

    # metadata
    COORDS_X = "X_centroid"
    COORDS_Y = "Y_centroid"
    INSTANCE_KEY = "CellID"


@unique
class MerscopeKeys(ModeEnum):
    """Keys for *MERSCOPE* data (Vizgen plateform)"""

    # files and directories
    IMAGES_DIR = "images"
    TRANSFORMATION_FILE = "micron_to_mosaic_pixel_transform.csv"
    TRANSCRIPTS_FILE = "detected_transcripts.csv"
    BOUNDARIES_FILE = "cell_boundaries.parquet"
    COUNTS_FILE = "cell_by_gene.csv"
    CELL_METADATA_FILE = "cell_metadata.csv"

    # VPT default outputs
    CELLPOSE_BOUNDARIES = "cellpose_micron_space.parquet"
    WATERSHED_BOUNDARIES = "watershed_micron_space.parquet"
    VPT_NAME_COUNTS = "cell_by_gene"
    VPT_NAME_OBS = "cell_metadata"
    VPT_NAME_BOUNDARIES = "cell_boundaries"

    # metadata
    INSTANCE_KEY = "EntityID"
    COUNTS_CELL_KEY = "cell"
    CELL_X = "center_x"
    CELL_Y = "center_y"
    GLOBAL_X = "global_x"
    GLOBAL_Y = "global_y"
    GLOBAL_Z = "global_z"
    Z_INDEX = "ZIndex"
