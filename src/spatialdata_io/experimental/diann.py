import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tabulate import tabulate
from loguru import logger

# Setup logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}",
)

# Silence warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# Set Scanpy settings
sc.settings.verbosity = 1
sc.set_figure_params(dpi=150)

# Helper functions
def DIANN_to_adata(DIANN_path, DIANN_sep="\t", metadata_path=None, metadata_sep=";", sample_id_column="Name"):
    """
    Convert DIANN output and metadata files into an AnnData object.

    DIANN file is assumed to be tab-delimited with the first 5 columns as metadata,
    and the remaining columns as protein expression data. Metadata file should have a
    column of sample names matching the DIANN data.

    Parameters:
        DIANN_path (str): Path to the DIANN output file.
        DIANN_sep (str): Delimiter for the DIANN output file.
        metadata_path (str): Path to the metadata file.
        metadata_sep (str): Delimiter for the metadata file.
        sample_id_column (str): Column in metadata with sample names.

    Returns:
        AnnData: Object containing protein expression and metadata.
    """
    print("Loading DIANN output file")
    df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    rawdata = df.iloc[:, 5:].transpose()

    print("Loading metadata file")
    sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep)

    if sample_id_column not in sample_metadata.columns:
        print(f"ERROR: {sample_id_column} not found in metadata.")
        return None
    sample_metadata.index = sample_metadata[sample_id_column]
    sample_metadata.drop(columns=sample_id_column, inplace=True)

    if rawdata.shape[0] != sample_metadata.shape[0]:
        print("ERROR: Sample counts in DIANN and metadata files do not match.")
        return None

    print("Loading protein metadata")
    protein_metadata = df.iloc[:, :5]
    protein_metadata.index = protein_metadata["Protein.Group"]
    protein_metadata.drop(columns="Protein.Group", inplace=True)

    print("Creating AnnData object")
    return ad.AnnData(X=rawdata.values, obs=sample_metadata, var=protein_metadata)


def filter_out_contaminants(adata, qc_export_path=None):
    """
    Remove contaminants from the AnnData object based on specific patterns in protein IDs and names.

    Contaminants are identified if their IDs or names contain specific patterns like "Cont_".
    Optionally, the filtered contaminants are saved to a file.

    Parameters:
        adata (AnnData): Input AnnData object.
        qc_export_path (str, optional): Path to save contaminants list.

    Returns:
        AnnData: Filtered object with contaminants removed.
    """
    print("Filtering out contaminants")
    condition = adata.var["Protein.Ids"].str.contains("Cont_") | adata.var_names.str.contains("Cont_")

    filtered_out = adata[:, condition]
    filtered_out.var["Species"] = filtered_out.var["Protein.Names"].str.split("_").str[-1]

    print("Contaminants filtered:")
    print(
        tabulate(
            filtered_out.var[["Genes", "Protein.Names", "Species"]].sort_values("Species").values,
            headers=["Genes", "Protein.Names", "Species"],
            tablefmt="psql",
        )
    )

    if qc_export_path:
        filtered_out.var[["Genes", "Protein.Names", "Species"]].sort_values("Species").to_csv(qc_export_path)

    return adata[:, ~condition]


def filter_invalid_proteins(adata, threshold=0.6, grouping=None, qc_export_path=None):
    """
    Filter proteins with NaN proportions above a specified threshold for each group in the grouping variable.

    Proteins with a proportion of NaNs above `threshold` are removed.
    If a grouping variable is provided, filtering is applied within each group.

    Parameters:
        adata (AnnData): Input AnnData object.
        threshold (float): NaN proportion threshold for filtering.
        grouping (str, optional): Column in `adata.obs` for group-based filtering.
        qc_export_path (str, optional): Path to save filtering results.

    Returns:
        AnnData: Filtered AnnData object.
    """
    logger.info("Filtering proteins with excessive NaNs")
    df_proteins = pd.DataFrame(index=adata.var_names, data=adata.var["Genes"]).fillna("None")

    if grouping:
        logger.info(f"Applying group-based filtering by {grouping}")
        for group in adata.obs[grouping].unique():
            adata_group = adata[adata.obs[grouping] == group]
            df_proteins[f"{group}_valid"] = (np.isnan(adata_group.X).mean(axis=0) < threshold)

        df_proteins["valid_in_any"] = df_proteins[[f"{group}_valid" for group in adata.obs[grouping].unique()]].any(axis=1)
        adata = adata[:, df_proteins["valid_in_any"]]
    else:
        df_proteins["valid"] = (np.isnan(adata.X).mean(axis=0) < threshold)
        adata = adata[:, df_proteins["valid"]]

    if qc_export_path:
        df_proteins.to_csv(qc_export_path)

    return adata


def imputation_gaussian(adata, mean_shift=-1.8, std_dev_shift=0.3, perSample=False):
    """
    Impute missing values in the AnnData object using a Gaussian distribution.

    Missing values are imputed by generating random values from a Gaussian distribution.
    The distribution's mean is shifted by `mean_shift` * standard deviation.

    Parameters:
        adata (AnnData): Input AnnData object with missing values.
        mean_shift (float): Shift factor for the Gaussian mean.
        std_dev_shift (float): Scaling factor for the Gaussian standard deviation.
        perSample (bool): Whether to impute values per sample (default is per protein).

    Returns:
        AnnData: Imputed AnnData object.
    """
    logger.info("Imputing missing values using Gaussian distribution")
    adata_copy = adata.copy()
    df = pd.DataFrame(adata_copy.X, columns=adata_copy.var.index, index=adata_copy.obs_names)

    if perSample:
        df = df.T

    for col in df.columns:
        col_mean, col_std = df[col].mean(), df[col].std()
        nan_mask = df[col].isnull()
        shifted_values = (col_mean + mean_shift * col_std) + (col_std * std_dev_shift) * np.random.randn(nan_mask.sum())
        df.loc[nan_mask, col] = shifted_values

    if perSample:
        df = df.T

    adata_copy.X = df.values
    return adata_copy


# Main workflow
if __name__ == "__main__":
    DIANN_path = r"C:\Users\mtrinh\Documents\coding\example_data\di\20231204_DiQP01E021b.pg_matrix.tsv"
    metadata_path = r"your_metadata_path_here.csv"
    adata = DIANN_to_adata(DIANN_path, metadata_path=metadata_path)

    # Log-transform and process data
    adata.X = np.log2(adata.X)
    adata = filter_invalid_proteins(adata, qc_export_path=r"C:\Users\mtrinh\Documents\coding\example_data\di\valid_filtered.csv")
    adata = imputation_gaussian(adata)
    adata.layers["z_score"] = sc.pp.scale(adata.X, zero_center=True, max_value=None, copy=True)
    adata.obs["Group"] = adata.obs["Group"].astype("category")
    adata.var.index = adata.var['Genes'].tolist()

    print("Processing complete.")
