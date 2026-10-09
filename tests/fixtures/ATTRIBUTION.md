# Test fixture attribution

The `xenium-*-csv-tiny` fixtures are small subsets of public 10x Genomics Xenium
example datasets, used only to test reading of CSV-only / pre-1.3.0 bundles.

Both source datasets are © 10x Genomics and licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## `xenium-1.0.2-csv-tiny/`

- **Source:** Fresh Frozen Mouse Brain for Xenium Explorer Demo (Xenium Onboard Analysis 1.0.2),
  region `Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX+HP`.
  <https://www.10xgenomics.com/datasets/fresh-frozen-mouse-brain-for-xenium-explorer-demo-1-standard>
- **Author:** 10x Genomics.
- **License:** CC BY 4.0.
- **Edits:** subset to 30 cells; kept only the CSV-form outputs
  (`cell_boundaries`, `nucleus_boundaries`, `cells`, `transcripts` as `.csv.gz`),
  the `cell_feature_matrix.h5` (subset to the same cells), and `experiment.xenium`;
  removed the zarr/parquet outputs, images, and auxiliary outputs. No values were altered.

## `xenium-3.0.0-csv-hex-tiny/`

- **Source:** Fresh Frozen Mouse Brain Hemisphere, 5K Mouse Pan Tissue & Pathways Panel
  (Xenium Prime, Xenium Onboard Analysis 3.0.0).
  <https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain>
- **Author:** 10x Genomics.
- **License:** CC BY 4.0.
- **Edits:** subset to 12 cells; kept only the CSV-form outputs
  (`cell_boundaries`, `nucleus_boundaries`, `cells` as `.csv.gz`, carrying the hex
  `cell_id` and `label_id` columns), the `cell_feature_matrix.h5` (subset to the same
  cells), and `experiment.xenium`; removed the zarr/parquet outputs, transcripts,
  images, and auxiliary outputs. No values were altered.
