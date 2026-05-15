from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class TestDataset:
    """Test dataset to be downloaded.

    Parameters
    ----------
    key : str
        Unique identifier for the dataset, used for referencing in tests and
        the CLI.
    group : str
        Logical grouping of the dataset, for example ``"xenium"``,
        ``"visium_hd"``, ``"seqfish"``, or ``"macsima"``.
    url : str
        Direct URL to the dataset archive, for example a ZIP file.
    archive_name : str
        Expected filename of the downloaded archive.
    extracted_dir : str
        Expected name of the directory created when the archive is extracted.
    source : str
        Human-readable description of the dataset source and license.
    test_path : str
        Optional path inside the extracted directory that should be passed to
        integration tests.
    """

    key: str
    group: str
    url: str
    archive_name: str
    extracted_dir: str
    source: str
    test_path: str = ""


DATASETS = (
    TestDataset(
        key="xenium_breast",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip",
        archive_name="Xenium_V1_human_Breast_2fov_outs.zip",
        extracted_dir="Xenium_V1_human_Breast_2fov_outs",
        source="10x Genomics Xenium 2.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="xenium_lung",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip",
        archive_name="Xenium_V1_human_Lung_2fov_outs.zip",
        extracted_dir="Xenium_V1_human_Lung_2fov_outs",
        source="10x Genomics Xenium 2.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="xenium_prime_mouse_ileum",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip",
        archive_name="Xenium_Prime_Mouse_Ileum_tiny_outs.zip",
        extracted_dir="Xenium_Prime_Mouse_Ileum_tiny_outs",
        source="10x Genomics Xenium 3.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="xenium_ovary",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Human_Ovary_tiny/Xenium_V1_Human_Ovary_tiny_outs.zip",
        archive_name="Xenium_V1_Human_Ovary_tiny_outs.zip",
        extracted_dir="Xenium_V1_Human_Ovary_tiny_outs",
        source="10x Genomics Xenium 4.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="xenium_multicell_ovary",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_MultiCellSeg_Human_Ovary_tiny/Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs.zip",
        archive_name="Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs.zip",
        extracted_dir="Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs",
        source="10x Genomics Xenium 4.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="xenium_protein_kidney",
        group="xenium",
        url="https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Protein_Human_Kidney_tiny/Xenium_V1_Protein_Human_Kidney_tiny_outs.zip",
        archive_name="Xenium_V1_Protein_Human_Kidney_tiny_outs.zip",
        extracted_dir="Xenium_V1_Protein_Human_Kidney_tiny_outs",
        source="10x Genomics Xenium 4.0.0, CC BY 4.0",
    ),
    TestDataset(
        key="visium_hd_tiny",
        group="visium_hd",
        url="https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Tiny_3prime_Dataset/Visium_HD_Tiny_3prime_Dataset_outs.zip",
        archive_name="Visium_HD_Tiny_3prime_Dataset_outs.zip",
        extracted_dir="Visium_HD_Tiny_3prime_Dataset_outs",
        source="10x Genomics Visium HD 4.0.1, CC BY 4.0",
    ),
    TestDataset(
        key="seqfish",
        group="seqfish",
        url="https://s3.embl.de/spatialdata/raw_data/seqfish-2-test-dataset.zip",
        archive_name="seqfish-2-test-dataset.zip",
        extracted_dir="seqfish-2-test-dataset",
        source="Spatial Genomics seqFISH v2, public test data",
        test_path="instrument 2 official",
    ),
    TestDataset(
        key="macsima_omap23",
        group="macsima",
        url="https://zenodo.org/api/records/18196452/files-archive",
        archive_name="OMAP23_small.zip",
        extracted_dir="OMAP23_small",
        source="MACSima OMAP23, CC BY 4.0",
    ),
    TestDataset(
        key="macsima_omap10",
        group="macsima",
        url="https://zenodo.org/api/records/18196366/files-archive",
        archive_name="OMAP10_small.zip",
        extracted_dir="OMAP10_small",
        source="MACSima OMAP10, CC BY 4.0",
    ),
)


@cache
def _datasets_by_key() -> dict[str, TestDataset]:
    return {dataset.key: dataset for dataset in DATASETS}


def get_dataset(key: str) -> TestDataset:
    """Return the dataset registered for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not a registered dataset key.
    """
    try:
        return _datasets_by_key()[key]
    except KeyError as exc:
        available = ", ".join(sorted(_datasets_by_key()))
        raise KeyError(f"Unknown test dataset key {key!r}. Available keys: {available}") from exc


def datasets_by_group(group: str) -> tuple[TestDataset, ...]:
    """Return datasets registered for ``group``."""
    return tuple(dataset for dataset in DATASETS if dataset.group == group)
