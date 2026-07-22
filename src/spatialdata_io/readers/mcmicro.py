from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from spatialdata.transformations import Identity, Translation, set_transformation
from yaml.loader import SafeLoader

from spatialdata_io._constants._constants import McmicroKeys, McmicroPipeline
from spatialdata_io.readers._utils._utils import _set_reader_metadata, parse_channels

if TYPE_CHECKING:
    from collections.abc import Mapping

    from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
    from spatial_image import SpatialImage

__all__ = ["mcmicro"]


def _get_transformation(
    tma: int | None = None,
    tma_centroids: pd.DataFrame | None = None,
    raster_data: SpatialImage | MultiscaleSpatialImage | None = None,
) -> dict[str, Identity]:
    if tma is None:
        assert tma_centroids is None
        return {"global": Identity()}
    else:
        assert tma_centroids is not None
        assert raster_data is not None
        xy_centroids = tma_centroids[["x", "y"]].loc[tma].to_numpy()
        x_offset = np.median(raster_data["x"])
        y_offset = np.median(raster_data["y"])
        xy = xy_centroids - np.array([x_offset, y_offset])
        return {"global": Translation(xy, axes=("x", "y"))}


def mcmicro(
    path: str | Path,
    pipeline: str | McmicroPipeline | None = None,
    markers_file: str | Path | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    labels_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read a *Mcmicro* output into a SpatialData object.

    Supports both the original `labsyspharm/mcmicro <https://mcmicro.org/>`_ Nextflow pipeline
    and the newer `nf-core/mcmicro <https://nf-co.re/mcmicro/>`_ pipeline. The two produce
    different output directory layouts; the reader auto-detects which one it is looking at,
    and either whole-slide-image (WSI) or tissue-microarray (TMA) mode. Multiple samples and
    multiple segmentation methods per run are supported.

    .. seealso::

        - `Mcmicro pipeline <https://mcmicro.org/>`_.
        - `nf-core/mcmicro pipeline <https://nf-co.re/mcmicro/>`_.

    Parameters
    ----------
    path
        Path to the dataset (the mcmicro output directory).
    pipeline
        Which pipeline produced the output, ``"labsyspharm"`` or ``"nfcore"``. When ``None``
        (default) the pipeline is auto-detected from the directory contents.
    markers_file
        Optional path to the markers CSV (with a ``marker_name`` column). When ``None`` the
        reader looks in the conventional locations for the detected pipeline, falling back to
        the channel names embedded in the registration OME-TIFF.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    labels_models_kwargs
        Keyword arguments to pass to the labels models

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    pipeline = _detect_pipeline(path) if pipeline is None else McmicroPipeline(pipeline)

    tma = _is_tma(path, pipeline)

    registration_dir = _registration_dir(path, pipeline)
    if not registration_dir.exists():
        raise ValueError(f"{path} does not contain a `{registration_dir.name}` registration directory")

    registration_samples = sorted(registration_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX))
    if not registration_samples:
        raise ValueError(f"No `{McmicroKeys.IMAGE_SUFFIX}` images found in {registration_dir}")

    # sample ids are the registration image stems; used to attribute masks / tables to a sample.
    sample_ids = [_image_stem(s) for s in registration_samples]

    # markers DataFrame (labsyspharm) and channel names for the registration images
    markers_df, image_channel_names = _load_markers(
        path, pipeline, markers_file, registration_samples[0], imread_kwargs
    )

    tma_centroids: pd.DataFrame | None = None
    if tma:
        centroids_file = _centroids_file(path, pipeline)
        tma_centroids = pd.read_csv(centroids_file, header=None, names=["y", "x"], index_col=False, sep=" ")
        tma_centroids.index = tma_centroids.index + 1

    images = {}
    if tma:
        # the registration image is the global mosaic containing all cores; keep it as `tma_map`
        # and then parse each individual core from the dearray directory.
        data = imread(registration_samples[0], **imread_kwargs)
        data = Image2DModel.parse(data, transformations=_get_transformation(), rgb=None, **image_models_kwargs)
        images["tma_map"] = data

        dearray_dir = _dearray_dir(path, pipeline)
        # per-core images have integer stems (1.tif, 2.tif, ...); skip TMA_MAP.tif and any extras.
        all_dearray = sorted(set(dearray_dir.glob("*" + McmicroKeys.IMAGE_SUFFIX)) | set(dearray_dir.glob("*.tif")))
        samples = sorted((s for s in all_dearray if _image_stem(s).isdigit()), key=lambda s: int(_image_stem(s)))
        samples_masks = sorted((dearray_dir / "masks").glob("*"))
    else:
        samples = registration_samples
        samples_masks = []

    core_ids = [_image_stem(s) for s in samples] if tma else []

    for sample in samples:
        core_id = _image_stem(sample)
        image_id = f"core_{core_id}" if tma else core_id

        data = imread(sample, **imread_kwargs)
        c_coords = image_channel_names if len(image_channel_names) == data.shape[0] else None
        data = Image2DModel.parse(data, c_coords=c_coords, rgb=None, **image_models_kwargs)
        transformations = _get_transformation(
            tma=int(core_id) if tma else None, tma_centroids=tma_centroids, raster_data=data
        )
        set_transformation(data, transformation=transformations, set_all=True)
        images[f"{image_id}_image"] = data

    # in exemplar-001 the raw images are aligned with the illumination images, not with the
    # registration image. These directories only exist for the labsyspharm layout.
    raw_dir = path / McmicroKeys.RAW_DIR
    if raw_dir.exists():
        for raw_image in raw_dir.glob("*"):
            raw_name = _image_stem(raw_image)
            data = imread(raw_image, **imread_kwargs)
            images[raw_name] = Image2DModel.parse(
                data, transformations={raw_name: Identity()}, rgb=None, **image_models_kwargs
            )

    illumination_dir = path / McmicroKeys.ILLUMINATION_DIR
    if illumination_dir.exists():
        for illumination_image in illumination_dir.glob("*"):
            illumination_name = _image_stem(illumination_image)
            raw_name = illumination_name.removesuffix(McmicroKeys.ILLUMINATION_SUFFIX_DFP)
            raw_name = raw_name.removesuffix(McmicroKeys.ILLUMINATION_SUFFIX_FFP)
            data = imread(illumination_image, **imread_kwargs)
            images[illumination_name] = Image2DModel.parse(
                data, transformations={raw_name: Identity()}, rgb=None, **image_models_kwargs
            )

    labels = _get_labels(
        path,
        pipeline,
        tma=tma,
        tma_centroids=tma_centroids,
        sample_ids=sample_ids,
        core_ids=core_ids,
        wsi_image_id=sample_ids[0] if not tma else None,
        samples_masks=samples_masks,
        imread_kwargs=imread_kwargs,
        labels_models_kwargs=labels_models_kwargs,
    )

    tables_dict = _get_tables(path, pipeline, markers_df, tma, sample_ids, core_ids, list(labels))

    sdata = SpatialData(images=images, labels=labels, tables=tables_dict)
    return _set_reader_metadata(sdata, "mcmicro")


def _detect_pipeline(path: Path) -> McmicroPipeline:
    """Infer which mcmicro pipeline produced ``path``."""
    if (path / McmicroKeys.PARAMS_FILE).exists():
        return McmicroPipeline.LABSYSPHARM
    if (path / McmicroKeys.NFCORE_PIPELINE_INFO).exists() or (path / McmicroKeys.NFCORE_IMAGES_DIR_WSI).exists():
        return McmicroPipeline.NFCORE
    raise ValueError(
        f"Could not detect the mcmicro pipeline for {path}. Pass `pipeline='labsyspharm'` or "
        f"`pipeline='nfcore'` explicitly."
    )


def _is_tma(path: Path, pipeline: McmicroPipeline) -> bool:
    if pipeline == McmicroPipeline.LABSYSPHARM:
        return bool(_load_params(path)["workflow"]["tma"])
    params = _load_nfcore_params(path)
    if params is not None and McmicroKeys.NFCORE_IMAGES_DIR_TMA.value in params:
        return bool(params[McmicroKeys.NFCORE_IMAGES_DIR_TMA.value])
    return (path / McmicroKeys.NFCORE_IMAGES_DIR_TMA).exists()


def _registration_dir(path: Path, pipeline: McmicroPipeline) -> Path:
    if pipeline == McmicroPipeline.LABSYSPHARM:
        return path / McmicroKeys.IMAGES_DIR_WSI
    return path / McmicroKeys.NFCORE_IMAGES_DIR_WSI


def _dearray_dir(path: Path, pipeline: McmicroPipeline) -> Path:
    if pipeline == McmicroPipeline.LABSYSPHARM:
        return path / McmicroKeys.IMAGES_DIR_TMA
    return path / McmicroKeys.NFCORE_IMAGES_DIR_TMA


def _centroids_file(path: Path, pipeline: McmicroPipeline) -> Path:
    if pipeline == McmicroPipeline.LABSYSPHARM:
        return path / McmicroKeys.COREOGRAPH_CENTROIDS
    return path / McmicroKeys.NFCORE_COREOGRAPH_CENTROIDS


def _image_stem(sample: Path) -> str:
    """Return the file stem with a (possibly double) suffix like ``.ome.tif`` stripped."""
    return sample.with_name(sample.stem).with_suffix("").stem


def _match_sample(name: str, sample_ids: list[str]) -> str:
    """Return the sample id that best matches ``name`` (longest one that is a prefix/substring)."""
    matches = [s for s in sample_ids if name.startswith(s) or s in name]
    if matches:
        return max(matches, key=len)
    return name


def _match_core(name: str, core_ids: list[str]) -> str:
    """Return the core id whose numeric token appears in ``name``.

    Core mask/table filenames embed the core number as a token, but also contain other digits
    (e.g. ``exemplar-002_1_mask``), so match against the known core ids rather than the first digit.
    """
    tokens = re.split(r"[_\-.]", name)
    for t in tokens:
        if t in core_ids:
            return t
    match = re.search(r"\d+", name)
    return match.group() if match else name


def _load_params(path: Path) -> Any:
    params_path = path / McmicroKeys.PARAMS_FILE
    with open(params_path) as fp:
        params = yaml.load(fp, SafeLoader)
    return params


def _load_nfcore_params(path: Path) -> dict[str, Any] | None:
    """Load the most recent ``pipeline_info/params*.json`` if present."""
    params_files = sorted((path / McmicroKeys.NFCORE_PIPELINE_INFO).glob(McmicroKeys.NFCORE_PARAMS_GLOB.value))
    if not params_files:
        return None
    with open(params_files[-1]) as fp:
        return json.load(fp)


def _read_marker_sheet(sheet_path: Path) -> pd.DataFrame | None:
    """Read a wide marker sheet, or return ``None`` if the file is not one.

    Some nf-core outputs contain a long-format MultiQC validation report named ``*markersheet*``
    that is not a usable marker table; such files (lacking a ``marker_name`` column) are skipped.
    """
    sep = "\t" if sheet_path.suffix == ".tsv" else ","
    markers = pd.read_csv(sheet_path, sep=sep)
    if McmicroKeys.MARKER_NAME.value not in markers.columns:
        return None
    if McmicroKeys.CHANNEL_NUMBER.value not in markers.columns:
        markers[McmicroKeys.CHANNEL_NUMBER.value] = range(1, len(markers) + 1)
    markers.index = markers[McmicroKeys.MARKER_NAME.value]
    return markers


def _load_markers(
    path: Path,
    pipeline: McmicroPipeline,
    markers_file: str | Path | None,
    fallback_image: Path,
    imread_kwargs: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load the marker table and the channel names to attach to the registration images.

    Returns a ``(markers_df, channel_names)`` tuple. ``markers_df`` (used to build the labsyspharm
    tables) may be ``None`` for nf-core, where table variables are derived from the quantification
    CSV columns instead. ``channel_names`` are attached to the images when their length matches the
    number of image channels.
    """
    if pipeline == McmicroPipeline.LABSYSPHARM:
        markers_path = Path(markers_file) if markers_file is not None else path / McmicroKeys.MARKERS_FILE
        if not markers_path.exists():
            # historically required; surface a clear error rather than silently falling back.
            raise FileNotFoundError(f"Expected a markers file at {markers_path}")
        markers = _read_marker_sheet(markers_path)
        if markers is None:
            raise ValueError(f"Markers file {markers_path} has no `{McmicroKeys.MARKER_NAME.value}` column")
        assert markers[McmicroKeys.CHANNEL_NUMBER.value].is_monotonic_increasing
        return markers, markers[McmicroKeys.MARKER_NAME.value].tolist()

    # nf-core: the registration image (full ashlar mosaic) and the quantification tables can have
    # different channel/marker sets (background subtraction drops channels). Collect every candidate
    # marker sheet, then pick the channel names for the image by matching the image's channel count.
    n_channels = int(imread(fallback_image, **imread_kwargs).shape[0])
    sheet_paths: list[Path] = []
    if markers_file is not None:
        sheet_paths.append(Path(markers_file))
    else:
        sheet_paths.append(path / McmicroKeys.MARKERS_FILE)
        sheet_paths.extend(
            sorted((path / McmicroKeys.NFCORE_BACKSUB_DIR).glob(McmicroKeys.NFCORE_BACKSUB_MARKERS_GLOB.value))
        )
        for glob_pattern in McmicroKeys.NFCORE_MARKERSHEET_GLOBS.value.split(";"):
            sheet_paths.extend(sorted(path.glob(glob_pattern)))
    sheets = [s for s in (_read_marker_sheet(p) for p in sheet_paths if p.exists()) if s is not None]

    # candidate name lists for the image, in priority order
    candidate_names = [_safe_parse_channels(fallback_image)]
    candidate_names += [s[McmicroKeys.MARKER_NAME.value].tolist() for s in sheets]
    image_names = next((n for n in candidate_names if len(n) == n_channels), None)
    if image_names is None:
        image_names = next((n for n in candidate_names if n), None)
        if image_names is None:
            image_names = [f"channel_{i}" for i in range(n_channels)]
            warnings.warn(
                f"No markers file found; using integer channel names for {fallback_image.name}.",
                UserWarning,
                stacklevel=2,
            )

    # keep a marker sheet (any is fine: table variables are matched to it by name) for table var
    markers_df = next((s for s in sheets if len(s) == n_channels), sheets[0] if sheets else None)
    return markers_df, image_names


def _safe_parse_channels(image: Path) -> list[str]:
    try:
        return parse_channels(image)
    except Exception:
        return []


def _labsyspharm_module(parent_stem: str, ref_id: str | None) -> str:
    """Extract the segmentation module from a `<module>-<sample_or_core>` directory name."""
    if ref_id is not None and parent_stem.endswith(f"-{ref_id}"):
        return parent_stem[: -(len(ref_id) + 1)]
    return parent_stem


def _get_labels(
    path: Path,
    pipeline: McmicroPipeline,
    tma: bool,
    tma_centroids: pd.DataFrame | None,
    sample_ids: list[str],
    core_ids: list[str],
    wsi_image_id: str | None,
    samples_masks: list[Path],
    imread_kwargs: Mapping[str, Any],
    labels_models_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    labels: dict[str, Any] = {}

    if pipeline == McmicroPipeline.LABSYSPHARM:
        samples_labels = list((path / McmicroKeys.LABELS_DIR).glob("*/*" + McmicroKeys.IMAGE_SUFFIX))
        for labels_path in samples_labels:
            segmentation_stem = _image_stem(labels_path)
            if not tma:
                # the segmentation subdir is `<module>-<sample>`; include the module so that
                # multiple segmenters (e.g. unmicst + ilastik) don't collide on the same key.
                module = _labsyspharm_module(labels_path.parent.stem, wsi_image_id)
                data = imread(labels_path, **imread_kwargs).squeeze()
                data = Labels2DModel.parse(data, transformations=_get_transformation(), **labels_models_kwargs)
                labels[f"{wsi_image_id}_{module}_{segmentation_stem}"] = data
            else:
                core_id_search = re.search(r"\d+$", labels_path.parent.stem)
                if core_id_search is None:
                    raise ValueError(f"Cannot infer core_id from {labels_path.parent}")
                core_id = core_id_search.group()
                module = _labsyspharm_module(labels_path.parent.stem, core_id)
                data = imread(labels_path, **imread_kwargs).squeeze()
                data = Labels2DModel.parse(data, **labels_models_kwargs)
                transformations = _get_transformation(tma=int(core_id), tma_centroids=tma_centroids, raster_data=data)
                set_transformation(data, transformation=transformations, set_all=True)
                labels[f"core_{core_id}_{module}_{segmentation_stem}"] = data
    else:  # nf-core: segmentation/<segmenter>/<any mask tif>
        for segmenter_dir in sorted(p for p in (path / McmicroKeys.LABELS_DIR).glob("*") if p.is_dir()):
            segmenter = segmenter_dir.stem
            mask_files = sorted(set(segmenter_dir.glob("*.tif")) | set(segmenter_dir.glob("*.tiff")))
            for mask_path in mask_files:
                data = imread(mask_path, **imread_kwargs).squeeze()
                if not tma:
                    sample_id = _match_sample(_image_stem(mask_path), sample_ids)
                    data = Labels2DModel.parse(data, transformations=_get_transformation(), **labels_models_kwargs)
                    labels[f"{sample_id}_{segmenter}"] = data
                else:
                    core_id = _match_core(_image_stem(mask_path), core_ids)
                    data = Labels2DModel.parse(data, **labels_models_kwargs)
                    transformations = _get_transformation(
                        tma=int(core_id), tma_centroids=tma_centroids, raster_data=data
                    )
                    set_transformation(data, transformation=transformations, set_all=True)
                    labels[f"core_{core_id}_{segmenter}"] = data

    # per-core dearray masks (shared by both layouts in TMA mode)
    if tma:
        for mask_path in samples_masks:
            mask_stem = mask_path.stem
            core_id = mask_stem.split("_")[0]
            data = imread(mask_path, **imread_kwargs).squeeze()
            data = Labels2DModel.parse(data, **labels_models_kwargs)
            transformations = _get_transformation(tma=int(core_id), tma_centroids=tma_centroids, raster_data=data)
            set_transformation(data, transformation=transformations, set_all=True)
            labels[f"core_dearray_{mask_stem}"] = data

    return labels


def _get_tables(
    path: Path,
    pipeline: McmicroPipeline,
    marker_df: pd.DataFrame | None,
    tma: bool,
    sample_ids: list[str],
    core_ids: list[str],
    label_keys: list[str],
) -> dict[str, AnnData]:
    coords = [McmicroKeys.COORDS_X.value, McmicroKeys.COORDS_Y.value]
    instance_key = McmicroKeys.INSTANCE_KEY.value
    tables_dict: dict[str, AnnData] = {}

    if pipeline == McmicroPipeline.LABSYSPHARM:
        # WSI: one table per CSV; TMA: all cores concatenated into a single table (legacy behavior).
        regions: list[str] = []
        adatas = None
        for table_path in (path / McmicroKeys.QUANTIFICATION_DIR).glob("*.csv"):
            name, region, var, var_df = _table_spec(
                table_path, None, pipeline, tma, marker_df, coords, sample_ids, core_ids, label_keys
            )
            adata = _build_anndata(table_path, var_df, var, coords, region)
            if not tma:
                tables_dict[name] = TableModel.parse(
                    adata, region=region, region_key="region", instance_key=instance_key
                )
            else:
                regions.append(region)
                adatas = adata if adatas is None else ad.concat([adatas, adata], index_unique="_")
                tables_dict["segmentation_table"] = TableModel.parse(
                    adatas, region=regions, region_key="region", instance_key=instance_key
                )
        return tables_dict

    # nf-core: quantification/mcquant/<segmenter>/<core-or-sample>.csv
    entries = [(p, p.parent.stem) for p in (path / McmicroKeys.NFCORE_QUANTIFICATION_DIR).glob("*/*.csv")]
    if not tma:
        for table_path, segmenter in entries:
            name, region, var, var_df = _table_spec(
                table_path, segmenter, pipeline, tma, marker_df, coords, sample_ids, core_ids, label_keys
            )
            tables_dict[name] = TableModel.parse(
                _build_anndata(table_path, var_df, var, coords, region),
                region=region,
                region_key="region",
                instance_key=instance_key,
            )
        return tables_dict

    # nf-core TMA: one table per segmenter, concatenating that segmenter's cores.
    by_segmenter: dict[str, list[Path]] = {}
    for table_path, segmenter in entries:
        by_segmenter.setdefault(segmenter, []).append(table_path)
    for segmenter, csvs in by_segmenter.items():
        regions = []
        adatas = None
        for table_path in sorted(csvs):
            _, region, var, var_df = _table_spec(
                table_path, segmenter, pipeline, tma, marker_df, coords, sample_ids, core_ids, label_keys
            )
            adata = _build_anndata(table_path, var_df, var, coords, region)
            regions.append(region)
            adatas = adata if adatas is None else ad.concat([adatas, adata], index_unique="_")
        tables_dict[f"{segmenter}_table"] = TableModel.parse(
            adatas, region=regions, region_key="region", instance_key=instance_key
        )
    return tables_dict


def _table_spec(
    csv_path: Path,
    segmenter: str | None,
    pipeline: McmicroPipeline,
    tma: bool,
    marker_df: pd.DataFrame | None,
    coords: list[str],
    sample_ids: list[str],
    core_ids: list[str],
    label_keys: list[str],
) -> tuple[str, str, list[str], pd.DataFrame]:
    """Return ``(table_name, region_value, var, var_df)`` for a quantification CSV."""
    if pipeline == McmicroPipeline.LABSYSPHARM:
        assert marker_df is not None
        var = marker_df[McmicroKeys.MARKER_NAME.value].tolist()
        # filename form: <SAMPLE_ID>--<MODULE>_<labels_NAME>, e.g. exemplar-001--unmicst_cell
        labels_basename = csv_path.stem.split("_")[-1]
        sample_id_search = re.search(r"^(.*?)--", csv_path.stem)
        sample_id = sample_id_search.groups()[0] if sample_id_search else None
        if not sample_id:
            raise ValueError(
                f"Csv filename should be in form <SAMPLE_ID>--<SEGMENTATION>_<labels_NAME>, got {csv_path.stem} "
            )
        rest = csv_path.stem[len(sample_id) + 2 :]  # drop "<sample_id>--"
        module = rest[: -(len(labels_basename) + 1)] if rest.endswith("_" + labels_basename) else rest
        prefix = f"core_{sample_id}" if tma else sample_id
        region_value = f"{prefix}_{module}_{labels_basename}"
        return csv_path.stem, region_value, var, marker_df

    # nf-core: variables are the CSV columns between CellID and the coordinate columns.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    var = _nfcore_marker_columns(header, coords)
    var_df = (
        marker_df.reindex(var)
        if marker_df is not None
        else pd.DataFrame(index=pd.Index(var, name=McmicroKeys.MARKER_NAME.value))
    )
    # in TMA mode the CSV corresponds to a core; in WSI mode it corresponds to a sample
    unit = _match_core(csv_path.stem, core_ids) if tma else _match_sample(csv_path.stem, sample_ids)
    # table names share a namespace with images/labels, so suffix to avoid colliding with the label
    table_name = f"{unit}_{segmenter}_table"
    region_value = _match_label_key(unit, segmenter, tma, label_keys)
    return table_name, region_value, var, var_df


def _nfcore_marker_columns(header: list[str], coords: list[str]) -> list[str]:
    """Marker columns are those after ``CellID`` and before the first coordinate column."""
    instance = McmicroKeys.INSTANCE_KEY.value
    start = header.index(instance) + 1 if instance in header else 0
    end = min((header.index(c) for c in coords if c in header), default=len(header))
    return header[start:end]


def _match_label_key(sample_id: str, segmenter: str | None, tma: bool, label_keys: list[str]) -> str:
    """Link a quantification table to the label element produced by the same segmenter.

    Segmentation and quantification directories do not always share the same segmenter name
    (e.g. ``deepcell_mesmer`` vs ``mesmer``), so we match by prefix and then by substring.
    """
    prefix = f"core_{sample_id}" if tma else sample_id
    exact = f"{prefix}_{segmenter}"
    if exact in label_keys:
        return exact
    candidates = [k for k in label_keys if k.startswith(f"{prefix}_")]
    for k in candidates:
        seg = k[len(prefix) + 1 :]
        if segmenter and (segmenter in seg or seg in segmenter):
            return k
    return exact


def _build_anndata(
    csv_path: Path,
    markers: pd.DataFrame,
    var: list[str],
    coords: list[str],
    region_value: str,
) -> AnnData:
    table = pd.read_csv(csv_path)
    adata = AnnData(
        table[var].to_numpy().astype(np.float32),
        obs=table.drop(columns=var + coords),
        var=markers,
        obsm={"spatial": table[coords].to_numpy()},
    )
    adata.obs["region"] = pd.Categorical([region_value] * len(adata))
    return adata
