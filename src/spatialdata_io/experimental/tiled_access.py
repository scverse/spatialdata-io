"""Opt-in spatial tiling for a SpatialData store.

Two entry points, both a single call:

:func:`add_spatial_tiling`
    Add the profile to a store that already exists.
:func:`xenium_spatially_tiled`
    Read raw Xenium and write a tiled store in one go.

The one-shot path prepares table annotations before the initial write, then tiles the
written vector Parquets and writes the tuned CSC buffers.

The operation is opt-in and modifies the store in place: canonical Parquets are
reordered, Shapes gain cell_code, and the table receives derived annotations and a
CSC layer. Original coordinates and geometries remain authoritative. read_zarr works
unchanged, but an ordinary SpatialData.write does not preserve the display profile
or tile row-group layout. Regenerate after saving or editing source data.

The multi-asset operation is not transactional. The existing-store entry point deletes
and rewrites the table when expression indexing is requested; the one-shot Xenium entry
point avoids that rewrite. Current support assumes Xenium-compatible instance identifiers,
centroid layout and display coordinates.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from spatialdata_io.experimental.display_colors import add_cluster_colors, add_gene_colors
from spatialdata_io.experimental.expression_index import (
    add_gene_statistics,
    write_csc_layer,
)
from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.manifest import (
    PROFILE_NAME,
    build_manifest,
    validate_manifest,
    write_manifest,
    write_root_manifest,
)
from spatialdata_io.experimental.points_parquet import (
    DisplayTransform,
    write_points_regular_grid,
)
from spatialdata_io.experimental.regular_grid import (
    DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    RegularGrid,
)
from spatialdata_io.experimental.shapes_parquet import (
    CELL_CODE_COLUMN,
    write_shapes_regular_grid,
)

__all__ = ["add_spatial_tiling", "xenium_spatially_tiled"]

#: Directory inside the store holding derived (non-canonical) profile assets.
PROFILE_DIR = "visualization"


def _grid_for(points: Any, transform: DisplayTransform, tile_size_px: float) -> RegularGrid:
    """Derive the grid covering the points element in display pixel space."""
    x = points["x"].max().compute() if hasattr(points["x"].max(), "compute") else points["x"].max()
    y = points["y"].max().compute() if hasattr(points["y"].max(), "compute") else points["y"].max()
    px, py = transform.apply(np.array([float(x)]), np.array([float(y)]))
    return RegularGrid.from_bounds(0, 0, float(np.rint(px[0])), float(np.rint(py[0])), tile_size_px)


def _cell_positions_for_shapes(table: Any, shapes_element: str) -> dict[Any, int]:
    """Map one Shapes element's instance IDs to absolute rows in the annotating table."""
    from spatialdata.models import get_table_keys

    regions, region_key, instance_key = get_table_keys(table)
    declared = [regions] if isinstance(regions, str) else list(regions)
    # Xenium's table annotates ``cell_labels`` while the boundary Shapes carry the same
    # cell instance IDs. When there is only one annotated region, those IDs provide an
    # unambiguous bridge to the requested boundaries. With several unrelated regions we
    # require an exact region match rather than guessing.
    if shapes_element in declared:
        selected_regions = {shapes_element}
    elif len(declared) == 1:
        selected_regions = {declared[0]}
    else:
        raise ValueError(
            f"table does not directly annotate shapes element {shapes_element!r}, and its "
            f"declared regions {declared} do not identify one unambiguous instance namespace"
        )

    positions: dict[Any, int] = {}
    for row_position, (region, instance_id) in enumerate(
        zip(table.obs[region_key], table.obs[instance_key], strict=True)
    ):
        if region not in selected_regions:
            continue
        if instance_id in positions:
            raise ValueError(
                f"table has duplicate {instance_key!r} value {instance_id!r} for region {shapes_element!r}"
            )
        positions[instance_id] = row_position
    return positions


def _prepare_expression_index(table: Any, cluster_column: str | None) -> tuple[Any | None, dict[str, Any]]:
    """Mutate a table with display metadata and retain a CSC matrix for tuned writing."""
    import scipy.sparse as sp

    stats = add_gene_statistics(table)
    colors = add_gene_colors(table)
    cluster_colors = add_cluster_colors(table, cluster_column) if cluster_column else None
    csc = table.X.tocsc() if sp.issparse(table.X) else None
    description: dict[str, Any] = {
        "var_statistics": stats,
        **({"gene_colors": colors} if colors else {}),
        **({"cluster_colors": cluster_colors} if cluster_colors else {}),
    }
    return csc, description


def add_spatial_tiling(
    store: str | Path,
    *,
    points_element: str = "transcripts",
    shapes_element: str | None = "cell_boundaries",
    table_element: str | None = "table",
    coordinate_system: str = "global",
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    feature_key: str = "feature_name",
    technology: str = "Xenium",
    index_expression: bool = True,
    cluster_column: str | None = None,
    profile_layout: str = "v1",
    compression: str = "zstd",
    _expression_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the regular-grid visualization profile to an existing SpatialData store.

    Re-runnable: running it again replaces the render columns and derived assets rather
    than duplicating them.

    Parameters
    ----------
    store
        Path to the ``.zarr`` store to tile, modified in place.
    points_element
        Name of the Points element holding transcripts.
    shapes_element
        Name of the Shapes element holding cell boundaries, or ``None`` to skip.
    table_element
        Name of the annotating table, used for gene ordering and expression.
        ``None`` derives codes from categorical observed features, includes their names
        in the manifest and does not advertise table-backed viewer components.
    coordinate_system
        Target coordinate system, assumed to coincide with reference-image pixels.
        The current code does not validate that assumption.
    tile_size_px
        Tile edge length in display pixels. The default of 250 gives roughly 20 cells per
        tile on Xenium-density tissue, which is the granularity the viewer fetches at.
    max_row_groups_per_file
        Row groups per chunk file.
    feature_key
        Column in the points element holding the feature name.
    technology
        Celldega technology string recorded in the manifest.
    index_expression
        Add per-gene statistics to ``var`` and a gene-major (CSC) copy of ``X`` as a layer,
        so a client can read one gene without downloading the whole matrix. Costs a second
        copy of the non-zeros.
    cluster_column
        Categorical ``obs`` column to colour cells by. Its palette is written to
        ``uns["<column>_colors"]``, the scanpy convention.
    profile_layout
        ``"v1"`` writes the display Parquets and a manifest file under
        ``visualization/grid_files_v1``. ``"canonical"`` writes neither: the canonical
        Parquets carry everything a viewer needs. Geometry is written as ``geoarrow`` so a
        browser can decode coordinate buffers without parsing WKB; point columns are ordered
        so a projection spans no unwanted column; and the manifest goes into the store's root
        attributes. No ``visualization/`` directory is created.
    compression
        Parquet compression codec.

    Returns
    -------
    The profile manifest.
    """
    import spatialdata

    store = Path(store)
    sdata = spatialdata.read_zarr(store)

    if points_element not in sdata.points:
        raise ValueError(f"points element {points_element!r} not found; have {list(sdata.points)}")
    points = sdata.points[points_element]
    table = sdata.tables[table_element] if table_element else None

    transform = DisplayTransform.from_element(points, coordinate_system)
    grid = _grid_for(points, transform, tile_size_px)

    if table is not None:
        catalog = FeatureCatalog.from_points_and_table(points, table, feature_key=feature_key)
    else:
        observed = points[feature_key]
        names = sorted(observed.cat.as_known().cat.categories) if hasattr(observed, "cat") else []
        catalog = FeatureCatalog(names=tuple(names), n_genes=len(names))

    canonical_only = profile_layout == "canonical"
    if profile_layout not in ("v1", "canonical"):
        raise ValueError(f"profile_layout must be 'v1' or 'canonical', not {profile_layout!r}")

    profile_dir = store / PROFILE_DIR / PROFILE_NAME
    if not canonical_only:
        profile_dir.mkdir(parents=True, exist_ok=True)

    # Reading x, y and feature_name costs 27.7 KiB with z between them and 19.7 KiB
    # without, because parquet-wasm coalesces a projection into one contiguous byte range.
    render_first = ["x", "y", feature_key] if canonical_only else None
    geometry_encoding = "geoarrow" if canonical_only else "WKB"

    # The render columns go to a standalone file inside the profile directory. A viewer
    # reads every column of it, so it needs no column projection, and the canonical
    # element is left free of nested Arrow columns.
    transcripts = None
    if not canonical_only:
        transcripts = write_points_regular_grid(
            points,
            profile_dir / "trx",
            catalog=catalog,
            grid=grid,
            display_transform=transform,
            feature_key=feature_key,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            render_only=True,
            overwrite=True,
        )
    # The canonical element is re-ordered into tile row groups but keeps only its own
    # columns, so it still round-trips through SpatialData.write() and normal reads.
    canonical_points = write_points_regular_grid(
        points,
        store / "points" / points_element / "points.parquet",
        catalog=catalog,
        grid=grid,
        display_transform=transform,
        feature_key=feature_key,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        column_order=render_first,
        overwrite=True,
    )

    if canonical_only:
        # Describe the canonical file instead of a display file. Coordinates are two
        # separate columns rather than one interleaved column, which is what a client has
        # to know to read them: there is no display_xy to fall back on.
        transcripts = {
            **{k: v for k, v in canonical_points.items() if not k.startswith("position_")},
            "directory": f"points/{points_element}/points.parquet",
            "position_encoding": "separate_columns",
            "position_columns": ["x", "y"],
            "feature_column": feature_key,
            "feature_encoding": "dictionary",
            "render_only": False,
        }

    cell_segmentation = None
    if shapes_element:
        if shapes_element not in sdata.shapes:
            raise ValueError(f"shapes element {shapes_element!r} not found; have {list(sdata.shapes)}")
        shapes = sdata.shapes[shapes_element]
        shapes_transform = DisplayTransform.from_element(shapes, coordinate_system)
        cell_positions = _cell_positions_for_shapes(table, shapes_element) if table is not None else None
        if not canonical_only:
            cell_segmentation = write_shapes_regular_grid(
                shapes,
                profile_dir / "cell_seg",
                grid=grid,
                display_transform=shapes_transform,
                cell_index=cell_positions,
                max_row_groups_per_file=max_row_groups_per_file,
                compression=compression,
                render_only=True,
                overwrite=True,
            )
        canonical_shapes = write_shapes_regular_grid(
            shapes,
            store / "shapes" / shapes_element / "shapes.parquet",
            grid=grid,
            display_transform=shapes_transform,
            cell_index=cell_positions,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            geometry_encoding=geometry_encoding,
            overwrite=True,
        )

        if canonical_only:
            cell_segmentation = {
                **{
                    k: v
                    for k, v in canonical_shapes.items()
                    if k not in {"directory", "path", "geometry_column", "geometry_is_lossy", "geometry_note"}
                },
                "geometry_column": "geometry",
                "geometry_encoding": "geoarrow.polygon",
                "cell_id_column": CELL_CODE_COLUMN,
                "display_transform": shapes_transform.to_manifest_dict(),
                "geometry_is_lossy": False,
                "render_only": False,
            }
            if "files" in canonical_shapes:
                cell_segmentation["directory"] = f"shapes/{shapes_element}/shapes.parquet"
            else:
                cell_segmentation["path"] = f"shapes/{shapes_element}/shapes.parquet"

    # Gene-major access and a gene list both require reading every non-zero from a CSR
    # matrix. Precomputing the statistics and storing a CSC copy turns "download the
    # whole matrix" into "read two chunks", which is what makes this scale.
    expression_index: dict[str, Any] | None = _expression_index
    if index_expression and table is not None and table_element:
        csc, expression_index = _prepare_expression_index(table, cluster_column)
        # SpatialData refuses to overwrite an element inside the store it was read from
        # (scverse/spatialdata#520), so the table is deleted and rewritten. The statistics
        # pass has already materialised X, so nothing is read back from the deleted path.
        sdata.delete_element_from_disk(table_element)
        sdata.write_element(table_element)

        # Written after the table, and directly, because the chunking is the whole point:
        # AnnData sizes chunks for whole-matrix reads, which costs 24x too much per gene.
        layer = write_csc_layer(store / "tables" / table_element, csc) if csc is not None else None
        if layer:
            expression_index["csc"] = layer

    native_components = ["images"]
    if table is not None:
        native_components[:0] = ["metadata", "cbg"]
    spatialdata_manifest: dict[str, Any] = {
        "store_url": "." if canonical_only else "../..",
        **({} if canonical_only else {"native": native_components}),
        **({"table": table_element} if table_element else {}),
        **({"cluster_column": cluster_column} if cluster_column else {}),
        **({"expression_index": expression_index} if expression_index else {}),
    }

    manifest = build_manifest(
        grid=grid,
        technology=technology,
        transcripts=transcripts,
        cell_segmentation=cell_segmentation,
        # Gene and cell metadata, expression and images are all read from the store itself,
        # so the profile declares where the store is rather than duplicating its contents.
        feature_catalog={
            "n_genes": catalog.n_genes,
            "extra_features": list(catalog.names[catalog.n_genes :]),
            **({"names": list(catalog.names)} if table is None else {}),
        },
        spatialdata=spatialdata_manifest,
        source={
            "store": store.name,
            "technology": technology,
            "points_element": points_element,
            "shapes_element": shapes_element,
            "table_element": table_element,
            "coordinate_system": coordinate_system,
            "tile_size_px": tile_size_px,
        },
    )
    if canonical_only:
        # The root attribute is a storage/access description, not a Celldega settings
        # file. The Celldega adapter supplies its own defaults after discovering this
        # profile. Keep the v1 file manifest unchanged for DegaFiles compatibility.
        for key in (
            "technology",
            "use_row_groups",
            "use_int_index",
            "segmentation_approach",
            "tile_size",
            "image_info",
            "image_format",
        ):
            manifest.pop(key, None)
        manifest["row_group_files"].pop("images", None)
        # Canonical paths are relative to the store root, which is also where this
        # manifest lives. Validate before publishing it so a malformed profile cannot
        # become a blank viewport in the browser.
        validate_manifest(manifest, base_path=store)
        write_root_manifest(store, manifest)
    else:
        validate_manifest(manifest, base_path=profile_dir)
        write_manifest(manifest, profile_dir)
    return manifest


def xenium_spatially_tiled(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    compression: str = "zstd",
    overwrite: bool = False,
    tiling: dict[str, Any] | None = None,
    **xenium_kwargs: Any,
) -> dict[str, Any]:
    """Read raw Xenium data and write a spatially tiled SpatialData store in one call.

    Parameters
    ----------
    raw_path
        Directory of raw Xenium output.
    output_path
        Destination ``.zarr`` store.
    tile_size_px
        Tile edge length in display pixels.
    max_row_groups_per_file
        Row groups per chunk file.
    compression
        Parquet compression codec.
    overwrite
        Replace ``output_path`` if it exists.
    tiling
        Extra keyword arguments for :func:`add_spatial_tiling`, for example
        ``{"shapes_element": "cell_polygons"}``. Kept separate from ``xenium_kwargs``
        because the two functions have distinct option sets.
    xenium_kwargs
        Forwarded to :func:`spatialdata_io.xenium`.

    Returns
    -------
    The profile manifest.
    """
    from spatialdata_io.readers.xenium import xenium

    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
        shutil.rmtree(output_path)

    tiling_options = dict(tiling or {})
    index_expression = bool(tiling_options.pop("index_expression", True))
    table_element = tiling_options.get("table_element", "table")
    cluster_column = tiling_options.get("cluster_column")

    sdata = xenium(raw_path, **xenium_kwargs)
    expression_index: dict[str, Any] | None = None
    csc = None
    if index_expression and table_element is not None:
        table = sdata.tables[table_element]
        csc, expression_index = _prepare_expression_index(table, cluster_column)
    sdata.write(output_path)

    # The one-shot path writes table annotations with the initial store, so it never
    # deletes and rewrites that table. Only the CSC buffers are replaced afterward to
    # give them the small chunks required for per-gene browser reads.
    if csc is not None and table_element is not None:
        layer = write_csc_layer(output_path / "tables" / table_element, csc)
        expression_index = expression_index or {}
        expression_index["csc"] = layer

    return add_spatial_tiling(
        output_path,
        tile_size_px=tile_size_px,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        index_expression=False,
        _expression_index=expression_index,
        **tiling_options,
    )
