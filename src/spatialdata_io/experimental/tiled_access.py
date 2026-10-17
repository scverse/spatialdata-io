"""Opt-in spatial tiling for a SpatialData store.

Two entry points, both a single call:

:func:`add_spatial_tiling`
    Add the profile to a store that already exists.
:func:`xenium_spatially_tiled`
    Read raw Xenium and write a tiled store in one go.

The one-shot path is internally ``read -> write -> tile``, because the tiling rewrites
*written* Parquet.

Everything here is additive and opt-in. A store that has been tiled is still an ordinary
SpatialData store: :func:`spatialdata.read_zarr` works unchanged, the canonical columns and
geometries are untouched, and a client that does not know about the profile simply ignores
the extra columns and the manifest.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from spatialdata_io.experimental.cbg_parquet import write_cbg_row_groups
from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.manifest import (
    PROFILE_NAME,
    build_manifest,
    validate_manifest,
    write_manifest,
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
    write_cell_metadata,
    write_shapes_regular_grid,
)

__all__ = ["add_spatial_tiling", "xenium_spatially_tiled"]

#: Directory inside the store holding derived (non-canonical) profile assets.
PROFILE_DIR = "visualization"

#: Display colours cycled through when a channel has none assigned. First is blue, which
#: is the conventional nuclear stain colour and usually channel 0 (DAPI).
_DEFAULT_CHANNEL_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def _channels_of(element: Any) -> list[Any]:
    """List an image element's channel names, falling back to indices."""
    level = element[next(iter(element.children))] if hasattr(element, "children") else element
    array = level[next(iter(level.data_vars))] if hasattr(level, "data_vars") else level
    coords = getattr(array, "coords", {})
    if "c" in coords:
        return [str(c) for c in coords["c"].values]
    size = dict(zip(array.dims, array.shape, strict=True)).get("c", 1)
    return list(range(size))


def _channel_label(channel: Any, index: int) -> str:
    """A filesystem- and URL-safe label for a channel.

    Xenium channel names include slashes ('ATP1A1/CD45/E-Cadherin'), which would other-
    wise create nested directories and break the manifest's relative paths.
    """
    if isinstance(channel, int):
        return f"channel_{channel}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(channel)).strip("_").lower()
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or f"channel_{index}"


def _grid_for(points: Any, transform: DisplayTransform, tile_size_px: float) -> RegularGrid:
    """Derive the grid covering the points element in display pixel space."""
    x = points["x"].max().compute() if hasattr(points["x"].max(), "compute") else points["x"].max()
    y = points["y"].max().compute() if hasattr(points["y"].max(), "compute") else points["y"].max()
    px, py = transform.apply(np.array([float(x)]), np.array([float(y)]))
    return RegularGrid.from_bounds(0, 0, float(np.rint(px[0])), float(np.rint(py[0])), tile_size_px)


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
    include_cbg: bool = True,
    image_element: str | None = None,
    image_channels: list[int | str] | None = None,
    image_colors: dict[str, tuple[int, int, int]] | None = None,
    image_tile_size: int = 512,
    compression: str = "zstd",
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
        Name of the annotating table, used for the gene order and the CBG. ``None`` skips
        the CBG and derives feature codes from the observed features alone.
    coordinate_system
        Coordinate system defining display pixel space.
    tile_size_px
        Tile edge length in display pixels. The default of 250 gives roughly 20 cells per
        tile on Xenium-density tissue, which is the granularity the viewer fetches at.
    max_row_groups_per_file
        Row groups per chunk file.
    feature_key
        Column in the points element holding the feature name.
    technology
        Celldega technology string recorded in the manifest.
    include_cbg
        Whether to write the gene-major cell-by-gene files.
    image_element
        Name of an image element to render into a WebP display pyramid, or ``None`` to
        skip images. The canonical OME-Zarr image is left untouched either way.
    image_channels
        Channels to render, by name or index. Defaults to every channel in the element.
    image_colors
        Optional display colour per channel name. Channels without an entry get a colour
        from a default palette.
    image_tile_size
        Image tile edge length in pixels.
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

    profile_dir = store / PROFILE_DIR / PROFILE_NAME
    profile_dir.mkdir(parents=True, exist_ok=True)

    # The render columns go to a standalone file inside the profile directory. A viewer
    # reads every column of it, so it needs no column projection, and the canonical
    # element is left free of nested Arrow columns.
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
    write_points_regular_grid(
        points,
        store / "points" / points_element / "points.parquet",
        catalog=catalog,
        grid=grid,
        display_transform=transform,
        feature_key=feature_key,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        overwrite=True,
    )

    cell_segmentation = None
    cell_metadata = None
    cell_names: list[str] | None = None
    if shapes_element:
        if shapes_element not in sdata.shapes:
            raise ValueError(f"shapes element {shapes_element!r} not found; have {list(sdata.shapes)}")
        shapes = sdata.shapes[shapes_element]
        shapes_transform = DisplayTransform.from_element(shapes, coordinate_system)
        # Cells are the overview representation, so the client needs every centroid up
        # front. Centroids are not stored anywhere in the SpatialData store (the Xenium
        # reader puts no x/y_centroid in obs), so they are derived from the geometry --
        # the same centroids already computed for tile assignment.
        cell_metadata = write_cell_metadata(
            shapes,
            profile_dir / "cell_metadata.parquet",
            display_transform=shapes_transform,
            cell_index=list(table.obs_names) if table is not None else None,
        )
        cell_names = [str(k) for k in (table.obs_names if table is not None else shapes.index)]
        cell_segmentation = write_shapes_regular_grid(
            shapes,
            profile_dir / "cell_seg",
            grid=grid,
            display_transform=shapes_transform,
            cell_index=list(table.obs_names) if table is not None else None,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            render_only=True,
            overwrite=True,
        )
        write_shapes_regular_grid(
            shapes,
            store / "shapes" / shapes_element / "shapes.parquet",
            grid=grid,
            display_transform=shapes_transform,
            cell_index=list(table.obs_names) if table is not None else None,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            overwrite=True,
        )

    cbg = None
    if include_cbg and table is not None:
        cbg = write_cbg_row_groups(
            table,
            profile_dir / "cbg",
            catalog=catalog,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            overwrite=True,
        )

    images: dict[str, Any] = {}
    image_info: list[dict[str, Any]] = []
    image_dimensions: dict[str, Any] | None = None
    max_pyramid_zoom: int | None = None
    if image_element:
        from spatialdata_io.experimental.webp_parquet import write_webp_pyramid

        if image_element not in sdata.images:
            raise ValueError(f"image element {image_element!r} not found; have {list(sdata.images)}")
        element = sdata.images[image_element]
        channels = image_channels if image_channels is not None else _channels_of(element)

        for index, channel in enumerate(channels):
            label = _channel_label(channel, index)
            pyramid = write_webp_pyramid(
                element,
                profile_dir / "images" / label,
                channel=channel,
                tile_size=image_tile_size,
                source_element=image_element,
                overwrite=True,
            )
            # ImageRowGroupReader resolves files as baseUrl/directory/file and reads the
            # per-zoom grid from the entry's zoom_info.
            pyramid["directory"] = f"images/{label}"
            images[label] = pyramid
            colour = (image_colors or {}).get(label) or _DEFAULT_CHANNEL_COLORS[index % len(_DEFAULT_CHANNEL_COLORS)]
            image_info.append({"name": label, "button_name": str(channel), "color": list(colour)})
            # Every channel of one element shares its dimensions and pyramid depth.
            image_dimensions = {
                "width": pyramid["source_width"],
                "height": pyramid["source_height"],
                "tile_size": image_tile_size,
            }
            max_pyramid_zoom = pyramid["max_zoom"]

    # The gene name is the index and must be preserved: a client reads the gene list from
    # it, and without it the viewer simply shows no transcript controls at all.
    catalog.to_frame(table.X if table is not None else None).to_parquet(profile_dir / "meta_gene.parquet")

    # Files a client reads at fixed paths rather than through the manifest. Writing them
    # is what lets the profile directory stand in for a DegaFiles root, so no client needs
    # to know it is looking at a SpatialData store.
    _write_micron_to_image_transform(profile_dir / "micron_to_image_transform.csv", transform)
    cluster_info = None
    if cell_names is not None:
        cluster_info = _write_cell_clusters(profile_dir / "cell_clusters", cell_names, None)

    manifest = build_manifest(
        grid=grid,
        technology=technology,
        transcripts=transcripts,
        cell_segmentation=cell_segmentation,
        cbg=cbg,
        images=images,
        image_info=image_info,
        image_dimensions=image_dimensions,
        max_pyramid_zoom=max_pyramid_zoom,
        fixed_path_assets={
            "meta_gene": "meta_gene.parquet",
            "micron_to_image_transform": "micron_to_image_transform.csv",
            **({"cell_metadata": cell_metadata} if cell_metadata else {}),
            **({"cell_clusters": cluster_info} if cluster_info else {}),
        },
        source={
            "store": store.name,
            "points_element": points_element,
            "shapes_element": shapes_element,
            "table_element": table_element,
            "image_element": image_element,
            "coordinate_system": coordinate_system,
            "tile_size_px": tile_size_px,
        },
    )
    validate_manifest(manifest, base_path=profile_dir)
    write_manifest(manifest, profile_dir)
    return manifest


def xenium_spatially_tiled(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    include_cbg: bool = True,
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
    include_cbg
        Whether to write the gene-major cell-by-gene files.
    compression
        Parquet compression codec.
    overwrite
        Replace ``output_path`` if it exists.
    tiling
        Extra keyword arguments for :func:`add_spatial_tiling`, for example
        ``{"image_element": "morphology_focus", "image_channel": "DAPI"}``. Kept separate
        from ``xenium_kwargs`` because the two functions have distinct option sets.
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

    sdata = xenium(raw_path, **xenium_kwargs)
    sdata.write(output_path)
    return add_spatial_tiling(
        output_path,
        tile_size_px=tile_size_px,
        max_row_groups_per_file=max_row_groups_per_file,
        include_cbg=include_cbg,
        compression=compression,
        **(tiling or {}),
    )


def _write_micron_to_image_transform(path: Path, transform: DisplayTransform) -> None:
    """Write the micron-to-image affine Celldega reads at a fixed path.

    Coordinates in the profile are already in display pixels, so this is not needed to
    place anything. It is needed for the scale bar and any physical-units readout, so it
    must be the real micron-to-pixel affine and not identity -- writing identity would
    make the scale bar wrong by the pixel size.
    """
    (a, b, c), (d, e, f) = transform.matrix
    rows = [f"{a} {b} {c}", f"{d} {e} {f}", "0.0 0.0 1.0"]
    path.write_text("\n".join(rows) + "\n")


def _write_cell_clusters(directory: Path, cell_names: list[str], clusters: Any | None) -> dict[str, Any]:
    """Write the cluster assignment and palette Celldega reads at a fixed path.

    SpatialData does not require a clustering, and the Xenium reader does not load one, so
    when none is supplied every cell is placed in a single group. That keeps the viewer's
    category machinery working instead of failing on a missing file; it is a placeholder,
    not a scientific result.
    """
    import colorsys

    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    if clusters is None:
        labels = pd.Series(["unclustered"] * len(cell_names), index=cell_names, dtype=object)
    else:
        labels = pd.Series([str(v) for v in clusters], index=cell_names, dtype=object)

    counts = labels.value_counts()
    palette = {}
    for i, name in enumerate(counts.index):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.6, 0.9)
        palette[name] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    pd.DataFrame({"cluster": labels}).to_parquet(directory / "cluster.parquet")
    pd.DataFrame(
        {"color": [palette[n] for n in counts.index], "count": counts.to_numpy()},
        index=list(counts.index),
    ).to_parquet(directory / "meta_cluster.parquet")

    return {"n_clusters": int(counts.size), "placeholder": clusters is None}
