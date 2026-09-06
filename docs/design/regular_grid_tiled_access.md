# Regular-grid tiled access profile (`celldega_regular_grid_v1`)

**Status:** experimental · **Profile version:** 0.1.0

A viewer-independent protocol for fetching spatially-local subsets of a SpatialData store
over HTTP range requests, without downloading whole files and without consulting Parquet
statistics.

Celldega is the first reference client. Nothing in this document is Celldega-specific;
another viewer (Vitessce, SpatialData.js, napari) could implement it from this text alone.

---

## 1. Motivation and scope

A viewer showing a 34,000 × 14,000 pixel tissue with 8 million transcripts must fetch only
what is on screen. Two things have to be true:

1. The client can compute **which bytes it needs** from the viewport alone — no index
   download, no metadata probing, no statistics.
2. Those bytes are **already in the form the GPU wants** — no per-point object
   construction, no coordinate zipping, no WKB parsing.

This profile achieves both by reordering rows into a deterministic grid of Parquet row
groups and adding a small number of render-oriented columns.

**In scope:** transcript points, cell polygons, gene-major expression, image tiles,
and the manifest that describes them.

**Out of scope:** clustering, linked views, annotation, neighbourhood analysis, and any
other viewer feature. This is a data-access protocol.

### Design invariants

- **The profile is opt-in.** Default SpatialData write behaviour is unchanged.
- **Canonical data is authoritative and preserved.** Render columns are additions; they
  never replace canonical coordinates, identifiers, geometries or annotations.
- **A store carrying the profile is still an ordinary SpatialData store.**
  `spatialdata.read_zarr()` works unchanged; a client that does not understand the profile
  ignores the extra columns and the manifest.

---

## 2. Coordinate system

All profile coordinates are **level-0 pixels of a chosen reference image**, referred to
here as *display pixel space*.

The mapping from canonical element coordinates to display pixels is the element's
SpatialData affine transformation into a named coordinate system (`global` by default).
It is recorded in the manifest:

```json
"display_transform": {
  "coordinate_space": "image-pixel",
  "coordinate_system": "global",
  "affine_matrix": [[4.70588235, 0.0, 0.0], [0.0, 4.70588235, 0.0]],
  "rounding": "nearest"
}
```

Producers **MUST** apply the affine in float64 and round half-to-even (`rint`). Applying
it in float32 loses pixel accuracy on large images, and in some environments a float32
array multiplied by a scalar stays float32.

Display coordinates **MUST** be non-negative and **MUST** fit the declared integer dtype.
A producer encountering values outside that range **MUST** fail rather than clamp: an
out-of-range coordinate indicates a mismatched transform, not a rounding artefact.

---

## 3. The grid

A non-overlapping regular square grid over display pixel space, defined by five numbers:

| field | meaning |
|---|---|
| `x_min`, `y_min` | grid origin, display pixels |
| `tile_size` | tile edge length, display pixels |
| `num_tiles_x`, `num_tiles_y` | grid dimensions |

### Tile assignment

```
tile_x = floor((x_px - x_min) / tile_size)
tile_y = floor((y_px - y_min) / tile_size)
```

Tile bounds are **half-open** `[min, max)`: a coordinate lying exactly on an internal
boundary belongs to the **upper** tile.

The single exception is the grid's outer edge. A coordinate equal to `x_max` or `y_max`
is clamped into the last tile, so that a point on the boundary of the dataset is not
lost. Coordinates beyond one tile past the extent are an error.

### Tile numbering

Tiles are numbered **x-major**:

```
tile_id = tile_x * num_tiles_y + tile_y
```

`tile_id` ranges over `[0, num_tiles_x * num_tiles_y)`.

### Choosing `tile_size`

`tile_size` is a tuning parameter, not a constant. It trades viewport granularity against
storage: smaller tiles fetch less off-screen data but fragment the file into more,
individually-compressed row groups.

The recommended target is **roughly 20 cells per tile**, which is the granularity at
which a viewer fetches. On Xenium-density tissue this is about **250 display pixels**.
Measured on a Xenium human pancreas section (140,702 cells):

| tile px | cells/tile | row groups | size vs untiled |
|---|---|---|---|
| 200 | 13.5 | 11,799 | +71% |
| **250** | **21.0** | **7,535** | **+60%** |
| 500 | 81.0 | 1,932 | +50% |

---

## 4. Row groups and files

### One tile, one row group

Each logical tile is written as **exactly one Parquet row group**, at index `tile_id`.
Tiles containing no rows are written as **zero-row row groups**, not skipped. This is what
lets a client address a tile by formula with no lookup table.

A conforming file therefore contains exactly `num_tiles_x * num_tiles_y` row groups across
all its parts.

### Multi-part files

Row groups are split across files:

```
file_index      = tile_id // max_row_groups_per_file
local_row_group = tile_id %  max_row_groups_per_file
```

`max_row_groups_per_file` defaults to **400**.

Splitting is **required**, not cosmetic. A Parquet reader must fetch a file's entire
footer before reading any row group, and footer size grows with row-group count. On the
pancreas dataset a single 7,535-row-group file has a **7.4 MB footer**; split into 19
files each footer is ~410 KB, and a client only fetches footers for files its viewport
actually touches.

### File naming

Chunk files are named `chunk_<n>.parquet` with `<n>` **zero-padded** to the width of the
largest index (`chunk_00.parquet` … `chunk_18.parquet`).

Padding is required because consumers disagree about ordering: a client indexes the
manifest's `files` array by position, but tools that glob a directory sort
lexicographically, where `chunk_10` precedes `chunk_2`. Padding makes the two agree.

> Existing Celldega DegaFiles use unpadded names. That is safe there because only the
> manifest-array consumer exists. Stores written under this profile use padded names.

### Statistics

Producers **SHOULD** write Parquet files with column statistics disabled. The tile formula
is the spatial index, so no conforming client reads column-chunk min/max, and statistics
inflate the footer the client must download before its first read.

### Compression

**zstd** is the recommended codec. On the pancreas dataset, snappy costs +38.5% over an
untiled store while zstd costs +4.9% for the same content, at no meaningful write cost.
Producers **MUST NOT** use a codec the target client cannot decode.

---

## 5. Transcript points

The canonical Points element gains two columns; every canonical column and the DataFrame
index are preserved. Physical row order changes (rows are grouped by tile), which is
permitted; rows **MUST NOT** be added, dropped or altered.

### `display_xy`

```
fixed_size_list<uint32>[2]
```

Integer display-pixel coordinates. The Arrow child buffer is therefore already
`[x0, y0, x1, y1, ...]`, directly usable as a deck.gl binary `getPosition` attribute.

A client **MUST NOT** need to interleave separate x and y arrays.

> A future revision may allow fixed-point sub-pixel coordinates
> (`stored = pixel * 16`, `scale = 0.0625`). Producers of v0.1.0 write integer pixels
> and declare `"scale": 1.0`.

### `feature_code`

```
uint16  (or uint32 when the catalog exceeds 65535 entries)
```

An index into the feature catalog (§7).

---

## 6. Cell polygons

The canonical Shapes element gains two columns. The canonical geometry column and its
GeoParquet `geo` metadata are preserved, so the file remains readable by
`geopandas.read_parquet` and by SpatialData.

### `display_geometry`

```
list<list<fixed_size_list<uint32>[2]>>
```

Polygon → rings → interleaved integer pixel vertices. A client lifts `getPolygon` from the
flat coordinate child buffer and `startIndices` from the list offsets:

```
start_index[i] = ring_offsets[polygon_offsets[i]]
```

`display_geometry` is explicitly a **lossy display representation**: it holds the exterior
ring only, and for a MultiPolygon only the largest part. The canonical geometry is retained
alongside it and is authoritative.

> **Interoperability warning.** GeoArrow permits `struct<x, y>` coordinates as well as
> interleaved `fixed_size_list`. A client walking the nesting blindly will, on struct
> coordinates, obtain the `x` child alone and render wrong polygons with no error. Clients
> **MUST** verify the vertex level is a `FixedSizeList` before treating the buffer as
> interleaved. (`geopandas.to_parquet` emits struct coordinates, so this is reachable in
> practice.)

### `cell_code`

```
uint32
```

Positional index into the annotating table's `obs` order, so cells can be coloured from an
expression vector without a string join in the client.

### Tile assignment

A cell is assigned to **exactly one tile**, by its **centroid** in display pixel space.

A polygon whose outline crosses into neighbouring tiles is **NOT** duplicated into them.
Duplication would inflate the file and make cell counts wrong. A client rendering a
viewport should expect polygons to overhang tile boundaries and, if it needs full coverage
at the edges, fetch one extra ring of tiles.

---

## 7. Feature catalog

An ordered vocabulary mapping feature names to `feature_code`, stored as
`meta_gene.parquet` with columns `name`, `feature_code`, `is_gene`.

Ordering is normative:

1. Codes `[0, n_genes)` are **genes**, in the annotating table's `var_names` order.
2. Codes `>= n_genes` are **non-gene features** (negative controls, unassigned codewords),
   sorted for reproducibility.

Because genes come first and in table order, **a gene's `feature_code` is also its CBG row
group** (§8). One integer addresses both a transcript's identity and its expression vector.

Non-gene features **MUST** be retained and **MUST NOT** be folded into a gene; doing so
would fabricate expression. `n_genes` is recorded in the manifest so a client can tell the
two apart.

---

## 8. Cell-by-gene expression

Gene-major, one row group per gene, so selecting a gene fetches one row group and touches
no transcript data.

Columns:

| column | type | meaning |
|---|---|---|
| `cell_id` | uint32 | cell code (§6), not a string barcode |
| `expression` | float32 | non-zero value |
| `gene` | string | gene name |

Zero values are omitted, including sparse *stored* zeros.

Row group `i` holds catalog gene `i`, including genes with no expression (written as a
zero-row row group), preserving the `feature_code == row group` invariant. The explicit
mapping is written both in the manifest and in the Parquet schema metadata:

```
gene_to_row_group   JSON object
num_genes           integer
storage_mode        "row_groups_cbg_chunked"
```

A client **MAY** use the mapping rather than assuming the identity.

> This is a **transpose**, not a redundant copy. SpatialData tables are stored cell-major
> (CSR); assembling one gene's vector from them requires reading the whole matrix or doing
> one random access per cell.

---

## 9. Image tiles

*(Not implemented in v0.1.0; specified here for forward compatibility.)*

Canonical OME-Zarr images are retained and authoritative. An optional derived WebP pyramid
may be stored as Parquet row groups with columns `zoom`, `tile_x`, `tile_y`, `image_data`
(encoded WebP bytes).

Image tiles at zoom 0 **MUST** use the same level-0 pixel coordinate system as
`display_xy` and `display_geometry`.

Per channel the manifest records: source image element, source dimensions and dtype,
reference pyramid level, tile size, per-zoom grid dimensions, display intensity min/max,
gamma, colour, downsampling method, and WebP lossless/lossy setting.

---

## 10. The manifest

A JSON document named `landscape_parameters.json`, conventionally at
`<store>.zarr/visualization/celldega_regular_grid_v1/`.

Paths inside it are **relative to the manifest's own directory**, so a client pointed at
that directory resolves into the store without knowing the zarr layout:

```json
{
  "technology": "Xenium",
  "use_row_groups": true,
  "profile": "celldega_regular_grid_v1",
  "profile_version": "0.1.0",
  "tile_grid": {
    "num_tiles_x": 137, "num_tiles_y": 55, "tile_size": 250.0,
    "x_min": 0.0, "y_min": 0.0, "x_max": 34250.0, "y_max": 13750.0
  },
  "row_group_files": {
    "transcripts": {
      "directory": "../../points/transcripts/points.parquet",
      "files": ["chunk_00.parquet", "..."],
      "max_row_groups_per_file": 400,
      "total_row_groups": 7535,
      "position_column": "display_xy",
      "feature_column": "feature_code",
      "columns": ["display_xy", "feature_code"]
    },
    "cell_segmentation": {
      "directory": "../../shapes/cell_boundaries/shapes.parquet",
      "geometry_column": "display_geometry",
      "cell_id_column": "cell_code",
      "columns": ["display_geometry", "cell_code"]
    },
    "cbg": { "directory": "cbg", "gene_to_row_group": {} },
    "images": {}
  },
  "image_info": []
}
```

`columns` is the projection a client should request. Honouring it is what keeps canonical
coordinates, identifiers and QC columns off the wire during rendering.

A client that does not recognise the profile-specific keys and finds no column names
declared **SHOULD** fall back to reading all columns.

---

## 11. Transport requirements

A conforming host **MUST** support:

- **HTTP range requests**, answering `Range: bytes=a-b` with `206 Partial Content` and a
  correct `Content-Range`.
- **Suffix ranges** (`Range: bytes=-8`), which is how a reader locates the Parquet footer.
- **CORS**, with `Access-Control-Allow-Origin` and `Access-Control-Expose-Headers`
  including `Content-Range`, for browser clients on another origin.

Hugging Face dataset `resolve/` URLs satisfy all three.

---

## 12. Invalidation

The profile is derived data and **MUST** be regenerated when any of the following change:

- transcript rows are added, removed, or spatially moved;
- the feature catalog or `var_names` order changes;
- canonical cell identifiers or table row order change (invalidates `cell_code`);
- cell geometries change;
- the reference image, or the transform into display pixel space, changes;
- `tile_size`, the grid origin, or `max_row_groups_per_file` change;
- image intensity windowing or WebP settings change (images only).

Changing a SpatialData transformation that does **not** affect the declared display
coordinate system does not require regeneration, but this **MUST** be verified rather than
assumed — compare the resulting affine against `display_transform.affine_matrix`.

Producers **SHOULD** record a source fingerprint in `source` so staleness is detectable.

---

## 13. Conformance checklist

A producer conforms if:

- [ ] total row groups equals `num_tiles_x * num_tiles_y`, empty tiles included
- [ ] row group index equals `tile_x * num_tiles_y + tile_y`
- [ ] every canonical row appears exactly once; index preserved
- [ ] canonical columns and geometries are bit-identical to the source
- [ ] `display_xy` is `fixed_size_list<uint32>[2]` with an interleaved child buffer
- [ ] `display_geometry` vertices are `fixed_size_list`, not `struct<x,y>`
- [ ] each cell appears exactly once, in its centroid's tile
- [ ] `feature_code` matches the catalog; genes precede non-genes
- [ ] CBG row group index equals `feature_code` for every gene
- [ ] the store still opens with `spatialdata.read_zarr()`
- [ ] the manifest validates and every declared file exists
