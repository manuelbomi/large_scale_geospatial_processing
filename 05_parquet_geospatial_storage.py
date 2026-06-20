"""
05_parquet_geospatial_storage.py
==================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Benchmark GeoParquet vs GeoJSON vs Shapefile for geospatial data storage.
    Demonstrate columnar storage advantages for attribute-only queries.
    Connect these concepts to Apache Iceberg data lakehouse architecture.

WHY GEOPARQUET IS BETTER FOR DATA LAKEHOUSES:
    1. SIZE: Columnar + Snappy compression = 5-10× smaller than GeoJSON
    2. SPEED: Columnar scan for attribute queries skips geometry bytes entirely
    3. SCHEMA: Typed columns with Parquet schema (no string-parsing overhead)
    4. PARTITIONING: Parquet files can be partitioned (by county, by date)
    5. ICEBERG: Parquet is the native storage format for Iceberg tables
    6. SPARK: Native Spark format; Sedona reads GeoParquet directly

GEOPARQUET SPEC:
    GeoParquet stores geometry as WKB (Well-Known Binary) in a binary column.
    A "geo" metadata key in the Parquet file metadata describes:
    - Which column contains geometry
    - The CRS (as WKT or PROJJSON)
    - Geometry types present
    - Bounding box of the dataset

APACHE ICEBERG CONNECTION:
    An Iceberg table over GeoParquet files looks like:
    s3://data-lake/
    ├── geospatial/
    │   ├── fields/
    │   │   ├── data/          <- GeoParquet data files
    │   │   │   ├── county=Imperial/
    │   │   │   │   ├── part-0000.parquet
    │   │   │   └── county=Riverside/
    │   │   │       └── part-0001.parquet
    │   │   └── metadata/      <- Iceberg table metadata (JSON manifests)
    │   │       ├── v1.metadata.json
    │   │       └── snap-*.avro

    Iceberg then provides:
    - Time travel: SELECT ... AS OF TIMESTAMP '2024-01-01'
    - Schema evolution: ALTER TABLE ... ADD COLUMN ...
    - Partition pruning: WHERE county='Imperial' skips all other partitions
    - ACID transactions: concurrent reads during writes
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box
import time
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# SECTION 1: Generate Test GeoDataFrame
# ---------------------------------------------------------------------------

N = 10_000
print(f"Generating {N:,} polygon GeoDataFrame for storage benchmark...")

rng = np.random.default_rng(42)
base_lon, base_lat = -115.5, 33.0

# Generate realistic field polygons with attributes
lons = rng.uniform(base_lon - 0.5, base_lon + 0.5, N)
lats = rng.uniform(base_lat - 0.5, base_lat + 0.5, N)
sizes = rng.uniform(0.001, 0.005, N)

polygons = [
    box(lon, lat, lon + size, lat + size * 0.7)
    for lon, lat, size in zip(lons, lats, sizes)
]

crop_types = rng.choice(["corn", "wheat", "soybean", "cotton", "alfalfa",
                          "lettuce", "tomatoes"], N)
counties = rng.choice(["Imperial", "Riverside", "San Diego", "Coachella"], N)
years = rng.choice([2023, 2024, 2025], N)

gdf = gpd.GeoDataFrame({
    "field_id":    [f"F{i:06d}" for i in range(N)],
    "crop_type":   crop_types,
    "county":      counties,
    "year":        years.astype(int),
    "area_ha":     rng.uniform(1, 200, N).round(2),
    "yield_tha":   rng.uniform(1, 20, N).round(3),
    "quality":     rng.uniform(0, 1, N).round(4),
    "elevation_m": rng.uniform(-50, 300, N).round(1),
}, geometry=polygons, crs="EPSG:4326")

print(f"GeoDataFrame: {gdf.shape[0]:,} rows × {gdf.shape[1]} columns")
print(f"Crop types: {gdf['crop_type'].value_counts().to_dict()}")

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------------------
# SECTION 2: Write to All Formats and Benchmark
# ---------------------------------------------------------------------------

def write_and_benchmark(label, write_fn, output_path):
    """Write data and return (elapsed_sec, file_size_mb)."""
    t0 = time.perf_counter()
    write_fn(output_path)
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [{label}] {elapsed:.3f}s | {size_mb:.2f} MB")
    return elapsed, size_mb

results = {}

# --- GeoJSON ---
print("\n--- WRITE BENCHMARKS ---")
geojson_path = "data/benchmark_fields.geojson"
t, s = write_and_benchmark(
    "GeoJSON",
    lambda p: gdf.to_file(p, driver="GeoJSON"),
    geojson_path
)
results["GeoJSON"] = {"write_sec": t, "size_mb": s}

# --- Shapefile ---
shp_path = "data/benchmark_fields.shp"
gdf_shp = gdf.copy()
gdf_shp.columns = [c[:10] for c in gdf_shp.columns]  # Truncate names
t, s = write_and_benchmark(
    "Shapefile",
    lambda p: gdf_shp.to_file(p, driver="ESRI Shapefile"),
    shp_path
)
results["Shapefile"] = {"write_sec": t, "size_mb": s}

# --- GeoPackage ---
gpkg_path = "data/benchmark_fields.gpkg"
t, s = write_and_benchmark(
    "GeoPackage",
    lambda p: gdf.to_file(p, driver="GPKG", layer="fields"),
    gpkg_path
)
results["GeoPackage"] = {"write_sec": t, "size_mb": s}

# --- GeoParquet (Snappy compression) ---
def write_geoparquet(output_path, compression="snappy"):
    """Write GeoDataFrame as GeoParquet with WKB geometry."""
    gdf_pq = gdf.copy()
    gdf_pq["geometry"] = gdf_pq.geometry.to_wkb()

    # Build geo metadata per GeoParquet spec v1.0
    geo_meta = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": gdf.crs.to_wkt(),
                "bbox": [float(gdf.total_bounds[0]), float(gdf.total_bounds[1]),
                         float(gdf.total_bounds[2]), float(gdf.total_bounds[3])],
            }
        }
    }

    # Non-geometry columns as PyArrow table
    df_no_geom = gdf_pq.drop(columns=["geometry"])
    pa_table = pa.Table.from_pandas(df_no_geom)

    # Add WKB geometry column
    geom_array = pa.array(gdf_pq["geometry"].tolist(), type=pa.binary())
    pa_table = pa_table.append_column("geometry", geom_array)

    # Embed geo metadata in Parquet file metadata
    existing_meta = pa_table.schema.metadata or {}
    meta_with_geo = {**existing_meta, b"geo": json.dumps(geo_meta).encode()}
    pa_table = pa_table.replace_schema_metadata(meta_with_geo)

    pq.write_table(pa_table, output_path, compression=compression)

parquet_path = "data/benchmark_fields.parquet"
t0 = time.perf_counter()
write_geoparquet(parquet_path)
t_pq = time.perf_counter() - t0
s_pq = os.path.getsize(parquet_path) / (1024 * 1024)
print(f"  [GeoParquet/Snappy] {t_pq:.3f}s | {s_pq:.2f} MB")
results["GeoParquet"] = {"write_sec": t_pq, "size_mb": s_pq}

# --- GeoParquet (uncompressed for comparison) ---
parquet_uncompressed_path = "data/benchmark_fields_uncompressed.parquet"
t0 = time.perf_counter()
write_geoparquet(parquet_uncompressed_path, compression="none")
t_pq_unc = time.perf_counter() - t0
s_pq_unc = os.path.getsize(parquet_uncompressed_path) / (1024 * 1024)
print(f"  [GeoParquet/None]   {t_pq_unc:.3f}s | {s_pq_unc:.2f} MB")
results["GeoParquet_uncompressed"] = {"write_sec": t_pq_unc, "size_mb": s_pq_unc}

# ---------------------------------------------------------------------------
# SECTION 3: READ BENCHMARKS
# ---------------------------------------------------------------------------

print("\n--- READ BENCHMARKS ---")

# Full file reads
for label, path, read_fn in [
    ("GeoJSON", geojson_path, lambda p: gpd.read_file(p)),
    ("Shapefile", shp_path, lambda p: gpd.read_file(p)),
    ("GeoPackage", gpkg_path, lambda p: gpd.read_file(p, layer="fields")),
    ("GeoParquet", parquet_path, lambda p: pq.read_table(p)),
]:
    t0 = time.perf_counter()
    data = read_fn(path)
    elapsed = time.perf_counter() - t0
    n_rows = len(data) if hasattr(data, "__len__") else "N/A"
    print(f"  [{label}] {elapsed:.3f}s | {n_rows:,} rows")
    if label in results:
        results[label]["read_full_sec"] = elapsed

# Attribute-only query (key columnar storage advantage!)
# In Parquet, reading only 'crop_type' and 'yield_tha' skips geometry bytes entirely
print("\n--- ATTRIBUTE-ONLY QUERY (crop_type + yield_tha only) ---")

# GeoJSON: must read entire file, then filter columns
t0 = time.perf_counter()
gdf_geojson = gpd.read_file(geojson_path)
result_geojson = gdf_geojson[["field_id", "crop_type", "yield_tha"]]
t_geojson_attr = time.perf_counter() - t0
print(f"  [GeoJSON attr-only]  {t_geojson_attr:.3f}s (reads ENTIRE file)")

# GeoParquet: reads only requested columns (columnar scan!)
t0 = time.perf_counter()
table_attr = pq.read_table(parquet_path, columns=["field_id", "crop_type", "yield_tha"])
result_parquet = table_attr.to_pandas()
t_parquet_attr = time.perf_counter() - t0
print(f"  [GeoParquet attr]    {t_parquet_attr:.3f}s (reads ONLY attribute columns)")

speedup_attr = t_geojson_attr / t_parquet_attr
print(f"  Columnar speedup: {speedup_attr:.1f}×")

# ---------------------------------------------------------------------------
# SECTION 4: Storage Efficiency Summary
# ---------------------------------------------------------------------------

geojson_size = results["GeoJSON"]["size_mb"]
print("\n" + "="*60)
print("  STORAGE COMPARISON")
print("="*60)
print(f"\n  {'Format':<25} {'Size (MB)':>10} {'vs GeoJSON':>12} {'Write (s)':>10}")
print(f"  {'-'*60}")
for fmt, data in results.items():
    ratio = data["size_mb"] / geojson_size
    print(f"  {fmt:<25} {data['size_mb']:>10.2f} {ratio:>11.1f}× {data['write_sec']:>10.3f}")

# ---------------------------------------------------------------------------
# SECTION 5: Apache Iceberg Architecture Notes
# ---------------------------------------------------------------------------

print("\n=== APACHE ICEBERG ARCHITECTURE ===")
print("""
GeoParquet is the BRIDGE between GeoPandas workflows and Apache Iceberg:

    GeoPandas (Python ETL)
         │
         │ write GeoParquet
         ▼
    S3 / ADLS / GCS
    └── geospatial/fields/
        ├── county=Imperial/
        │   └── 2024/part-0000.parquet   ← GeoParquet file
        └── county=Riverside/
            └── 2024/part-0001.parquet

         │
         │ Iceberg registers table over S3 path
         ▼
    Apache Iceberg Table (REST catalog / Glue catalog)
         │
         ├── Apache Spark + Sedona  ← complex spatial SQL at scale
         ├── DuckDB                 ← fast ad-hoc queries
         └── Trino / Athena        ← serverless SQL
""")

# ---------------------------------------------------------------------------
# SECTION 6: Visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

formats = [k for k in results if "_uncompressed" not in k]
sizes = [results[k]["size_mb"] for k in formats]
write_times = [results[k]["write_sec"] for k in formats]

# Storage size bar chart
ax1 = axes[0]
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
bars1 = ax1.bar(formats, sizes, color=colors, edgecolor="white", linewidth=1.5)
for bar, sz in zip(bars1, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{sz:.2f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylabel("File Size (MB)", fontsize=12)
ax1.set_title(f"Storage Size Comparison\n{N:,} Field Polygons + Attributes",
              fontsize=12, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# Annotate GeoJSON as reference
ax1.text(0.02, 0.98,
         f"GeoJSON = {geojson_size:.2f} MB\n(reference baseline)",
         transform=ax1.transAxes, va="top", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Write time bar chart
ax2 = axes[1]
bars2 = ax2.bar(formats, write_times, color=colors, edgecolor="white", linewidth=1.5)
for bar, t in zip(bars2, write_times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{t:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylabel("Write Time (seconds)", fontsize=12)
ax2.set_title("Write Performance Comparison",
              fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("GeoParquet vs GeoJSON vs Shapefile vs GeoPackage\n"
             "Emmanuel Oyekanlu — Data Lakehouse Geospatial Engineering",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("storage_format_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: storage_format_benchmark.png")

print("\n=== Script 05 Complete ===")
