"""
07_dask_geospatial.py
======================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Demonstrate Dask for parallel geospatial processing. Dask extends
    pandas/numpy with lazy evaluation and parallel execution — making it
    possible to process datasets larger than RAM on a single machine,
    or to parallelize across CPU cores for speedup.

DASK CONCEPTS:
    - dask.dataframe: pandas-like API but lazy (builds computation graph)
    - dask.delayed: wrap any Python function to make it lazy
    - compute(): trigger actual execution of the lazy computation graph
    - Partitions: each partition is a regular pandas/GeoDataFrame in memory

GEOSPATIAL + DASK APPROACH:
    GeoPandas + Dask don't have native integration in older versions.
    The pattern demonstrated here:
    1. Split GeoDataFrame into spatial grid tiles (partitions)
    2. Apply geometry operations on each tile using dask.delayed
    3. Collect results in parallel

    In production (using dask-geopandas package):
    import dask_geopandas
    ddf = dask_geopandas.from_geopandas(gdf, npartitions=8)
    result = ddf.geometry.area.compute()

FALLBACK:
    If dask is not installed, demonstrates the partitioning strategy
    using pandas/geopandas alone (serial but same structure).

SCALING TO PETABYTES:
    Dask on a single machine → Dask on a cluster (Kubernetes, YARN)
    → Spark on EMR / Databricks for true petabyte scale
    The programming model is similar — Dask is a great bridge.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import time
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# CHECK DASK AVAILABILITY
# ---------------------------------------------------------------------------

DASK_AVAILABLE = False
try:
    import dask
    import dask.dataframe as dd
    from dask import delayed, compute
    DASK_AVAILABLE = True
    print(f"Dask available: version {dask.__version__}")
except ImportError:
    print("Dask not installed. Running serial fallback.")
    print("Install with: pip install 'dask[dataframe]'")
    print("Showing the same partitioning strategy with pandas.\n")

# ---------------------------------------------------------------------------
# SECTION 1: Generate Large Test GeoDataFrame
# ---------------------------------------------------------------------------

N = 20_000
print(f"Generating {N:,} test polygons...")

rng = np.random.default_rng(42)

# Create polygons spread across a 200km × 200km area
x0 = rng.uniform(0, 200_000, N)
y0 = rng.uniform(0, 200_000, N)
widths = rng.uniform(100, 1000, N)
heights = rng.uniform(100, 800, N)

polygons = [box(x, y, x + w, y + h)
            for x, y, w, h in zip(x0, y0, widths, heights)]

gdf = gpd.GeoDataFrame({
    "field_id":  [f"F{i:06d}" for i in range(N)],
    "crop_type": rng.choice(["corn", "wheat", "soy", "cotton"], N),
    "area_ha":   rng.uniform(1, 200, N).round(2),
    "yield_tha": rng.uniform(1, 20, N).round(3),
}, geometry=polygons, crs="EPSG:32611")

print(f"GeoDataFrame: {gdf.shape}")

# ---------------------------------------------------------------------------
# SECTION 2: Spatial Grid Partitioning
# ---------------------------------------------------------------------------

def partition_by_spatial_grid(gdf: gpd.GeoDataFrame, n_cols: int, n_rows: int):
    """
    Partition a GeoDataFrame into a spatial grid of n_cols × n_rows tiles.
    Features are assigned to the tile containing their centroid.
    Returns a dict of {tile_id: GeoDataFrame}.
    """
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds

    # Compute cell dimensions
    cell_w = (maxx - minx) / n_cols
    cell_h = (maxy - miny) / n_rows

    centroids_x = gdf.geometry.centroid.x
    centroids_y = gdf.geometry.centroid.y

    # Assign each feature to a grid cell
    col_idx = np.clip(((centroids_x - minx) / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip(((centroids_y - miny) / cell_h).astype(int), 0, n_rows - 1)
    tile_ids = row_idx * n_cols + col_idx

    gdf_copy = gdf.copy()
    gdf_copy["_tile_id"] = tile_ids

    partitions = {}
    for tid in sorted(gdf_copy["_tile_id"].unique()):
        partitions[tid] = gdf_copy[gdf_copy["_tile_id"] == tid].drop(
            columns=["_tile_id"]
        ).reset_index(drop=True)

    return partitions

# Create 4×4 = 16 spatial partitions
N_COLS, N_ROWS = 4, 4
partitions = partition_by_spatial_grid(gdf, N_COLS, N_ROWS)

print(f"\nSpatial partitions: {len(partitions)} tiles ({N_COLS}×{N_ROWS} grid)")
tile_sizes = [len(p) for p in partitions.values()]
print(f"Features per tile: min={min(tile_sizes)}, max={max(tile_sizes)}, "
      f"mean={np.mean(tile_sizes):.0f}")

# ---------------------------------------------------------------------------
# SECTION 3: Define Processing Function (applied per partition)
# ---------------------------------------------------------------------------

def process_geospatial_partition(gdf_partition: gpd.GeoDataFrame,
                                  partition_id: int) -> pd.DataFrame:
    """
    Compute geometry features for one spatial partition.
    This function is stateless — can run in parallel.

    Returns a DataFrame with derived attributes (no geometry).
    """
    if len(gdf_partition) == 0:
        return pd.DataFrame()

    result = gdf_partition[["field_id", "crop_type", "area_ha"]].copy()

    # Vectorized geometry operations (fast)
    geoms = gdf_partition.geometry
    result["computed_area_m2"] = geoms.area
    result["computed_area_ha"] = result["computed_area_m2"] / 10000
    result["perimeter_m"] = geoms.length
    result["compactness"] = (
        4 * math.pi * result["computed_area_m2"]
        / result["perimeter_m"].pow(2)
    ).round(6)

    centroid_x = geoms.centroid.x
    centroid_y = geoms.centroid.y
    result["centroid_x"] = centroid_x.values
    result["centroid_y"] = centroid_y.values

    # Distance to tile center (partition-relative metric)
    tile_cx = centroid_x.mean()
    tile_cy = centroid_y.mean()
    result["dist_to_tile_center_m"] = np.sqrt(
        (centroid_x - tile_cx)**2 + (centroid_y - tile_cy)**2
    ).round(2)

    result["partition_id"] = partition_id
    result["n_in_partition"] = len(gdf_partition)

    return result

# ---------------------------------------------------------------------------
# SECTION 4A: SERIAL Processing (baseline)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("APPROACH 1: Serial Processing (single thread)")
print("="*60)

start_serial = time.perf_counter()

serial_results = []
for tile_id, partition in sorted(partitions.items()):
    result = process_geospatial_partition(partition, tile_id)
    serial_results.append(result)

df_serial = pd.concat(serial_results, ignore_index=True)
elapsed_serial = time.perf_counter() - start_serial

print(f"Time: {elapsed_serial:.3f}s")
print(f"Output rows: {len(df_serial):,}")
print(f"Partitions processed: {len(partitions)}")

# ---------------------------------------------------------------------------
# SECTION 4B: DASK Processing (parallel)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
if DASK_AVAILABLE:
    print("APPROACH 2: Dask Parallel Processing (dask.delayed)")
    print("="*60)

    start_dask = time.perf_counter()

    # Wrap the processing function with @delayed to make it lazy
    # Each delayed call creates a node in the computation graph
    delayed_tasks = []
    for tile_id, partition in sorted(partitions.items()):
        # @delayed wraps the function — does NOT run it yet
        task = delayed(process_geospatial_partition)(partition, tile_id)
        delayed_tasks.append(task)

    # compute() executes the full graph, automatically parallelizing
    # across available CPU cores using Dask's task scheduler
    print(f"  Submitting {len(delayed_tasks)} delayed tasks...")
    start_compute = time.perf_counter()
    results_list = compute(*delayed_tasks)  # Tuple of results
    elapsed_dask = time.perf_counter() - start_compute

    df_dask = pd.concat(results_list, ignore_index=True)
    elapsed_dask_total = time.perf_counter() - start_dask

    speedup = elapsed_serial / elapsed_dask_total
    print(f"Compute time: {elapsed_dask:.3f}s")
    print(f"Total time (setup + compute): {elapsed_dask_total:.3f}s")
    print(f"Output rows: {len(df_dask):,}")
    print(f"Theoretical speedup potential: {len(partitions)} cores")
    print(f"Actual speedup: {speedup:.1f}×")

    # Verify correctness
    serial_sorted = df_serial.sort_values("field_id").reset_index(drop=True)
    dask_sorted = df_dask.sort_values("field_id").reset_index(drop=True)

    if serial_sorted["field_id"].equals(dask_sorted["field_id"]):
        print("Correctness check: field_id lists match")
    else:
        print("WARNING: Result mismatch between serial and dask")

else:
    print("APPROACH 2: Simulating Dask with concurrent.futures (fallback)")
    print("="*60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_parallel = time.perf_counter()

    parallel_results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_geospatial_partition, partition, tile_id): tile_id
            for tile_id, partition in sorted(partitions.items())
        }
        for future in as_completed(futures):
            tile_id = futures[future]
            parallel_results[tile_id] = future.result()

    df_parallel = pd.concat(
        [parallel_results[k] for k in sorted(parallel_results.keys())],
        ignore_index=True
    )
    elapsed_parallel = time.perf_counter() - start_parallel
    speedup = elapsed_serial / elapsed_parallel

    print(f"Time (4 threads): {elapsed_parallel:.3f}s")
    print(f"Speedup: {speedup:.1f}×")
    print(f"Output rows: {len(df_parallel):,}")

# ---------------------------------------------------------------------------
# SECTION 5: Dask Scaling Architecture Commentary
# ---------------------------------------------------------------------------

print("\n=== DASK SCALING ARCHITECTURE ===")
print("""
Single machine (this demo):
    dask.delayed → ThreadPoolExecutor → N CPU cores
    Typical speedup: 2-8× on 8-core machine

Multi-machine cluster:
    from dask.distributed import Client
    client = Client("tcp://scheduler:8786")
    # Now compute() distributes work across cluster workers
    # Typical speedup: 10-100× on 10-100 node cluster

Data Lakehouse integration:
    # Dask reads Parquet directly from S3 in parallel:
    import dask.dataframe as dd
    ddf = dd.read_parquet("s3://data-lake/fields/county=*/")
    # Each Parquet file partition processed by a different worker
    result = ddf.groupby("crop_type")["yield_tha"].mean().compute()

    # For spatial operations at true scale, use Apache Sedona on Spark:
    # spark.sql("SELECT ST_Area(geom) FROM iceberg.fields WHERE county='Imperial'")
""")

# ---------------------------------------------------------------------------
# SECTION 6: Visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Spatial partition map
ax1 = axes[0]
bounds = gdf.total_bounds
colors = plt.cm.tab20(np.linspace(0, 1, len(partitions)))

for (tile_id, partition), color in zip(sorted(partitions.items()), colors):
    if len(partition) > 0:
        # Plot a sample of features from this partition
        sample = partition.sample(min(50, len(partition)), random_state=42)
        for _, row in sample.iterrows():
            geom = row.geometry
            x, y = geom.exterior.xy
            ax1.fill(x, y, color=color, alpha=0.4)

# Draw grid lines
minx, miny, maxx, maxy = bounds
cell_w = (maxx - minx) / N_COLS
cell_h = (maxy - miny) / N_ROWS

for i in range(N_COLS + 1):
    x = minx + i * cell_w
    ax1.axvline(x, color="black", linewidth=1.5, zorder=5)
for j in range(N_ROWS + 1):
    y = miny + j * cell_h
    ax1.axhline(y, color="black", linewidth=1.5, zorder=5)

# Label tiles
for row in range(N_ROWS):
    for col in range(N_COLS):
        tile_id = row * N_COLS + col
        cx = minx + col * cell_w + cell_w / 2
        cy = miny + row * cell_h + cell_h / 2
        n = tile_sizes[tile_id] if tile_id < len(tile_sizes) else 0
        ax1.text(cx, cy, f"T{tile_id}\n({n})", ha="center", va="center",
                 fontsize=7, fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

ax1.set_title(f"Spatial Grid Partitioning\n{N_COLS}×{N_ROWS} tiles, {N:,} features",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("X (meters)")
ax1.set_ylabel("Y (meters)")

# Right: Performance comparison
ax2 = axes[1]
methods = ["Serial\n(1 thread)", "Parallel\n(4 threads/Dask)"]
times = [elapsed_serial, elapsed_dask_total if DASK_AVAILABLE else elapsed_parallel]
colors_bar = ["#e74c3c", "#2ecc71"]

bars = ax2.bar(methods, times, color=colors_bar, edgecolor="white",
               linewidth=2, width=0.4)
for bar, t in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{t:.3f}s", ha="center", va="bottom",
             fontsize=12, fontweight="bold")

spd = times[0] / times[1] if times[1] > 0 else 1
ax2.set_title(f"Processing Time Comparison\n{spd:.1f}× parallel speedup",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Time (seconds)")
ax2.grid(axis="y", alpha=0.3)

ax2.text(0.5, 0.85,
         "Scale to cluster:\n100×+ speedup possible\nwith Dask distributed",
         transform=ax2.transAxes, ha="center", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.8))

plt.suptitle(
    "Dask Parallel Geospatial Processing\n"
    f"Spatial Grid Partitioning → Parallel Execution | Emmanuel Oyekanlu",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("dask_parallel_processing.png", dpi=150, bbox_inches="tight")
print("\nSaved: dask_parallel_processing.png")

print("\n=== Script 07 Complete ===")
