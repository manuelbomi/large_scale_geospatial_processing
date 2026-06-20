"""
02_spatial_indexing_optimization.py
=====================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Demonstrate the dramatic speedup from using spatial indexes for
    nearest-neighbor queries. This is one of the most impactful
    optimizations in geospatial engineering.

THE PROBLEM:
    "For each of N query points (sensor readings), find the nearest
    polygon (field boundary) from a set of M polygons."

    Use case at Corning: "Which warehouse zone is the AGV currently in?"
    Use case at Bayer: "Which field does this soil sensor reading belong to?"

BRUTE FORCE vs STRtree:
    Brute force: Check every query point against every polygon → O(N × M)
    STRtree:     Build index once O(M log M), then query O(log M) per point

R-TREE CONCEPT:
    An R-tree (Rectangle tree) stores geometry bounding boxes in a tree.
    To find which polygon contains point P:
    1. Check P against root bounding box (1 comparison)
    2. If inside, descend to child nodes (log(M) levels)
    3. At leaf level, do exact geometric test on ~3-5 candidates
    Total: O(log M) instead of O(M)

    STRtree = Sort-Tile-Recursive R-tree: better cache performance than
    standard R-tree for static datasets (no inserts after build).

BENCHMARK SETUP:
    - M = 1000 polygon "fields" (random rectangles)
    - N = 1000 query points (random sensor locations)
    - Measure brute force vs STRtree time on nearest-polygon queries
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely import STRtree
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# SECTION 1: Generate Test Data
# ---------------------------------------------------------------------------

N_POLYGONS = 1000   # Number of reference polygons (field boundaries)
N_QUERIES = 1000    # Number of query points (sensor readings)

print(f"Generating {N_POLYGONS:,} polygons and {N_QUERIES:,} query points...")

# Generate random rectangles in a 100km × 100km area (UTM coordinates in meters)
AREA_MIN, AREA_MAX = 0, 100_000  # 100km extent

rng = np.random.default_rng(42)

polygon_list = []
for i in range(N_POLYGONS):
    x = rng.uniform(AREA_MIN, AREA_MAX - 200)
    y = rng.uniform(AREA_MIN, AREA_MAX - 200)
    w = rng.uniform(50, 400)    # 50-400m wide
    h = rng.uniform(50, 300)    # 50-300m tall
    polygon_list.append(box(x, y, x + w, y + h))

gdf_polygons = gpd.GeoDataFrame(
    {"poly_id": range(N_POLYGONS),
     "crop_type": rng.choice(["corn", "wheat", "soy", "cotton"], N_POLYGONS)},
    geometry=polygon_list,
    crs="EPSG:32611"
)

# Generate random query points
query_x = rng.uniform(AREA_MIN, AREA_MAX, N_QUERIES)
query_y = rng.uniform(AREA_MIN, AREA_MAX, N_QUERIES)
query_points = [Point(x, y) for x, y in zip(query_x, query_y)]

gdf_queries = gpd.GeoDataFrame(
    {"query_id": range(N_QUERIES)},
    geometry=query_points,
    crs="EPSG:32611"
)

print(f"  Polygons: {len(gdf_polygons):,}")
print(f"  Query points: {len(gdf_queries):,}")
print(f"  Brute-force comparisons needed: {N_POLYGONS * N_QUERIES:,}")

# ---------------------------------------------------------------------------
# SECTION 2: BRUTE FORCE — O(N × M)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("APPROACH 1: Brute Force O(N × M)")
print("="*60)

all_polygon_geoms = list(gdf_polygons.geometry)
all_polygon_ids = list(gdf_polygons["poly_id"])

start_bf = time.perf_counter()

brute_force_results = []
for qi, query_pt in enumerate(query_points):
    min_dist = float("inf")
    nearest_poly_id = None

    for pi, poly_geom in enumerate(all_polygon_geoms):
        # Exact distance from point to polygon boundary/interior
        dist = query_pt.distance(poly_geom)
        if dist < min_dist:
            min_dist = dist
            nearest_poly_id = all_polygon_ids[pi]

    brute_force_results.append({
        "query_id": qi,
        "nearest_poly_id": nearest_poly_id,
        "distance_m": min_dist
    })

elapsed_bf = time.perf_counter() - start_bf
df_bf = pd.DataFrame(brute_force_results)

print(f"Time: {elapsed_bf:.3f}s")
print(f"Comparisons: {N_POLYGONS * N_QUERIES:,}")
print(f"Rate: {N_QUERIES / elapsed_bf:.0f} queries/second")
print(f"Results sample:\n{df_bf.head(5).to_string()}")

# ---------------------------------------------------------------------------
# SECTION 3: STRtree — O((N + M) × log M)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("APPROACH 2: Shapely STRtree (Sort-Tile-Recursive R-tree)")
print("="*60)

# Build index — this is a one-time cost O(M log M)
build_start = time.perf_counter()
tree = STRtree(all_polygon_geoms)
build_time = time.perf_counter() - build_start
print(f"Index build time: {build_time:.4f}s (one-time cost)")

# Query index — O(log M) per query
query_start = time.perf_counter()

strtree_results = []
for qi, query_pt in enumerate(query_points):
    # nearest() returns the INDEX of the nearest geometry in the original list
    nearest_idx = tree.nearest(query_pt)
    if nearest_idx is not None:
        nearest_geom = all_polygon_geoms[nearest_idx]
        dist = query_pt.distance(nearest_geom)
        nearest_id = all_polygon_ids[nearest_idx]
    else:
        nearest_id = None
        dist = float("inf")

    strtree_results.append({
        "query_id": qi,
        "nearest_poly_id": nearest_id,
        "distance_m": dist
    })

elapsed_strtree_query = time.perf_counter() - query_start
elapsed_strtree_total = build_time + elapsed_strtree_query

df_strtree = pd.DataFrame(strtree_results)

print(f"Query time: {elapsed_strtree_query:.4f}s")
print(f"Total time (build + query): {elapsed_strtree_total:.4f}s")
print(f"Rate: {N_QUERIES / elapsed_strtree_query:.0f} queries/second")

# ---------------------------------------------------------------------------
# SECTION 4: Correctness Verification
# ---------------------------------------------------------------------------

print("\n=== Correctness Check ===")
# Verify that STRtree gives same results as brute force
match_count = (df_bf["nearest_poly_id"].values == df_strtree["nearest_poly_id"].values).sum()
dist_match = np.allclose(df_bf["distance_m"].values, df_strtree["distance_m"].values, rtol=1e-5)

print(f"Nearest polygon ID match: {match_count}/{N_QUERIES} ({match_count/N_QUERIES*100:.1f}%)")
print(f"Distance values match: {dist_match}")

# Note: STRtree.nearest returns one nearest — in edge cases with equidistant polygons,
# the chosen index may differ from brute force while distance is identical.

# ---------------------------------------------------------------------------
# SECTION 5: Performance Summary
# ---------------------------------------------------------------------------

speedup_query = elapsed_bf / elapsed_strtree_query
speedup_total = elapsed_bf / elapsed_strtree_total

print("\n" + "="*60)
print("  PERFORMANCE COMPARISON")
print("="*60)
print(f"  {'Metric':<35} {'Brute Force':>14} {'STRtree':>14} {'Speedup':>12}")
print(f"  {'-'*75}")
print(f"  {'Query time (s)':<35} {elapsed_bf:>14.3f} {elapsed_strtree_query:>14.4f} {speedup_query:>11.1f}×")
print(f"  {'Total time inc. build (s)':<35} {elapsed_bf:>14.3f} {elapsed_strtree_total:>14.4f} {speedup_total:>11.1f}×")
print(f"  {'Queries per second':<35} {N_QUERIES/elapsed_bf:>14.0f} {N_QUERIES/elapsed_strtree_query:>14.0f}")

print("\nKEY INSIGHT:")
print(f"  STRtree is {speedup_query:.0f}× faster for queries alone.")
print(f"  For repeated queries (Kafka stream, Airflow batch), build once, query many times.")
print(f"  At 100K queries: brute force ≈ {elapsed_bf * 100:.0f}s, STRtree ≈ {elapsed_strtree_query * 100:.1f}s")

# ---------------------------------------------------------------------------
# SECTION 6: Visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: spatial layout
ax = axes[0]
gdf_polygons.plot(ax=ax, color="lightblue", edgecolor="steelblue",
                  alpha=0.4, linewidth=0.5, label="Fields")
gdf_queries.plot(ax=ax, color="red", markersize=3, alpha=0.5,
                 label="Query points")

# Draw nearest-neighbor lines for first 30 queries
for _, qrow in df_strtree.head(30).iterrows():
    qpt = query_points[int(qrow["query_id"])]
    poly_geom = all_polygon_geoms[int(qrow["nearest_poly_id"])]
    nearest_on_poly = poly_geom.centroid
    ax.plot([qpt.x, nearest_on_poly.x], [qpt.y, nearest_on_poly.y],
            "k-", linewidth=0.5, alpha=0.3)

ax.set_title(f"Nearest-Neighbor Queries\n{N_POLYGONS} fields, {N_QUERIES} sensors",
             fontsize=11, fontweight="bold")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.legend(fontsize=9)

# Right: performance comparison bar chart
ax2 = axes[1]
categories = ["Brute Force\n(O(N×M))", "STRtree Query\n(O(log M))", "STRtree Total\n(inc. build)"]
times = [elapsed_bf, elapsed_strtree_query, elapsed_strtree_total]
colors = ["#e74c3c", "#2ecc71", "#27ae60"]

bars = ax2.bar(categories, times, color=colors, edgecolor="white", linewidth=1.5)
for bar, t in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + elapsed_bf * 0.01,
             f"{t:.4f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_ylabel("Time (seconds)", fontsize=11)
ax2.set_title(f"Nearest-Neighbor Query Time\n{N_QUERIES:,} points vs {N_POLYGONS:,} polygons",
              fontsize=11, fontweight="bold")
ax2.set_ylim(0, elapsed_bf * 1.2)

# Add speedup annotation
ax2.annotate(f"{speedup_query:.0f}× speedup",
             xy=(1, elapsed_strtree_query), xytext=(1.5, elapsed_bf * 0.5),
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=13, fontweight="bold", color="#2ecc71")

ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Spatial Index Optimization: Brute Force vs STRtree\n"
             "Emmanuel Oyekanlu — Principal Data Engineer",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("spatial_index_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: spatial_index_benchmark.png")

print("\n=== Script 02 Complete ===")
