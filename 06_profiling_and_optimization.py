"""
06_profiling_and_optimization.py
==================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Demonstrate professional profiling of a geospatial pipeline to find
    bottlenecks, then show how to fix them. This directly addresses the
    Bayer job requirement: "generating time and memory profiling reports."

    PROFILING TOOLS:
    - cProfile: Function-level time profiling (which functions are slow?)
    - pstats: Parse and format cProfile output
    - tracemalloc: Memory allocation tracking (which code allocates most RAM?)
    - timeit: Microbenchmarking of specific operations

    WORKFLOW:
    1. Run slow pipeline → generate profiling report
    2. Identify top bottlenecks from cProfile output
    3. Apply fixes (spatial index, vectorization, chunking)
    4. Run fast pipeline → compare reports
    5. Print before/after summary

AT CORNING:
    This profiling workflow was mandatory before deploying any pipeline
    processing >100K features. The optimization step alone typically
    reduced AGV telemetry processing time from 45 minutes to <3 minutes.
"""

import cProfile
import pstats
import io
import tracemalloc
import time
import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from shapely import STRtree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# SECTION 1: Generate Test Data
# ---------------------------------------------------------------------------

N_POLYGONS = 500
N_QUERIES = 200

rng = np.random.default_rng(42)
polys = [box(rng.uniform(0, 100000), rng.uniform(0, 100000),
             rng.uniform(100, 500) + rng.uniform(0, 100000),
             rng.uniform(100, 400) + rng.uniform(0, 100000))
         for _ in range(N_POLYGONS)]

gdf_ref = gpd.GeoDataFrame(
    {"poly_id": range(N_POLYGONS),
     "crop_type": rng.choice(["corn", "wheat", "cotton"], N_POLYGONS),
     "value": rng.uniform(1, 100, N_POLYGONS)},
    geometry=polys,
    crs="EPSG:32611"
)

query_pts = [Point(rng.uniform(0, 100000), rng.uniform(0, 100000))
             for _ in range(N_QUERIES)]

gdf_queries = gpd.GeoDataFrame(
    {"query_id": range(N_QUERIES)},
    geometry=query_pts,
    crs="EPSG:32611"
)

print(f"Test data: {N_POLYGONS} reference polygons, {N_QUERIES} query points")

# ---------------------------------------------------------------------------
# SECTION 2: SLOW PIPELINE (unoptimized)
# ---------------------------------------------------------------------------

def slow_pipeline(gdf_ref, gdf_queries):
    """
    SLOW VERSION: Python loops, no spatial index, row-by-row operations.
    This is what a naive implementation looks like.
    """
    all_geoms = list(gdf_ref.geometry)
    all_ids = list(gdf_ref["poly_id"])
    all_crops = list(gdf_ref["crop_type"])

    results = []

    for qi, qrow in gdf_queries.iterrows():
        qpt = qrow.geometry

        # BAD: O(N) distance computation for EVERY query point
        min_dist = float("inf")
        nearest_id = None
        nearest_crop = None

        for pi, (poly_geom, poly_id, crop) in enumerate(
            zip(all_geoms, all_ids, all_crops)
        ):
            # BAD: Python-level Shapely call in inner loop
            dist = qpt.distance(poly_geom)
            if dist < min_dist:
                min_dist = dist
                nearest_id = poly_id
                nearest_crop = crop

        # BAD: Computing area in a Python loop (not vectorized)
        qbuffer = qpt.buffer(500)
        nearby_area = sum(
            g.intersection(qbuffer).area
            for g in all_geoms
            if g.intersects(qbuffer)
        )

        results.append({
            "query_id": int(qi),
            "nearest_poly_id": nearest_id,
            "min_distance_m": min_dist,
            "nearest_crop": nearest_crop,
            "nearby_covered_area_m2": nearby_area,
        })

    return results

# ---------------------------------------------------------------------------
# SECTION 3: FAST PIPELINE (optimized)
# ---------------------------------------------------------------------------

def fast_pipeline(gdf_ref, gdf_queries):
    """
    FAST VERSION: STRtree spatial index, vectorized operations.
    """
    all_geoms = list(gdf_ref.geometry)
    all_ids = list(gdf_ref["poly_id"])
    all_crops = list(gdf_ref["crop_type"])

    # GOOD: Build spatial index ONCE outside the query loop
    tree = STRtree(all_geoms)

    results = []

    for qi, qrow in gdf_queries.iterrows():
        qpt = qrow.geometry

        # GOOD: O(log N) nearest neighbor via STRtree
        nearest_idx = tree.nearest(qpt)
        nearest_id = all_ids[nearest_idx]
        nearest_geom = all_geoms[nearest_idx]
        min_dist = qpt.distance(nearest_geom)
        nearest_crop = all_crops[nearest_idx]

        # GOOD: Use STRtree to find candidates within buffer before computing intersection
        qbuffer = qpt.buffer(500)
        candidate_indices = tree.query(qbuffer, predicate="intersects")

        # Only compute intersection for actual candidates (not all polygons)
        nearby_area = sum(
            all_geoms[i].intersection(qbuffer).area
            for i in candidate_indices
        )

        results.append({
            "query_id": int(qi),
            "nearest_poly_id": nearest_id,
            "min_distance_m": min_dist,
            "nearest_crop": nearest_crop,
            "nearby_covered_area_m2": nearby_area,
        })

    return results

# ---------------------------------------------------------------------------
# SECTION 4: Profile SLOW Pipeline with cProfile
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("PROFILING: Slow Pipeline")
print("="*60)

# tracemalloc: track memory allocations
tracemalloc.start()
slow_mem_start = tracemalloc.get_traced_memory()

# cProfile: time each function
slow_profiler = cProfile.Profile()
slow_profiler.enable()

t_slow_start = time.perf_counter()
slow_results = slow_pipeline(gdf_ref, gdf_queries)
t_slow = time.perf_counter() - t_slow_start

slow_profiler.disable()
slow_mem_peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

print(f"Slow pipeline time: {t_slow:.3f}s")
print(f"Peak memory: {slow_mem_peak / 1024:.1f} KB")

# Save cProfile output to string and file
slow_stream = io.StringIO()
slow_stats = pstats.Stats(slow_profiler, stream=slow_stream)
slow_stats.sort_stats("cumulative")
slow_stats.print_stats(15)  # Top 15 functions
slow_profile_text = slow_stream.getvalue()

with open("profiling_before.txt", "w") as f:
    f.write(f"SLOW PIPELINE PROFILE\n{'='*60}\n")
    f.write(f"Total time: {t_slow:.3f}s\n")
    f.write(f"Peak memory: {slow_mem_peak / 1024:.1f} KB\n\n")
    f.write(slow_profile_text)

print("\nTop functions (slow pipeline):")
# Print a cleaner version of top functions
print(slow_profile_text[:2000])

# ---------------------------------------------------------------------------
# SECTION 5: Profile FAST Pipeline with cProfile
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("PROFILING: Fast Pipeline")
print("="*60)

tracemalloc.start()

fast_profiler = cProfile.Profile()
fast_profiler.enable()

t_fast_start = time.perf_counter()
fast_results = fast_pipeline(gdf_ref, gdf_queries)
t_fast = time.perf_counter() - t_fast_start

fast_profiler.disable()
fast_mem_peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

print(f"Fast pipeline time: {t_fast:.3f}s")
print(f"Peak memory: {fast_mem_peak / 1024:.1f} KB")

fast_stream = io.StringIO()
fast_stats = pstats.Stats(fast_profiler, stream=fast_stream)
fast_stats.sort_stats("cumulative")
fast_stats.print_stats(15)
fast_profile_text = fast_stream.getvalue()

with open("profiling_after.txt", "w") as f:
    f.write(f"FAST PIPELINE PROFILE\n{'='*60}\n")
    f.write(f"Total time: {t_fast:.3f}s\n")
    f.write(f"Peak memory: {fast_mem_peak / 1024:.1f} KB\n\n")
    f.write(fast_profile_text)

# ---------------------------------------------------------------------------
# SECTION 6: tracemalloc Memory Report
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("MEMORY PROFILING with tracemalloc (fast pipeline)")
print("="*60)

tracemalloc.start()
_ = fast_pipeline(gdf_ref, gdf_queries)
snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()

# Get top 10 memory allocations
top_stats = snapshot.statistics("lineno")
memory_report_lines = [
    f"Top 10 memory allocations in fast_pipeline:\n"
]
for i, stat in enumerate(top_stats[:10]):
    memory_report_lines.append(
        f"  #{i+1}: {stat.traceback.format()[-1].strip()} "
        f"→ {stat.size / 1024:.1f} KB"
    )

memory_report = "\n".join(memory_report_lines)
print(memory_report)

with open("memory_trace.txt", "w") as f:
    f.write(memory_report + "\n")
    f.write("\nAll memory allocations:\n")
    for stat in top_stats[:20]:
        f.write(f"{stat}\n")

# ---------------------------------------------------------------------------
# SECTION 7: Before/After Comparison Report
# ---------------------------------------------------------------------------

speedup = t_slow / t_fast if t_fast > 0 else float("inf")
memory_reduction = slow_mem_peak / fast_mem_peak if fast_mem_peak > 0 else float("inf")

performance_report = {
    "pipeline": "geospatial_zone_lookup",
    "n_reference_polygons": N_POLYGONS,
    "n_query_points": N_QUERIES,
    "before_optimization": {
        "time_sec": round(t_slow, 3),
        "peak_memory_kb": round(slow_mem_peak / 1024, 1),
        "bottlenecks": [
            "O(N×M) brute force nearest neighbor search",
            "Python loop for distance computation",
            "No spatial index for buffer intersection filter",
        ]
    },
    "after_optimization": {
        "time_sec": round(t_fast, 3),
        "peak_memory_kb": round(fast_mem_peak / 1024, 1),
        "optimizations_applied": [
            "STRtree spatial index (O(log N) nearest neighbor)",
            "STRtree.query() for intersection candidate filtering",
            "Build index once, reuse for all queries",
        ]
    },
    "improvements": {
        "speedup_factor": round(speedup, 1),
        "memory_reduction_factor": round(memory_reduction, 2),
        "time_saved_sec": round(t_slow - t_fast, 3),
    }
}

with open("performance_comparison.json", "w") as f:
    json.dump(performance_report, f, indent=2)

print("\n" + "="*60)
print("  BEFORE/AFTER PERFORMANCE REPORT")
print("="*60)
print(f"  {'Metric':<30} {'Before':>12} {'After':>12} {'Improvement':>14}")
print(f"  {'-'*68}")
print(f"  {'Time (seconds)':<30} {t_slow:>12.3f} {t_fast:>12.3f} {speedup:>13.1f}×")
print(f"  {'Peak Memory (KB)':<30} {slow_mem_peak/1024:>12.1f} {fast_mem_peak/1024:>12.1f} {memory_reduction:>13.2f}×")

print(f"\n  Files generated:")
print(f"  - profiling_before.txt   (cProfile of slow pipeline)")
print(f"  - profiling_after.txt    (cProfile of fast pipeline)")
print(f"  - memory_trace.txt       (tracemalloc allocation report)")
print(f"  - performance_comparison.json  (structured report)")

# ---------------------------------------------------------------------------
# SECTION 8: Visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics = ["Time (s)", "Peak Memory (KB)"]
before_vals = [t_slow, slow_mem_peak / 1024]
after_vals = [t_fast, fast_mem_peak / 1024]

for ax, metric, bval, aval in zip(axes, metrics, before_vals, after_vals):
    bars = ax.bar(["Before\n(Unoptimized)", "After\n(Optimized)"],
                  [bval, aval],
                  color=["#e74c3c", "#2ecc71"],
                  edgecolor="white", linewidth=2, width=0.5)

    for bar, val in zip(bars, [bval, aval]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    improvement = bval / aval if aval > 0 else 0
    ax.set_title(f"{metric}\n({improvement:.1f}× improvement)", fontsize=12)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    f"Geospatial Pipeline Optimization — cProfile + tracemalloc\n"
    f"{N_POLYGONS} polygons × {N_QUERIES} queries | Emmanuel Oyekanlu",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("profiling_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: profiling_comparison.png")

print("\n=== Script 06 Complete ===")
