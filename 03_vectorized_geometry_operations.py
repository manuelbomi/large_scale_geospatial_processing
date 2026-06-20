"""
03_vectorized_geometry_operations.py
======================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Demonstrate the speedup from vectorized GeoPandas operations
    vs Python row-by-row Shapely loops.

    GeoPandas uses Cython/C-level GEOS calls internally when you use
    its vectorized API (.area, .centroid, .distance, etc.).
    Python loops break this optimization by falling back to Python-speed
    calls for each individual geometry.

OPERATIONS BENCHMARKED:
    1. Area computation (10,000 polygons)
    2. Centroid computation (10,000 polygons)
    3. Perimeter computation (10,000 polygons)
    4. Distance to a fixed point (10,000 polygons)
    5. Buffer operation (10,000 polygons)

EXPECTED SPEEDUP:
    10-50× depending on the operation and GeoPandas version.
    GeoPandas 0.12+ uses Shapely 2.0 which is especially fast.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
import timeit
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# SECTION 1: Generate Test GeoDataFrame
# ---------------------------------------------------------------------------

N = 10_000  # Number of polygons
print(f"Generating {N:,} test polygons...")

rng = np.random.default_rng(42)
x_origins = rng.uniform(0, 100_000, N)
y_origins = rng.uniform(0, 100_000, N)
widths = rng.uniform(100, 1000, N)
heights = rng.uniform(100, 800, N)

polygons = [box(x, y, x + w, y + h)
            for x, y, w, h in zip(x_origins, y_origins, widths, heights)]

gdf = gpd.GeoDataFrame(
    {"poly_id": range(N), "value": rng.uniform(0, 100, N)},
    geometry=polygons,
    crs="EPSG:32611"
)

reference_point = Point(50_000, 50_000)
print(f"GeoDataFrame created: {gdf.shape}")

# ---------------------------------------------------------------------------
# SECTION 2: Benchmark Functions
# ---------------------------------------------------------------------------

def time_operation(label: str, loop_fn, vec_fn, n_repeat: int = 3):
    """
    Time both loop and vectorized implementations, return comparison dict.
    """
    # Time loop version
    loop_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        loop_result = loop_fn()
        loop_times.append(time.perf_counter() - t0)
    t_loop = min(loop_times)

    # Time vectorized version
    vec_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        vec_result = vec_fn()
        vec_times.append(time.perf_counter() - t0)
    t_vec = min(vec_times)

    speedup = t_loop / t_vec if t_vec > 0 else float("inf")

    print(f"\n  {label}:")
    print(f"    Loop:        {t_loop:.4f}s")
    print(f"    Vectorized:  {t_vec:.4f}s")
    print(f"    Speedup:     {speedup:.1f}×")

    return {
        "operation": label,
        "loop_sec": t_loop,
        "vec_sec": t_vec,
        "speedup": speedup,
    }

all_geoms = list(gdf.geometry)

print("\n" + "="*60)
print("BENCHMARK: Row-by-row Shapely loop vs GeoPandas vectorized")
print(f"N = {N:,} polygons")
print("="*60)

results = []

# --- Operation 1: Area ---
r = time_operation(
    "Area computation (area in m²)",
    loop_fn=lambda: [g.area for g in all_geoms],
    vec_fn=lambda: gdf.geometry.area
)
results.append(r)

# --- Operation 2: Centroid ---
r = time_operation(
    "Centroid computation",
    loop_fn=lambda: [g.centroid for g in all_geoms],
    vec_fn=lambda: gdf.geometry.centroid
)
results.append(r)

# --- Operation 3: Perimeter (length) ---
r = time_operation(
    "Perimeter computation",
    loop_fn=lambda: [g.length for g in all_geoms],
    vec_fn=lambda: gdf.geometry.length
)
results.append(r)

# --- Operation 4: Distance to a reference point ---
r = time_operation(
    "Distance to reference point",
    loop_fn=lambda: [g.distance(reference_point) for g in all_geoms],
    vec_fn=lambda: gdf.geometry.distance(reference_point)
)
results.append(r)

# --- Operation 5: Buffer (1m buffer) ---
r = time_operation(
    "Buffer(1m) computation",
    loop_fn=lambda: [g.buffer(1) for g in all_geoms],
    vec_fn=lambda: gdf.geometry.buffer(1)
)
results.append(r)

# --- Operation 6: Bounds (bounding box) ---
r = time_operation(
    "Bounds extraction",
    loop_fn=lambda: [g.bounds for g in all_geoms],
    vec_fn=lambda: gdf.geometry.bounds
)
results.append(r)

# --- Operation 7: Is valid check ---
r = time_operation(
    "Validity check",
    loop_fn=lambda: [g.is_valid for g in all_geoms],
    vec_fn=lambda: gdf.geometry.is_valid
)
results.append(r)

# ---------------------------------------------------------------------------
# SECTION 3: Computed Area Correctness Check
# ---------------------------------------------------------------------------

loop_areas = np.array([g.area for g in all_geoms])
vec_areas = gdf.geometry.area.values

print("\n=== Correctness Check ===")
print(f"Max absolute difference: {np.abs(loop_areas - vec_areas).max():.6f} m²")
print(f"Results identical: {np.allclose(loop_areas, vec_areas)}")

# ---------------------------------------------------------------------------
# SECTION 4: Summary Table and Plot
# ---------------------------------------------------------------------------

df_results = pd.DataFrame(results)

print("\n" + "="*60)
print("  PERFORMANCE SUMMARY")
print("="*60)
print(f"\n  {'Operation':<40} {'Loop (s)':>10} {'Vec (s)':>10} {'Speedup':>10}")
print(f"  {'-'*72}")
for _, row in df_results.iterrows():
    print(f"  {row['operation']:<40} {row['loop_sec']:>10.4f} {row['vec_sec']:>10.4f} "
          f"{row['speedup']:>9.1f}×")

print(f"\n  Mean speedup: {df_results['speedup'].mean():.1f}×")
print(f"  Max speedup:  {df_results['speedup'].max():.1f}×")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: absolute times
ax1 = axes[0]
x = np.arange(len(df_results))
width = 0.35
bars1 = ax1.bar(x - width/2, df_results["loop_sec"], width, label="Loop", color="#e74c3c", alpha=0.8)
bars2 = ax1.bar(x + width/2, df_results["vec_sec"], width, label="Vectorized", color="#2ecc71", alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(
    [r["operation"].split("(")[0].strip()[:20] for _, r in df_results.iterrows()],
    rotation=30, ha="right", fontsize=8
)
ax1.set_ylabel("Time (seconds)")
ax1.set_title(f"Computation Time — Loop vs Vectorized\nN = {N:,} polygons",
              fontsize=11, fontweight="bold")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
ax1.set_yscale("log")

# Bar chart: speedup
ax2 = axes[1]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_results)))
bars = ax2.bar(x, df_results["speedup"], color=colors, edgecolor="white")
for bar, val in zip(bars, df_results["speedup"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(
    [r["operation"].split("(")[0].strip()[:20] for _, r in df_results.iterrows()],
    rotation=30, ha="right", fontsize=8
)
ax2.set_ylabel("Speedup (×)")
ax2.set_title("Vectorization Speedup\n(higher = better)",
              fontsize=11, fontweight="bold")
ax2.axhline(y=1, color="red", linewidth=1.5, linestyle="--", label="1× (no improvement)")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("GeoPandas Vectorized vs Python Loop Performance\n"
             "Emmanuel Oyekanlu — Principal Data Engineer",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("vectorization_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: vectorization_benchmark.png")

print("\nKEY TAKEAWAY:")
print("  ALWAYS use gdf.geometry.area, .length, .centroid, etc.")
print("  NEVER write 'for geom in gdf.geometry: geom.area'")
print("  The vectorized path uses C-level GEOS calls, avoiding Python overhead.")

print("\n=== Script 03 Complete ===")
