"""
04_crs_batch_reprojection.py
==============================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Benchmark CRS reprojection at scale and demonstrate correctness
    requirements (always_xy=True) for production use.

    CRS reprojection is one of the most common geospatial operations.
    Every time data from different sources is combined, CRS harmonization
    is required. Doing it wrong (row-by-row, incorrect axis order) is
    a common source of bugs in geospatial pipelines.

METHODS COMPARED:
    1. Row-by-row pyproj Transformer (naive)
    2. GeoPandas to_crs() (vectorized, internally uses pyproj batch API)
    3. pyproj Transformer.transform() batch call directly (for reference)

CRITICAL: always_xy=True
    Many CRS definitions have latitude-first axis order (lat, lon).
    When you do proj.transform(lon, lat), you might actually need
    proj.transform(lat, lon) depending on the CRS.

    always_xy=True forces (longitude, latitude) input order regardless
    of the CRS's native axis order. This is CRITICAL to get right —
    swapping lat/lon is a common silent bug that puts coordinates in
    the wrong ocean.

    Proof: EPSG:4326 native order is (latitude, longitude) but
    always_xy=True forces it to interpret input as (longitude, latitude).
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pyproj import Transformer
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# SECTION 1: Generate 10,000 Point Coordinates in WGS84
# ---------------------------------------------------------------------------

N = 10_000
print(f"Generating {N:,} points for reprojection benchmark...")

rng = np.random.default_rng(42)

# Generate random points in the Imperial Valley / Southern California area
lons = rng.uniform(-116.5, -114.5, N)  # Longitude (WGS84)
lats = rng.uniform(32.5, 34.0, N)      # Latitude (WGS84)

points_wgs84 = [Point(lon, lat) for lon, lat in zip(lons, lats)]

gdf_wgs84 = gpd.GeoDataFrame(
    {"point_id": range(N), "lon_wgs84": lons, "lat_wgs84": lats},
    geometry=points_wgs84,
    crs="EPSG:4326"  # WGS84 geographic
)

print(f"Source CRS: EPSG:4326 (WGS84)")
print(f"Target CRS: EPSG:32611 (WGS84 / UTM Zone 11N)")
print(f"  -> Meters, appropriate for Imperial Valley, CA")

# ---------------------------------------------------------------------------
# SECTION 2: METHOD 1 — Row-by-Row pyproj (Naive)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 1: Row-by-row pyproj Transformer (NAIVE)")
print("="*60)

# CRITICAL: always_xy=True ensures consistent (lon, lat) input order
# without this, results depend on CRS axis conventions — silent bugs!
transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:32611",
    always_xy=True  # ALWAYS include this for predictable behavior
)

start_rowbyrow = time.perf_counter()

utm_coords_rowbyrow = []
for lon, lat in zip(lons, lats):
    easting, northing = transformer.transform(lon, lat)
    utm_coords_rowbyrow.append((easting, northing))

elapsed_rowbyrow = time.perf_counter() - start_rowbyrow

print(f"Time: {elapsed_rowbyrow:.4f}s")
print(f"Rate: {N / elapsed_rowbyrow:.0f} points/second")
print(f"Sample: {utm_coords_rowbyrow[0]}")

# ---------------------------------------------------------------------------
# SECTION 3: METHOD 2 — pyproj Batch Transform
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 2: pyproj Transformer.transform() on full arrays (BATCH)")
print("="*60)

start_batch_pyproj = time.perf_counter()

# Pass entire arrays at once — pyproj internally calls C-level loop
eastings_batch, northings_batch = transformer.transform(lons, lats)

elapsed_batch_pyproj = time.perf_counter() - start_batch_pyproj

print(f"Time: {elapsed_batch_pyproj:.4f}s")
print(f"Rate: {N / elapsed_batch_pyproj:.0f} points/second")
print(f"Sample: ({eastings_batch[0]:.2f}, {northings_batch[0]:.2f})")

# Verify correctness
row_vs_batch_error = np.max(
    np.abs(eastings_batch - [c[0] for c in utm_coords_rowbyrow])
)
print(f"Max error vs row-by-row: {row_vs_batch_error:.6f} m (should be ~0)")

# ---------------------------------------------------------------------------
# SECTION 4: METHOD 3 — GeoPandas to_crs() (Recommended)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 3: GeoPandas gdf.to_crs() (VECTORIZED — RECOMMENDED)")
print("="*60)

start_geopandas = time.perf_counter()

gdf_utm = gdf_wgs84.to_crs("EPSG:32611")

elapsed_geopandas = time.perf_counter() - start_geopandas

print(f"Time: {elapsed_geopandas:.4f}s")
print(f"Rate: {N / elapsed_geopandas:.0f} points/second")
print(f"Result CRS: {gdf_utm.crs}")
sample_pt = gdf_utm.geometry.iloc[0]
print(f"Sample: ({sample_pt.x:.2f}, {sample_pt.y:.2f})")

# Verify against batch pyproj
gpd_vs_batch_error = np.max(
    np.abs(gdf_utm.geometry.x.values - eastings_batch)
)
print(f"Max error vs batch pyproj: {gpd_vs_batch_error:.6f} m (should be ~0)")

# ---------------------------------------------------------------------------
# SECTION 5: CORRECTNESS DEMONSTRATION — always_xy=True
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("CORRECTNESS CHECK: always_xy=True vs False")
print("="*60)

# Sample point: Golden Gate Bridge
lon_ggb = -122.4784
lat_ggb = 37.8199

# CORRECT: always_xy=True (longitude first, latitude second)
t_correct = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
e_correct, n_correct = t_correct.transform(lon_ggb, lat_ggb)
print(f"CORRECT  (always_xy=True):  Easting={e_correct:.2f}, Northing={n_correct:.2f}")

# INCORRECT: without always_xy (may interpret as lat,lon for some CRS)
t_wrong = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=False)
e_wrong, n_wrong = t_wrong.transform(lon_ggb, lat_ggb)
print(f"INCORRECT (always_xy=False): Easting={e_wrong:.2f}, Northing={n_wrong:.2f}")

# The incorrect version may produce different values because EPSG:4326
# has native axis order (latitude, longitude) in PROJ 6+
error_m = np.sqrt((e_correct - e_wrong)**2 + (n_correct - n_wrong)**2)
if error_m > 100:
    print(f"Error caused by wrong axis order: {error_m:,.0f} m = {error_m/1000:.1f} km!")
    print("This is a SILENT bug — no error message, just wrong coordinates!")
else:
    print(f"For this CRS pair, axis order doesn't cause issue (error={error_m:.0f}m)")
    print("But always use always_xy=True as a defensive practice!")

# ---------------------------------------------------------------------------
# SECTION 6: Performance Summary
# ---------------------------------------------------------------------------

speedup_batch = elapsed_rowbyrow / elapsed_batch_pyproj
speedup_gpd = elapsed_rowbyrow / elapsed_geopandas

print("\n" + "="*60)
print("  REPROJECTION PERFORMANCE SUMMARY")
print("="*60)
print(f"  {'Method':<40} {'Time (s)':>10} {'Speedup':>10} {'Points/sec':>15}")
print(f"  {'-'*78}")
print(f"  {'Row-by-row pyproj':<40} {elapsed_rowbyrow:>10.4f} {'1.0×':>10} "
      f"{N/elapsed_rowbyrow:>15,.0f}")
print(f"  {'Batch pyproj (arrays)':<40} {elapsed_batch_pyproj:>10.4f} "
      f"{speedup_batch:>9.1f}× {N/elapsed_batch_pyproj:>15,.0f}")
print(f"  {'GeoPandas to_crs()':<40} {elapsed_geopandas:>10.4f} "
      f"{speedup_gpd:>9.1f}× {N/elapsed_geopandas:>15,.0f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
methods = ["Row-by-row\npyproj", "Batch\npyproj", "GeoPandas\nto_crs()"]
times = [elapsed_rowbyrow, elapsed_batch_pyproj, elapsed_geopandas]
colors = ["#e74c3c", "#f39c12", "#2ecc71"]

bars = ax.bar(methods, times, color=colors, edgecolor="white", linewidth=2, width=0.5)
for bar, t, spd in zip(bars, times, [1.0, speedup_batch, speedup_gpd]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{t:.4f}s\n({spd:.0f}×)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title(f"CRS Reprojection Performance\n{N:,} Points: EPSG:4326 → EPSG:32611",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_yscale("log")

ax.text(0.98, 0.02,
        "ALWAYS use always_xy=True\nwhen creating Transformer!",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#e74c3c",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("crs_reprojection_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: crs_reprojection_benchmark.png")

print("\n=== Script 04 Complete ===")
