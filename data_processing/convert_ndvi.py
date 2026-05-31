"""
build_ndvi_csv.py
-----------------
Fast conversion of a multi-band NDVI GeoTIFF → CSV aligned to elevation_grid.csv.

Key optimisations over the original:
  1. Read each band as a full numpy array (one disk read per band, not one per point).
  2. Convert grid lat/lon → pixel row/col once, reuse for every band.
  3. Parallelise band processing across CPU cores with ProcessPoolExecutor.
  4. Write output incrementally (one chunk per worker result) → low peak RAM.

Usage:
    python convert_ndvi.py
        # defaults read/write under <project>/data/
        --workers 8          # defaults to os.cpu_count()
        --start-year 2012
"""

import argparse
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Precompute pixel indices for every grid point (done ONCE)
# ---------------------------------------------------------------------------
def compute_pixel_indices(transform, inv_crs_needed, src_crs,
                           lons, lats):
    """
    Convert WGS-84 lon/lat → integer pixel (row, col) in the raster.
    If the raster is already EPSG:4326 the reprojection is a no-op.
    Returns arrays of rows and cols (clipped to valid raster extent).
    """
    from rasterio.warp import transform as reproj_transform

    if inv_crs_needed:
        xs, ys = reproj_transform("EPSG:4326", src_crs, lons, lats)
    else:
        xs, ys = lons, lats

    rows, cols = rowcol(transform, xs, ys)          # vectorised!
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


# ---------------------------------------------------------------------------
# Worker function – runs in a separate process
# ---------------------------------------------------------------------------
def process_band(tif_path, band_idx, time_label, rows, cols,
                 grid_ids, lats, lons, nodata):
    """
    Opens the TIF (each worker holds its own file handle), reads one band,
    and samples it at the precomputed pixel indices.
    Returns a list of (grid_id, time, lat, lon, ndvi) tuples.
    """
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width

        # Clip indices to valid extent (guard against edge points)
        r = np.clip(rows, 0, height - 1)
        c = np.clip(cols, 0, width  - 1)

        arr = src.read(band_idx)                    # whole band, one read

    ndvi_vals = arr[r, c].astype(np.float64)

    # Mask nodata
    if nodata is not None:
        ndvi_vals[ndvi_vals == nodata] = np.nan

    # Round to 4 dp – NDVI never needs more precision, saves ~30 % CSV size
    ndvi_vals = np.round(ndvi_vals, 4)

    # Convert to plain Python floats/empty-strings so csv.writer renders correctly.
    # np.nan written directly becomes the string "nan"; "" gives a blank cell.
    ndvi_out = [
        "" if np.isnan(v) else float(v)
        for v in ndvi_vals
    ]

    return time_label, list(zip(grid_ids, [time_label]*len(grid_ids),
                                lats, lons, ndvi_out))


# ---------------------------------------------------------------------------
# Time index helper
# ---------------------------------------------------------------------------
def generate_time_index(num_bands, start_year=2012):
    times, year, month = [], start_year, 1
    for _ in range(num_bands):
        times.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_multiband_ndvi(ndvi_path, grid_path, out_csv,
                          start_year=2012, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"Workers: {max_workers}")
    print("Loading grid …")
    grid = pd.read_csv(grid_path)
    grid["lat"] = grid["lat"].astype(float)
    grid["lon"] = grid["lon"].astype(float)

    lons = grid["lon"].values
    lats = grid["lat"].values
    grid_ids = grid["grid_id"].values

    print("Opening raster to read metadata …")
    with rasterio.open(ndvi_path) as src:
        num_bands = src.count
        transform = src.transform
        src_crs   = src.crs
        nodata    = src.nodata
        is_4326   = src_crs and src_crs.to_epsg() == 4326

    print(f"Bands: {num_bands}  |  Grid points: {len(grid):,}")
    times = generate_time_index(num_bands, start_year)

    # Pixel indices computed once in the main process
    print("Computing pixel indices (once) …")
    rows, cols = compute_pixel_indices(
        transform, not is_4326, src_crs, lons, lats
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print(f"Processing {num_bands} bands with {max_workers} workers …")
    completed = 0

    # Submit all bands; write results as they finish to keep RAM low
    with open(out_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["grid_id", "time", "lat", "lon", "ndvi"])

        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for b in range(1, num_bands + 1):
                fut = pool.submit(
                    process_band,
                    ndvi_path, b, times[b - 1],
                    rows, cols,
                    grid_ids, lats, lons, nodata
                )
                futures[fut] = times[b - 1]

            for fut in as_completed(futures):
                time_label, rows_out = fut.result()
                writer.writerows(rows_out)
                completed += 1
                if completed % 10 == 0 or completed == num_bands:
                    print(f"  {completed}/{num_bands} bands done …")

    size_mb = os.path.getsize(out_csv) / 1_048_576
    print(f"\nSaved → {out_csv}  ({size_mb:.1f} MB)")
    print("Done ✓")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndvi",       default=str(DATA_DIR / "raw" / "ndvi.tif"))
    parser.add_argument("--grid",       default=str(DATA_DIR / "cleaned" / "elevation_grid.csv"))
    parser.add_argument("--out",        default=str(DATA_DIR / "cleaned" / "ndvi.csv"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--workers",    type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    args = parser.parse_args()

    build_multiband_ndvi(
        ndvi_path   = args.ndvi,
        grid_path   = args.grid,
        out_csv     = args.out,
        start_year  = args.start_year,
        max_workers = args.workers,
    )
