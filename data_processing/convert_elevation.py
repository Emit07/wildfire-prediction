# convert_elevation.py

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows

from data_loading import make_ca_grid

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sample_mean(src, bounds):

    win = rasterio.windows.from_bounds(
        *bounds,
        transform=src.transform
    )

    data = src.read(1, window=win)

    if src.nodata is not None:
        data = data[data != src.nodata]

    data = data[np.isfinite(data)]

    if data.size == 0:
        return np.nan

    return float(np.mean(data))


grid = make_ca_grid()

results = []

with rasterio.open(DATA_DIR / "raw" / "california_dem.tif") as dem_src, \
     rasterio.open(DATA_DIR / "raw" / "california_slope.tif") as slope_src, \
     rasterio.open(DATA_DIR / "raw" / "california_aspect.tif") as aspect_src:

    for i, row in grid.iterrows():

        bounds = (
            row["lon_min"],
            row["lat_min"],
            row["lon_max"],
            row["lat_max"]
        )

        elevation = sample_mean(dem_src, bounds)
        slope = sample_mean(slope_src, bounds)
        aspect = sample_mean(aspect_src, bounds)

        results.append({
            "grid_id": row["grid_id"],
            "lat": row["lat"],
            "lon": row["lon"],
            "elevation_m": elevation,
            "slope_deg": slope,
            "aspect_deg": aspect
        })

        if i % 100 == 0:
            print(i)

out = pd.DataFrame(results)

out["elevation_m"] = out["elevation_m"].fillna(
    out["elevation_m"].median()
)

out["slope_deg"] = out["slope_deg"].fillna(0)

out["aspect_deg"] = out["aspect_deg"].fillna(180)

out.to_csv(
    DATA_DIR / "cleaned" / "elevation_grid.csv",
    index=False
)

print(f"Saved {len(out):,} rows")
