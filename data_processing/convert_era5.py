from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ERA5_FILE = DATA_DIR / "raw" / "era5_california.nc"
GRID_FILE = DATA_DIR / "cleaned" / "elevation_grid.csv"
OUTPUT_FILE = DATA_DIR / "cleaned" / "california_monthly_weather.csv"

print("Loading datasets...")

# drop auxiliary coords that can interfere with masking
ds = xr.open_dataset(ERA5_FILE, drop_variables=["expver", "number"])

grid = pd.read_csv(GRID_FILE)

# filter grid to ERA5 extent
lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())
lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())

print(f"ERA5 latitude range:  {lat_min} to {lat_max}")
print(f"ERA5 longitude range: {lon_min} to {lon_max}")

grid = grid[
    (grid["lat"] >= lat_min) & (grid["lat"] <= lat_max) &
    (grid["lon"] >= lon_min) & (grid["lon"] <= lon_max)
].copy()

print(f"Grid cells in ERA5 bounds: {len(grid):,}")

# select nearest ERA5 cell for every grid point (vectorized)
lats = xr.DataArray(grid["lat"].values, dims="point")
lons = xr.DataArray(grid["lon"].values, dims="point")

sampled = ds.sel(latitude=lats, longitude=lons, method="nearest")

# result shape is (n_time, n_points); ravel() in C order →
# [t0p0, t0p1, ..., t0pN, t1p0, ..., t1pN, ...]
n_time = ds.sizes["valid_time"]
n_points = len(grid)

times = pd.to_datetime(ds.valid_time.values)
year_months = times.strftime("%Y-%m")

# unit conversions
temp_c = (sampled["t2m"].values - 273.15)           # K → °C
precip_mm = (sampled["tp"].values * 1000.0)          # m → mm
wind_speed_ms = np.sqrt(
    sampled["u10"].values ** 2 + sampled["v10"].values ** 2
)
soil_moisture = sampled["swvl1"].values
solar_radiation = sampled["ssrd"].values

df = pd.DataFrame({
    "grid_id":       np.tile(grid["grid_id"].values, n_time),
    "lat":           np.tile(grid["lat"].values, n_time),
    "lon":           np.tile(grid["lon"].values, n_time),
    "year_month":    np.repeat(year_months, n_points),
    "temp_c":        temp_c.ravel(),
    "precip_mm":     precip_mm.ravel(),
    "wind_speed_ms": wind_speed_ms.ravel(),
    "soil_moisture": soil_moisture.ravel(),
    "solar_radiation": solar_radiation.ravel(),
})

# drop ocean/sea grid cells (ERA5 land-sea mask fills them with NaN)
n_before = len(df)
df = df.dropna(subset=["temp_c", "precip_mm", "wind_speed_ms", "soil_moisture", "solar_radiation"])
print(f"Dropped {n_before - len(df):,} NaN rows (ERA5 ocean cells)")

df = df.sort_values(["grid_id", "year_month"]).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)
print(df.head(10).to_string())
print(f"\nSaved {len(df):,} rows → {OUTPUT_FILE}")
