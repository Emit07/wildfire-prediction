# convert_frap.py
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# =====================================================
# 1. LOAD GRID
# =====================================================

grid_df = pd.read_csv(DATA_DIR / "cleaned" / "elevation_grid.csv")

grid = gpd.GeoDataFrame(
    grid_df,
    geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat),
    crs="EPSG:4326"
)

# =====================================================
# 2. LOAD FIRE DATA
# =====================================================

fires = gpd.read_file(DATA_DIR / "raw" / "frap")
fires = fires.to_crs("EPSG:4326")

# =====================================================
# 3. USE ALARM_DATE (IMPORTANT FIX)
# =====================================================

fires["date"] = pd.to_datetime(fires["ALARM_DATE"], errors="coerce")

# drop invalid dates
fires = fires.dropna(subset=["date"])

# =====================================================
# 4. FILTER EXACT TIME WINDOW
# =====================================================

start = pd.Timestamp("2012-01-20")
end   = pd.Timestamp("2025-01-20")

fires = fires[(fires["date"] >= start) & (fires["date"] <= end)].copy()

# =====================================================
# 5. MONTH LABEL
# =====================================================

fires["year_month"] = fires["date"].dt.to_period("M").astype(str)

# =====================================================
# 6. FIRE AREA
# =====================================================

fires_proj = fires.to_crs("EPSG:3857")
fires["burned_area_m2"] = fires_proj.geometry.area

# =====================================================
# 7. SPATIAL JOIN (fire → grid)
# =====================================================

joined = gpd.sjoin(
    fires,
    grid,
    how="inner",
    predicate="intersects"
)

# =====================================================
# 8. AGGREGATE PER GRID + MONTH
# =====================================================

agg = joined.groupby(
    ["grid_id", "lat", "lon", "year_month"]
).agg(
    fire_count=("geometry", "count"),
    burned_area_m2=("burned_area_m2", "sum")
).reset_index()

# convert to binary label
agg["fire_occurred"] = (agg["fire_count"] > 0).astype(int)

# keep only needed columns
agg = agg[["grid_id", "lat", "lon", "year_month", "fire_occurred", "burned_area_m2"]]

# =====================================================
# 9. BUILD FULL GRID × MONTH PANEL
# =====================================================

months = pd.period_range(
    start="2012-01",
    end="2025-01",
    freq="M"
).astype(str)

grid_month = pd.MultiIndex.from_product(
    [grid_df["grid_id"], months],
    names=["grid_id", "year_month"]
).to_frame(index=False)

grid_month = grid_month.merge(
    grid_df[["grid_id", "lat", "lon"]],
    on="grid_id",
    how="left"
)

# =====================================================
# 10. MERGE FIRE LABELS
# =====================================================

final = grid_month.merge(
    agg,
    on=["grid_id", "lat", "lon", "year_month"],
    how="left"
)

final["fire_occurred"] = final["fire_occurred"].fillna(0).astype(int)
final["burned_area_m2"] = final["burned_area_m2"].fillna(0)

# =====================================================
# 11. SAVE
# =====================================================

final.to_csv(
    DATA_DIR / "cleaned" / "ca_fire_monthly_2012_2025.csv",
    index=False
)

print("Done: monthly wildfire dataset created")
