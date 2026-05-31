# Wildfire Prediction using Machine Learning

Using machine learning to predict wildfire risk and how climate change affects risk in California.

> **Note**
>
> Many files are still being added to this repo, more to come.

## Examples

![July 2024 Example](media/july2024example.png)

## Data sources

* Historical Fire Data: https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters
* NDVI: Modis + Google Earth Engine
* Weather: Copernicus ERA5
* Topography: scraped 30m dem (3dep)

## Models

* XGBoost: Model we ended up using
* LightGBM: would only classify 0
* RandomForest: Took to long to train
