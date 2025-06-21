# Google Earth Engine 10 Meter LST Prediction

Land Surface Temperature 10-Meter Resolution Prediction Using Machine Learning Random Forest Regression

---

## Overview

This project provides a comprehensive [Google Earth Engine](https://earthengine.google.com/) (GEE) script to **downscale and predict Land Surface Temperature (LST) at 10-meter resolution**. The method leverages machine learning, specifically Random Forest Regression, to predict high-resolution LST using the 30m Landsat Surface Temperature (ST_B10) band as reference and a suite of spectral, topographic, and land cover predictors derived from Sentinel-2 and DEM data.

---

## Key Features

- **Downscaling LST:** Predicts 10m LST from 30m Landsat ST_B10 using machine learning.
- **Multi-source Data Fusion:** Integrates Landsat 8/9, Sentinel-2, DEM, and Cloud Score+ dataset.
- **Spectral Indices:** Computes NDVI, NDWI, NDMI, BSI, NDBI, SAVI, OSAVI, MSAVI, UI, NDDI, NMDI, MNDWI, and more.
- **Topographic Features:** Includes elevation, slope, and elevation stratification.
- **Land Cover Derivatives:** Wetness, greenness, and dryness maps.
- **Feature Selection:** Automated correlation and multicollinearity analysis to select optimal predictors.
- **Model Validation:** Outputs R², MAE, RMSE, error distributions, and 1:1 plots.
- **Urban Heat Analysis:** Calculates Urban Heat Island (UHI) and Urban Thermal Field Variance Index (UTFVI).
- **Export:** Prepares export tasks for LST prediction, UHI, and UTFVI maps.

---

## Workflow

1. **Configuration:** Set temporal range, model parameters, processing scale, and predictor list.
2. **Data Loading:** Load and preprocess Landsat and Sentinel-2 imagery, cloud mask, and DEM.
3. **Feature Engineering:** Calculate spectral indices and land cover/topographic features.
4. **Sampling:** Extract stratified samples for model training and validation.
5. **Feature Selection:** Analyze correlation and multicollinearity to select best predictors.
6. **Model Training:** Train Random Forest regression model on selected features.
7. **Validation:** Assess model performance with test data and visualize results.
8. **Prediction:** Generate 10m LST prediction map for the study area.
9. **Urban Heat Analysis:** Compute UHI and UTFVI indices and classes.
10. **Export:** Set up export tasks for results.

---

## Usage

1. **Open the Script:**
   - Use the [10m-LST-Prediction.js](10m-LST-Prediction.js) file in the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).

2. **Set Required Assets:**
   - Define your study area (`aoi`) at the top of the script or in the GEE Assets tab.

3. **Configure Parameters:**
   - Adjust `CONFIG` for your date range, sample size, and other settings as needed.

4. **Run the Script:**
   - Click "Run" in the GEE Code Editor.
   - Monitor the Console for progress, validation results, and summary statistics.
   - View LST predictions and indices in the Layers panel.

5. **Export Results:**
   - Go to the "Tasks" tab in GEE and start the export tasks for LST, UHI, and UTFVI outputs.

---

## Predictor Variables

- **Spectral Bands:** B2, B3, B4, B8, B11, B12 (Sentinel-2)
- **Spectral Indices:** NDVI, NDWI, NDMI, BSI, NDBI, SAVI, OSAVI, MSAVI, UI, NDDI, NMDI, MNDWI1, MNDWI2
- **Topography:** Slope, Elevation Class (using quantile based stratification)
- **Land Cover:** Wetness, Greenness, Dryness

---

## Model & Validation

- **Algorithm:** Random Forest Regression (`ee.Classifier.smileRandomForest`)
- **Feature Selection:** Based on correlation with LST and multicollinearity reduction
- **Validation Metrics:** R², MAE, RMSE, error percentiles, scatter and histogram plots

---

## Outputs

- **LST Prediction Map (10m)**
- **Urban Heat Island (UHI) Map**
- **Urban Thermal Field Variance Index (UTFVI) Map (continuous and classified)**
- **Validation and summary statistics**

---

## Requirements

- [Google Earth Engine account](https://signup.earthengine.google.com/)
- Defined study area (`aoi`)

---
