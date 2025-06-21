// =======================================================================================================
//                           LAND SURFACE TEMPERATURE 10 METER RESOLUTION PREDICTION
//                               USING MACHINE LEARNING RANDOM FOREST REGRESSION
//                                        Google Earth Engine Script
// =======================================================================================================

// =======================================================================================================
// 1. CONFIGURATION AND CONSTANTS
// =======================================================================================================

var CONFIG = {
  // Temporal parameters
  dates: {
    start: "yyyy-mm-dd",
    end: "yyyy-mm-dd",
  },

  // Model parameters
  model: {
    trainRatio: 0.8,
    finalPredictorsSize: 10,
    numTrees: 100,
    sampleSize: 1000,
  },

  // Processing parameters
  processing: {
    scale: 10,
    maxPixels: 1e13,
    focalRadius: 10,
    tileSize: 256,
    parallel: true
  },

  // Units and labels
  label: "LST",
  unit: "Celsius",

  // Predictors list
  predictors: [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDMI",
    "BSI",
    "NDWI",
    "MNDWI1",
    "MNDWI2",
    "NDVI",
    "NDBI",
    "SAVI",
    "OSAVI",
    "MSAVI",
    "NMDI",
    "UI",
    "NDDI",
    "elevation",
    "slope",
    "wetness",
    "greeness",
    "dryness",
  ],
  
  // Validation settings
  validation: {
    errorThreshold: 2, // °C
    minR2: 0.7
  },
  
  // Export settings
  export: {
    maxPixels: 1e13,
    cloudOptimized: true,
    formatOptions: {
      noData: 0
    }
  }
};

// Visualization parameters
var VIS_PARAMS = {
  lst: {
    min: 20,
    max: 40,
    palette: ["black", "purple", "blue", "cyan", "green", "yellow", "red"],
  },

  // Updated color gradients for better perceptual separation
  rdylgn: ["#a50026", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
  rdbl: ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4", "#313695"],

  uhi: {
    min: -2,
    max: 2,
    palette: [
      "#313695", "#4575b4", "#74add1", "#fed976", "#feb24c",
      "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"
    ],
  },

  utfviClass: {
    min: 1,
    max: 6,
    palette: ["#313695", "#74add1", "#fed976", "#fd8d3c", "#fc4e2a", "#b10026"],
  },
};

// Data collections (can be replace by assets)
var COLLECTIONS = {
  landsat8: ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
  landsat9: ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"),
  sentinel2: ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
  cloud: ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"),
  dem: ee.Image("USGS/SRTMGL1_003"),
};

// Study area
var roi = aoi;
Map.centerObject(roi);

// ================================================================================================
// 2. UTILITY FUNCTIONS
// ================================================================================================

/**
 * Cloud masking function for Landsat data
 * @param {ee.Image} image - Landsat image
 * @returns {ee.Image} - Cloud-masked LST image
 */
function cloudMaskLandsat(image) {
  var qa = image.select("QA_PIXEL");
  var mask = ee
    .Image(
      [1, 2, 3, 4].map(function (num) {
        return qa.bitwiseAnd(1 << num).eq(0);
      })
    )
    .reduce(ee.Reducer.allNonZero());

  return image
    .select(["ST_B10"], [CONFIG.label])
    .updateMask(mask)
    .multiply(0.00341802)
    .add(149)
    .add(-273.15); // Convert Kelvin to Celsius
}

/**
 * Cloud masking function for Sentinel-2 data
 * @param {ee.Image} image - Sentinel-2 image
 * @returns {ee.Image} - Cloud-masked image
 */
function cloudMaskS2(image) {
  var mask = image.select("cs").gt(0.6);
  return image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).updateMask(mask);
}
function scaleBands(image) {
  var opticalBands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']);
  return image.addBands(opticalBands.divide(10000), null, true);
}

/**
 * Create correlation scatter chart
 * @param {ee.FeatureCollection} sample - Sample data
 * @param {string} xVar - X variable name
 * @param {string} yVar - Y variable name
 * @param {string} title - Chart title
 * @returns {ui.Chart} - Scatter chart
 */
function createCorrelationChart(sample, xVar, yVar, title) {
  return ui.Chart.feature
    .byFeature(sample.randomColumn().limit(1000, "random"), xVar, [yVar])
    .setChartType("ScatterChart")
    .setOptions({
      title: title,
      hAxis: { title: xVar + " (" + CONFIG.unit + ")" },
      vAxis: { title: yVar },
      dataOpacity: 0.3,
      trendlines: [{ showR2: true, visibleInLegend: true, opacity: 1 }],
    });
}

/**
 * Create elevation stratification image
 * @param {ee.Image} elevation - DEM image
 * @returns {ee.Image} - Stratified elevation classes using quantile interval approach
 */
function createElevationStrata(elevation) {
  // Calculate quantiles for the elevation data
  var quantiles = elevation.reduceRegion({
    reducer: ee.Reducer.percentile([0, 20, 40, 60, 80, 100]),
    geometry: roi,
    scale: CONFIG.processing.scale,
    maxPixels: CONFIG.processing.maxPixels
  });
  
  // Convert dictionary to sorted list
  var breaks = ee.List(quantiles.values()).sort();
  
  // Create empty image for stratification
  var strat = ee.Image(0);

  // Create elevation classes based on quantile breaks
  for (var i = 0; i < 5; i++) {
    var lowerBreak = ee.Number(breaks.get(i));
    var upperBreak = ee.Number(breaks.get(i + 1));
    
    if (i === 0) {
      // First class
      strat = strat.where(elevation.lte(upperBreak), 1);
    } else if (i === 4) {
      // Last class
      strat = strat.where(elevation.gt(lowerBreak), 5);
    } else {
      // Middle classes
      strat = strat.where(
        elevation.gt(lowerBreak).and(elevation.lte(upperBreak)), 
        i + 1
      );
    }
  }

  // Return masked stratification
  return strat.selfMask().rename("elevation_class");
}

/**
 * Export image to Google Drive
 * @param {ee.Image} image - Image to export
 * @param {string} description - Export description
 * @param {string} folder - Drive folder name
 * @param {Object} options - Additional export options
 */
function exportImageToDrive(image, description, folder, options) {
  options = options || {};

  Export.image.toDrive({
    image: image,
    description: description,
    folder: folder,
    fileNamePrefix: options.fileNamePrefix || description,
    region: roi,
    scale: CONFIG.processing.scale,
    maxPixels: CONFIG.processing.maxPixels,
    fileFormat: options.fileFormat || "GeoTIFF",
    formatOptions: options.formatOptions || { cloudOptimized: true },
  });
}

// ================================================================================================
// 3. DATA PREPROCESSING
// ================================================================================================

/**
 * Load and preprocess satellite data
 */
function loadSatelliteData() {
  // Create filter for temporal and spatial constraints
  var filter = ee.Filter.and(
    ee.Filter.bounds(roi),
    ee.Filter.date(CONFIG.dates.start, CONFIG.dates.end)
  );

  // Generate Landsat LST composite
  var lstLandsat = COLLECTIONS.landsat8
    .filter(filter)
    .merge(COLLECTIONS.landsat9.filter(filter))
    .map(cloudMaskLandsat)
    .mean()
    .clip(roi);

  Map.addLayer(lstLandsat, VIS_PARAMS.lst, "LST Landsat");

  // Generate Sentinel-2 composite
  var s2Image = COLLECTIONS.sentinel2
    .filter(filter)
    .linkCollection(COLLECTIONS.cloud.filter(filter), "cs")
    .map(cloudMaskS2)
    .map(scaleBands)
    .median()
    .clip(roi);

  Map.addLayer(
    s2Image,
    { min: 0, max: 3000, bands: ["B8", "B11", "B12"] },
    "S2 Image",
    false
  );

  return {
    lstLandsat: lstLandsat,
    s2Image: s2Image,
  };
}

/**
 * Calculate spectral indices
 * @param {ee.Image} s2Image - Sentinel-2 image
 * @returns {ee.Image} - Image with calculated indices
 */
function calculateSpectralIndices(s2Image) {
  var indices = [
     {
      name: "BSI",
      formula: "((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE))",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "NDMI",
      formula: "(NIR - SWIR1) / (NIR + SWIR1)",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "NDWI",
      formula: "(GREEN - NIR) / (GREEN + NIR)",
      palette: VIS_PARAMS.rdbl,
    },
     {
      name: "MNDWI1",
      formula: "(GREEN - SWIR1) / (GREEN + SWIR1)",
      palette: VIS_PARAMS.rdbl,
    },
    {
      name: "MNDWI2",
      formula: "(GREEN - SWIR2) / (GREEN + SWIR2)",
      palette: VIS_PARAMS.rdbl,
    },
    {
      name: "NDVI",
      formula: "(NIR - RED) / (NIR + RED)",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "NDBI",
      formula: "(SWIR1 - NIR) / (SWIR1 + NIR)",
      palette: VIS_PARAMS.rdbl,
    },
    {
      name: "SAVI",
      formula: "((NIR - RED) * (1 + 0.5)) / (NIR + RED + 0.5)",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "OSAVI",
      formula: "(NIR - RED) / (NIR + RED + 0.16)",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "MSAVI",
      formula:
        "((2 * NIR + 1) - (sqrt((((2 * NIR + 1) * (2 * NIR + 1)) - 8 * (NIR - RED))))) / 2",
      palette: VIS_PARAMS.rdylgn,
    },
    {
      name: "UI",
      formula: "(SWIR2 - NIR) / (SWIR2 + NIR)",
      palette: VIS_PARAMS.rdbl,
    },
    {
      name: "NDDI",
      formula: "(SWIR2 - BLUE) / (SWIR2 + BLUE)",
      palette: VIS_PARAMS.rdbl,
    },
    {
      name: "NMDI",
      formula: "(NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))",
      palette: VIS_PARAMS.rdylgn,
    },
  ];

  var indicesImages = indices.map(function (dict) {
    var image = s2Image
      .expression(dict.formula, {
        NIR: s2Image.select("B8"),
        RED: s2Image.select("B4"),
        GREEN: s2Image.select("B3"),
        BLUE: s2Image.select("B2"),
        SWIR1: s2Image.select("B11"),
        SWIR2: s2Image.select("B12"),
      })
      .rename(dict.name);

    Map.addLayer(
      image,
      { min: -1, max: 1, palette: dict.palette },
      dict.name,
      false
    );
    return image;
  });

  return ee.Image.cat(indicesImages);
}

/**
 * Create land cover derivative maps
 * @param {ee.Image} indicesImage - Image with spectral indices
 * @returns {Object} - Object containing wetness, greenness and dryness maps
 */
function createLandCoverMaps(indicesImage) {
  var wet = indicesImage.select("MNDWI2").gt(0);
  var green = indicesImage.select("NDMI").gt(0.1).and(wet.eq(0));
  var dry = wet.eq(0).and(green.eq(0));

  var wetness = wet
    .focalMean(1, "square", "pixels", CONFIG.processing.focalRadius)
    .clip(roi)
    .rename("wetness");
  var greeness = green
    .focalMean(1, "square", "pixels", CONFIG.processing.focalRadius)
    .clip(roi)
    .rename("greeness");
  var dryness = dry
    .focalMean(1, "square", "pixels", CONFIG.processing.focalRadius)
    .clip(roi)
    .rename("dryness");

  Map.addLayer(
    wetness,
    { min: 0, max: 1, palette: VIS_PARAMS.rdbl },
    "Wetness",
    false
  );
  Map.addLayer(
    greeness,
    { min: 0, max: 1, palette: VIS_PARAMS.rdylgn },
    "Greeness",
    false
  );
  Map.addLayer(
    dryness,
    { min: 0, max: 1, palette: VIS_PARAMS.lst.palette },
    "Dryness",
    false
  );

  return {
    wetness: wetness,
    greeness: greeness,
    dryness: dryness,
  };
}

/**
 * Process topographic data
 * @returns {Object} - Object containing elevation, slope, and stratification
 */

function processTopographicData() {
  var elevation = COLLECTIONS.dem
    .select('elevation')
    .clip(roi)
    .unmask(0)

  // Use more efficient operations
  var slope = ee.Terrain.slope(elevation)
    .reproject({
      crs: elevation.projection(),
      scale: CONFIG.processing.scale
    });

  // Optimize memory usage in elevation strata calculation
  var elevationStrata = createElevationStrata(elevation)
    .reproject({
      crs: elevation.projection(),
      scale: CONFIG.processing.scale
    })
    .uint8(); // Use smaller data type
    
  return {
    elevation: elevation,
    slope: slope,
    elevationStrata: elevationStrata
  };
}

// ================================================================================================
// 4. FEATURE SELECTION AND CORRELATION ANALYSIS
// ================================================================================================

/**
 * Perform correlation analysis and feature selection
 * @param {ee.FeatureCollection} sample - Sample data
 * @returns {ee.List} - List of selected features
 */
function performFeatureSelection(sample) {
  // Create predictors + LST list for correlation analysis
  var predictorsLst = [];
  for (var i = 0; i < CONFIG.predictors.length; i++) {
    predictorsLst.push(CONFIG.predictors[i]);
    predictorsLst.push(CONFIG.label);
  }

  // Calculate correlation with LST
  var correlationReduce = sample.reduceColumns(
    ee.Reducer.pearsonsCorrelation().repeat(CONFIG.predictors.length),
    predictorsLst
  );

  // Create correlation table
  var correlationTable = ee.FeatureCollection(
    CONFIG.predictors.map(function (band, index) {
      var corr = ee.Number(
        ee.List(correlationReduce.get("correlation")).get(index)
      );
      return ee.Feature(null, {
        feature: band,
        r2: corr.pow(2),
      });
    })
  );

  // Get top correlated features
  var predictorsSize = CONFIG.model.finalPredictorsSize * 2;
  var topCorrelation = correlationTable
    .limit(predictorsSize, "r2", false)
    .aggregate_array("feature");

  // Print top correlated features first
  print(
    ee
      .String("Top " + predictorsSize + " related features with LST: ")
      .cat(topCorrelation.join(", "))
  );

  // Compute correlation matrix table between top features immediately after
  var correlationMatrixTable = ee.FeatureCollection(
    topCorrelation.map(function (main) {
      main = ee.String(main);
      var correlations = topCorrelation.map(function (other) {
        other = ee.String(other);
        var reducer = ee.Reducer.pearsonsCorrelation();
        var columns = [main, other];
        var corr = ee.Number(
          sample.reduceColumns(reducer, columns).get("correlation")
        );
        return ee.Dictionary().set(other, corr);
      });

      var merged = ee.Dictionary(
        correlations.iterate(function (item, acc) {
          return ee.Dictionary(acc).combine(ee.Dictionary(item), true);
        }, ee.Dictionary({}))
      );

      var values = merged.values();
      var avg = ee.Number(values.reduce(ee.Reducer.mean()));
      var med = ee.Number(values.reduce(ee.Reducer.median()));

      return ee.Feature(
        null,
        merged
          .set("main_variable", main)
          .set("averageCorrelation", avg)
          .set("medianCorrelation", med)
      );
    })
  );

  // Show correlation matrix right after top features list
  topCorrelation.evaluate(function (list) {
    var correlationMatrixChart = ui.Chart.feature
      .byFeature(correlationMatrixTable, "main_variable", list)
      .setChartType("Table")
      .setOptions({
        title: 'Correlation Matrix Between Top Features',
        allowHtml: true,
        cssClassNames: {
          headerRow: 'large-font',
          tableRow: 'large-font',
          oddTableRow: 'large-font'
        }
      });
    print(correlationMatrixChart);
  });

  // Create correlation chart
  var corrChart = ui.Chart.feature
    .byFeature(correlationTable, "feature", ["r2"])
    .setChartType("ColumnChart")
    .setOptions({
      title: "Correlation Coefficient R² with LST",
      hAxis: { title: "Feature" },
      vAxis: { title: "R²" },
    });
  print(corrChart);

  // Remove duplicate correlation matrix visualization
  // Chart average and median correlation
  var averageCorrelationMatrixChart = ui.Chart.feature
    .byFeature(correlationMatrixTable, "main_variable", [
      "averageCorrelation",
      "medianCorrelation",
    ])
    .setChartType("ColumnChart")
    .setOptions({
      title: "Average/Median Correlation Between Features (Multicollinearity)",
      hAxis: { title: "Feature" },
      vAxis: { title: "Correlation Value" },
    });
  print(averageCorrelationMatrixChart);

  // Select lowest average correlation features
  var finalPredictorsSize = CONFIG.model.finalPredictorsSize;
  var lowestCorrelationMatrix = correlationMatrixTable
    .limit(finalPredictorsSize, "averageCorrelation")
    .aggregate_array("main_variable");

  print(
    ee
      .String("Selected Features with Lowest Average Correlation: ")
      .cat(lowestCorrelationMatrix.join(", "))
  );

  return lowestCorrelationMatrix;
}
/**
 * Perform correlation matrix analysis to reduce multicollinearity
 * @param {ee.FeatureCollection} sample - Sample data
 * @param {ee.List} topCorrelation - Top correlated features
 * @returns {ee.List} - Final selected features
 */
function performCorrelationMatrixAnalysis(sample, topCorrelation) {
  var sequenceCorrelation = ee.List.sequence(
    0,
    topCorrelation.size().subtract(1)
  );
  var correlationLists = sequenceCorrelation
    .map(function (num0) {
      return sequenceCorrelation.map(function (num1) {
        return [topCorrelation.get(num0), topCorrelation.get(num1)];
      });
    })
    .flatten();

  // Calculate correlation matrix
  var correlationMatrixReduce = ee.List(
    sample
      .reduceColumns(
        ee.Reducer.pearsonsCorrelation().repeat(
          sequenceCorrelation.size().pow(2)
        ),
        correlationLists
      )
      .get("correlation")
  );

  // Create correlation matrix table
  var correlationMatrixTable = ee.FeatureCollection(
    sequenceCorrelation.map(function (index1) {
      index1 = ee.Number(index1);
      var variable = topCorrelation.get(index1);
      var values = sequenceCorrelation.map(function (index2) {
        index2 = ee.Number(index2);
        return correlationMatrixReduce.get(
          index1.multiply(sequenceCorrelation.size()).add(index2)
        );
      });
      var dict = ee.Dictionary.fromLists(topCorrelation, values);
      return ee.Feature(null, dict).set({
        main_variable: variable,
        averageCorrelation: values.reduce(ee.Reducer.mean()),
        medianCorrelation: values.reduce(ee.Reducer.median()),
      });
    })
  );

  // Display correlation matrix
  topCorrelation.evaluate(function (list) {
    var correlationMatrixChart = ui.Chart.feature
      .byFeature(correlationMatrixTable, "main_variable", list)
      .setChartType("Table");
    print("Correlation Matrix Between Top Features", correlationMatrixChart);
  });

  // Select features with lowest average correlation matrix
  var lowestCorrelationMatrix = correlationMatrixTable
    .limit(CONFIG.model.finalPredictorsSize, "averageCorrelation")
    .aggregate_array("main_variable");

  print(
    ee
      .String(
        "Final " + CONFIG.model.finalPredictorsSize + " selected features: "
      )
      .cat(lowestCorrelationMatrix.join(", "))
  );

  return lowestCorrelationMatrix;
}

// ================================================================================================
// 5. MACHINE LEARNING MODEL
// ================================================================================================

/**
 * Train and validate machine learning model
 * @param {ee.FeatureCollection} sample - Sample data
 * @param {ee.List} selectedFeatures - Selected features for modeling
 * @returns {Object} - Trained model and validation results
 */
function trainAndValidateModel(sample, selectedFeatures) {
  // Split sample into train and test
  sample = sample.randomColumn();
  var train = sample.filter(ee.Filter.lte("random", CONFIG.model.trainRatio));
  var test = sample.filter(ee.Filter.gt("random", CONFIG.model.trainRatio));

  // Train model
  var model = ee.Classifier.smileRandomForest(CONFIG.model.numTrees)
    .setOutputMode("REGRESSION")
    .train(train, CONFIG.label, selectedFeatures);

  // Compute feature importance
  var importance = ee.Dictionary(model.explain().get("importance"));
  var keys = importance.keys();
  var values = importance.values();
  var sum = values.reduce(ee.Reducer.sum());

  // Compute relative importance
  var relative = values.map(function (value) {
    return ee.Number(value).divide(sum).multiply(100);
  });

  // Create a list of features from keys and relative importance
  var features = ee.FeatureCollection(
    keys
      .zip(relative)
      .map(function (pair) {
        pair = ee.List(pair);
        return ee.Feature(null, {
          feature: pair.get(0),
          importance: pair.get(1),
        });
      })
  );

  // Create a bar chart using ui.Chart.feature.byFeature()
  var chart = ui.Chart.feature
    .byFeature(features, "feature", "importance")
    .setChartType("ColumnChart")
    .setOptions({
      title: "Relative Variable Importance (%)",
      hAxis: { title: "Variable" },
      vAxis: { title: "Importance (%)" },
      legend: { position: "none" },
      colors: ["#1f77b4"],
    });

  // Print chart
  print(chart);

  // Validate model
  var validationResults = validateModel(test, model, selectedFeatures);

  return {
    model: model,
    validation: validationResults,
    selectedFeatures: selectedFeatures,
  };
}

/**
 * Validate model performance
 * @param {ee.FeatureCollection} test - Test dataset
 * @param {ee.Classifier} model - Trained model
 * @param {ee.List} selectedFeatures - Selected features
 * @returns {Object} - Validation statistics
 */
function validateModel(test, model, selectedFeatures) {
  // Apply model to test data
  var testApply = test.classify(model, "prediction").map(function (feat) {
    var ref = ee.Number(feat.get(CONFIG.label));
    var pred = ee.Number(feat.get("prediction"));
    var error = ref.subtract(pred);
    return feat.set({
      error: error,
      errorAbs: error.abs(),
      line: ref,
    });
  });

  // Calculate validation statistics
  var stats = testApply.reduceColumns(
    ee.Reducer.mean().combine(ee.Reducer.pearsonsCorrelation()),
    ["errorAbs", CONFIG.label, "prediction"]
  );

  var mae = ee.Number(stats.get("mean"));
  var r2 = ee.Number(stats.get("correlation")).pow(2);
  var mse = testApply
    .map(function (feature) {
      var error = ee.Number(feature.get("error"));
      return feature.set("squaredError", error.pow(2));
    })
    .reduceColumns(ee.Reducer.mean(), ["squaredError"])
    .get("mean");
  var rmse = ee.Number(mse).sqrt();

  // Print validation results and 1:1 plot
  print("=== MODEL VALIDATION RESULTS ===");
  print(ee.String("R²: ").cat(r2));
  print(ee.String("MAE: ").cat(mae).cat(" ").cat(CONFIG.unit));
  print(ee.String("RMSE: ").cat(rmse).cat(" ").cat(CONFIG.unit));

  // Create and print 1:1 validation plot
  var plot = ui.Chart.feature
    .byFeature(
      testApply.randomColumn("chart").limit(1000, "chart"),
      CONFIG.label,
      ["prediction", "line"]
    )
    .setChartType("ScatterChart")
    .setOptions({
      title: "LST Reference vs Prediction",
      hAxis: {
        title: "Reference (" + CONFIG.unit + ")",
        viewWindow: { min: 20, max: 50 },
      },
      vAxis: {
        title: "Prediction (" + CONFIG.unit + ")",
        viewWindow: { min: 20, max: 50 },
      },
      dataOpacity: 0.3,
      series: [{}, { pointsVisible: false }],
      trendlines: [
        { showR2: true, visibleInLegend: true, opacity: 0.3 },
        { showR2: false, visibleInLegend: true, label: "1:1 line", opacity: 1 },
      ],
    });
  print(plot);

  // Add error distribution plots
  var errorHistogram = ui.Chart.feature
    .histogram({
      features: testApply,
      property: "error",
      maxBuckets: 30,
    })
    .setOptions({
      title: "Distribution of Prediction Errors",
      hAxis: { title: "Error (" + CONFIG.unit + ")" },
      vAxis: { title: "Frequency" },
      legend: { position: "none" },
      colors: ['#FFA500']
    });
  print(errorHistogram);

  // Absolute error histogram
  var absErrorHistogram = ui.Chart.feature
    .histogram({
      features: testApply,
      property: "errorAbs",
      maxBuckets: 30,
    })
    .setOptions({
      title: "Distribution of Absolute Errors",
      hAxis: { title: "Absolute Error (" + CONFIG.unit + ")" },
      vAxis: { title: "Frequency" },
      legend: { position: "none" },
      colors: ['#FFA500']
    });
  print(absErrorHistogram);

  // Calculate and print error percentiles
  var errorPercentiles = testApply.reduceColumns({
    reducer: ee.Reducer.percentile([10, 25, 50, 75, 90, 95]),
    selectors: ["errorAbs"],
  });
  print("Error Percentiles:", errorPercentiles);

  return {
    r2: r2,
    mae: mae,
    rmse: rmse,
    testData: testApply,
  };
}

// ================================================================================================
// 6. URBAN HEAT ANALYSIS
// ================================================================================================

/**
 * Calculate Urban Heat Island (UHI) and Urban Thermal Field Variance Index (UTFVI)
 * @param {ee.Image} lstPrediction - Predicted LST image
 * @returns {Object} - UHI and UTFVI images with statistics
 */
function calculateUrbanHeatIndices(lstPrediction) {
  // Calculate mean and standard deviation
  var meanLST = lstPrediction
    .reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: CONFIG.processing.scale,
      maxPixels: CONFIG.processing.maxPixels,
    })
    .getNumber(CONFIG.label);

  var stdLST = lstPrediction
    .reduceRegion({
      reducer: ee.Reducer.stdDev(),
      geometry: roi,
      scale: CONFIG.processing.scale,
      maxPixels: CONFIG.processing.maxPixels,
    })
    .getNumber(CONFIG.label);

  // Create constant images for calculations
  var meanLSTImage = ee.Image.constant(meanLST);
  var stdLSTImage = ee.Image.constant(stdLST);

  // Calculate UHI (normalized using Z-score)
  var uhi = lstPrediction
    .subtract(meanLSTImage)
    .divide(stdLSTImage)
    .rename("UHI_Normalized");

  // Calculate UTFVI
  var utfvi = lstPrediction
    .subtract(meanLSTImage)
    .divide(meanLSTImage)
    .rename("UTFVI");

  // Classify UTFVI
  var utfviClassified = ee
    .Image(0)
    .where(utfvi.lt(0), 1)
    .where(utfvi.gte(0).and(utfvi.lt(0.005)), 2)
    .where(utfvi.gte(0.005).and(utfvi.lt(0.01)), 3)
    .where(utfvi.gte(0.01).and(utfvi.lt(0.015)), 4)
    .where(utfvi.gte(0.015).and(utfvi.lt(0.02)), 5)
    .where(utfvi.gte(0.02), 6)
    .selfMask()
    .rename("UTFVI_class");

  // Add layers to map
  Map.addLayer(uhi, VIS_PARAMS.uhi, "UHI Normalized");
  Map.addLayer(utfviClassified, VIS_PARAMS.utfviClass, "UTFVI Classes");

  // Calculate and print statistics
  var uhiStats = uhi.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), "", true)
      .combine(ee.Reducer.minMax(), "", true),
    geometry: roi,
    scale: CONFIG.processing.scale,
    maxPixels: CONFIG.processing.maxPixels,
  });

  var utfviStats = utfvi.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), "", true)
      .combine(ee.Reducer.minMax(), "", true),
    geometry: roi,
    scale: CONFIG.processing.scale,
    maxPixels: CONFIG.processing.maxPixels,
  });

  print("=== URBAN HEAT ANALYSIS ===");
  print("UHI Statistics:", uhiStats);
  print("UTFVI Statistics:", utfviStats);

  return {
    uhi: uhi,
    utfvi: utfvi,
    utfviClassified: utfviClassified,
  };
}

// ================================================================================================
// 7. MAIN EXECUTION WORKFLOW
// ================================================================================================

/**
 * Generate final statistics and visualizations
 */
function generateFinalStatistics(
  lstReference,
  lstPrediction,
  sample,
  selectedFeatures
) {
  // LST statistics comparison
  var refStats = lstReference.reduceRegion({
    reducer: ee.Reducer.minMax().combine(ee.Reducer.mean(), null, true),
    geometry: roi,
    scale: 30,
    maxPixels: CONFIG.processing.maxPixels,
  });

  var predStats = lstPrediction.reduceRegion({
    reducer: ee.Reducer.min()
      .combine({
        reducer2: ee.Reducer.max(),
        sharedInputs: true,
      })
      .combine({
        reducer2: ee.Reducer.mean(),
        sharedInputs: true,
      })
      .combine({
        reducer2: ee.Reducer.stdDev(),
        sharedInputs: true,
      }),
    geometry: roi,
    scale: CONFIG.processing.scale,
    maxPixels: CONFIG.processing.maxPixels,
  });

  print("=== FINAL STATISTICS ===");
  print("LST Reference Statistics:", refStats);
  print("LST Prediction Statistics:", predStats);

  // Histogram comparison
  var histogram = ui.Chart.image
    .histogram({
      image: ee.Image([
        lstReference.rename("Reference"),
        lstPrediction.rename("Prediction"),
      ]),
      region: roi,
      scale: 100,
      maxPixels: CONFIG.processing.maxPixels,
    })
    .setOptions({
      title: "Histogram: LST Reference vs Prediction",
    });
  print(histogram);

  // Individual feature correlation charts
  selectedFeatures.evaluate(function (list) {
    list.forEach(function (feature) {
      var chart = createCorrelationChart(
        sample,
        CONFIG.label,
        feature,
        "LST vs " + feature + " Correlation"
      );
      print(chart);
    });
  });
}

/**
 * Setup export tasks
 */
function setupExports(lstPrediction, heatIndices) {
  // Export LST prediction
  exportImageToDrive(lstPrediction, "LST_Prediction", "GEE_LST_Results", {
    fileNamePrefix: "LST_Prediction",
  });

  // Export UHI
  exportImageToDrive(heatIndices.uhi, "UHI_Normalized", "GEE_LST_Results", {
    fileNamePrefix: "UHI_Normalized",
  });

  // Export UTFVI classes
  exportImageToDrive(
    heatIndices.utfviClassified,
    "UTFVI_Classes",
    "GEE_LST_Results",
    { fileNamePrefix: "UTFVI_Classes" }
  );

  // Export UTFVI continuous
  exportImageToDrive(heatIndices.utfvi, "UTFVI_Continuous", "GEE_LST_Results", {
    fileNamePrefix: "UTFVI_Continuous",
  });
  
  print("Export tasks have been created. Check the Tasks tab to run them.");
}

/**
 * Main execution function
 */
function main() {
  print("=== STARTING LST PREDICTION WORKFLOW ===");

  // 1. Load and preprocess satellite data
  print("1. Loading satellite data...");
  var satelliteData = loadSatelliteData();

  // 2. Calculate spectral indices
  print("2. Calculating spectral indices...");
  var indicesImage = calculateSpectralIndices(satelliteData.s2Image);

  // 3. Create land cover maps
  print("3. Creating land cover derivative maps...");
  var landCoverMaps = createLandCoverMaps(indicesImage);

  // 4. Process topographic data
  print("4. Processing topographic data...");
  var topoData = processTopographicData();

  // 5. Combine all features
  print("5. Combining features...");
  var features = ee.Image([
    satelliteData.s2Image,
    indicesImage,
    topoData.elevation,
    topoData.slope,
    topoData.elevationStrata,
    landCoverMaps.wetness,
    landCoverMaps.greeness,
    landCoverMaps.dryness,
  ]);

  // 6. Extract samples
  print("6. Extracting samples...");
  var sample = features.addBands(satelliteData.lstLandsat).stratifiedSample({
    scale: CONFIG.processing.scale,
    numPoints: CONFIG.model.sampleSize,
    region: roi,
    classBand: "elevation_class",
  });

  // 7. Feature selection
  print("7. Performing feature selection...");
  var selectedFeatures = performFeatureSelection(sample);

  // 8. Train and validate model
  print("8. Training and validating model...");
  var modelResults = trainAndValidateModel(sample, selectedFeatures);

  // 9. Generate LST prediction
  print("9. Generating LST prediction...");
  var lstPrediction = features.classify(modelResults.model, CONFIG.label);
  Map.addLayer(lstPrediction, VIS_PARAMS.lst, "LST Prediction");

  // 10. Urban heat analysis
  print("10. Performing urban heat analysis...");
  var heatIndices = calculateUrbanHeatIndices(lstPrediction);

  // 11. Generate final summary report
  generateSummaryReport(modelResults, heatIndices);

  // Add final predictors output
  modelResults.selectedFeatures.evaluate(function (features) {
    print("Final predictors:", features.join(", "));
  });

  // 12. Generate visual comparison statistics
  print("11. Generating final statistics...");
  generateFinalStatistics(
    satelliteData.lstLandsat,
    lstPrediction,
    sample,
    selectedFeatures
  );

  // 13. Export results
  print("12. Setting up exports...");
  setupExports(lstPrediction, heatIndices);

  print("=== WORKFLOW COMPLETED ===");
}

// ================================================================================================
// 8. ADDITIONAL ANALYSIS FUNCTIONS
// ================================================================================================

/**
 * Generate comprehensive summary report
 * @param {Object} modelResults - Model training results
 * @param {Object} heatIndices - Urban heat analysis results
 */
function generateSummaryReport(modelResults, heatIndices) {
  print("=== COMPREHENSIVE SUMMARY REPORT ===");
  print("Analysis Period:", ee.Date(CONFIG.dates.start).get("year"));
  print("Processing Scale:", CONFIG.processing.scale, "meters");
  print("Sample Size:", CONFIG.model.sampleSize, "points");
  print("Number of Trees:", CONFIG.model.numTrees);
  print(
    "Train/Test Split:",
    CONFIG.model.trainRatio,
    "/",
    1 - CONFIG.model.trainRatio
  );

  // Classify model performance using multiple metrics
  var r2Value = modelResults.validation.r2;
  var maeValue = modelResults.validation.mae;
  var rmseValue = modelResults.validation.rmse;

  // Performance classification based on R²
  var r2Class = ee.String(ee.Algorithms.If(
    r2Value.gt(0.90), "Very Good (R² > 0.90)",
    ee.Algorithms.If(
      r2Value.gt(0.80), "Good (R² 0.80-0.90)",
      ee.Algorithms.If(
        r2Value.gt(0.70), "Average (R² 0.70-0.80)",
        "Poor (R² < 0.70)"
      )
    )
  ));

  // Performance classification based on MAE
  var maeClass = ee.String(ee.Algorithms.If(
    maeValue.lt(1), "Very Good (MAE < 1°C)",
    ee.Algorithms.If(
      maeValue.lt(2), "Good (MAE 1-2°C)",
      ee.Algorithms.If(
        maeValue.lt(3), "Average (MAE 2-3°C)",
        "Poor (MAE > 3°C)"
      )
    )
  ));

  // Performance classification based on RMSE
  var rmseClass = ee.String(ee.Algorithms.If(
    rmseValue.lt(1.5), "Very Good (RMSE < 1.5°C)",
    ee.Algorithms.If(
      rmseValue.lt(2.5), "Good (RMSE 1.5-2.5°C)",
      ee.Algorithms.If(
        rmseValue.lt(3.5), "Average (RMSE 2.5-3.5°C)",
        "Poor (RMSE > 3.5°C)"
      )
    )
  ));

  // Modify print statements to show only classifications without numbers
  print("--- Model Performance ---");
  print("Individual Metric Classifications:");
  print("• R²:", r2Class);
  print("• MAE:", maeClass);
  print("• RMSE:", rmseClass);
}

// ================================================================================================
// 9. EXECUTE MAIN WORKFLOW
// ================================================================================================

// Run the main analysis
main();

print("=== SCRIPT EXECUTION COMPLETED ===");
print("Check the Console for results and the Layers panel for visualizations.");
print("Export tasks are available in the Tasks tab.");

/** ===============================================================================================
 * @ 2025
 * Zidd's note:
 * The result of 8 months of pain, caffeine, and emotional damage.
 * There's a github repo for the source code in 
 * https://github.com/itsZidd/gee-lst-10m-prediction
 * please give it a star if this script help you. Thanks :)
================================================================================================
*/
