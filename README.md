# Traffic Demand Prediction Submission

## Overview

This submission predicts traffic demand for each row in `test.csv`. The final
prediction file is `submission.csv`, formatted with the required columns:

```text
Index,demand
```

## Dataset

- Training rows: `77,299`
- Test rows: `41,778`
- Target column: `demand`
- Evaluation metric: R2 score

## Missing Values

| Feature | Missing rows |
| --- | ---: |
| Temperature | 2,495 |
| Weather | 797 |
| RoadType | 600 |

Missing numeric values are filled with the training median. Missing categorical
values are represented by the `Missing` category.

## Feature Engineering

The pipeline creates the following features:

- Timestamp: `hour`, `minute`, `minutes_since_midnight`, `time_step`
- Cyclical time: `hour_sin`, `hour_cos`
- Day: `month`, `dayofweek`, `weekend`
- Traffic periods: `rush_hour`, `night_flag`
- Weather: cleaned categorical `Weather` and imputed `Temperature`
- Geohash: decoded latitude, longitude, 4-character prefix, and 5-character prefix

The dataset provides an ordinal `day` value rather than an actual calendar
date. Therefore, `month` and `dayofweek` are inferred from this sequence.

## Exploratory Data Analysis

The `eda_outputs` folder contains:

- `correlation_heatmap.png`
- `feature_distributions.png`
- `demand_vs_weather.png`
- `demand_vs_hour.png`
- `demand_vs_roadtype.png`
- `missing_values.csv`
- `demand_by_hour.csv`

## Model Comparison

A chronological validation split was used because the test data continues
forward in time after the training data.

| Model | Validation R2 |
| --- | ---: |
| RandomForestRegressor | 0.816569 |
| CatBoostRegressor | 0.802698 |
| XGBoostRegressor | 0.787585 |
| LightGBMRegressor | 0.758822 |
| Weighted top-3 ensemble | **0.818166** |

## Final Ensemble

The submission uses an optimized weighted ensemble of the top three models:

| Model | Weight |
| --- | ---: |
| RandomForestRegressor | 0.756822 |
| CatBoostRegressor | 0.243178 |
| XGBoostRegressor | 0.000000 |

Although XGBoost was included among the top three candidates, validation-based
weight optimization assigned it a near-zero contribution.

## Files

| File | Purpose |
| --- | --- |
| `submission.csv` | Prediction file to upload |
| `traffic_demand_pipeline.py` | Complete EDA, training, ensemble, and prediction pipeline |
| `EDA_REPORT.md` | Detailed generated report |
| `model_results.json` | Validation scores and ensemble weights |
| `eda_outputs/` | EDA plots and summary tables |

## Run Instructions

Install dependencies:

```powershell
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn xgboost catboost lightgbm
```

Run the full pipeline:

```powershell
python traffic_demand_pipeline.py
```

Upload `submission.csv` as the prediction file.
