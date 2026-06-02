# Traffic Demand Prediction EDA and Modeling Report

## Dataset

- Training shape: `(77299, 11)`
- Target: `demand`
- Mean demand: `0.093942`
- Standard deviation: `0.142191`

## Missing Values

| Column | Missing count | Missing percent |
| --- | ---: | ---: |
| Temperature | 2495 | 3.228% |
| Weather | 797 | 1.031% |
| RoadType | 600 | 0.776% |
| day | 0 | 0.000% |
| geohash | 0 | 0.000% |
| Index | 0 | 0.000% |
| timestamp | 0 | 0.000% |
| NumberofLanes | 0 | 0.000% |
| demand | 0 | 0.000% |
| Landmarks | 0 | 0.000% |
| LargeVehicles | 0 | 0.000% |

## EDA Artifacts

- `correlation_heatmap.png`
- `feature_distributions.png`
- `demand_vs_weather.png`
- `demand_vs_hour.png`
- `demand_vs_roadtype.png`
- `demand_by_hour.csv`
- `missing_values.csv`

## Feature Engineering

- Parsed `timestamp` into `hour`, `minute`, `minutes_since_midnight`, cyclical hour features, and `time_step`.
- Derived `month`, `dayofweek`, and `weekend` from the supplied ordinal `day`.
- Added `rush_hour` and `night_flag`.
- Decoded `geohash` into latitude/longitude and added 4-character and 5-character geohash prefixes.
- Filled missing numeric values with training medians and categorical values with `Missing`.

The dataset does not contain an actual calendar date. `month` and `dayofweek` therefore use the provided
ordinal `day` as an inferred sequence rather than claiming a real-world calendar mapping.

## Chronological Validation R2

The final training intervals were reserved as validation data to reflect the time-forward test set.

| Model | R2 |
| --- | ---: |
| RandomForestRegressor | 0.816569 |
| XGBoostRegressor | 0.787585 |
| LightGBMRegressor | 0.758822 |
| CatBoostRegressor | 0.802698 |
| WeightedTop3Ensemble | 0.818166 |

## Top-3 Weighted Ensemble

| Model | Weight |
| --- | ---: |
| RandomForestRegressor | 0.756822 |
| CatBoostRegressor | 0.243178 |
| XGBoostRegressor | 0.000000 |

The final `submission.csv` is produced by refitting each top-3 model on all training rows and applying the
optimized validation weights to their test predictions.
