from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
EDA_DIR = ROOT / "eda_outputs"
EDA_DIR.mkdir(exist_ok=True)
TARGET = "demand"
RANDOM_STATE = 42
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def decode_geohash(value: str) -> tuple[float, float]:
    """Decode a geohash to the center point of its latitude/longitude box."""
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    for char in str(value).lower():
        number = BASE32.index(char)
        for mask in (16, 8, 4, 2, 1):
            interval = lon if even else lat
            midpoint = (interval[0] + interval[1]) / 2
            if number & mask:
                interval[0] = midpoint
            else:
                interval[1] = midpoint
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    time_parts = data["timestamp"].str.split(":", expand=True).astype(int)
    data["hour"] = time_parts[0]
    data["minute"] = time_parts[1]
    data["minutes_since_midnight"] = data["hour"] * 60 + data["minute"]
    data["time_step"] = data["day"] * 96 + data["minutes_since_midnight"] // 15

    # The source supplies an ordinal day, not a real calendar date.
    data["month"] = ((data["day"] - 1) // 30 % 12) + 1
    data["dayofweek"] = (data["day"] - 1) % 7
    data["weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["rush_hour"] = data["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    data["night_flag"] = ((data["hour"] < 6) | (data["hour"] >= 22)).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["minutes_since_midnight"] / 1440)
    data["hour_cos"] = np.cos(2 * np.pi * data["minutes_since_midnight"] / 1440)

    unique_geohashes = data["geohash"].dropna().unique()
    decoded = {item: decode_geohash(item) for item in unique_geohashes}
    data["geohash_lat"] = data["geohash"].map(lambda item: decoded[item][0])
    data["geohash_lon"] = data["geohash"].map(lambda item: decoded[item][1])
    data["geohash_prefix4"] = data["geohash"].str[:4]
    data["geohash_prefix5"] = data["geohash"].str[:5]

    data = data.drop(columns=["Index", "timestamp"], errors="ignore")
    return data


def run_eda(train: pd.DataFrame) -> dict:
    missing = train.isna().sum().sort_values(ascending=False)
    missing_table = pd.DataFrame(
        {"missing_count": missing, "missing_percent": (missing / len(train) * 100).round(3)}
    )
    missing_table.to_csv(EDA_DIR / "missing_values.csv")

    sns.set_theme(style="whitegrid")
    numeric = train.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    distribution_columns = ["demand", "Temperature", "NumberofLanes", "day"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, axis in zip(distribution_columns, axes.flat):
        sns.histplot(data=train, x=column, kde=True, ax=axis)
        axis.set_title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "feature_distributions.png", dpi=160)
    plt.close()

    clean_weather = train.fillna({"Weather": "Missing"})
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=clean_weather, x="Weather", y=TARGET)
    plt.title("Demand vs Weather")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "demand_vs_weather.png", dpi=160)
    plt.close()

    eda_features = make_features(train)
    hour_summary = eda_features.groupby("hour")[TARGET].agg(["mean", "median", "count"])
    hour_summary.to_csv(EDA_DIR / "demand_by_hour.csv")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hour_summary, x=hour_summary.index, y="mean", marker="o")
    plt.title("Average Demand vs Hour")
    plt.ylabel("Average demand")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "demand_vs_hour.png", dpi=160)
    plt.close()

    clean_road = train.fillna({"RoadType": "Missing"})
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=clean_road, x="RoadType", y=TARGET)
    plt.title("Demand vs RoadType")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "demand_vs_roadtype.png", dpi=160)
    plt.close()

    return {
        "train_shape": list(train.shape),
        "missing_values": missing_table.to_dict(orient="index"),
        "demand_mean": float(train[TARGET].mean()),
        "demand_std": float(train[TARGET].std()),
        "weather_demand_mean": clean_weather.groupby("Weather")[TARGET].mean().round(6).to_dict(),
        "roadtype_demand_mean": clean_road.groupby("RoadType")[TARGET].mean().round(6).to_dict(),
        "hour_demand_mean": hour_summary["mean"].round(6).to_dict(),
    }


def build_models(categorical_columns: list[str], numeric_columns: list[str]) -> dict:
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return {
        "XGBoostRegressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=450,
                        learning_rate=0.045,
                        max_depth=9,
                        min_child_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LightGBMRegressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=700,
                        learning_rate=0.035,
                        num_leaves=80,
                        max_depth=-1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=0.1,
                        verbosity=-1,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def optimize_weights(predictions: np.ndarray, actual: pd.Series) -> np.ndarray:
    count = predictions.shape[1]

    def objective(weights: np.ndarray) -> float:
        return -r2_score(actual, predictions @ weights)

    result = minimize(
        objective,
        np.repeat(1 / count, count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
    )
    return result.x


def write_report(eda: dict, scores: dict, top_names: list[str], weights: np.ndarray) -> None:
    missing_rows = [
        f"| {column} | {values['missing_count']} | {values['missing_percent']:.3f}% |"
        for column, values in eda["missing_values"].items()
    ]
    score_rows = [f"| {name} | {score:.6f} |" for name, score in scores.items()]
    ensemble_rows = [f"| {name} | {weight:.6f} |" for name, weight in zip(top_names, weights)]
    report = f"""# Traffic Demand Prediction EDA and Modeling Report

## Dataset

- Training shape: `{tuple(eda["train_shape"])}`
- Target: `demand`
- Mean demand: `{eda["demand_mean"]:.6f}`
- Standard deviation: `{eda["demand_std"]:.6f}`

## Missing Values

| Column | Missing count | Missing percent |
| --- | ---: | ---: |
{chr(10).join(missing_rows)}

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
{chr(10).join(score_rows)}

## Base Weighted Ensemble

| Model | Weight |
| --- | ---: |
{chr(10).join(ensemble_rows)}

The base fallback predictions are produced by refitting the weighted ensemble on all training rows.
The final `submission.csv` then applies a day-49 correction blend for rows with an exact previous-day
same-location/same-time demand signal, while preserving the base ensemble for unmatched rows.
"""
    (ROOT / "EDA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    eda = run_eda(train)

    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    all_features = make_features(combined)
    features = all_features.iloc[: len(train)].copy()
    test_features = all_features.iloc[len(train) :].copy()
    target = train[TARGET].copy()

    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    numeric_columns = features.columns.difference(categorical_columns).tolist()
    numeric_medians = features[numeric_columns].median().fillna(0)
    for frame in (features, test_features):
        frame[categorical_columns] = frame[categorical_columns].fillna("Missing").astype(str)
        frame[numeric_columns] = frame[numeric_columns].fillna(numeric_medians)

    # Keep the final 12 intervals as a time-forward validation window.
    validation_start = features["time_step"].max() - 11
    train_mask = features["time_step"] < validation_start
    validation_mask = ~train_mask
    x_train, x_valid = features.loc[train_mask], features.loc[validation_mask]
    y_train, y_valid = target.loc[train_mask], target.loc[validation_mask]
    print(f"Chronological split: train={len(x_train)}, validation={len(x_valid)}")

    models = build_models(categorical_columns, numeric_columns)
    scores: dict[str, float] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        validation_predictions[name] = model.predict(x_valid)
        scores[name] = float(r2_score(y_valid, validation_predictions[name]))
        print(f"{name}: R2={scores[name]:.6f}")

    ranked = sorted(scores, key=scores.get, reverse=True)
    top_names = ranked[:3]
    prediction_matrix = np.column_stack([validation_predictions[name] for name in top_names])
    weights = optimize_weights(prediction_matrix, y_valid)
    ensemble_score = float(r2_score(y_valid, prediction_matrix @ weights))
    scores["WeightedTop3Ensemble"] = ensemble_score
    print("Top models:", top_names)
    print("Ensemble weights:", dict(zip(top_names, weights.round(6))))
    print(f"WeightedTop3Ensemble: R2={ensemble_score:.6f}")

    test_prediction_map: dict[str, np.ndarray] = {}
    for name in top_names:
        print(f"Refitting {name} on all training rows...")
        model = models[name]
        model.fit(features, target)
        test_prediction_map[name] = model.predict(test_features)

    test_prediction_matrix = np.column_stack([test_prediction_map[name] for name in top_names])
    base_test_predictions = np.clip(test_prediction_matrix @ weights, 0, None)
    base_submission = pd.DataFrame(
        {
            sample.columns[0]: test["Index"],
            sample.columns[1]: base_test_predictions,
        }
    )
    base_submission.to_csv(ROOT / "submission_base_ensemble.csv", index=False)

    correction_features = features.copy()
    correction_test_features = test_features.copy()
    previous_day = features[["geohash", "day", "minutes_since_midnight"]].copy()
    previous_day["prev_day_same_slot_demand"] = target.to_numpy()
    previous_day["day"] += 1
    correction_features = correction_features.merge(
        previous_day,
        on=["geohash", "day", "minutes_since_midnight"],
        how="left",
    )
    correction_test_features = correction_test_features.merge(
        previous_day,
        on=["geohash", "day", "minutes_since_midnight"],
        how="left",
    )
    for frame in (correction_features, correction_test_features):
        frame["prev_day_missing"] = frame["prev_day_same_slot_demand"].isna().astype(int)
        frame["prev_day_to_geohash_lat"] = frame["prev_day_same_slot_demand"] / (
            frame["geohash_lat"].abs() + 1e-6
        )

    correction_rows = correction_features["prev_day_same_slot_demand"].notna()
    correction_train = correction_features.loc[correction_rows].copy()
    correction_target = target.loc[correction_rows]
    correction_validation_mask = correction_train["minutes_since_midnight"] > 75
    correction_train_mask = ~correction_validation_mask

    correction_categorical = correction_train.select_dtypes(include="object").columns.tolist()
    correction_numeric = correction_train.columns.difference(correction_categorical).tolist()
    correction_medians = (
        correction_train.loc[correction_train_mask, correction_numeric].median().fillna(0)
    )
    for frame in (correction_train, correction_test_features):
        frame[correction_categorical] = frame[correction_categorical].fillna("Missing").astype(str)
        frame[correction_numeric] = frame[correction_numeric].fillna(correction_medians)

    correction_weights = np.array([0.2573, 0.7427])
    if correction_validation_mask.any() and correction_train_mask.any():
        correction_xgb_validator = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                correction_categorical,
                            ),
                            ("numeric", "passthrough", correction_numeric),
                        ]
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        correction_cat_validator = CatBoostRegressor(
            iterations=400,
            learning_rate=0.035,
            depth=5,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
            thread_count=-1,
            allow_writing_files=False,
        )
        correction_xgb_validator.fit(
            correction_train.loc[correction_train_mask],
            correction_target.loc[correction_train_mask],
        )
        correction_cat_validator.fit(
            correction_train.loc[correction_train_mask],
            correction_target.loc[correction_train_mask],
            cat_features=correction_categorical,
        )
        correction_valid_matrix = np.column_stack(
            [
                correction_xgb_validator.predict(correction_train.loc[correction_validation_mask]),
                correction_cat_validator.predict(correction_train.loc[correction_validation_mask]),
            ]
        )
        correction_weights = optimize_weights(
            correction_valid_matrix,
            correction_target.loc[correction_validation_mask],
        )
        correction_valid_predictions = correction_valid_matrix @ correction_weights
        correction_score = float(
            r2_score(
                correction_target.loc[correction_validation_mask],
                correction_valid_predictions,
            )
        )
        scores["Day49CorrectionHoldout"] = correction_score
        scores["Day49CorrectionXGBoostWeight"] = float(correction_weights[0])
        scores["Day49CorrectionCatBoostWeight"] = float(correction_weights[1])
        print("Day49 correction weights:", correction_weights.round(6).tolist())
        print(f"Day49CorrectionHoldout: R2={correction_score:.6f}")

    correction_xgb_model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            correction_categorical,
                        ),
                        ("numeric", "passthrough", correction_numeric),
                    ]
                ),
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.03,
                    max_depth=4,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    correction_cat_model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.035,
        depth=5,
        loss_function="RMSE",
        verbose=False,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        allow_writing_files=False,
    )
    correction_xgb_model.fit(correction_train, correction_target)
    correction_cat_model.fit(
        correction_train,
        correction_target,
        cat_features=correction_categorical,
    )
    corrected_test_matrix = np.column_stack(
        [
            correction_xgb_model.predict(correction_test_features),
            correction_cat_model.predict(correction_test_features),
        ]
    )
    corrected_test_predictions = np.clip(corrected_test_matrix @ correction_weights, 0, None)
    has_correction = correction_test_features["prev_day_missing"].to_numpy() == 0
    final_test_predictions = base_test_predictions.copy()
    final_test_predictions[has_correction] = corrected_test_predictions[has_correction]

    submission = pd.DataFrame(
        {
            sample.columns[0]: test["Index"],
            sample.columns[1]: final_test_predictions,
        }
    )
    submission.to_csv(ROOT / "submission.csv", index=False)
    write_report(eda, scores, top_names, weights)

    results = {
        "validation_rows": int(validation_mask.sum()),
        "scores": scores,
        "ranked_base_models": ranked,
        "ensemble_models": top_names,
        "ensemble_weights": dict(zip(top_names, map(float, weights))),
        "submission_shape": list(submission.shape),
    }
    (ROOT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
