from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
TARGET = "demand"
RANDOM_STATE = 42
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
CAT_FEATURES = [
    "geohash",
    "geohash_prefix4",
    "geohash_prefix5",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "timestamp",
]

warnings.filterwarnings("ignore")


def decode_geohash(value: str) -> tuple[float, float]:
    """Decode the geohash to the center latitude/longitude."""
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


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    time_parts = data["timestamp"].str.split(":", expand=True).astype(int)
    data["hour"] = time_parts[0]
    data["minute"] = time_parts[1]
    data["minutes_since_midnight"] = data["hour"] * 60 + data["minute"]
    data["quarter"] = data["minutes_since_midnight"] // 15
    data["day_index"] = data["day"] - 48
    data["dayofweek"] = (data["day"] - 1) % 7
    data["weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["rush_hour"] = data["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    data["night_flag"] = ((data["hour"] < 6) | (data["hour"] >= 22)).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["minutes_since_midnight"] / 1440)
    data["hour_cos"] = np.cos(2 * np.pi * data["minutes_since_midnight"] / 1440)
    unique_geohashes = data["geohash"].dropna().unique()
    decoded = {gh: decode_geohash(gh) for gh in unique_geohashes}
    data["geohash_lat"] = data["geohash"].map(lambda gh: decoded.get(gh, (0.0, 0.0))[0])
    data["geohash_lon"] = data["geohash"].map(lambda gh: decoded.get(gh, (0.0, 0.0))[1])
    data["geohash_prefix4"] = data["geohash"].str[:4]
    data["geohash_prefix5"] = data["geohash"].str[:5]
    data["temp_missing"] = data["Temperature"].isna().astype(int)
    data["weather_missing"] = data["Weather"].isna().astype(int)
    return data


def fill_missing_values(frame: pd.DataFrame, numeric_medians: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    frame["Temperature"] = frame["Temperature"].fillna(numeric_medians["Temperature"])
    frame["RoadType"] = frame["RoadType"].fillna("Missing").astype(str)
    frame["Weather"] = frame["Weather"].fillna("Missing").astype(str)
    frame["LargeVehicles"] = frame["LargeVehicles"].fillna("Missing").astype(str)
    frame["Landmarks"] = frame["Landmarks"].fillna("Missing").astype(str)
    return frame


def add_prev_day_demand(source_train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    prev_day = source_train[["geohash", "day", "timestamp", TARGET]].copy()
    prev_day["day"] = prev_day["day"] + 1
    prev_day = prev_day.rename(columns={TARGET: "prev_day_demand"})
    frame = frame.merge(prev_day, on=["geohash", "day", "timestamp"], how="left")
    frame["prev_day_missing"] = frame["prev_day_demand"].isna().astype(int)
    return frame


def add_group_stats(source_train: pd.DataFrame, frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    frame = frame.copy()
    for name, cols in groups.items():
        group_mean = source_train.groupby(cols)[TARGET].mean()
        frame[name] = frame.set_index(cols).index.map(group_mean).astype(float)
        frame[name] = frame[name].fillna(source_train[TARGET].mean())
    return frame


def build_catboost_model(iterations: int, learning_rate: float, depth: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


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


def build_feature_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")

    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    features = build_features(combined)

    train_features = features.iloc[: len(train)].copy()
    test_features = features.iloc[len(train) :].copy()
    target = train[TARGET].copy()

    numeric_columns = [
        "day_index",
        "hour",
        "minute",
        "minutes_since_midnight",
        "quarter",
        "dayofweek",
        "geohash_lat",
        "geohash_lon",
        "Temperature",
        "NumberofLanes",
        "temp_missing",
        "weather_missing",
    ]
    numeric_medians = train_features[numeric_columns].median().fillna(0)
    train_features = fill_missing_values(train_features, numeric_medians)
    test_features = fill_missing_values(test_features, numeric_medians)

    train_features = add_prev_day_demand(train, train_features)
    test_features = add_prev_day_demand(train, test_features)

    return train_features, test_features, target, sample, numeric_medians


def make_validation_split(train_features: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    day49 = train_features[train_features["day"] == 49].index
    validation = train_features.loc[day49].sample(frac=0.2, random_state=RANDOM_STATE)
    validation_idx = validation.index
    train_idx = train_features.index.difference(validation_idx)
    return train_idx, validation_idx


def main() -> None:
    train_features, test_features, target, sample, _ = build_feature_set()

    train_idx, valid_idx = make_validation_split(train_features)
    train_with_target = train_features.loc[train_idx].copy()
    train_with_target[TARGET] = target.loc[train_idx]
    stat_groups = {
        "gh_mean": ["geohash"],
        "ts_mean": ["timestamp"],
        "gh_ts_mean": ["geohash", "timestamp"],
        "gh4_mean": ["geohash_prefix4"],
        "gh5_mean": ["geohash_prefix5"],
        "road_mean": ["RoadType"],
        "weather_mean": ["Weather"],
        "gh_hour_mean": ["geohash", "hour"],
        "gh_quarter_mean": ["geohash", "quarter"],
    }
    train_features = add_group_stats(train_with_target, train_features, stat_groups)
    test_features = add_group_stats(train_with_target, test_features, stat_groups)

    x_train = train_features.loc[train_idx]
    x_valid = train_features.loc[valid_idx]
    y_train = target.loc[train_idx]
    y_valid = target.loc[valid_idx]

    feature_columns = [
        "day_index",
        "hour",
        "minute",
        "minutes_since_midnight",
        "quarter",
        "dayofweek",
        "weekend",
        "rush_hour",
        "night_flag",
        "hour_sin",
        "hour_cos",
        "geohash_lat",
        "geohash_lon",
        "prev_day_missing",
        "prev_day_demand",
        "temp_missing",
        "weather_missing",
        "Temperature",
        "NumberofLanes",
        "geohash",
        "geohash_prefix4",
        "geohash_prefix5",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "timestamp",
        "gh_mean",
        "ts_mean",
        "gh_ts_mean",
        "gh4_mean",
        "gh5_mean",
        "road_mean",
        "weather_mean",
        "gh_hour_mean",
        "gh_quarter_mean",
    ]

    print(f"Training rows: {len(x_train)}, validation rows: {len(x_valid)}")

    base_model = build_catboost_model(iterations=900, learning_rate=0.035, depth=6)
    base_model.fit(
        x_train[feature_columns],
        y_train,
        cat_features=CAT_FEATURES,
    )
    base_valid = base_model.predict(x_valid[feature_columns])
    base_score = float(r2_score(y_valid, base_valid))
    print(f"Base CatBoost R2: {base_score:.6f}")

    correction_train = x_train[x_train["prev_day_missing"] == 0].copy()
    correction_valid = x_valid[x_valid["prev_day_missing"] == 0].copy()
    weights = np.array([1.0])
    correction_score = None
    correction_model = None

    if len(correction_valid) > 0 and len(correction_train) > 0:
        correction_model = build_catboost_model(iterations=650, learning_rate=0.04, depth=6)
        correction_model.fit(
            correction_train[feature_columns],
            y_train.loc[correction_train.index],
            cat_features=CAT_FEATURES,
        )
        correction_valid_pred = correction_model.predict(correction_valid[feature_columns])
        correction_score = float(r2_score(y_valid.loc[correction_valid.index], correction_valid_pred))
        print(f"Correction CatBoost R2: {correction_score:.6f}")

        matched_valid = x_valid["prev_day_missing"] == 0
        base_valid_matched = base_valid[matched_valid.to_numpy()]
        combined_preds = np.column_stack([base_valid_matched, correction_valid_pred])
        weights = optimize_weights(combined_preds, y_valid.loc[correction_valid.index])
        blended_score = float(r2_score(y_valid.loc[correction_valid.index], combined_preds @ weights))
        print(f"Weighted correction blend R2: {blended_score:.6f}")

    base_model.fit(train_features[feature_columns], target, cat_features=CAT_FEATURES)

    if correction_model is not None:
        full_correction_train = train_features[train_features["prev_day_missing"] == 0].copy()
        correction_model.fit(
            full_correction_train[feature_columns],
            target.loc[full_correction_train.index],
            cat_features=CAT_FEATURES,
        )

    test_predictions = base_model.predict(test_features[feature_columns])
    combined_predictions = test_predictions.copy()
    if correction_model is not None:
        matched = test_features["prev_day_missing"] == 0
        corr_pred = correction_model.predict(test_features.loc[matched, feature_columns])
        if len(weights) == 1:
            combined_predictions[matched] = corr_pred
        else:
            base_test_pred = test_predictions[matched]
            combined_predictions[matched] = np.clip(
                weights[0] * base_test_pred + weights[1] * corr_pred,
                0,
                None,
            )

    submission = pd.DataFrame(
        {
            sample.columns[0]: test_features["Index"].astype(int),
            sample.columns[1]: np.clip(combined_predictions, 0, None),
        }
    )
    submission.to_csv(ROOT / "submission_best.csv", index=False)

    results = {
        "base_r2": base_score,
        "correction_r2": correction_score,
        "weights": weights.tolist(),
        "matched_test_fraction": float((test_features["prev_day_missing"] == 0).mean()),
        "submission_shape": list(submission.shape),
    }
    (ROOT / "model_results_best.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
