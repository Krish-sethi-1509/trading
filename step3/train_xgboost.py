"""Train an XGBoost classifier for 4-hour XAU/USD direction.

Input is the feature CSV produced by feature_engineering.py. The target is
computed as the future close at timestamp + horizon; an observation is used
only when a bar exists within --label-tolerance of that time. Rows are split
chronologically, with the most recent fraction reserved for testing.

Labels use the future simple return: Down below -neutral-threshold, Up above
+neutral-threshold, otherwise Neutral. The default threshold is 0.10%.

Example:
  python train_xgboost.py --input data/features.csv --output-dir artifacts/xgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CLASS_NAMES = ["Down", "Neutral", "Up"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
NON_FEATURE_COLUMNS = {
    "timestamp", "target", "target_class", "future_close", "future_return",
    "label", "label_id", "bar_date", "id",
}


def make_target(
    frame: pd.DataFrame,
    horizon: pd.Timedelta,
    tolerance: pd.Timedelta,
    neutral_threshold: float,
) -> pd.DataFrame:
    """Attach the future close, return, and categorical direction to each bar."""
    if "timestamp" not in frame or "close" not in frame:
        raise ValueError("Input must contain timestamp and close columns")
    if neutral_threshold < 0:
        raise ValueError("neutral threshold cannot be negative")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["close"] = pd.to_numeric(result["close"], errors="raise")
    if result["timestamp"].duplicated().any():
        raise ValueError("Input timestamps must be unique")
    result = result.sort_values("timestamp").reset_index(drop=True)

    future = result[["timestamp", "close"]].rename(
        columns={"timestamp": "future_timestamp", "close": "future_close"}
    )
    result["target_timestamp"] = result["timestamp"] + horizon
    aligned = pd.merge_asof(
        result.sort_values("target_timestamp"),
        future.sort_values("future_timestamp"),
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="nearest",
        tolerance=tolerance,
    )
    aligned["future_return"] = aligned["future_close"] / aligned["close"] - 1.0
    aligned["target_class"] = np.select(
        [aligned["future_return"] < -neutral_threshold, aligned["future_return"] > neutral_threshold],
        ["Down", "Up"],
        default="Neutral",
    ).astype(object)
    aligned.loc[aligned["future_close"].isna(), "target_class"] = np.nan
    return aligned.sort_values("timestamp").reset_index(drop=True)


def make_pipeline(seed: int, estimators: int, max_depth: int, learning_rate: float) -> Pipeline:
    """Create train-only imputation/scaling followed by a multiclass booster."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "xgboost",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(CLASS_NAMES),
                    eval_metric="mlogloss",
                    n_estimators=estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def train(
    data: pd.DataFrame,
    *,
    horizon_hours: float,
    label_tolerance_minutes: float,
    neutral_threshold: float,
    test_size: float,
    seed: int,
    estimators: int,
    max_depth: int,
    learning_rate: float,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float]]:
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test size must be between 0.05 and 0.5")
    labeled = make_target(
        data,
        pd.Timedelta(hours=horizon_hours),
        pd.Timedelta(minutes=label_tolerance_minutes),
        neutral_threshold,
    ).dropna(subset=["target_class"])
    if len(labeled) < 20:
        raise ValueError("Fewer than 20 usable labeled rows; provide more intraday history")

    candidates = [
        column for column in labeled.columns
        if column not in NON_FEATURE_COLUMNS
        and column not in {"target_timestamp", "future_timestamp"}
        and not column.startswith(("target_", "future_", "label_"))
        and pd.api.types.is_numeric_dtype(labeled[column])
    ]
    features = labeled[candidates].replace([np.inf, -np.inf], np.nan)
    # Remove features with no observed values in the full historical sample;
    # all remaining imputation/scaling is learned inside the training pipeline.
    feature_columns = features.columns[features.notna().any()].tolist()
    features = features[feature_columns]
    if not feature_columns:
        raise ValueError("No numeric feature columns found in the input")

    labels = labeled["target_class"].map(CLASS_TO_ID).astype("int64")
    split_at = int(len(labeled) * (1.0 - test_size))
    if split_at <= 0 or split_at >= len(labeled):
        raise ValueError("Chronological split produced an empty train or test set")
    x_train, x_test = features.iloc[:split_at], features.iloc[split_at:]
    y_train, y_test = labels.iloc[:split_at], labels.iloc[split_at:]
    if y_train.nunique() < 2:
        raise ValueError("Training partition contains fewer than two target classes")

    observed_classes = sorted(y_train.unique().tolist())
    encoded_train = y_train.map({class_id: index for index, class_id in enumerate(observed_classes)})
    pipeline = make_pipeline(seed, estimators, max_depth, learning_rate)
    pipeline.set_params(xgboost__num_class=len(observed_classes))
    pipeline.fit(x_train, encoded_train)
    encoded_predictions = pipeline.predict(x_test).astype(int)
    predicted = np.asarray([observed_classes[index] for index in encoded_predictions], dtype=int)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
    }
    # Metadata used by inference so it applies exactly the same feature order.
    pipeline.gold_feature_columns_ = feature_columns
    pipeline.gold_class_names_ = CLASS_NAMES
    pipeline.gold_trained_class_ids_ = observed_classes
    results = labeled.iloc[split_at:][["timestamp", "close", "future_close", "future_return", "target_class"]].copy()
    results["predicted_class"] = [ID_TO_CLASS[int(value)] for value in predicted]
    results["predicted_class_id"] = predicted
    results.index = range(len(results))
    details = pd.DataFrame(report).transpose()
    return pipeline, results, details, matrix, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Feature CSV from feature_engineering.py")
    parser.add_argument("--output-dir", required=True, help="Directory for model and evaluation artifacts")
    parser.add_argument("--horizon-hours", type=float, default=4.0)
    parser.add_argument("--label-tolerance-minutes", type=float, default=5.0)
    parser.add_argument("--neutral-threshold", type=float, default=0.001,
                        help="Absolute 4-hour return cutoff for Neutral (default 0.001 = 0.10%%)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Most recent fraction held out for testing (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()
    if args.horizon_hours <= 0 or args.label_tolerance_minutes < 0:
        parser.error("horizon must be positive and label tolerance nonnegative")
    if args.estimators < 1 or args.max_depth < 1 or args.learning_rate <= 0:
        parser.error("estimators/depth must be positive and learning rate must be positive")

    data = pd.read_csv(args.input)
    model, predictions, report, matrix, metrics = train(
        data,
        horizon_hours=args.horizon_hours,
        label_tolerance_minutes=args.label_tolerance_minutes,
        neutral_threshold=args.neutral_threshold,
        test_size=args.test_size,
        seed=args.seed,
        estimators=args.estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    from joblib import dump

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(model, output_dir / "xgboost_pipeline.joblib")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    report.to_csv(output_dir / "classification_report.csv")
    matrix_frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    matrix_frame.index.name = "actual\\predicted"
    matrix_frame.to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                **metrics,
                "horizon_hours": args.horizon_hours,
                "neutral_threshold": args.neutral_threshold,
                "class_order": CLASS_NAMES,
                "features": model.gold_feature_columns_,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Chronological holdout confusion matrix (rows = actual, columns = predicted):")
    print(matrix_frame.to_string())
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Train rows: {metrics['train_rows']}; test rows: {metrics['test_rows']}; features: {metrics['feature_count']}")
    print(f"Artifacts saved to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
