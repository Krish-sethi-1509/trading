import os
import sys
import unittest
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.append(str(ROOT / "step3"))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from train_xgboost import make_target  # noqa: E402
import services  # noqa: E402
import live_features  # noqa: E402
from live_features import LiveFeatureUnavailable, build_live_feature_snapshot  # noqa: E402
from step3.feature_engineering import build_features  # noqa: E402
from models import Base, PredictionLog, PriceHistory  # noqa: E402


class ForwardTargetTests(unittest.TestCase):
    def test_target_uses_first_bar_after_horizon_not_a_pre_horizon_bar(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        frame = __import__("pandas").DataFrame({
            "timestamp": [start, start + timedelta(hours=4) - timedelta(minutes=1),
                          start + timedelta(hours=4) + timedelta(minutes=4)],
            "close": [100.0, 101.0, 102.0],
        })
        result = make_target(frame, timedelta(hours=4), timedelta(minutes=5), 0.001)
        self.assertEqual(float(result.iloc[0]["future_close"]), 102.0)

    def test_no_future_bar_within_tolerance_means_no_label(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        frame = __import__("pandas").DataFrame({
            "timestamp": [start, start + timedelta(hours=4) - timedelta(minutes=1)],
            "close": [100.0, 101.0],
        })
        result = make_target(frame, timedelta(hours=4), timedelta(minutes=5), 0.001)
        self.assertTrue(__import__("pandas").isna(result.iloc[0]["target_class"]))


class PredictionFreshnessTests(unittest.TestCase):
    def test_prediction_rejects_stale_features_before_model_loading(self):
        now = datetime.now(timezone.utc)
        with patch.object(services, "_latest_feature_vector",
                          return_value=({"close": 2000.0}, 2000.0, now - timedelta(seconds=901))), \
             patch.object(services, "utc_now", return_value=now), \
             patch.object(services, "_load_artifacts", side_effect=AssertionError("model must not load")):
            with self.assertRaises(services.ServiceUnavailable):
                services.create_prediction(session=None)


class OutcomeToleranceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def _pending_prediction(self, session, target):
        row = PredictionLog(
            timestamp=target - timedelta(hours=4),
            target_timestamp=target,
            predicted_direction="UP",
            confidence_score=0.8,
            reference_price=2000.0,
        )
        session.add(row)
        return row

    def _price(self, session, stamp, price=2001.0):
        session.add(PriceHistory(
            symbol="XAU/USD", timestamp=stamp, open=price, high=price,
            low=price, close=price, volume=None, macro_features={}, feature_vector={},
        ))

    def test_does_not_score_using_a_price_beyond_tolerance(self):
        target = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        with self.Session() as session:
            prediction = self._pending_prediction(session, target)
            self._price(session, target + timedelta(hours=2))
            session.commit()
            with patch.object(services, "utc_now", return_value=target + timedelta(hours=3)), \
                 patch.dict(os.environ, {"OUTCOME_MAX_DELAY_SECONDS": "900"}):
                services.update_prediction_outcomes(session)
            session.refresh(prediction)
            self.assertIsNone(prediction.actualized_at)

    def test_scores_a_price_inside_tolerance(self):
        target = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        with self.Session() as session:
            prediction = self._pending_prediction(session, target)
            self._price(session, target + timedelta(minutes=5), price=2004.0)
            session.commit()
            with patch.object(services, "utc_now", return_value=target + timedelta(minutes=6)), \
                 patch.dict(os.environ, {"OUTCOME_MAX_DELAY_SECONDS": "900"}):
                services.update_prediction_outcomes(session)
            session.refresh(prediction)
            self.assertEqual(prediction.actual_outcome, "UP")
            self.assertTrue(prediction.accuracy_flag)



class LiveFeaturePipelineTests(unittest.TestCase):
    def _bars(self, now, final_direction=1):
        import pandas as pd
        bars = []
        for index in range(80):
            stamp = now - timedelta(minutes=5 * (80 - index))
            close = 2000.0 + index * 0.2
            open_price = close - 2.0 if index == 79 and final_direction > 0 else close + 2.0 if index == 79 else close - 0.1
            bars.append({
                "timestamp": stamp, "open": open_price, "high": max(open_price, close) + 0.2,
                "low": min(open_price, close) - 0.2, "close": close, "volume": 123.0,
            })
        return pd.DataFrame(bars)

    def test_builds_features_from_recent_completed_bars_without_fabricating_volume(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        features, close, feature_timestamp = build_live_feature_snapshot(self._bars(now), now=now)
        self.assertEqual(feature_timestamp, now - timedelta(minutes=5))
        self.assertEqual(close, 2015.8)
        self.assertIn("atr_14", features)
        self.assertIn("relative_volume", features)
        self.assertIsNone(features["relative_volume"])
        self.assertIsNone(features["liquidity_high_sweep"])


    def test_live_macro_join_observes_the_one_day_publication_lag(self):
        import pandas as pd
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        tips = pd.DataFrame({
            "timestamp": [now - timedelta(days=2), now - timedelta(days=1)],
            "tips_yield": [3.0, 4.0],
        })
        features, _, _ = build_live_feature_snapshot(self._bars(now), tips=tips, now=now)
        # The observation from one day ago is not available until its next-day
        # availability timestamp; the just-published value is still in the future.
        self.assertEqual(features["tips_yield"], 3.0)

    def test_refuses_stale_live_candles(self):
        now = datetime(2026, 1, 3, 12, tzinfo=timezone.utc)
        with self.assertRaises(LiveFeatureUnavailable):
            build_live_feature_snapshot(
                self._bars(datetime(2026, 1, 1, 12, tzinfo=timezone.utc)),
                now=now,
                max_age_seconds=900,
            )

    def test_feature_lookup_never_falls_back_to_historical_csv(self):
        class EmptySession:
            def scalars(self, _statement):
                return SimpleNamespace(all=lambda: [])

        with patch.dict(os.environ, {"FEATURES_CSV_PATH": "historical/features.csv"}):
            with self.assertRaises(services.ServiceUnavailable):
                services._latest_feature_vector(EmptySession())

    def test_feature_lookup_returns_the_live_bar_timestamp_and_close(self):
        stamp = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        row = SimpleNamespace(
            timestamp=stamp,
            feature_vector={
                "atr_14": 1.2,
                "_source_timestamp": stamp.isoformat(),
                "_source_price": 2042.5,
                "_source": "twelve_data_5min",
            },
        )
        class SnapshotSession:
            def scalars(self, _statement):
                return SimpleNamespace(all=lambda: [row])

        features, price, feature_timestamp = services._latest_feature_vector(SnapshotSession())
        self.assertEqual(features, {"atr_14": 1.2})
        self.assertEqual(price, 2042.5)
        self.assertEqual(feature_timestamp, stamp)

    def test_training_and_live_builder_emit_identical_feature_schema_and_volume_mode(self):
        import pandas as pd
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        bars = self._bars(now)
        training_row = build_features(bars).iloc[-1]
        live_features, _, _ = build_live_feature_snapshot(bars, now=now)
        excluded = {"timestamp", "target_timestamp", "future_timestamp", "target_class", "future_close", "future_return"}
        training_schema = {
            str(name) for name, value in training_row.items()
            if name not in excluded and isinstance(value, (int, float, __import__("numpy").integer, __import__("numpy").floating))
        }
        self.assertEqual(set(live_features), training_schema)
        self.assertIsNone(live_features["relative_volume"])
        self.assertIsNone(live_features["liquidity_high_sweep"])

    def test_rejects_gaps_in_the_recent_provider_candles(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        bars = self._bars(now)
        bars = bars.drop(index=75)
        with self.assertRaisesRegex(LiveFeatureUnavailable, "incomplete"):
            build_live_feature_snapshot(bars, now=now)

    def test_twelve_data_bars_flow_through_features_and_model_and_market_change_changes_output(self):
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        outcomes = []
        class Response:
            def __init__(self, payload): self.payload = payload
            def raise_for_status(self): pass
            def json(self): return self.payload
        class DirectionClassifier:
            classes_ = np.array([0, 1, 2])
            def fit(self, values, labels=None): return self
            def predict_proba(self, values):
                return np.array([[0.90, 0.05, 0.05] if row[0] < 0 else [0.05, 0.05, 0.90] for row in values])

        class PredictionSession:
            def __init__(self, vector):
                self.row = SimpleNamespace(feature_vector=vector, timestamp=now)
                self.added = []
            def scalars(self, _statement):
                return SimpleNamespace(all=lambda: [self.row])
            def add(self, value): self.added.append(value)
            def commit(self): pass

        for final_direction in (1, -1):
            raw = self._bars(now, final_direction=final_direction)
            values = []
            for row in raw.to_dict("records"):
                values.append({
                    "datetime": row["timestamp"].isoformat(), "open": str(row["open"]),
                    "high": str(row["high"]), "low": str(row["low"]),
                    "close": str(row["close"]), "volume": str(row["volume"]),
                })
            with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": "test-key", "FEATURE_MAX_AGE_SECONDS": "900"}), \
                 patch.object(live_features.requests, "get", return_value=Response({"values": values})):
                provider_bars = live_features._download_recent_bars()
            features, reference_price, feature_time = build_live_feature_snapshot(provider_bars, now=now)
            ordered_names = ["candlestick_body_atr"]
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", DirectionClassifier()),
            ]).fit(np.array([[-2.0], [2.0]]), np.array([0, 2]))
            model.gold_feature_columns_ = ordered_names
            model.gold_trained_class_ids_ = [0, 1, 2]
            vector = {
                **features, "_source_timestamp": feature_time.isoformat(),
                "_source_price": reference_price, "_source": "twelve_data_5min",
            }
            session = PredictionSession(vector)
            with patch.object(services, "utc_now", return_value=now), \
                 patch.object(services, "_load_artifacts", return_value=(model, None, ordered_names)):
                result = services.create_prediction(session)
            self.assertEqual(len(session.added), 1)
            outcomes.append(result["direction"])
        self.assertEqual(outcomes, ["UP", "DOWN"])

    def test_current_failed_refresh_does_not_reuse_an_older_cached_feature_row(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        old = SimpleNamespace(
            timestamp=now - timedelta(minutes=1),
            feature_vector={"atr_14": 1.2, "_source_timestamp": (now - timedelta(minutes=5)).isoformat(), "_source_price": 2000.0},
        )
        latest = SimpleNamespace(timestamp=now, feature_vector={})
        class FailedRefreshSession:
            def scalars(self, _statement):
                return SimpleNamespace(all=lambda: [latest, old])
        with self.assertRaises(services.ServiceUnavailable):
            services._latest_feature_vector(FailedRefreshSession())

    def test_model_feature_order_is_preserved_and_missing_columns_are_rejected(self):
        ordered = services._prepare_feature_frame({"second": 2.0, "first": 1.0}, object(), ["first", "second"])
        self.assertEqual(list(ordered.columns), ["first", "second"])
        with self.assertRaisesRegex(services.ServiceUnavailable, "missing model inputs"):
            services._prepare_feature_frame({"first": 1.0}, object(), ["first", "second"])



if __name__ == "__main__":
    unittest.main()
