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
from live_features import LiveFeatureUnavailable, build_live_feature_snapshot  # noqa: E402
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
    def _bars(self, now):
        import pandas as pd
        bars = []
        for index in range(80):
            stamp = now - timedelta(minutes=5 * (80 - index))
            close = 2000.0 + index * 0.2
            bars.append({
                "timestamp": stamp, "open": close - 0.1, "high": close + 0.2,
                "low": close - 0.2, "close": close, "volume": None,
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



if __name__ == "__main__":
    unittest.main()
