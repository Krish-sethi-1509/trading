import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from market_service import build_market_snapshot


class MarketSnapshotTests(unittest.TestCase):
    def test_daily_range_and_change_are_derived_from_current_utc_day(self):
        now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        rows = []
        for index in range(1_441):
            stamp = now - timedelta(minutes=1_440 - index)
            close = 100.0 if index < 1_440 else 101.0
            high = max(close, 102.0 if index == 1_430 else close)
            low = min(close, 99.0 if index == 1_431 else close)
            rows.append(SimpleNamespace(timestamp=stamp, open=close, high=high, low=low, close=close, volume=None))

        snapshot = build_market_snapshot(rows, now=now)

        self.assertAlmostEqual(snapshot["daily_change_pct"], 1.0)
        self.assertEqual(snapshot["day_high"], 102.0)
        self.assertEqual(snapshot["day_low"], 99.0)
        self.assertEqual(snapshot["sentiment"], None)


if __name__ == "__main__":
    unittest.main()
