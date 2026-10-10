import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from step3.download_twelvedata_xauusd import coverage_report, download_history


class TwelveDataCoverageTests(unittest.TestCase):
    def _bars(self):
        parts = []
        for day in pd.date_range("2026-01-05", "2026-01-07", freq="D", tz="UTC"):
            parts.append(pd.DataFrame({
                "timestamp": pd.date_range(day, periods=276, freq="5min"),
                "open": 2000.0, "high": 2001.0, "low": 1999.0,
                "close": 2000.5, "volume": float("nan"),
            }))
        return pd.concat(parts, ignore_index=True)

    def test_full_coverage_is_accepted(self):
        report = coverage_report(self._bars(), date(2026, 1, 5), date(2026, 1, 7))
        self.assertTrue(report["complete"])
        self.assertEqual(report["actual_5m_bars"], 828)
        self.assertEqual(report["missing_or_short_weekdays"], [])

    def test_short_period_is_rejected(self):
        report = coverage_report(self._bars().iloc[:276], date(2026, 1, 5), date(2026, 1, 7))
        self.assertFalse(report["complete"])
        self.assertEqual(report["missing_or_short_weekdays"], ["2026-01-06", "2026-01-07"])

    def test_six_month_range_rejects_one_month_sample(self):
        report = coverage_report(self._bars(), date(2026, 1, 5), date(2026, 6, 30))
        self.assertFalse(report["complete"])
        self.assertFalse(report["endpoint_coverage_ok"])

    def test_download_uses_requested_period_chunks_and_volume_free_schema(self):
        bars = self._bars().iloc[:276].copy()
        with patch("step3.download_twelvedata_xauusd.fetch_window", return_value=bars) as fetch:
            with patch("step3.download_twelvedata_xauusd.time.sleep"):
                result = download_history(date(2026, 1, 5), date(2026, 1, 6), "test-key")
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result.columns.tolist(), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertTrue(result["volume"].isna().all())


if __name__ == "__main__":
    unittest.main()
