import lzma
import struct
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from datetime import date

import pandas as pd

from step3.download_historical_xauusd import (decode_bi5, fetch_side, midpoint_bars, resample_five_minutes, validate_history_coverage)


class DukascopyDecoderTests(unittest.TestCase):
    def test_decode_lzma_minute_bar(self):
        row = struct.pack(">IIIIIf", 60, 2_650_000, 2_651_000, 2_649_000, 2_652_000, 12.5)
        payload = lzma.compress(row, format=lzma.FORMAT_ALONE)
        decoded = decode_bi5(payload, date(2026, 1, 2))
        self.assertEqual(decoded.loc[0, "timestamp"], pd.Timestamp("2026-01-02T00:01:00Z"))
        self.assertAlmostEqual(decoded.loc[0, "open"], 2650.0)
        self.assertAlmostEqual(decoded.loc[0, "high"], 2652.0)
        self.assertAlmostEqual(decoded.loc[0, "low"], 2649.0)
        self.assertAlmostEqual(decoded.loc[0, "close"], 2651.0)
        self.assertAlmostEqual(decoded.loc[0, "volume"], 12.5)

    def test_midpoint_and_resample_skip_sparse_buckets(self):
        timestamps = pd.date_range("2026-01-02T00:00:00Z", periods=7, freq="min")
        bid = pd.DataFrame({
            "timestamp": timestamps, "open": [100.0] * 7, "high": [101.0] * 7,
            "low": [99.0] * 7, "close": [100.5] * 7, "volume": [4.0] * 7,
        })
        ask = bid.assign(open=102.0, high=103.0, low=101.0, close=102.5, volume=6.0)
        minute = midpoint_bars(bid, ask)
        five = resample_five_minutes(minute, minimum_minutes=3)
        self.assertEqual(len(five), 1)
        self.assertEqual(five.loc[0, "timestamp"], pd.Timestamp("2026-01-02T00:00:00Z"))
        self.assertEqual(five.loc[0, "open"], 101.0)
        self.assertEqual(five.loc[0, "high"], 102.0)
        self.assertEqual(five.loc[0, "low"], 100.0)
        self.assertEqual(five.loc[0, "close"], 101.5)
        self.assertEqual(five.loc[0, "volume"], 25.0)

    def test_fetch_retries_transient_http_503(self):
        row = struct.pack(">IIIIIf", 60, 2_650_000, 2_651_000, 2_649_000, 2_652_000, 12.5)
        payload = lzma.compress(row, format=lzma.FORMAT_ALONE)
        transient = HTTPError("https://example.test", 503, "busy", None, None)
        with patch("step3.download_historical_xauusd.urlopen", side_effect=[transient, BytesIO(payload)]) as request:
            with patch("step3.download_historical_xauusd.time.sleep"):
                result = fetch_side(date(2026, 1, 2), "BID")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "close"], 2651.0)

    def test_reject_malformed_record(self):
        with self.assertRaisesRegex(ValueError, "Malformed BI5"):
            decode_bi5(lzma.compress(b"bad", format=lzma.FORMAT_ALONE), date(2026, 1, 2))

    def _complete_three_weekdays(self):
        chunks = []
        for day in pd.date_range("2026-01-05", "2026-01-07", freq="D", tz="UTC"):
            chunks.append(pd.DataFrame({
                "timestamp": pd.date_range(day, periods=276, freq="5min"),
                "open": 2000.0, "high": 2001.0, "low": 1999.0,
                "close": 2000.5, "volume": 1.0,
            }))
        return pd.concat(chunks, ignore_index=True)

    def test_coverage_report_proves_requested_dates_and_expected_bars(self):
        report = validate_history_coverage(
            self._complete_three_weekdays(), date(2026, 1, 5), date(2026, 1, 7)
        )
        self.assertEqual(report["actual_5m_bars"], 828)
        self.assertEqual(report["missing_or_short_weekdays"], [])
        self.assertTrue(report["endpoint_coverage_ok"])
        self.assertEqual(report["bar_coverage"], 1.0)

    def test_coverage_rejects_a_missing_weekday(self):
        bars = self._complete_three_weekdays()
        bars = bars[bars["timestamp"].dt.date != date(2026, 1, 6)]
        with self.assertRaisesRegex(RuntimeError, "2026-01-06"):
            validate_history_coverage(bars, date(2026, 1, 5), date(2026, 1, 7))

    def test_coverage_rejects_short_history_for_six_month_request(self):
        with self.assertRaisesRegex(RuntimeError, "do not cover expected endpoints"):
            validate_history_coverage(
                self._complete_three_weekdays(), date(2026, 1, 5), date(2026, 6, 30)
            )


if __name__ == "__main__":
    unittest.main()
