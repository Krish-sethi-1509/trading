import lzma
import struct
import unittest
from datetime import date

import pandas as pd

from step3.download_historical_xauusd import decode_bi5, midpoint_bars, resample_five_minutes


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
            "timestamp": timestamps, "open": [100.0] * 8, "high": [101.0] * 8,
            "low": [99.0] * 8, "close": [100.5] * 8, "volume": [4.0] * 8,
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

    def test_reject_malformed_record(self):
        with self.assertRaisesRegex(ValueError, "Malformed BI5"):
            decode_bi5(lzma.compress(b"bad", format=lzma.FORMAT_ALONE), date(2026, 1, 2))


if __name__ == "__main__":
    unittest.main()
