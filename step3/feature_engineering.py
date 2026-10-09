"""Build leakage-aware institutional and technical features from market data.

Gold input must be intraday OHLCV (UTC timestamps) to calculate the London/NY
overlap VWAP. Expected CSV columns: timestamp, open, high, low, close, volume.
Optional CSVs provide daily DXY and 10-year TIPS yields, COMEX futures, COT
positioning, option strike open interest, and macro release timestamps.

Examples:
  python feature_engineering.py --gold data/xauusd_5m.csv --tips data/tips.csv \
      --dxy data/dxy.csv --output data/features.csv

All optional data files use a `timestamp` or `date` column and the value column
documented in --help. Times with no UTC offset are interpreted as UTC. Features
are causal: rolling reference levels are shifted, and external daily series are
joined backward with merge_asof. This module computes research features, not
trade recommendations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OHLCV = ("open", "high", "low", "close", "volume")


def _utc_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    return parsed


def _load_csv(path: str | Path, value_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_col = "timestamp" if "timestamp" in frame else "date" if "date" in frame else None
    if time_col is None:
        raise ValueError(f"{path} must contain a 'timestamp' or 'date' column")
    frame["timestamp"] = _utc_timestamp(frame[time_col])
    if time_col != "timestamp":
        frame = frame.drop(columns=[time_col])
    missing = set(value_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _asof_join(
    gold: pd.DataFrame,
    path: str | None,
    value_columns: tuple[str, ...],
    availability_lag: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    if path is None:
        return gold
    external = _load_csv(path, value_columns)
    external = external[["timestamp", *value_columns]]
    external["timestamp"] = external["timestamp"] + availability_lag
    collisions = (set(external.columns) & set(gold.columns)) - {"timestamp"}
    external = external.rename(columns={column: f"{column}_external" for column in collisions})
    return pd.merge_asof(
        gold.sort_values("timestamp"), external.sort_values("timestamp"),
        on="timestamp", direction="backward", allow_exact_matches=True,
    )


def _rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minimum = min_periods if min_periods is not None else max(5, window // 4)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _compute_overlap_vwap(frame: pd.DataFrame) -> None:
    """Session-reset 12:00–16:00 UTC VWAP and volume-weighted sigma bands."""
    utc = frame["timestamp"].dt
    minute_of_day = utc.hour * 60 + utc.minute
    in_overlap = minute_of_day.ge(12 * 60) & minute_of_day.lt(16 * 60)
    frame["is_london_ny_overlap"] = in_overlap.astype("int8")
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].clip(lower=0).fillna(0)
    session = frame["timestamp"].dt.strftime("%Y-%m-%d")
    weights = volume.where(in_overlap, 0.0)
    weighted_price = (typical * weights).groupby(session).cumsum()
    cumulative_volume = weights.groupby(session).cumsum()
    vwap = weighted_price.div(cumulative_volume.replace(0, np.nan))
    weighted_sq = (typical.pow(2) * weights).groupby(session).cumsum()
    variance = (weighted_sq.div(cumulative_volume.replace(0, np.nan)) - vwap.pow(2)).clip(lower=0)
    sigma = np.sqrt(variance)
    # Keep overlap measures missing outside the target session; this avoids
    # carrying an expired session's VWAP into later hours.
    frame["overlap_vwap"] = vwap.where(in_overlap)
    frame["overlap_vwap_sigma"] = sigma.where(in_overlap)
    for multiple in (1, 2, 3):
        frame[f"overlap_vwap_plus_{multiple}sigma"] = (vwap + multiple * sigma).where(in_overlap)
        frame[f"overlap_vwap_minus_{multiple}sigma"] = (vwap - multiple * sigma).where(in_overlap)
    frame["overlap_vwap_deviation_sigma"] = ((frame["close"] - vwap) / sigma.replace(0, np.nan)).where(in_overlap)
    frame["overlap_vwap_plus2_reversion_sell"] = (
        in_overlap & frame["close"].ge(vwap + 2 * sigma) & sigma.gt(0)
    ).astype("int8")


def _compute_liquidity_sweeps(frame: pd.DataFrame, lookback: int, volume_window: int) -> None:
    """Flag prior rolling high/low raids that close back through the level."""
    prior_high = frame["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = frame["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    frame["prior_liquidity_high"] = prior_high
    frame["prior_liquidity_low"] = prior_low
    volume_mean = frame["volume"].rolling(volume_window, min_periods=max(3, volume_window // 3)).mean().shift(1)
    frame["relative_volume"] = frame["volume"] / volume_mean.replace(0, np.nan)
    high_sweep = frame["high"].gt(prior_high) & frame["close"].lt(prior_high)
    low_sweep = frame["low"].lt(prior_low) & frame["close"].gt(prior_low)
    elevated_volume = frame["relative_volume"].ge(1.5)
    frame["liquidity_high_sweep"] = (high_sweep & elevated_volume).astype("int8")
    frame["liquidity_low_sweep"] = (low_sweep & elevated_volume).astype("int8")
    # A downside break followed by a close back above support on elevated
    # volume is a reproducible bullish ChoCH/reclaim proxy.
    frame["choch_bullish_reclaim"] = frame["liquidity_low_sweep"]
    frame["choch_bearish_rejection"] = frame["liquidity_high_sweep"]
    frame["equal_high_cluster"] = (
        frame["high"].sub(prior_high).abs().le(frame["close"].abs() * 0.0005) & prior_high.notna()
    ).astype("int8")
    frame["equal_low_cluster"] = (
        frame["low"].sub(prior_low).abs().le(frame["close"].abs() * 0.0005) & prior_low.notna()
    ).astype("int8")


def _compute_yield_divergence(frame: pd.DataFrame, window: int) -> None:
    if "tips_yield" not in frame:
        return
    gold_return = frame["close"].pct_change()
    yield_change = frame["tips_yield"].diff()
    frame["gold_return"] = gold_return
    frame["real_yield_change"] = yield_change
    # A negative yield change is supportive for gold; positive gold-return
    # response is expected. Positive divergence means yields fell but gold
    # underperformed its rolling yield-implied response.
    rolling_beta = gold_return.rolling(window, min_periods=max(10, window // 3)).cov(yield_change) / yield_change.rolling(
        window, min_periods=max(10, window // 3)
    ).var()
    frame["gold_yield_rolling_beta"] = rolling_beta
    expected_gold_return = rolling_beta * yield_change
    frame["real_yield_divergence"] = expected_gold_return - gold_return
    frame["real_yield_divergence_z"] = _rolling_zscore(frame["real_yield_divergence"], window)
    # Yield drops with a muted/negative gold response are candidate catch-up
    # gaps. A z-score makes the trigger comparable across price regimes.
    frame["real_yield_gold_lag_signal"] = (
        yield_change.lt(0) & frame["real_yield_divergence_z"].ge(1.0)
    ).astype("int8")
    if "dxy_close" in frame:
        frame["dxy_return"] = frame["dxy_close"].pct_change()
        frame["gold_dxy_relative_return"] = gold_return - frame["dxy_return"]


def build_features(
    gold: pd.DataFrame,
    *,
    tips: str | None = None,
    dxy: str | None = None,
    futures: str | None = None,
    cot: str | None = None,
    options: str | None = None,
    macro_releases: str | None = None,
    sweep_lookback: int = 20,
    volume_window: int = 20,
    return_window: int = 48,
    macro_window_minutes: int = 30,
    round_number_step: float = 100.0,
) -> pd.DataFrame:
    """Return a chronologically sorted feature frame from gold OHLCV data."""
    required = {"timestamp", *OHLCV}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Gold data is missing columns: {', '.join(sorted(missing))}")
    frame = gold.copy()
    frame["timestamp"] = _utc_timestamp(frame["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be positive")

    # Daily close and end-of-day yield observations are only made available
    # after their observation date; lag them one day for intraday joins.
    frame = _asof_join(frame, tips, ("tips_yield",), pd.Timedelta(days=1))
    frame = _asof_join(frame, dxy, ("dxy_close",), pd.Timedelta(days=1))
    if futures:
        frame = _asof_join(frame, futures, ("futures_close",))
    if cot:
        frame = _asof_join(frame, cot, ("cot_net_long",))
    if options:
        frame = _asof_join(frame, options, ("options_strike", "options_oi"))

    _compute_overlap_vwap(frame)
    _compute_liquidity_sweeps(frame, sweep_lookback, volume_window)
    _compute_yield_divergence(frame, return_window)

    # ATR and price-action features.
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    frame["candlestick_body_atr"] = (frame["close"] - frame["open"]) / frame["atr_14"].replace(0, np.nan)
    frame["bullish_momentum"] = (frame["close"].gt(frame["open"]) & frame["candlestick_body_atr"].ge(0.5)).astype("int8")
    frame["bearish_momentum"] = (frame["close"].lt(frame["open"]) & frame["candlestick_body_atr"].le(-0.5)).astype("int8")
    # Three-candle fair value gaps; emitted on the confirming candle.
    frame["bullish_fvg"] = (frame["low"].gt(frame["high"].shift(2))).astype("int8")
    frame["bullish_fvg_size"] = (frame["low"] - frame["high"].shift(2)).clip(lower=0)
    frame["bearish_fvg"] = (frame["high"].lt(frame["low"].shift(2))).astype("int8")
    frame["bearish_fvg_size"] = (frame["low"].shift(2) - frame["high"]).clip(lower=0)

    # EFP/carry proxy: futures minus spot less theoretical financing carry.
    if "futures_close" in frame:
        if "risk_free_rate" not in frame:
            frame["risk_free_rate"] = 0.0
        rate = pd.to_numeric(frame["risk_free_rate"], errors="coerce")
        frame["efp_basis"] = frame["futures_close"] - frame["close"]
        frame["carry_adjusted_basis"] = frame["efp_basis"] - frame["close"] * rate / 100.0 * (1.0 / 252.0)
        frame["basis_zscore"] = _rolling_zscore(frame["carry_adjusted_basis"], return_window)

    # Options OI strike proximity and round-number acceleration proxy. The
    # options file is timestamped and contains strike-level OI observations.
    if "options_strike" in frame and "options_oi" in frame:
        frame["options_strike_distance"] = frame["options_strike"] - frame["close"]
        frame["options_oi_weighted_proximity"] = frame["options_oi"] / (1.0 + frame["options_strike_distance"].abs())
    nearest_round = (frame["close"] / round_number_step).round() * round_number_step
    frame["nearest_round_number"] = nearest_round
    frame["round_number_distance"] = nearest_round - frame["close"]
    frame["price_acceleration"] = frame["close"].pct_change().diff()
    frame["round_number_squeeze_proxy"] = (
        frame["round_number_distance"].abs().le(round_number_step * 0.02)
        & frame["price_acceleration"].abs().gt(frame["price_acceleration"].rolling(return_window, min_periods=10).std())
    ).astype("int8")

    if "cot_net_long" in frame:
        frame["cot_net_long_change"] = frame["cot_net_long"].diff()
        frame["cot_net_long_zscore"] = _rolling_zscore(frame["cot_net_long"], return_window)

    if macro_releases:
        releases = _load_csv(macro_releases)
        release_times = releases["timestamp"].dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        bar_times = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        half_window = np.timedelta64(macro_window_minutes, "m")
        indices = np.searchsorted(release_times, bar_times)
        near_release = np.zeros(len(frame), dtype=bool)
        for offset in (0, -1):
            candidates = indices + offset
            valid = (candidates >= 0) & (candidates < len(release_times))
            near_release[valid] |= np.abs(bar_times[valid] - release_times[candidates[valid]]) <= half_window
        frame["macro_release_window"] = near_release.astype("int8")
        frame["institutional_buying_proxy"] = (
            near_release & frame["close"].gt(frame["open"]) & frame["relative_volume"].ge(1.5)
        ).astype("int8")
    else:
        frame["macro_release_window"] = 0
        frame["institutional_buying_proxy"] = 0

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Intraday XAU/USD CSV with timestamp/open/high/low/close/volume")
    parser.add_argument("--tips", help="CSV with timestamp,tips_yield (DFII10 in percent)")
    parser.add_argument("--dxy", help="CSV with timestamp,dxy_close")
    parser.add_argument("--futures", help="CSV with timestamp,futures_close and optional risk_free_rate (percent annualized)")
    parser.add_argument("--cot", help="CSV with timestamp,cot_net_long")
    parser.add_argument("--options", help="CSV with timestamp,options_strike,options_oi")
    parser.add_argument("--macro-releases", help="CSV with timestamp column, one row per macro release")
    parser.add_argument("--output", required=True, help="Destination feature CSV")
    parser.add_argument("--sweep-lookback", type=int, default=20)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--return-window", type=int, default=48)
    parser.add_argument("--macro-window-minutes", type=int, default=30)
    parser.add_argument("--round-number-step", type=float, default=100.0)
    args = parser.parse_args()
    if min(args.sweep_lookback, args.volume_window, args.return_window) < 2:
        parser.error("lookback windows must be at least 2")
    if args.round_number_step <= 0 or args.macro_window_minutes < 0:
        parser.error("round-number step must be positive and macro window nonnegative")
    gold = _load_csv(args.gold, OHLCV)
    result = build_features(
        gold,
        tips=args.tips,
        dxy=args.dxy,
        futures=args.futures,
        cot=args.cot,
        options=args.options,
        macro_releases=args.macro_releases,
        sweep_lookback=args.sweep_lookback,
        volume_window=args.volume_window,
        return_window=args.return_window,
        macro_window_minutes=args.macro_window_minutes,
        round_number_step=args.round_number_step,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} feature rows and {len(result.columns):,} columns to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
