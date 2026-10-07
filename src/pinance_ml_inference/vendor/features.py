"""Vendored from Pinance_ML@4de49a7 (src/pinance_ml/features/pipeline.py).

`compute_features` below is a byte-for-byte copy -- see LOCK.md for why it's
vendored rather than an installed dependency, and how to re-sync it if the
source changes. `feature_columns` is inference's own small helper (the
training-side version in `dataset.py` also excludes target columns, which
never exist here).
"""

import numpy as np
import pandas as pd
import ta

# README: "Лаги: returns и объёмы t-1..t-24 (2 часа истории)"
LAG_STEPS = range(1, 25)

# README: "Rolling: mean/std/min/max на окнах 6/12/36/144 свечей (30м/1ч/3ч/12ч)"
ROLLING_WINDOWS = (6, 12, 36, 144)

RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2
ATR_WINDOW = 14
# Windowed OBV rate-of-change (see compute_features): cumulative OBV depends on
# where the series starts, so it can't match between training (full history)
# and serving (last MIN_WARMUP_CANDLES candles).
OBV_ROC_WINDOW = 36

NON_FEATURE_COLUMNS = {"ts", "open", "high", "low", "close", "volume"}


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def compute_features(candles: pd.DataFrame, btc_candles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pure feature-engineering function: raw OHLCV -> feature matrix.

    Every feature at row t is computed only from candles at or before t, so
    this produces identical output whether called on the full history (as in
    training) or incrementally as each new candle arrives (as in serving) --
    the property `tests/test_pipeline.py::test_no_lookahead` (Pinance_ML)
    checks directly.

    `candles` is a single-symbol OHLCV frame (ts, open, high, low, close,
    volume), sorted or not -- it is sorted internally. `btc_candles` is the
    same shape for BTCUSDT, used for the cross-asset feature; pass None when
    `candles` already *is* BTCUSDT, or when it's simply unavailable.

    Rows near the start of the series carry NaNs where a lag/rolling/
    indicator window extends before the first candle (e.g. the first 143
    rows have no `ret_std_144`). This function does not drop them -- that's
    a modeling-time decision, not a feature-computation one.
    """
    df = candles.sort_values("ts").reset_index(drop=True).copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    ret = np.log(close / close.shift(1))

    for k in LAG_STEPS:
        df[f"ret_lag_{k}"] = ret.shift(k)
        df[f"vol_lag_{k}"] = volume.shift(k)

    for w in ROLLING_WINDOWS:
        ret_roll = ret.rolling(w)
        df[f"ret_mean_{w}"] = ret_roll.mean()
        df[f"ret_std_{w}"] = ret_roll.std()
        df[f"ret_min_{w}"] = ret_roll.min()
        df[f"ret_max_{w}"] = ret_roll.max()

        vol_roll = volume.rolling(w)
        df[f"vol_mean_{w}"] = vol_roll.mean()
        df[f"vol_std_{w}"] = vol_roll.std()
        df[f"vol_min_{w}"] = vol_roll.min()
        df[f"vol_max_{w}"] = vol_roll.max()

    df["rsi"] = ta.momentum.RSIIndicator(close, window=RSI_WINDOW).rsi()

    macd = ta.trend.MACD(close, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=BB_WINDOW, window_dev=BB_STD)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=ATR_WINDOW).average_true_range()
    # OBV[t] - OBV[t-w] is just the sum of signed volume over the last w candles,
    # so it needs no cumulative state. Normalised by traded volume in the same
    # window -> bounded in [-1, 1] and comparable across symbols.
    signed_volume = np.sign(close.diff()) * volume
    df[f"obv_roc_{OBV_ROC_WINDOW}"] = (
        signed_volume.rolling(OBV_ROC_WINDOW).sum() / volume.rolling(OBV_ROC_WINDOW).sum()
    )

    df["ret_sq"] = ret**2
    # ta's AverageTrueRange zero-fills its warmup rows (index < window-1) instead
    # of using NaN like its other indicators, so ret/atr would produce +/-inf
    # there instead of a clean "undefined" value.
    df["ret_atr_norm"] = ret / df["atr"].replace(0.0, np.nan)

    hour_of_day = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["hour_sin"], df["hour_cos"] = _cyclical(hour_of_day, 24)
    df["dow_sin"], df["dow_cos"] = _cyclical(df["ts"].dt.dayofweek, 7)

    if btc_candles is not None:
        btc = btc_candles.sort_values("ts").reset_index(drop=True)
        btc_ret = np.log(btc["close"] / btc["close"].shift(1))
        btc_ret_by_ts = pd.Series(btc_ret.to_numpy(), index=btc["ts"])
        df["btc_ret"] = df["ts"].map(btc_ret_by_ts)

    return df


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Every column in a `compute_features` output that's a model input."""
    return [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
