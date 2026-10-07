import numpy as np
import pandas as pd

from pinance_ml_inference.vendor.features import compute_features, feature_columns

# Brief §9's exact 93-column contract for BTC (94th, btc_ret, only applies to
# non-BTC symbols). If this list and what compute_features actually produces
# ever diverge, serving is silently feeding models a differently-shaped
# feature vector than they were trained on.
BRIEF_BTC_FEATURE_COLUMNS = [
    "ret_lag_1", "vol_lag_1", "ret_lag_2", "vol_lag_2", "ret_lag_3", "vol_lag_3", "ret_lag_4", "vol_lag_4",
    "ret_lag_5", "vol_lag_5", "ret_lag_6", "vol_lag_6", "ret_lag_7", "vol_lag_7", "ret_lag_8", "vol_lag_8",
    "ret_lag_9", "vol_lag_9", "ret_lag_10", "vol_lag_10", "ret_lag_11", "vol_lag_11", "ret_lag_12", "vol_lag_12",
    "ret_lag_13", "vol_lag_13", "ret_lag_14", "vol_lag_14", "ret_lag_15", "vol_lag_15", "ret_lag_16", "vol_lag_16",
    "ret_lag_17", "vol_lag_17", "ret_lag_18", "vol_lag_18", "ret_lag_19", "vol_lag_19", "ret_lag_20", "vol_lag_20",
    "ret_lag_21", "vol_lag_21", "ret_lag_22", "vol_lag_22", "ret_lag_23", "vol_lag_23", "ret_lag_24", "vol_lag_24",
    "ret_mean_6", "ret_std_6", "ret_min_6", "ret_max_6", "vol_mean_6", "vol_std_6", "vol_min_6", "vol_max_6",
    "ret_mean_12", "ret_std_12", "ret_min_12", "ret_max_12", "vol_mean_12", "vol_std_12", "vol_min_12", "vol_max_12",
    "ret_mean_36", "ret_std_36", "ret_min_36", "ret_max_36", "vol_mean_36", "vol_std_36", "vol_min_36", "vol_max_36",
    "ret_mean_144", "ret_std_144", "ret_min_144", "ret_max_144",
    "vol_mean_144", "vol_std_144", "vol_min_144", "vol_max_144",
    "rsi", "macd", "macd_signal", "macd_diff", "bb_width", "atr", "obv_roc_36", "ret_sq", "ret_atr_norm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _synthetic_candles(n=300, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(scale=0.001, size=n)))
    return pd.DataFrame(
        {
            "ts": ts,
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": rng.uniform(1, 100, size=n),
        }
    )


def test_feature_columns_match_brief_contract_for_btc():
    candles = _synthetic_candles()
    features = compute_features(candles, btc_candles=None)
    assert feature_columns(features) == BRIEF_BTC_FEATURE_COLUMNS
    assert len(BRIEF_BTC_FEATURE_COLUMNS) == 93


def test_btc_ret_appended_only_when_btc_candles_given():
    candles = _synthetic_candles(seed=1)
    btc = _synthetic_candles(seed=2)

    without_btc = feature_columns(compute_features(candles, btc_candles=None))
    with_btc = feature_columns(compute_features(candles, btc_candles=btc))

    assert "btc_ret" not in without_btc
    assert with_btc == without_btc + ["btc_ret"]
    assert len(with_btc) == 94


def test_no_lookahead_last_row_stable_when_future_candles_appended():
    """Same property Pinance_ML's own test_no_lookahead checks: a feature at
    row t must not change when rows after t are added -- otherwise serving
    incrementally (as new candles arrive) would produce different features
    than training saw for the same historical row.
    """
    candles = _synthetic_candles(n=300, seed=3)
    truncated = candles.iloc[:250]

    full_features = compute_features(candles)
    truncated_features = compute_features(truncated)

    row_249_full = full_features.iloc[249]
    row_249_truncated = truncated_features.iloc[249]
    pd.testing.assert_series_equal(row_249_full, row_249_truncated, check_names=False)


def test_warmup_rows_are_nan_not_dropped():
    # 20 rows: enough for ta's AverageTrueRange (window=14) to not index out
    # of bounds, still far short of rolling_144's window -- so ret_std_144
    # should be NaN throughout without any rows being dropped.
    candles = _synthetic_candles(n=20)
    features = compute_features(candles)
    assert len(features) == 20
    assert features["ret_std_144"].isna().all()
