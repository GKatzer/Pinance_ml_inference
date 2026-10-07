import time

import numpy as np
import pandas as pd

from pinance_ml_inference import model_registry
from pinance_ml_inference.config import CANDLE_INTERVAL_MINUTES, HORIZONS, MIN_WARMUP_CANDLES
from pinance_ml_inference.vendor.features import compute_features


class InsufficientHistoryError(ValueError):
    """Raised when a symbol (or BTC, for the cross-asset feature) doesn't have
    MIN_WARMUP_CANDLES of history in the request -- see brief §3: serving
    anyway would hand the model a row where most of its 93 features are NaN.
    """


class AsOfMismatchError(ValueError):
    """Raised when the request's as_of_ts doesn't match the last candle's ts.

    predict() must be a pure function of (as_of_ts, candles), nothing else --
    that's what lets the backend backfill a candle it missed the first time
    around by simply calling this endpoint again later with the same body,
    instead of only ever being able to ask "what's the prediction right now".
    Silently predicting on whatever candle happens to be last, while claiming
    a different as_of_ts in the response, would defeat that guarantee in a
    way that's much harder to notice than an outright missing-data error.
    """


# Curated drift-monitoring subset of the ~93 model input columns (see
# vendor/features.py), one per signal axis rather than e.g. all 24
# autocorrelated ret_lag_* columns: btc_ret (the only cross-asset input),
# atr/bb_width (two independent volatility measures -- amplitude vs.
# relative-to-MA), rsi (momentum), ret_std_144 (slowest realized-vol window,
# a regime indicator), macd_diff (trend).
#
# Must stay byte-for-byte identical to Pinance_ml_training's own
# DRIFT_FEATURE_COLUMNS (src/pinance_ml/config.py) -- that's the set
# export_models.py computes feature_baseline over (metadata.json, surfaced
# below via model_set.metadata.get("feature_baseline")). This service only
# reports the baseline and the live feature_snapshot side by side; it does
# not compare them, and no consumer currently does. A consumer that wanted
# to would pair baseline and snapshot entries by column name, so a change
# here without a matching change in training would leave entries unpaired
# silently rather than raise. The two repos deploy independently with no
# shared import.
DRIFT_FEATURE_COLUMNS = ("btc_ret", "atr", "bb_width", "rsi", "ret_std_144", "macd_diff")


def _feature_snapshot(last: pd.DataFrame) -> dict:
    """Raw values of DRIFT_FEATURE_COLUMNS for the row being predicted on.

    `btc_ret` doesn't exist in `last` at all when serving BTCUSDT itself --
    compute_features only adds that column when btc_candles is passed --
    reported as None rather than omitted, so the response shape stays
    uniform across symbols. Any NaN (a btc_ret timestamp that didn't align,
    since MIN_WARMUP_CANDLES already rules out the ordinary warmup case) is
    also None -- float('nan') isn't valid JSON.
    """
    snapshot = {}
    for col in DRIFT_FEATURE_COLUMNS:
        if col not in last.columns:
            snapshot[col] = None
            continue
        value = last[col].iloc[0]
        snapshot[col] = float(value) if pd.notna(value) else None
    return snapshot


def predict(
    symbol: str,
    as_of_ts: pd.Timestamp,
    candles: pd.DataFrame,
    btc_candles: pd.DataFrame | None = None,
    *,
    slot: str = "production",
) -> dict:
    """Predict r_1..r_12 and their price-space equivalents for the candle at
    `as_of_ts`. Brief §7's serving recipe: take the caller-supplied history
    (+BTC, if `symbol` isn't BTC), compute features, run 12 horizon models,
    convert log-returns back to price via close[t] * exp(r_h).

    Pure function of its arguments -- no DB, no clock, no live feed. That's
    what makes backend backfill possible: a candle predict() was never asked
    about (service down, network hiccup, whatever) can be requested again with
    the same `as_of_ts` and `candles` at any later time and get exactly the
    prediction it would have gotten at the time, instead of the caller only
    ever being able to ask for "the current one".

    `candles` must have >= MIN_WARMUP_CANDLES rows, ascending by ts, with the
    last row's ts equal to `as_of_ts`. `btc_candles` is the same shape for
    BTCUSDT, required (same minimum) whenever `symbol != "BTCUSDT"`, unused
    otherwise.

    `slot` selects which model set to serve: "production" (default) or
    "candidate" for shadow serving. Raises model_registry.ModelNotLoadedError
    if nothing's currently loaded for that slot -- schema validation already
    happened when the model was polled in, not here (see model_registry.py).
    """
    # Cheap in-memory check first: no point validating/computing anything for
    # a slot with nothing loaded (the normal state for "candidate" most of
    # the time).
    model_set = model_registry.get_model_set(symbol, slot)

    candles = candles.sort_values("ts").reset_index(drop=True)
    if len(candles) < MIN_WARMUP_CANDLES:
        raise InsufficientHistoryError(f"{symbol}: only {len(candles)} candles in request (need >= {MIN_WARMUP_CANDLES})")

    if symbol != "BTCUSDT":
        btc_len = 0 if btc_candles is None else len(btc_candles)
        if btc_len < MIN_WARMUP_CANDLES:
            raise InsufficientHistoryError(
                f"BTCUSDT: only {btc_len} candles in request (need >= {MIN_WARMUP_CANDLES}), "
                f"required for {symbol}'s btc_ret feature"
            )
        btc_candles = btc_candles.sort_values("ts").reset_index(drop=True)

    last_ts = candles["ts"].iloc[-1]
    if last_ts != as_of_ts:
        raise AsOfMismatchError(
            f"{symbol}: as_of_ts={as_of_ts.isoformat()} does not match the last candle's "
            f"ts={last_ts.isoformat()} -- candles[symbol][-1] must be the as_of_ts candle"
        )

    features = compute_features(candles, btc_candles=btc_candles)

    last = features.iloc[[-1]]
    x = last[model_set.feature_columns]
    last_close = float(candles["close"].iloc[-1])
    feature_snapshot = _feature_snapshot(last)

    predictions = []
    inference_start = time.perf_counter()
    for h in HORIZONS:
        r_by_quantile = {q: float(model_set.boosters[(h, q)].predict(x)[0]) for q in model_set.quantiles}
        entry = {
            "horizon": h,
            "target_ts": (last_ts + pd.Timedelta(minutes=CANDLE_INTERVAL_MINUTES * h)).isoformat(),
        }
        if len(r_by_quantile) == 1:
            # No corridor for this model set (quantile_levels was empty/absent
            # -- see models.quantiles_from_metadata) -- point prediction only.
            r_pred = r_by_quantile[0.5]
            entry["r_pred"] = r_pred
            entry["price_pred"] = last_close * float(np.exp(r_pred))
        else:
            # Quantile crossing protection: each (horizon, quantile) model is
            # trained independently, so nothing guarantees q10 < q90 for a
            # given row even though it's calibrated in aggregate. Sorting in
            # log-return space is enough -- price = close * exp(r) is
            # strictly monotonic in r, so an ascending sort here is an
            # ascending sort of the derived prices too. "median" is
            # therefore the middle value after sorting, not necessarily
            # whatever the q=0.5 booster output.
            r_q10, r_pred, r_q90 = sorted(r_by_quantile.values())
            entry["r_pred"] = r_pred
            entry["price_pred"] = last_close * float(np.exp(r_pred))
            entry["r_q10"] = r_q10
            entry["r_q90"] = r_q90
            entry["price_q10"] = last_close * float(np.exp(r_q10))
            entry["price_q90"] = last_close * float(np.exp(r_q90))
        predictions.append(entry)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    return {
        "symbol": symbol,
        "as_of_ts": last_ts.isoformat(),
        "close": last_close,
        "model_version": model_set.metadata.get("model_version"),
        "quantile_model_version": model_set.metadata.get("quantile_model_version"),
        "schema_version": model_set.metadata.get("schema_version"),
        "inference_ms": inference_ms,
        "feature_count": len(model_set.feature_columns),
        "feature_baseline": model_set.metadata.get("feature_baseline"),
        "feature_snapshot": feature_snapshot,
        "predictions": predictions,
    }
