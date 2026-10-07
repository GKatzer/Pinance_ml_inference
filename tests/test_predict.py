import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from pinance_ml_inference import predict as predict_module
from pinance_ml_inference.config import HORIZONS
from pinance_ml_inference.model_registry import ModelNotLoadedError
from pinance_ml_inference.models import ModelSet
from pinance_ml_inference.predict import AsOfMismatchError, InsufficientHistoryError, predict

QUANTILES = (0.1, 0.5, 0.9)

FEAT_COLS = ["ret_lag_1"]  # a genuine column vendor/features.py always produces


def _flat_candles(n):
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}
    )


class _ConstantBooster:
    """Stub with the same `.predict(x)` interface as lgb.Booster, always
    returning a fixed value regardless of input -- lets tests control exactly
    what each (horizon, quantile) model "predicts" without fighting a real
    LightGBM model into a specific output.
    """

    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.array([self.value])


def _fake_model_set(metadata):
    """A ModelSet good enough to exercise predict()'s wiring end to end,
    without needing the real ~93-column feature schema -- `last[feature_columns]`
    resolves against the real computed features via FEAT_COLS. Same booster
    for every (horizon, quantile): quantiles come out equal, so crossing
    protection is a no-op here -- see test_predict_sorts_crossed_quantiles
    below for the case that actually exercises it.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 1))
    y = rng.normal(size=20)
    train_set = lgb.Dataset(X, label=y, feature_name=FEAT_COLS)
    booster = lgb.train({"objective": "regression", "verbosity": -1}, train_set, num_boost_round=2)
    return ModelSet(
        symbol="BTCUSDT",
        feature_columns=FEAT_COLS,
        boosters={(h, q): booster for h in HORIZONS for q in QUANTILES},
        metadata=metadata,
        quantiles=QUANTILES,
    )


def _stub_model_registry(monkeypatch, model_set=None, captured=None):
    """predict() checks the registry before touching `candles`/`btc_candles`
    (see predict.py), so every test needs a stand-in model_set -- or None to
    simulate ModelNotLoadedError -- even ones only exercising the
    candle-history checks below.
    """

    def fake_get_model_set(symbol, slot):
        if captured is not None:
            captured["slot"] = slot
        if model_set is None:
            raise ModelNotLoadedError(f"no model for {symbol}/{slot}")
        return model_set

    monkeypatch.setattr(predict_module.model_registry, "get_model_set", fake_get_model_set)


def test_predict_raises_when_symbol_history_too_short(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(10)

    with pytest.raises(InsufficientHistoryError):
        predict("BTCUSDT", candles["ts"].iloc[-1], candles)


def test_predict_raises_when_btc_history_too_short_for_altcoin(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)
    btc_candles = _flat_candles(10)

    with pytest.raises(InsufficientHistoryError):
        predict("ETHUSDT", candles["ts"].iloc[-1], candles, btc_candles)


def test_predict_raises_when_btc_candles_is_none_for_altcoin(monkeypatch):
    """None (not just "too short") is how a caller represents "didn't send
    a BTCUSDT window at all" -- must fail the same way as too-short, not
    crash on `len(None)`.
    """
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    with pytest.raises(InsufficientHistoryError):
        predict("ETHUSDT", candles["ts"].iloc[-1], candles, None)


def test_predict_raises_when_as_of_ts_mismatches_last_candle(monkeypatch):
    """The core purity invariant: predict() must refuse to serve a
    prediction for a candle other than the one the caller claims it's
    asking about, since that's what backend backfill relies on.
    """
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)
    wrong_as_of_ts = candles["ts"].iloc[-2]  # one candle before the real last one

    with pytest.raises(AsOfMismatchError):
        predict("BTCUSDT", wrong_as_of_ts, candles)


def test_predict_sorts_unsorted_candles(monkeypatch):
    """Contract says candles arrive ascending by ts, but predict() sorts
    defensively rather than trusting the caller -- so as_of_ts still
    resolves to the chronologically last candle even if the request body
    didn't order it that way.
    """
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200).iloc[::-1].reset_index(drop=True)
    as_of_ts = candles["ts"].max()

    result = predict("BTCUSDT", as_of_ts, candles)

    assert result["as_of_ts"] == as_of_ts.isoformat()


def test_predict_includes_model_version_from_metadata(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({"model_version": "202607280001-e58289db"}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["model_version"] == "202607280001-e58289db"


def test_predict_model_version_is_none_when_metadata_lacks_it(monkeypatch):
    """Old metadata.json (pre model_version field) shouldn't break serving."""
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["model_version"] is None


def test_predict_includes_quantile_model_version_from_metadata(monkeypatch):
    _stub_model_registry(
        monkeypatch,
        model_set=_fake_model_set({"model_version": "202607280001-e58289db", "quantile_model_version": "q-202607290002"}),
    )
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["model_version"] == "202607280001-e58289db"
    assert result["quantile_model_version"] == "q-202607290002"


def test_predict_quantile_model_version_is_none_when_metadata_lacks_it(monkeypatch):
    """Point-only slot (no corridor ever pushed) -- must not break serving."""
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({"model_version": "v1"}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["quantile_model_version"] is None


def test_predict_includes_feature_count(monkeypatch):
    """model_set.feature_columns is the full model input schema (~93/94
    columns for the real vendored pipeline) -- FEAT_COLS here is a stand-in,
    but the field is just len() of whatever the loaded model actually uses."""
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["feature_count"] == len(FEAT_COLS)


def test_predict_includes_feature_baseline_from_metadata(monkeypatch):
    baseline = {"rsi": {"mean": 50.0, "std": 10.0, "bin_edges": [0, 100], "bin_fractions": [1.0]}, "btc_ret": None}
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({"feature_baseline": baseline}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["feature_baseline"] == baseline


def test_predict_feature_baseline_is_none_when_metadata_lacks_it(monkeypatch):
    """Metadata from a model pushed before export_models.py started computing
    feature_baseline -- must not break serving."""
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert result["feature_baseline"] is None


def test_predict_includes_feature_snapshot(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    snapshot = result["feature_snapshot"]
    assert set(snapshot) == {"btc_ret", "atr", "bb_width", "rsi", "ret_std_144", "macd_diff"}
    # BTCUSDT is the base asset itself -- compute_features never adds a
    # btc_ret column when serving it (predict() passes btc_candles=None),
    # unlike every other symbol.
    assert snapshot["btc_ret"] is None
    for col in ("atr", "bb_width", "rsi", "ret_std_144", "macd_diff"):
        assert isinstance(snapshot[col], float)
        assert snapshot[col] == snapshot[col]  # not NaN -- NaN isn't valid JSON


def test_predict_feature_snapshot_includes_btc_ret_for_altcoin(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)
    btc_candles = _flat_candles(200)

    result = predict("ETHUSDT", candles["ts"].iloc[-1], candles, btc_candles)

    btc_ret = result["feature_snapshot"]["btc_ret"]
    assert isinstance(btc_ret, float)
    assert btc_ret == btc_ret  # not NaN


def test_predict_includes_inference_ms(monkeypatch):
    """Measures only the horizon loop (the model.predict() calls), not
    feature computation -- the one place VDS2 can report model time honestly
    rather than a VDS1-side round-trip."""
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert isinstance(result["inference_ms"], float)
    assert result["inference_ms"] >= 0.0


def test_predict_defaults_slot_to_production(monkeypatch):
    captured = {}
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}), captured=captured)
    candles = _flat_candles(200)

    predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    assert captured["slot"] == "production"


def test_predict_passes_explicit_slot_through(monkeypatch):
    captured = {}
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}), captured=captured)
    candles = _flat_candles(200)

    predict("BTCUSDT", candles["ts"].iloc[-1], candles, slot="candidate")

    assert captured["slot"] == "candidate"


def test_predict_raises_model_not_loaded_when_candidate_slot_empty(monkeypatch):
    """Mirrors the /predict/{symbol}/shadow 404 case: no candidate loaded.
    The registry check must fail before predict() ever looks at `candles`.
    """
    _stub_model_registry(monkeypatch, model_set=None)
    candles = _flat_candles(200)

    with pytest.raises(ModelNotLoadedError):
        predict("BTCUSDT", candles["ts"].iloc[-1], candles, slot="candidate")


def test_predict_includes_quantile_fields(monkeypatch):
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({}))
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    first = result["predictions"][0]
    assert set(first) == {"horizon", "target_ts", "r_pred", "price_pred", "r_q10", "r_q90", "price_q10", "price_q90"}


def test_predict_sorts_crossed_quantiles(monkeypatch):
    """The mandatory case: each (horizon, quantile) model is trained
    independently, so nothing stops q10's raw output from landing above
    q90's for a given row. Build a ModelSet where that's deliberately true
    for horizon 1 and assert the response still comes back monotonic.
    """
    boosters = {
        (1, 0.1): _ConstantBooster(0.05),  # q10 "predicts" the highest r
        (1, 0.5): _ConstantBooster(0.0),
        (1, 0.9): _ConstantBooster(-0.05),  # q90 "predicts" the lowest r
    }
    for h in HORIZONS:
        if h == 1:
            continue
        for q in QUANTILES:
            boosters[(h, q)] = _ConstantBooster(0.0)
    model_set = ModelSet(symbol="BTCUSDT", feature_columns=FEAT_COLS, boosters=boosters, metadata={}, quantiles=QUANTILES)
    _stub_model_registry(monkeypatch, model_set=model_set)
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    h1 = next(p for p in result["predictions"] if p["horizon"] == 1)
    assert h1["r_q10"] <= h1["r_pred"] <= h1["r_q90"]
    assert h1["price_q10"] <= h1["price_pred"] <= h1["price_q90"]
    # values got reordered, not dropped: the raw {0.05, 0.0, -0.05} set is
    # still exactly what comes out, just relabeled into ascending slots.
    assert {h1["r_q10"], h1["r_pred"], h1["r_q90"]} == {0.05, 0.0, -0.05}


def test_predict_omits_quantile_fields_when_model_set_is_point_only(monkeypatch):
    """quantile_levels empty/absent (e.g. a retrain run without
    --include-quantiles) is a valid state -- must still serve a point
    prediction, without r_q10/r_q90/price_q10/price_q90."""
    booster = _ConstantBooster(0.02)
    model_set = ModelSet(
        symbol="BTCUSDT",
        feature_columns=FEAT_COLS,
        boosters={(h, 0.5): booster for h in HORIZONS},
        metadata={},
        quantiles=(0.5,),
    )
    _stub_model_registry(monkeypatch, model_set=model_set)
    candles = _flat_candles(200)

    result = predict("BTCUSDT", candles["ts"].iloc[-1], candles)

    first = result["predictions"][0]
    assert set(first) == {"horizon", "target_ts", "r_pred", "price_pred"}
    assert first["r_pred"] == 0.02
