"""HTTP-level tests for the /predict routes and the lifespan hook.

test_predict.py calls predict() directly; these go through FastAPI so the
status-code mapping in api.py (404 for no model, 422 for a bad window or
body) is exercised too. The registry is stubbed exactly as in
test_predict.py -- no MinIO, no database.
"""

import pytest
from fastapi.testclient import TestClient

from pinance_ml_inference import api
from tests.test_predict import _fake_model_set, _flat_candles, _stub_model_registry


def _items(df):
    return [
        {"ts": r.ts.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
        for r in df.itertuples()
    ]


def _body(n=200, btc_n=None, as_of_offset=0):
    candles = _flat_candles(n)
    body = {
        "as_of_ts": (candles["ts"].iloc[-1] + as_of_offset * candles["ts"].diff().iloc[-1]).isoformat(),
        "candles": {"BTCUSDT": _items(candles), "ETHUSDT": _items(candles)},
    }
    if btc_n is not None:
        body["candles"]["BTCUSDT"] = _items(_flat_candles(btc_n))
    return body


@pytest.fixture
def client():
    # No `with`: lifespan (initial poll, poll thread) is not started here.
    return TestClient(api.app)


@pytest.fixture
def loaded(monkeypatch):
    captured = {}
    _stub_model_registry(monkeypatch, model_set=_fake_model_set({"model_version": "v1"}), captured=captured)
    return captured


def test_predict_returns_200_with_twelve_horizons(client, loaded):
    r = client.post("/predict/BTCUSDT", json=_body())

    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["model_version"] == "v1"
    assert [p["horizon"] for p in body["predictions"]] == list(range(1, 13))
    assert loaded["slot"] == "production"


def test_symbol_in_path_is_uppercased(client, loaded):
    r = client.post("/predict/btcusdt", json=_body())

    assert r.status_code == 200
    assert r.json()["symbol"] == "BTCUSDT"


def test_shadow_route_serves_the_candidate_slot(client, loaded):
    r = client.post("/predict/BTCUSDT/shadow", json=_body())

    assert r.status_code == 200
    assert loaded["slot"] == "candidate"


def test_altcoin_with_btc_window_is_served(client, loaded):
    r = client.post("/predict/ETHUSDT", json=_body())

    assert r.status_code == 200
    assert r.json()["symbol"] == "ETHUSDT"


def test_timestamps_without_offset_are_treated_as_utc(client, loaded):
    body = _body()
    body["as_of_ts"] = body["as_of_ts"].replace("+00:00", "")
    for items in body["candles"].values():
        for c in items:
            c["ts"] = c["ts"].replace("+00:00", "")

    assert client.post("/predict/BTCUSDT", json=body).status_code == 200


@pytest.mark.parametrize("route", ["/predict/BTCUSDT", "/predict/BTCUSDT/shadow"])
def test_no_model_loaded_is_404(client, monkeypatch, route):
    _stub_model_registry(monkeypatch, model_set=None)

    r = client.post(route, json=_body())

    assert r.status_code == 404
    assert "no model" in r.json()["detail"]


def test_missing_model_wins_over_a_bad_window(client, monkeypatch):
    _stub_model_registry(monkeypatch, model_set=None)

    assert client.post("/predict/BTCUSDT", json=_body(n=10)).status_code == 404


def test_short_window_is_422(client, loaded):
    r = client.post("/predict/BTCUSDT", json=_body(n=149))

    assert r.status_code == 422
    assert "149" in r.json()["detail"]


def test_altcoin_with_short_btc_window_is_422(client, loaded):
    r = client.post("/predict/ETHUSDT", json=_body(btc_n=10))

    assert r.status_code == 422
    assert "BTCUSDT" in r.json()["detail"]


def test_altcoin_without_btc_window_is_422(client, loaded):
    body = _body()
    del body["candles"]["BTCUSDT"]

    assert client.post("/predict/ETHUSDT", json=body).status_code == 422


def test_as_of_ts_not_matching_last_candle_is_422(client, loaded):
    r = client.post("/predict/BTCUSDT", json=_body(as_of_offset=1))

    assert r.status_code == 422
    assert "as_of_ts" in r.json()["detail"]


def test_symbol_without_a_window_in_the_body_is_422(client, loaded):
    body = _body()
    del body["candles"]["BTCUSDT"]

    assert client.post("/predict/BTCUSDT", json=body).status_code == 422


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("as_of_ts"),
        lambda b: b.pop("candles"),
        lambda b: b["candles"]["BTCUSDT"][0].pop("close"),
        lambda b: b["candles"]["BTCUSDT"][0].update(close="not a number"),
    ],
)
def test_malformed_body_is_422(client, loaded, mutate):
    body = _body()
    mutate(body)

    assert client.post("/predict/BTCUSDT", json=body).status_code == 422


def test_app_starts_when_the_initial_poll_fails(monkeypatch, caplog):
    def boom():
        raise RuntimeError("database down")

    monkeypatch.setattr(api, "list_symbols", boom)
    monkeypatch.setattr(api, "poll_loop", lambda stop_event, interval: stop_event.wait())

    with caplog.at_level("ERROR"), TestClient(api.app) as c:  # `with` runs lifespan
        assert c.get("/health").json() == {"status": "ok"}

    assert "Initial model poll failed" in caplog.text
