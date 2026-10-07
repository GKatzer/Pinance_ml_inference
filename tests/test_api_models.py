from pinance_ml_inference import api


def test_get_models_returns_status_payload(monkeypatch):
    seen = {}

    def fake_status(symbols):
        seen["symbols"] = symbols
        return {"expected_schema_version": {"BTCUSDT": "abc"}, "models": {}}

    monkeypatch.setattr(api, "list_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(api, "models_status", fake_status)

    assert api.get_models() == {"expected_schema_version": {"BTCUSDT": "abc"}, "models": {}}
    assert seen["symbols"] == ["BTCUSDT"]


def test_get_models_still_answers_when_db_is_down(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    seen = {}
    monkeypatch.setattr(api, "list_symbols", boom)
    monkeypatch.setattr(api, "models_status", lambda symbols: seen.setdefault("symbols", symbols) and {})

    api.get_models()

    assert seen["symbols"] == []  # falls back to whatever the registry already holds


def test_models_route_is_registered():
    assert any(r.path == "/models" and "GET" in r.methods for r in api.app.routes)
