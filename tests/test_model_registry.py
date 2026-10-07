import logging

import lightgbm as lgb
import numpy as np
import pytest

from pinance_ml_inference import minio_client, model_registry
from pinance_ml_inference.model_registry import ModelNotLoadedError, ModelRegistry, fetch_and_register, poll_once
from pinance_ml_inference.models import quantiles_from_metadata, schema_version

FEAT_COLS = ["a", "b"]
HORIZONS_PER_SLOT = 2


def _tiny_booster_text(feature_names):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(feature_names)))
    y = rng.normal(size=20)
    train_set = lgb.Dataset(X, label=y, feature_name=feature_names)
    booster = lgb.train({"objective": "regression", "verbosity": -1}, train_set, num_boost_round=2)
    return booster.model_to_string()


def _metadata(model_version, feature_columns=FEAT_COLS, horizons=(1, 2), quantile_levels=(0.1, 0.9)):
    return {
        "feature_columns": feature_columns,
        "horizons": list(horizons),
        "quantile_levels": list(quantile_levels),
        "quantile_schema_version": schema_version(feature_columns),
        "model_version": model_version,
        "source_commit": "deadbeef",
    }


class FakeMinioSource:
    """In-memory stand-in for minio_client: {(symbol, slot): metadata dict}."""

    def __init__(self):
        self.slots: dict[tuple[str, str], dict] = {}
        self.booster_fetch_count = 0

    def fetch_metadata(self, symbol, slot):
        return self.slots.get((symbol, slot))

    def fetch_booster_text(self, symbol, slot, horizon, quantile):
        self.booster_fetch_count += 1
        return _tiny_booster_text(self.slots[(symbol, slot)]["feature_columns"])


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """The registry is a module-level singleton -- isolate each test."""
    monkeypatch.setattr(model_registry, "registry", ModelRegistry())


@pytest.fixture
def fake_source(monkeypatch):
    source = FakeMinioSource()
    monkeypatch.setattr(minio_client, "fetch_metadata", source.fetch_metadata)
    monkeypatch.setattr(minio_client, "fetch_booster_text", source.fetch_booster_text)
    monkeypatch.setattr(model_registry, "_expected_feature_columns", lambda symbol: tuple(FEAT_COLS))
    return source


def test_registry_get_raises_when_nothing_loaded():
    registry = ModelRegistry()
    with pytest.raises(ModelNotLoadedError):
        registry.get("BTCUSDT", "production")


def test_registry_get_or_none_returns_none_when_nothing_loaded():
    registry = ModelRegistry()
    assert registry.get_or_none("BTCUSDT", "production") is None


def test_fetch_and_register_loads_new_version(fake_source):
    metadata = _metadata("v1")
    fake_source.slots[("BTCUSDT", "production")] = metadata

    fetch_and_register("BTCUSDT", "production")

    model_set = model_registry.get_model_set("BTCUSDT", "production")
    assert model_set.metadata["model_version"] == "v1"
    assert set(model_set.boosters.keys()) == {(h, q) for h in (1, 2) for q in quantiles_from_metadata(metadata)}


def test_fetch_and_register_skips_redownload_when_version_unchanged(fake_source):
    metadata = _metadata("v1")
    fake_source.slots[("BTCUSDT", "production")] = metadata
    fetch_and_register("BTCUSDT", "production")
    expected_count = HORIZONS_PER_SLOT * len(quantiles_from_metadata(metadata))
    assert fake_source.booster_fetch_count == expected_count

    fetch_and_register("BTCUSDT", "production")  # same version again

    assert fake_source.booster_fetch_count == expected_count  # no redundant downloads


def test_fetch_and_register_picks_up_version_change(fake_source):
    metadata = _metadata("v1")
    fake_source.slots[("BTCUSDT", "production")] = metadata
    fetch_and_register("BTCUSDT", "production")

    fake_source.slots[("BTCUSDT", "production")] = _metadata("v2")
    fetch_and_register("BTCUSDT", "production")

    assert model_registry.get_model_set("BTCUSDT", "production").metadata["model_version"] == "v2"
    expected_count = HORIZONS_PER_SLOT * len(quantiles_from_metadata(metadata))
    assert fake_source.booster_fetch_count == 2 * expected_count


def test_fetch_and_register_leaves_registry_empty_when_slot_missing(fake_source):
    """Normal state for "candidate" most of the time -- nothing pushed yet."""
    fetch_and_register("BTCUSDT", "candidate")

    with pytest.raises(ModelNotLoadedError):
        model_registry.get_model_set("BTCUSDT", "candidate")


def test_fetch_and_register_skips_swap_on_feature_schema_mismatch(monkeypatch, caplog):
    """Deliberately don't patch _expected_feature_columns: the real vendored
    column list will never match ["a", "b"], simulating genuine schema drift.
    """
    source = FakeMinioSource()
    source.slots[("BTCUSDT", "candidate")] = _metadata("v1")
    monkeypatch.setattr(minio_client, "fetch_metadata", source.fetch_metadata)
    monkeypatch.setattr(minio_client, "fetch_booster_text", source.fetch_booster_text)

    with caplog.at_level(logging.ERROR):
        fetch_and_register("BTCUSDT", "candidate")

    assert any(r.levelno == logging.ERROR for r in caplog.records)
    with pytest.raises(ModelNotLoadedError):
        model_registry.get_model_set("BTCUSDT", "candidate")


def test_fetch_and_register_loads_point_only_when_no_corridor_pushed(fake_source):
    """auto_retrain.py run without --include-quantiles pushes quantile_levels
    empty -- a valid, expected state (model_storage.py's contract), not a
    schema mismatch. Must still load and serve the point model."""
    fake_source.slots[("BTCUSDT", "production")] = _metadata("v1", quantile_levels=())

    fetch_and_register("BTCUSDT", "production")

    model_set = model_registry.get_model_set("BTCUSDT", "production")
    assert model_set.quantiles == (0.5,)
    assert set(model_set.boosters.keys()) == {(h, 0.5) for h in (1, 2)}


def test_fetch_and_register_reloads_on_quantile_only_version_change(fake_source):
    """The point and corridor pipelines push independently -- a later run
    that only updates quantile_model_version (same model_version, an
    already-loaded slot) must still trigger a reload, not be mistaken for
    "nothing changed"."""
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v1", quantile_levels=(0.1, 0.9)) | {
        "quantile_model_version": "q1"
    }
    fetch_and_register("BTCUSDT", "candidate")
    assert model_registry.get_model_set("BTCUSDT", "candidate").metadata["quantile_model_version"] == "q1"

    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v1", quantile_levels=(0.1, 0.9)) | {
        "quantile_model_version": "q2"
    }
    fetch_and_register("BTCUSDT", "candidate")

    assert model_registry.get_model_set("BTCUSDT", "candidate").metadata["quantile_model_version"] == "q2"


def test_fetch_and_register_keeps_prior_version_on_later_schema_mismatch(fake_source, monkeypatch):
    """A good version is loaded, then a later bad push must not evict it."""
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v1")
    fetch_and_register("BTCUSDT", "candidate")

    monkeypatch.setattr(model_registry, "_expected_feature_columns", lambda symbol: ("different",))
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v2")
    fetch_and_register("BTCUSDT", "candidate")

    assert model_registry.get_model_set("BTCUSDT", "candidate").metadata["model_version"] == "v1"


def test_poll_once_continues_after_one_symbol_fails(monkeypatch):
    monkeypatch.setattr(model_registry, "_expected_feature_columns", lambda symbol: tuple(FEAT_COLS))
    good_metadata = _metadata("v1")

    def fake_fetch_metadata(symbol, slot):
        if symbol == "BROKEN":
            raise RuntimeError("boom")
        return good_metadata if (symbol, slot) == ("BTCUSDT", "production") else None

    monkeypatch.setattr(minio_client, "fetch_metadata", fake_fetch_metadata)
    monkeypatch.setattr(
        minio_client, "fetch_booster_text", lambda symbol, slot, horizon, quantile: _tiny_booster_text(FEAT_COLS)
    )

    poll_once(["BROKEN", "BTCUSDT"])

    assert model_registry.get_model_set("BTCUSDT", "production").metadata["model_version"] == "v1"


def test_fetch_and_register_serves_point_when_corridor_schema_is_foreign(fake_source, caplog):
    """The real post-retrain state: point pushed on the new schema, corridor
    still on the old one. Point must load; corridor must be ignored (and not
    downloaded), with a warning."""
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v1") | {"quantile_schema_version": "old000000000"}

    with caplog.at_level(logging.WARNING):
        fetch_and_register("BTCUSDT", "candidate")

    model_set = model_registry.get_model_set("BTCUSDT", "candidate")
    assert model_set.quantiles == (0.5,)
    assert model_set.quantile_status == "ignored_schema_mismatch"
    assert fake_source.booster_fetch_count == HORIZONS_PER_SLOT
    assert any("corridor ignored" in r.getMessage() for r in caplog.records)


def test_models_status_reports_loaded_rejected_and_expected(fake_source):
    fake_source.slots[("BTCUSDT", "production")] = _metadata("v1") | {"quantile_schema_version": "old000000000"}
    fetch_and_register("BTCUSDT", "production")
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v2", feature_columns=["x"])
    fetch_and_register("BTCUSDT", "candidate")  # schema mismatch -> rejected

    status = model_registry.models_status(["BTCUSDT", "ETHUSDT"])

    assert status["expected_schema_version"] == {s: schema_version(FEAT_COLS) for s in ("BTCUSDT", "ETHUSDT")}
    prod = status["models"]["BTCUSDT"]["production"]
    assert prod["loaded"] and prod["matches_expected"] and prod["schema_version"] == schema_version(FEAT_COLS)
    assert prod["quantile"]["status"] == "ignored_schema_mismatch"
    assert "old000000000" in prod["last_load_error"]  # corridor-ignored reason stays visible
    cand = status["models"]["BTCUSDT"]["candidate"]
    assert cand["loaded"] is False
    assert "does not match" in cand["last_load_error"]
    eth = status["models"]["ETHUSDT"]["production"]
    assert eth == {"loaded": False, "last_load_error": None}


def test_rejection_error_clears_once_a_good_version_loads(fake_source):
    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v1", feature_columns=["x"])
    fetch_and_register("BTCUSDT", "candidate")
    assert model_registry.models_status([])["models"]["BTCUSDT"]["candidate"]["last_load_error"]

    fake_source.slots[("BTCUSDT", "candidate")] = _metadata("v2")
    fetch_and_register("BTCUSDT", "candidate")

    assert model_registry.models_status([])["models"]["BTCUSDT"]["candidate"]["last_load_error"] is None
