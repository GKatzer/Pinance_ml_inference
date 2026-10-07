import lightgbm as lgb
import numpy as np
import pytest

from pinance_ml_inference.models import (
    QUANTILE_ABSENT,
    QUANTILE_IGNORED,
    QUANTILE_LOADED,
    SchemaMismatchError,
    booster_keys_from_metadata,
    build_model_set,
    quantiles_from_metadata,
    schema_version,
)

HORIZONS = (1, 2)


def _tiny_booster_text(feature_names):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, len(feature_names)))
    y = rng.normal(size=50)
    train_set = lgb.Dataset(X, label=y, feature_name=feature_names)
    booster = lgb.train({"objective": "regression", "verbosity": -1}, train_set, num_boost_round=2)
    return booster.model_to_string()


def _metadata(symbol, feature_columns, horizons=HORIZONS, quantile_levels=(0.1, 0.9), quantile_schema_version="same"):
    """`quantile_schema_version="same"` -> a corridor trained on exactly these features."""
    if quantile_schema_version == "same":
        quantile_schema_version = schema_version(feature_columns)
    return {
        "symbol": symbol,
        "feature_columns": feature_columns,
        "horizons": list(horizons),
        "quantile_levels": list(quantile_levels),
        "quantile_schema_version": quantile_schema_version,
        "source_commit": "deadbeef",
    }


def _booster_texts(feature_columns, horizons=HORIZONS, quantiles=(0.1, 0.5, 0.9)):
    return {(h, q): _tiny_booster_text(feature_columns) for h in horizons for q in quantiles}


def test_quantiles_from_metadata_always_includes_median():
    assert quantiles_from_metadata(_metadata("X", ["a"])) == (0.5, 0.1, 0.9)


def test_quantiles_from_metadata_is_point_only_when_levels_empty():
    """Empty/absent quantile_levels means point-only, per model_storage.py's
    contract -- not an error, same as the pre-corridor format."""
    assert quantiles_from_metadata({"quantile_levels": []}) == (0.5,)
    assert quantiles_from_metadata({}) == (0.5,)


def test_booster_keys_from_metadata_uses_quantile_horizons_when_present():
    """Point and corridor have independent horizon lists by construction
    (Pinance_ML's model_storage.py) -- must not assume they're the same
    list just because they coincide today."""
    metadata = _metadata("X", ["a"], horizons=(1, 2, 3)) | {"quantile_horizons": [1, 2]}

    keys = set(booster_keys_from_metadata(metadata))

    assert keys == {(1, 0.5), (2, 0.5), (3, 0.5), (1, 0.1), (1, 0.9), (2, 0.1), (2, 0.9)}


def test_booster_keys_from_metadata_falls_back_to_horizons_when_quantile_horizons_absent():
    """Metadata pushed before quantile_horizons existed as its own key."""
    metadata = _metadata("X", ["a"])

    keys = set(booster_keys_from_metadata(metadata))

    assert keys == {(1, 0.5), (2, 0.5), (1, 0.1), (1, 0.9), (2, 0.1), (2, 0.9)}


def test_booster_keys_from_metadata_point_only():
    metadata = {"horizons": [1, 2], "quantile_levels": []}

    assert set(booster_keys_from_metadata(metadata)) == {(1, 0.5), (2, 0.5)}


def test_build_model_set_succeeds_when_schema_matches():
    feat_cols = ["a", "b", "c"]

    model_set = build_model_set("TESTUSDT", _metadata("TESTUSDT", feat_cols), _booster_texts(feat_cols), tuple(feat_cols))

    assert model_set.feature_columns == feat_cols
    assert model_set.quantiles == (0.5, 0.1, 0.9)
    assert set(model_set.boosters.keys()) == {(h, q) for h in HORIZONS for q in (0.1, 0.5, 0.9)}


def test_build_model_set_rejects_feature_schema_mismatch():
    feat_cols = ["a", "b", "c"]

    with pytest.raises(SchemaMismatchError):
        build_model_set(
            "TESTUSDT2",
            _metadata("TESTUSDT2", feat_cols),
            _booster_texts(feat_cols),
            ("a", "b", "different"),
        )


def test_build_model_set_is_point_only_when_quantile_levels_empty():
    """No corridor pushed (e.g. auto_retrain.py run without
    --include-quantiles) is a valid, expected state -- must still build a
    servable point-only ModelSet, not raise."""
    feat_cols = ["a", "b", "c"]
    metadata = _metadata("TESTUSDT3", feat_cols, quantile_levels=())

    model_set = build_model_set("TESTUSDT3", metadata, _booster_texts(feat_cols, quantiles=(0.5,)), tuple(feat_cols))

    assert model_set.quantiles == (0.5,)
    assert set(model_set.boosters.keys()) == {(h, 0.5) for h in HORIZONS}


def test_build_model_set_ignores_corridor_with_foreign_schema_but_keeps_point():
    """Point and corridor ship on independent schedules, so a corridor
    trained on an older feature list is a normal state, not a load failure:
    serve the point model alone and never fetch/assemble the tails."""
    feat_cols = ["a", "b", "c"]
    metadata = _metadata("TESTUSDT4", feat_cols, quantile_schema_version="old000000000")

    model_set = build_model_set("TESTUSDT4", metadata, _booster_texts(feat_cols, quantiles=(0.5,)), tuple(feat_cols))

    assert model_set.quantile_status == QUANTILE_IGNORED
    assert model_set.quantiles == (0.5,)
    assert set(model_set.boosters.keys()) == {(h, 0.5) for h in HORIZONS}
    assert set(booster_keys_from_metadata(metadata)) == {(h, 0.5) for h in HORIZONS}


def test_corridor_without_quantile_schema_version_is_ignored():
    """Can't be verified against the point features -> not served."""
    metadata = _metadata("X", ["a", "b"])
    del metadata["quantile_schema_version"]

    assert quantiles_from_metadata(metadata) == (0.5,)


def test_quantile_status_loaded_and_absent():
    feat_cols = ["a", "b"]
    loaded = build_model_set("X", _metadata("X", feat_cols), _booster_texts(feat_cols), tuple(feat_cols))
    absent = build_model_set(
        "X", _metadata("X", feat_cols, quantile_levels=()), _booster_texts(feat_cols, quantiles=(0.5,)), tuple(feat_cols)
    )

    assert loaded.quantile_status == QUANTILE_LOADED
    assert absent.quantile_status == QUANTILE_ABSENT


def test_build_model_set_rejects_schema_version_label_that_contradicts_feature_columns():
    feat_cols = ["a", "b"]
    metadata = _metadata("X", feat_cols) | {"schema_version": "000000000000"}

    with pytest.raises(SchemaMismatchError):
        build_model_set("X", metadata, _booster_texts(feat_cols), tuple(feat_cols))


def test_schema_version_is_order_sensitive_12_hex():
    assert len(schema_version(["a", "b"])) == 12
    assert schema_version(["a", "b"]) != schema_version(["b", "a"])


def _texts_with_names(point_names, tail_names, horizons=HORIZONS):
    out = {}
    for h in horizons:
        out[(h, 0.5)] = _tiny_booster_text(point_names)
        for q in (0.1, 0.9):
            out[(h, q)] = _tiny_booster_text(tail_names)
    return out


def test_point_booster_with_renamed_feature_is_rejected_with_diff():
    cols = ["a", "b", "c"]
    texts = _texts_with_names(["a", "b", "renamed"], cols)

    with pytest.raises(SchemaMismatchError, match=r"only in booster: \['renamed'\], only in expected: \['c'\]"):
        build_model_set("X", _metadata("X", cols), texts, tuple(cols))


def test_point_booster_with_reordered_features_is_rejected():
    """Same set, different order is still a mismatch -- LightGBM is positional."""
    cols = ["a", "b", "c"]
    texts = _texts_with_names(["a", "c", "b"], cols)

    with pytest.raises(SchemaMismatchError, match="different order"):
        build_model_set("X", _metadata("X", cols), texts, tuple(cols))


def test_quantile_booster_with_foreign_names_disables_corridor_but_serves_point():
    """metadata claims the corridor matches (hash agrees) but the file was
    trained on other names -- e.g. a mixed push. Names in the file win."""
    cols = ["a", "b", "c"]
    texts = _texts_with_names(cols, ["a", "b", "old"])

    model_set = build_model_set("X", _metadata("X", cols), texts, tuple(cols))

    assert model_set.quantile_status == QUANTILE_IGNORED
    assert model_set.quantiles == (0.5,)
    assert set(model_set.boosters) == {(h, 0.5) for h in HORIZONS}
    assert "only in booster: ['old'], only in expected: ['c']" in model_set.quantile_error


def test_matching_booster_names_pass_with_no_error():
    cols = ["a", "b", "c"]

    model_set = build_model_set("X", _metadata("X", cols), _texts_with_names(cols, cols), tuple(cols))

    assert model_set.quantile_status == QUANTILE_LOADED
    assert model_set.quantile_error is None
    assert model_set.quantiles == (0.5, 0.1, 0.9)


def test_hash_mismatched_corridor_reports_both_versions():
    cols = ["a", "b", "c"]
    metadata = _metadata("X", cols, quantile_schema_version="old000000000")

    model_set = build_model_set("X", metadata, _booster_texts(cols, quantiles=(0.5,)), tuple(cols))

    assert f"old000000000 != schema_version={schema_version(cols)}" in model_set.quantile_error
