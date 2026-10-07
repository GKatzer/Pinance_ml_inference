"""Cross-project check: this service must fingerprint features exactly like
Pinance_ML's training does.

tests/fixtures/training_schema_vectors.json holds real (feature_columns ->
schema_version) pairs taken from training-pushed candidate metadata.json
(model_version ...-4de49a70, i.e. the pinned commit in vendor/LOCK.md).
Training's `_schema_version` is hashed over `feature_columns` in order, and
the vendored compute_features must produce that same list per symbol --
otherwise training's auto-promote (which compares against GET /models'
expected_schema_version) would never match, or worse, match wrongly.
"""

import json
from pathlib import Path

import pytest

from pinance_ml_inference import model_registry
from pinance_ml_inference.models import schema_version

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "training_schema_vectors.json").read_text())


@pytest.mark.parametrize("symbol", sorted(VECTORS))
def test_schema_version_matches_training_hash(symbol):
    vector = VECTORS[symbol]
    assert schema_version(vector["feature_columns"]) == vector["schema_version"]


@pytest.mark.parametrize("symbol", sorted(VECTORS))
def test_vendored_features_produce_the_training_column_list(symbol):
    vector = VECTORS[symbol]
    model_registry._expected_feature_columns.cache_clear()

    assert list(model_registry._expected_feature_columns(symbol)) == vector["feature_columns"]
    assert model_registry.expected_schema_version(symbol) == vector["schema_version"]
