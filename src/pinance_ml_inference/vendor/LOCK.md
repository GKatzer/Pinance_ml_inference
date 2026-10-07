# Vendored feature code

`features.py` in this directory is a byte-for-byte copy of
`src/pinance_ml/features/pipeline.py` from the `Pinance_ml_training` repo, pinned at:

```
commit 4de49a70ae41f92f2701a0c7949a27b71b2cd4db
```

Why vendored instead of an installed dependency: the serving host (this service's
deployment target) only clones this one repo — no sibling checkout of
`Pinance_ml_training` to depend on. `feature_columns()` is a small reimplementation
(not copied) since inference never computes targets (`dataset.py`'s version
excludes target columns that don't exist here).

**Feature parity is enforced automatically, not just by convention**: every
exported model's `metadata.json` records the exact `feature_columns` list
used at training time, and `models.load_model_set` refuses to load a model
whose list doesn't match what this vendored code computes right now (see
`pinance_ml_inference/models.py`). If that check starts failing, it means
`Pinance_ml_training`'s `features/pipeline.py` changed after the pinned commit — re-vendor
`features.py` from the new commit and update the hash above.

**Schema fingerprint**: `models.schema_version()` mirrors training's
`export_models.py::_schema_version` (sha256 of comma-joined `feature_columns`,
first 12 hex). `tests/test_schema_parity.py` pins real training vectors in
`tests/fixtures/training_schema_vectors.json` — when you re-vendor, refresh
that fixture from a freshly pushed candidate's `metadata.json`.
`GET /models` exposes `expected_schema_version` per symbol for training's
auto-promote. A corridor (q10/q90) whose `quantile_schema_version` differs
from the point schema is ignored (point keeps serving), not rejected.
