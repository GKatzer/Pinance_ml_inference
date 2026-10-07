# Design decisions

Each entry: the decision, why, and what the alternative would have cost. Test names refer to [`tests/`](../tests).

## 1. `predict` is a pure function of its arguments

**Decision.** `predict(symbol, as_of_ts, candles, btc_candles, slot)` reads no database, no clock and no live feed. The caller sends the window; the last candle must be the `as_of_ts` candle, otherwise the call fails with 422 (`AsOfMismatchError`).

**Why.** The backend can lose a forecast (service down, network error). If the service could only answer "what is the forecast now", the gap would stay a gap. With a self-contained request the backend asks again later, with the same body, and gets the answer it would have got at the time. The `as_of_ts` check exists because predicting on whatever candle happens to be last while reporting a different timestamp would break this guarantee in a way that is hard to notice.

**Cost.** The service cannot verify that the candles are correct; it checks length, order and alignment only. Request bodies are large (150+ candles, twice for non-BTC symbols).

Tests: `test_predict_raises_when_as_of_ts_mismatches_last_candle`, `test_predict_sorts_unsorted_candles`.

## 2. The feature code is vendored, not installed

**Decision.** `vendor/features.py` is a byte-for-byte copy of the training pipeline, pinned to a commit recorded in [`vendor/LOCK.md`](../src/pinance_ml_inference/vendor/LOCK.md). Only `feature_columns()` is reimplemented, because the training version also drops target columns that never exist here.

**Why.** The serving host clones only this repository. A dependency on a sibling checkout was not possible there; copying keeps the service self-contained and the pin makes the copy auditable.

**Cost.** A copy can drift. That risk is not left to discipline: decisions 3 to 5 turn drift into a load-time rejection.

## 3. The model carries its own schema, and the service refuses a mismatch

**Decision.** Every pushed model has `feature_columns` in `metadata.json`. `build_model_set` raises `SchemaMismatchError` unless that list equals what the vendored code produces for the symbol (`test_build_model_set_rejects_feature_schema_mismatch`). A `schema_version` label that contradicts the model's own list is rejected too (`test_build_model_set_rejects_schema_version_label_that_contradicts_feature_columns`).

**Why.** A model fed a differently shaped feature vector still returns numbers. The only defence is to compare before serving.

**Where it runs.** At poll time, not per request. A rejected push keeps the previously loaded set and stores the reason (`test_fetch_and_register_keeps_prior_version_on_later_schema_mismatch`), so a bad push cannot become a stream of 500 errors. The reason is visible in `GET /models` as `last_load_error`.

## 4. Feature names inside the booster files are verified as well

**Decision.** After metadata passes, every booster is loaded and `booster.feature_name()` is compared with `feature_columns`. A point-model mismatch rejects the set; a corridor mismatch disables only the corridor. Names *and order* are compared (`test_point_booster_with_renamed_feature_is_rejected_with_diff`, `test_point_booster_with_reordered_features_is_rejected`).

**Why.** Metadata and files are pushed separately and can disagree (a mixed push into one slot). The files are what actually runs, so they get the last word. The check matters because LightGBM does not do it: [`docs/examples/lightgbm_names_check.py`](examples/lightgbm_names_check.py) trains a three-feature booster and predicts one row three ways (lightgbm 4.7.0, numpy 2.5.1, pandas 3.0.3):

```
booster.feature_name(): ['a', 'b', 'c']
own names      : -0.16358501101230444
renamed column : -0.16358501101230444
order swapped  : 0.4349148217212994
```

A renamed column gives exactly the same output (names are ignored, position is used) and a swapped order gives a different output without any error. A rename with the same width therefore produces silently wrong forecasts unless the service compares the names itself.

## 5. A corridor trained on another schema is ignored, not fatal

**Decision.** The point models and the quantile corridor are trained and pushed by independent pipelines, so a corridor lagging one schema behind is a normal moment. `quantile_status_from_metadata` returns `loaded`, `absent` (no `quantile_levels`) or `ignored_schema_mismatch` (`quantile_schema_version` differs from the fingerprint of the point features, or is missing). In the last case the point forecast keeps serving without `r_q10`/`r_q90` fields, and `GET /models` explains why.

**Why.** Rejecting the whole slot would take down a healthy point forecast because of an unrelated component.

Tests: `test_build_model_set_ignores_corridor_with_foreign_schema_but_keeps_point`, `test_corridor_without_quantile_schema_version_is_ignored`, `test_quantile_booster_with_foreign_names_disables_corridor_but_serves_point`, `test_fetch_and_register_serves_point_when_corridor_schema_is_foreign`.

## 6. The fingerprint is shared with the trainer and pinned by real vectors

**Decision.** `schema_version(columns)` is the first 12 hex characters of the SHA-256 of the comma-joined names (order matters). The trainer computes the same value when it exports a model. `tests/fixtures/training_schema_vectors.json` holds the feature lists and fingerprints of four real exported models (BTCUSDT: 93 features, `38b98fc7516d`; ETHUSDT, SOLUSDT, BNBUSDT: 94 features, all three `f72fb5b8ba97`; `model_version` ending `-4de49a70`); `tests/test_schema_parity.py` checks that the hash function reproduces them and that the vendored features produce exactly those lists.

**Why.** The two repositories deploy independently and share no import. Training's auto-promotion compares a candidate's `schema_version` with `expected_schema_version` from `GET /models`; if the two hash functions or feature lists ever differed, promotion would either never match or match wrongly.

**Fail closed.** The docstring of `models_status` states that a symbol absent from `/models` must not be promoted. That rule is applied by the training side; it is not checked here.

## 7. Why the history of `obv` matters here

An earlier feature, cumulative on-balance volume, depends on where the series starts. Training computed it over years of history, serving over a 150-candle window, so the two values differed (about 73 % apart in the synthetic regression test written up in [`Pinance_ml_training`](https://github.com/GKatzer/Pinance_ml_training); not measured on real candles, nor re-measured here). It was replaced by `obv_roc_36`, a windowed rate of change that needs no cumulative state. In this repository the property is protected by:

- `test_no_lookahead_last_row_stable_when_future_candles_appended`: the last feature row does not change when later candles are appended;
- `test_feature_columns_match_brief_contract_for_btc` and `test_btc_ret_appended_only_when_btc_candles_given`: the exact 93 columns for BTCUSDT and the 94th, `btc_ret`, only for the other symbols;
- the minimum window of 150 candles (`MIN_WARMUP_CANDLES`): `rolling(144)` needs 144 rows, the rest is headroom against an off-by-one. Below it, most of the 93 features are NaN, an input the models never saw in training, so the service answers 422 instead of predicting.

## 8. Quantile crossing is repaired, not hidden

**Decision.** With a corridor, the three raw outputs per horizon are sorted ascending and assigned to `r_q10 ≤ r_pred ≤ r_q90`. Sorting in log-return space is enough because `close * exp(r)` is monotonic.

**Why.** Each `(horizon, quantile)` model is trained on its own. Calibrated in aggregate does not mean ordered for a given row, and a response with `q10 > q90` would break the chart that draws the band.

**Side effect.** `r_pred` is the middle value after sorting, which is not necessarily what the q = 0.5 booster returned in the crossed case. `test_predict_sorts_crossed_quantiles` checks that values are reordered, not dropped.

## 9. Candidate and production are the same code with a different slot

**Decision.** `/predict/{symbol}` and `/predict/{symbol}/shadow` call the same function with `slot="production"` or `"candidate"`. An empty candidate slot is the normal state and answers 404.

**Why.** A candidate has to be judged under exactly the code that would serve it after promotion. Two code paths would make a good shadow result meaningless.

## 10. Polling instead of push, and a thread instead of a scheduler

**Decision.** A daemon thread polls MinIO every 600 s (configurable), metadata first, boosters only when the version pair changed.

**Why.** Training and serving do not need a shared message channel for something that changes on a retraining schedule (daily for the point models, weekly for the corridor, per the training repository); ten minutes of staleness is acceptable, and a failed poll leaves the previous models in place. The initial blocking poll removes the one case where staleness hurts (cold start).

**Cost.** A new model can wait up to one interval before it is served; the poll thread is in-process, so with several uvicorn workers each would poll and hold its own copy (the unit file starts one process).

## What was not done

- No authentication; the service is meant to be reachable only from a private network (see [deployment](deployment.md)).
- No feature-drift computation. The response carries `feature_baseline` (recorded at training and carried in the metadata) and `feature_snapshot` (six raw feature values for the predicted row) so that a consumer *could* compare them; nothing in this repository does. The comment above `DRIFT_FEATURE_COLUMNS` in `predict.py` says the same.
- No tests against a real MinIO server or with model files from the real training pipeline; the HTTP tests ([`tests/test_api_predict.py`](../tests/test_api_predict.py)) run in-process with a stubbed registry (see [Tests](../README.md#tests-and-quality)).
