# Offline demo: toy models and a fake MinIO

The real service needs models and an object store that are not part of this repository. [`docs/examples/toy_demo.py`](examples/toy_demo.py) removes both requirements so the service can be started and exercised on any machine, with real HTTP requests and responses.

## What is real and what is not

| Real (the code under test) | Replaced |
|---|---|
| vendored feature code, schema checks, `ModelRegistry`, polling thread and hot swap, `predict()`, the FastAPI routes and error mapping, a real `uvicorn` process | **the object store**: `minio_client.fetch_metadata` and `fetch_booster_text` are swapped at start-up for functions that read a local directory laid out like the bucket (`{symbol}/{slot}/metadata.json`, `h1.txt`, `h1_q0.1.txt`, …) |
| LightGBM boosters in the native text format, the same 37 files per symbol and slot that the trainer writes (12 medians + 24 corridor tails + metadata) | **the models**: 20 trees, 7 leaves, trained on a seeded random walk. They carry no information about any market |
| the real feature lists: 93 features for BTCUSDT, 94 for ETHUSDT, and the real fingerprints `38b98fc7516d` / `f72fb5b8ba97` | **the database**: the symbol list is fixed to `BTCUSDT`, `ETHUSDT` |

Forecast values, corridor widths and `feature_snapshot` numbers in the outputs below are therefore meaningless; what the demo shows is the behaviour of the service. The captured outputs are in [`docs/examples/toy_demo_outputs/`](examples/toy_demo_outputs/).

## Run it

Requirements as in the README (`uv pip install -r requirements.txt`). Commands are run from the repository root; `toy_store` is any empty directory.

```bash
python docs/examples/toy_demo.py build-models toy_store --version toy-1
python docs/examples/toy_demo.py make-request candles.json
python docs/examples/toy_demo.py serve toy_store          # listens on 127.0.0.1:8011, polls every 2 s
```

Output of the first two commands:

```
BTCUSDT/production: version=toy-1 variant=ok features=93 files=37
ETHUSDT/production: version=toy-1 variant=ok features=94 files=37
wrote candles.json: as_of_ts=2026-01-11T09:55:00+00:00, 200 candles for BTCUSDT, ETHUSDT
```

(`build-models` takes about 3 s on a 16-core machine.) In another terminal:

```bash
curl -s 127.0.0.1:8011/models
curl -s -X POST 127.0.0.1:8011/predict/BTCUSDT -H 'Content-Type: application/json' -d @candles.json
```

### 1. Healthy state

[`models-healthy.json`](examples/toy_demo_outputs/models-healthy.json), shortened to BTCUSDT:

```json
{
  "expected_schema_version": { "BTCUSDT": "38b98fc7516d", "ETHUSDT": "f72fb5b8ba97" },
  "models": { "BTCUSDT": {
      "production": { "loaded": true, "last_load_error": null, "model_version": "toy-1",
                      "schema_version": "38b98fc7516d", "matches_expected": true,
                      "quantile": { "status": "loaded", "quantile_model_version": "toy-1",
                                    "quantile_schema_version": "38b98fc7516d" } },
      "candidate":  { "loaded": false, "last_load_error": null } } }
}
```

The prediction ([`predict-btcusdt.json`](examples/toy_demo_outputs/predict-btcusdt.json), first horizon only; `…` marks cut content):

```json
{ "symbol": "BTCUSDT", "as_of_ts": "2026-01-11T09:55:00+00:00", "close": 98.39831624068697,
  "model_version": "toy-1", "quantile_model_version": "toy-1", "schema_version": "38b98fc7516d",
  "inference_ms": 21.9, "feature_count": 93, "feature_baseline": null,
  "feature_snapshot": { "btc_ret": null, "atr": 0.1779…, "bb_width": 0.0119…, "rsi": 70.55…,
                        "ret_std_144": 0.00096…, "macd_diff": 0.0544… },
  "predictions": [ { "horizon": 1, "target_ts": "2026-01-11T10:00:00+00:00",
                     "r_pred": 4.30e-05, "price_pred": 98.4025…,
                     "r_q10": -0.001278, "r_q90": 0.001266,
                     "price_q10": 98.2726…, "price_q90": 98.5229… }, … ] }
```

`feature_baseline` is `null` because the toy metadata has none; `btc_ret` is `null` because BTCUSDT has no cross-asset feature. The ETHUSDT response ([`predict-ethusdt.json`](examples/toy_demo_outputs/predict-ethusdt.json)) has `feature_count: 94` and a numeric `btc_ret`.

### 2. Errors

[`errors.txt`](examples/toy_demo_outputs/errors.txt):

| Request | Result |
|---|---|
| `/predict/BTCUSDT/shadow` before any candidate exists | 404 `No candidate model currently loaded for BTCUSDT` |
| BTCUSDT window of 100 candles | 422 `BTCUSDT: only 100 candles in request (need >= 150)` |
| `as_of_ts` five minutes before the last candle | 422 `… does not match the last candle's ts=… -- candles[symbol][-1] must be the as_of_ts candle` |
| ETHUSDT request without a BTCUSDT window | 422 `BTCUSDT: only 0 candles in request (need >= 150), required for ETHUSDT's btc_ret feature` |
| symbol with no model (`DOGEUSDT`) | 404 `No production model currently loaded for DOGEUSDT` |
| body without `candles` | 422, FastAPI validation error (`Field required`) |

### 3. Hot swap and rejection

While the service runs, new candidates are "pushed" by writing into the directory. Transcript: [`hot-swap-and-rejection.txt`](examples/toy_demo_outputs/hot-swap-and-rejection.txt).

```bash
python docs/examples/toy_demo.py build-models toy_store --slot candidate --version toy-2 --variant ok
```
After one poll, `candidate` for BTCUSDT is `loaded: true, model_version: "toy-2", matches_expected: true`, and `/predict/BTCUSDT/shadow` answers 200 with the candidate's version. No restart happened.

```bash
python docs/examples/toy_demo.py build-models toy_store --slot candidate --version toy-3 --variant renamed-booster
```
`toy-3` has correct metadata but its booster files were trained with the column `obv_roc_36` named `obv`. It is rejected; `toy-2` keeps serving, and `/models` explains:

```
"model_version": "toy-2",
"last_load_error": "BTCUSDT: point model h1 was trained on different features than metadata.feature_columns
  (schema_version=21c8984f3440 vs 38b98fc7516d): only in booster: ['obv'], only in expected: ['obv_roc_36']"
```

```bash
python docs/examples/toy_demo.py build-models toy_store --slot candidate --version toy-4 --variant foreign-corridor
```
`toy-4` has a corridor whose `quantile_schema_version` is another schema. The point model is served and the corridor is dropped:

```
"model_version": "toy-4", "last_load_error": "corridor ignored: quantile_schema_version=000000000000 != schema_version=38b98fc7516d",
"quantile": { "status": "ignored_schema_mismatch", … }
```

The next `/predict/BTCUSDT/shadow` response has only `horizon`, `target_ts`, `r_pred` and `price_pred` per horizon, no `r_q10`/`r_q90`.

These three cases are the same ones the unit tests assert ([decisions 3 to 5](design-decisions.md)); here they are seen through the HTTP interface of a running process.

### 4. Timing on toy models

```bash
python docs/examples/toy_demo.py bench candles.json -n 50
```

```
BTCUSDT, n=50 after 3 warm-up calls, toy models, same machine
round trip ms: p50=49.2 p95=95.2 max=97.7
inference_ms : p50=24.32 p95=70.55 max=72.40
ETHUSDT, n=50 after 3 warm-up calls, toy models, same machine
round trip ms: p50=50.8 p95=93.6 max=99.1
inference_ms : p50=23.84 p95=26.34 max=27.16
```

One run on a 16-core Linux desktop, client and server on the same machine, single sequential client. **This is not a performance figure for the production models**: these have 20 trees each, the production ones are larger, and the production host is different. The spread between p50 and p95 (the BTCUSDT `inference_ms` p95 is three times its median while the ETHUSDT p95 is not) shows how noisy 50 samples are. It only confirms that the service overhead around 36 small booster calls is of the order of tens of milliseconds. For real latency see the backend's measurements of the whole call ([`Pinance_backend`](https://github.com/GKatzer/Pinance_backend)).
