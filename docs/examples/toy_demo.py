"""Offline demo of the inference service with toy models and a fake MinIO.

Nothing here is a real model: the "market" is a seeded random walk and the
LightGBM boosters are tiny (20 trees). What is real is the service code: the
vendored features, schema checks, registry, polling, hot swap, and the HTTP
layer. The only thing replaced is the object store: `minio_client` reads from a
local directory laid out exactly like the bucket ({symbol}/{slot}/metadata.json,
h1.txt, h1_q0.1.txt, ...).

    python docs/examples/toy_demo.py build-models  STORE_DIR [--slot candidate --variant ok|renamed-booster|foreign-corridor --version V]
    python docs/examples/toy_demo.py make-request  OUT.json  [--symbols BTCUSDT ETHUSDT]
    python docs/examples/toy_demo.py serve         STORE_DIR [--port 8011]
    python docs/examples/toy_demo.py bench         [--url http://127.0.0.1:8011] REQUEST.json [-n 50]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://readonly_user:CHANGE_ME@127.0.0.1:5432/unused")
os.environ.setdefault("MINIO_ENDPOINT", "127.0.0.1:9")
os.environ.setdefault("MINIO_ACCESS_KEY", "CHANGE_ME")
os.environ.setdefault("MINIO_SECRET_KEY", "CHANGE_ME")
os.environ.setdefault("MODEL_POLL_INTERVAL_SECONDS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from pinance_ml_inference.models import schema_version
from pinance_ml_inference.vendor.features import compute_features, feature_columns

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = list(range(1, 13))
QUANTILES = (0.1, 0.9)
N_CANDLES = 3000
START = "2026-01-01"


def synthetic_candles(symbol: str, n: int = N_CANDLES) -> pd.DataFrame:
    """Seeded random-walk OHLCV; same timestamps for every symbol."""
    rng = np.random.default_rng(abs(hash_seed(symbol)))
    ts = pd.date_range(START, periods=n, freq="5min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(scale=0.001, size=n)))
    spread = rng.uniform(0.0002, 0.0015, size=n)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": np.roll(close, 1),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.uniform(1, 100, size=n),
        }
    )


def hash_seed(symbol: str) -> int:
    return sum(ord(c) * 31**i for i, c in enumerate(symbol)) % 2**31


def _train_symbol(symbol: str, variant: str, version: str):
    candles = synthetic_candles(symbol)
    btc = None if symbol == "BTCUSDT" else synthetic_candles("BTCUSDT")
    feats = compute_features(candles, btc_candles=btc)
    cols = feature_columns(feats)
    close = feats["close"]

    booster_names = list(cols)
    if variant == "renamed-booster":  # files trained on a differently named column, metadata untouched
        booster_names[booster_names.index("obv_roc_36")] = "obv"

    files = {}
    for h in HORIZONS:
        y = np.log(close.shift(-h) / close)
        keep = feats[cols].notna().all(axis=1) & y.notna()
        X, y = feats.loc[keep, cols].to_numpy(), y[keep].to_numpy()
        for q, objective in [(0.5, {"objective": "regression_l1"})] + [
            (q, {"objective": "quantile", "alpha": q}) for q in QUANTILES
        ]:
            params = {**objective, "num_leaves": 7, "learning_rate": 0.1, "verbosity": -1, "seed": 0}
            ds = lgb.Dataset(X, label=y, feature_name=booster_names)
            text = lgb.train(params, ds, num_boost_round=20).model_to_string()
            files[f"h{h}.txt" if q == 0.5 else f"h{h}_q{q}.txt"] = text

    sv = schema_version(cols)
    metadata = {
        "model_version": version,
        "quantile_model_version": version,
        "schema_version": sv,
        "quantile_schema_version": "000000000000" if variant == "foreign-corridor" else sv,
        "feature_columns": cols,
        "horizons": HORIZONS,
        "quantile_horizons": HORIZONS,
        "quantile_levels": list(QUANTILES),
        "source_commit": "toy-demo",
    }
    return metadata, files


def build_models(args):
    for symbol in SYMBOLS:
        metadata, files = _train_symbol(symbol, args.variant, args.version)
        out = Path(args.store) / symbol / args.slot
        out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (out / name).write_text(text)
        (out / "metadata.json").write_text(json.dumps(metadata))  # last, so a poll never sees a half-written slot
        print(f"{symbol}/{args.slot}: version={args.version} variant={args.variant} "
              f"features={len(metadata['feature_columns'])} files={len(files) + 1}")


def make_request(args):
    end = N_CANDLES - 1
    window = slice(end - args.window + 1, end + 1)
    candles = {}
    for symbol in args.symbols:
        df = synthetic_candles(symbol).iloc[window]
        candles[symbol] = [
            {"ts": r.ts.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
            for r in df.itertuples()
        ]
    as_of = candles[args.symbols[0]][-1]["ts"]
    Path(args.out).write_text(json.dumps({"as_of_ts": as_of, "candles": candles}))
    print(f"wrote {args.out}: as_of_ts={as_of}, {args.window} candles for {', '.join(args.symbols)}")


def serve(args):
    import uvicorn

    from pinance_ml_inference import api, minio_client, model_registry

    store = Path(args.store)

    def fetch_metadata(symbol, slot):
        path = store / symbol / slot / "metadata.json"
        return json.loads(path.read_text()) if path.exists() else None

    def fetch_booster_text(symbol, slot, horizon, quantile):
        name = f"h{horizon}.txt" if quantile == 0.5 else f"h{horizon}_q{quantile}.txt"
        return (store / symbol / slot / name).read_text()

    # The fake MinIO: the only substitution. Everything above and below it is the real service code.
    minio_client.fetch_metadata = fetch_metadata
    minio_client.fetch_booster_text = fetch_booster_text
    api.list_symbols = model_registry.list_symbols = lambda: SYMBOLS

    uvicorn.run(api.app, host="127.0.0.1", port=args.port, log_level="warning")


def bench(args):
    import httpx

    body = json.loads(Path(args.request).read_text())
    wall, infer = [], []
    with httpx.Client(timeout=30) as client:
        for _ in range(args.n + 3):
            t0 = time.perf_counter()
            r = client.post(f"{args.url}/predict/{args.symbol}", json=body)
            wall.append((time.perf_counter() - t0) * 1000)
            infer.append(r.json()["inference_ms"])
    wall, infer = wall[3:], infer[3:]  # drop 3 warm-up calls
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    print(f"{args.symbol}, n={len(wall)} after 3 warm-up calls, toy models, same machine")
    print(f"round trip ms: p50={statistics.median(wall):.1f} p95={q(wall, .95):.1f} max={max(wall):.1f}")
    print(f"inference_ms : p50={statistics.median(infer):.2f} p95={q(infer, .95):.2f} max={max(infer):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-models"); b.add_argument("store")
    b.add_argument("--slot", default="production", choices=["production", "candidate"])
    b.add_argument("--variant", default="ok", choices=["ok", "renamed-booster", "foreign-corridor"])
    b.add_argument("--version", default="toy-1"); b.set_defaults(fn=build_models)
    m = sub.add_parser("make-request"); m.add_argument("out")
    m.add_argument("--symbols", nargs="+", default=SYMBOLS); m.add_argument("--window", type=int, default=200)
    m.set_defaults(fn=make_request)
    s = sub.add_parser("serve"); s.add_argument("store"); s.add_argument("--port", type=int, default=8011)
    s.set_defaults(fn=serve)
    n = sub.add_parser("bench"); n.add_argument("request"); n.add_argument("--url", default="http://127.0.0.1:8011")
    n.add_argument("--symbol", default="BTCUSDT"); n.add_argument("-n", type=int, default=50); n.set_defaults(fn=bench)
    a = p.parse_args(); a.fn(a)
