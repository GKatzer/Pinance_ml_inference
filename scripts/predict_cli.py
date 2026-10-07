"""Quick local smoke test for the predict pipeline, no HTTP server needed.

    python scripts/predict_cli.py BTCUSDT
    python scripts/predict_cli.py BTCUSDT --shadow
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinance_ml_inference.config import FETCH_CANDLES
from pinance_ml_inference.db import load_recent_candles
from pinance_ml_inference.model_registry import fetch_and_register
from pinance_ml_inference.predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument(
        "--shadow", action="store_true", help="read the candidate slot instead of production"
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    slot = "candidate" if args.shadow else "production"

    # Standalone script, no background poller running -- fetch this one
    # (symbol, slot) from MinIO directly before predicting.
    fetch_and_register(symbol, slot)

    # predict() is now a pure function of (as_of_ts, candles) -- production
    # serving gets those from the POST body (see api.py); this CLI is the
    # one place still allowed to pull straight from the DB, for convenience.
    candles = load_recent_candles(symbol, limit=FETCH_CANDLES)
    btc_candles = None if symbol == "BTCUSDT" else load_recent_candles("BTCUSDT", limit=FETCH_CANDLES)
    as_of_ts = candles["ts"].iloc[-1]

    result = predict(symbol, as_of_ts, candles, btc_candles, slot=slot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
