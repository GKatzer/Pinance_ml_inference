import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# Model artifacts live in MinIO, not on local disk -- see model_registry.py.
# Fixed keys per symbol: {symbol}/production/... and {symbol}/candidate/...
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
MINIO_MODELS_BUCKET = os.environ.get("MINIO_MODELS_BUCKET", "pinance-models")

# How often the background poller checks MinIO for a model_version change,
# per symbol per slot (production/candidate). Brief/README: "раз в 10 минут".
MODEL_POLL_INTERVAL_SECONDS = int(os.environ.get("MODEL_POLL_INTERVAL_SECONDS", "600"))

# Target definition per inference brief: r_h = log(P[t + 5h min] / P[t]).
CANDLE_INTERVAL_MINUTES = 5
HORIZONS = list(range(1, 13))

# Brief §3: below this, most of the 93 features are NaN (rolling_144/lag_24
# warm-up) -- an input the model essentially never saw in training. Refuse
# to serve a prediction rather than pass one through. This is also the
# minimum window size the caller (VDS1) must send in POST /predict's
# `candles[symbol]` -- rolling_144 needs 144 candles up to and including the
# as_of_ts row, so 150 leaves headroom against an off-by-one.
MIN_WARMUP_CANDLES = 150

# How many recent candles scripts/predict_cli.py pulls from the DB for a
# manual local run. Production serving never fetches candles itself -- see
# predict.py -- this is a dev-tool convenience only, with headroom above
# MIN_WARMUP_CANDLES so a manual run never falls short by an off-by-one.
FETCH_CANDLES = 250
