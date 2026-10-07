import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pinance_ml_inference.config import MODEL_POLL_INTERVAL_SECONDS
from pinance_ml_inference.db import list_symbols
from pinance_ml_inference.model_registry import ModelNotLoadedError, models_status, poll_loop, poll_once
from pinance_ml_inference.predict import AsOfMismatchError, InsufficientHistoryError, predict

logger = logging.getLogger(__name__)

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class Candle(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    """Body for POST /predict/{symbol} (and its /shadow twin).

    `candles[symbol]` is the window predict() runs on -- ascending by ts,
    last row's ts == as_of_ts, >= MIN_WARMUP_CANDLES rows (predict.py
    enforces both, so a malformed window comes back as a 422, not a 500 or
    a silently wrong prediction). `candles["BTCUSDT"]` is the same shape for
    the cross-asset btc_ret feature, required whenever symbol != BTCUSDT and
    ignored (need not be present) when predicting BTCUSDT itself.
    """

    as_of_ts: datetime
    candles: dict[str, list[Candle]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocking, so a fresh restart doesn't 404 everything for up to 10 minutes.
    # A failure here (database down, MinIO unreachable) must not stop the app
    # from starting: poll_loop below polls again straight away and then every
    # interval, and /health and /models stay answerable meanwhile.
    try:
        poll_once(list_symbols())
    except Exception:
        logger.exception("Initial model poll failed; continuing, the poll loop will retry")
    stop_event = threading.Event()
    thread = threading.Thread(
        target=poll_loop, args=(stop_event, MODEL_POLL_INTERVAL_SECONDS), daemon=True
    )
    thread.start()
    yield
    stop_event.set()
    thread.join(timeout=5)


app = FastAPI(title="Pinance ML Inference (local)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    """Loaded model versions/schemas per symbol and slot, plus the schema
    fingerprint this service expects per symbol (auto-promote's precondition).
    """
    try:
        symbols = list_symbols()
    except Exception:  # DB down shouldn't hide what's loaded in memory
        symbols = []
    return models_status(symbols)


def _as_utc_timestamp(dt: datetime) -> pd.Timestamp:
    """Normalize to a tz-aware UTC pd.Timestamp, matching _candles_df's `ts`
    column so the two compare equal in predict()'s as_of_ts check. A naive
    input (no offset in the request JSON) is treated as already-UTC, same as
    pandas' own `utc=True` handling of naive datetimes below -- not shifted.
    """
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _candles_df(items: list[Candle] | None) -> pd.DataFrame:
    """Empty-but-correctly-shaped frame for a missing key (rather than None)
    so predict()'s MIN_WARMUP_CANDLES check is what rejects it -- a 422 with
    a clear count, not an AttributeError.
    """
    if not items:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame([c.model_dump() for c in items])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _predict_or_404(symbol: str, slot: str, body: PredictRequest):
    symbol = symbol.upper()
    as_of_ts = _as_utc_timestamp(body.as_of_ts)
    candles = _candles_df(body.candles.get(symbol))
    btc_candles = None if symbol == "BTCUSDT" else _candles_df(body.candles.get("BTCUSDT"))
    try:
        return predict(symbol, as_of_ts, candles, btc_candles, slot=slot)
    except (InsufficientHistoryError, AsOfMismatchError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict/{symbol}")
def post_prediction(symbol: str, body: PredictRequest):
    return _predict_or_404(symbol, "production", body)


@app.post("/predict/{symbol}/shadow")
def post_shadow_prediction(symbol: str, body: PredictRequest):
    """Candidate model's prediction, same request/response shape as
    /predict/{symbol}.

    404 when no candidate is loaded (the normal state -- there isn't always
    something to shadow-test); the caller (VDS1) already skips a cycle on
    that from fetch_prediction.
    """
    return _predict_or_404(symbol, "candidate", body)
