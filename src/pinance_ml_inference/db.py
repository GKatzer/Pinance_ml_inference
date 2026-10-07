from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from pinance_ml_inference.config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(DATABASE_URL)


def list_symbols() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM candles ORDER BY symbol"))
        return [r[0] for r in rows]


def load_recent_candles(symbol: str, limit: int) -> pd.DataFrame:
    """Most recent `limit` closed candles for a symbol, ascending by ts.

    Serving only ever needs a trailing window, not full history (that's
    training's job) -- ORDER BY ts DESC + LIMIT keeps this a fast, constant-
    size query no matter how many years of candles have accumulated.
    """
    query = """
        SELECT ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = %(symbol)s
        ORDER BY ts DESC
        LIMIT %(limit)s
    """
    df = pd.read_sql(query, get_engine(), params={"symbol": symbol, "limit": limit})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)
