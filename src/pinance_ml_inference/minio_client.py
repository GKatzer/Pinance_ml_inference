import json
from functools import lru_cache

import urllib3
from minio import Minio
from minio.error import S3Error

from pinance_ml_inference import config

_NOT_FOUND_CODES = {"NoSuchKey", "NoSuchBucket"}

# A bad/unreachable endpoint (wrong host, DNS failure, network down) must
# fail fast, not hang: lifespan's initial poll (api.py) blocks app startup on
# this, and it runs once per (symbol, slot) -- urllib3's default retry/
# backoff behavior would otherwise turn a single bad endpoint into a long,
# multiplied startup delay.
_CONNECT_TIMEOUT_SECONDS = 2.0
_READ_TIMEOUT_SECONDS = 5.0


@lru_cache(maxsize=1)
def _client() -> Minio:
    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT_SECONDS, read=_READ_TIMEOUT_SECONDS),
        retries=urllib3.Retry(total=0),
    )
    return Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_SECURE,
        http_client=http_client,
    )


def fetch_metadata(symbol: str, slot: str) -> dict | None:
    """`{symbol}/{slot}/metadata.json`, or None if that slot is empty.

    None is the normal state for `candidate` most of the time -- there isn't
    always a pushed candidate to test. Any other error (auth, network, a
    genuinely broken bucket) propagates; that's a real problem, not "no
    candidate".
    """
    try:
        response = _client().get_object(config.MINIO_MODELS_BUCKET, f"{symbol}/{slot}/metadata.json")
        try:
            return json.loads(response.data)
        finally:
            response.close()
            response.release_conn()
    except S3Error as e:
        if e.code in _NOT_FOUND_CODES:
            return None
        raise


def _booster_key(symbol: str, slot: str, horizon: int, quantile: float) -> str:
    """The median (q=0.5) reuses the original unsuffixed h{n}.txt -- whatever
    the training side puts there, a real quantile-loss model or a renamed
    copy of the old point regressor, inference doesn't care. Only the two
    tails get a _q{quantile} suffix, using the literal quantile value (e.g.
    `h1_q0.1.txt`) -- exactly how export_models.py names the file it saves
    (`f"h{h}_q{q}.txt"`, `q` being the raw float from CORRIDOR_QUANTILES),
    not a rounded percentage. This is the one place that asymmetry lives.
    """
    if quantile == 0.5:
        return f"{symbol}/{slot}/h{horizon}.txt"
    return f"{symbol}/{slot}/h{horizon}_q{quantile}.txt"


def fetch_booster_text(symbol: str, slot: str, horizon: int, quantile: float) -> str:
    """LightGBM's native text dump for one (horizon, quantile) model.

    Only called once `fetch_metadata` has confirmed the slot is populated,
    so a missing object here is a real inconsistency, not a normal state --
    left to propagate.
    """
    response = _client().get_object(config.MINIO_MODELS_BUCKET, _booster_key(symbol, slot, horizon, quantile))
    try:
        return response.data.decode("utf-8")
    finally:
        response.close()
        response.release_conn()
