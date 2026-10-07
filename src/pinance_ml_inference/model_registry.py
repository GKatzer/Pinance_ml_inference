import logging
import threading
from functools import lru_cache

import pandas as pd

from pinance_ml_inference import minio_client
from pinance_ml_inference.config import MIN_WARMUP_CANDLES
from pinance_ml_inference.db import list_symbols
from pinance_ml_inference.models import (
    QUANTILE_IGNORED,
    ModelSet,
    SchemaMismatchError,
    booster_keys_from_metadata,
    build_model_set,
    schema_version,
)
from pinance_ml_inference.vendor.features import compute_features, feature_columns

logger = logging.getLogger(__name__)

SLOTS = ("production", "candidate")

# Enough rows that every rolling/indicator window in vendor/features.py
# produces a value instead of erroring on short input (ta's AverageTrueRange
# indexes `window` rows in directly and raises otherwise) -- only column
# *names* matter here, never these values, so the content is arbitrary.
_SYNTHETIC_CANDLES = pd.DataFrame(
    {
        "ts": pd.date_range("2026-01-01", periods=MIN_WARMUP_CANDLES, freq="5min", tz="UTC"),
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
    }
)


@lru_cache(maxsize=None)
def _expected_feature_columns(symbol: str) -> tuple[str, ...]:
    """Column names this service's vendored feature code would currently
    produce for `symbol` -- for the poll-time schema check, so a bad push
    never reaches the registry. Distinct from predict.py's real feature
    computation, which needs actual candle history for real values; this
    only needs column *names*, which depend solely on whether `symbol` gets
    the cross-asset btc_ret feature (i.e. whether it's BTCUSDT itself).
    """
    btc_candles = None if symbol == "BTCUSDT" else _SYNTHETIC_CANDLES
    features = compute_features(_SYNTHETIC_CANDLES, btc_candles=btc_candles)
    return tuple(feature_columns(features))


def expected_schema_version(symbol: str) -> str:
    """Fingerprint of the feature list this service computes for `symbol` --
    what training's auto-promote compares a candidate's `schema_version`
    against (GET /models). Per symbol because BTCUSDT has no btc_ret column.
    """
    return schema_version(_expected_feature_columns(symbol))


class ModelNotLoadedError(LookupError):
    """No model currently loaded for (symbol, slot).

    Either it's never been successfully polled, or -- the normal state most
    of the time for "candidate" -- nothing's been pushed there yet.
    """


class ModelRegistry:
    """Thread-safe in-memory store of the current ModelSet per (symbol, slot).

    A full ModelSet is swapped in atomically by `set()`; readers never see a
    partially-built one. Production and candidate are just two slot values
    of the same key, so they update completely independently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sets: dict[tuple[str, str], ModelSet] = {}
        self._errors: dict[tuple[str, str], str] = {}

    def get(self, symbol: str, slot: str) -> ModelSet:
        model_set = self.get_or_none(symbol, slot)
        if model_set is None:
            raise ModelNotLoadedError(f"No {slot} model currently loaded for {symbol}")
        return model_set

    def get_or_none(self, symbol: str, slot: str) -> ModelSet | None:
        with self._lock:
            return self._sets.get((symbol, slot))

    def set(self, symbol: str, slot: str, model_set: ModelSet) -> None:
        with self._lock:
            self._sets[(symbol, slot)] = model_set
            self._errors.pop((symbol, slot), None)

    def set_error(self, symbol: str, slot: str, error: str | None) -> None:
        """Remember why the latest push for (symbol, slot) was rejected (None
        clears it) -- the registry only holds loaded sets, so without this a
        rejected slot is indistinguishable from an empty one.
        """
        with self._lock:
            if error is None:
                self._errors.pop((symbol, slot), None)
            else:
                self._errors[(symbol, slot)] = error

    def snapshot(self) -> tuple[dict[tuple[str, str], ModelSet], dict[tuple[str, str], str]]:
        with self._lock:
            return dict(self._sets), dict(self._errors)


registry = ModelRegistry()


def get_model_set(symbol: str, slot: str) -> ModelSet:
    return registry.get(symbol, slot)


def _version_key(metadata: dict) -> tuple:
    """Point and corridor are independently trained/pushed pipelines that
    can each update a slot at different times (model_storage.py's
    disjoint-keys design: `model_version` for the point pipeline,
    `quantile_model_version` for the corridor) -- a version key that only
    looked at `model_version` would silently miss a quantile-only update
    to an already-loaded slot.
    """
    return (metadata.get("model_version"), metadata.get("quantile_model_version"))


def fetch_and_register(symbol: str, slot: str) -> None:
    """Fetch `symbol`'s `slot` from MinIO and swap it into the registry if
    its (model_version, quantile_model_version) changed.

    No-op if the slot is empty (the normal state for "candidate" most of the
    time) or if the version is unchanged (no redundant booster downloads).
    On a schema mismatch, logs and keeps whatever was already registered
    (possibly nothing) rather than raising -- validation happens once here,
    at poll time, so a bad push can never turn into a per-request 500.
    """
    metadata = minio_client.fetch_metadata(symbol, slot)
    if metadata is None:
        registry.set_error(symbol, slot, None)
        return

    current = registry.get_or_none(symbol, slot)
    if current is not None and _version_key(metadata) == _version_key(current.metadata):
        return

    booster_texts = {
        (h, q): minio_client.fetch_booster_text(symbol, slot, h, q) for h, q in booster_keys_from_metadata(metadata)
    }
    try:
        model_set = build_model_set(symbol, metadata, booster_texts, _expected_feature_columns(symbol))
    except SchemaMismatchError as e:
        logger.error("Schema mismatch loading %s/%s, keeping prior state: %s", symbol, slot, e)
        registry.set_error(symbol, slot, str(e))
        return

    registry.set(symbol, slot, model_set)
    if model_set.quantile_status == QUANTILE_IGNORED:
        # Point is serving; keep the reason visible in GET /models (set() cleared it).
        registry.set_error(symbol, slot, model_set.quantile_error)
        logger.warning("%s/%s: %s; serving point only", symbol, slot, model_set.quantile_error)
    logger.info(
        "Loaded %s/%s model_version=%s quantile_model_version=%s",
        symbol,
        slot,
        metadata.get("model_version"),
        metadata.get("quantile_model_version"),
    )


def models_status(symbols: list[str]) -> dict:
    """Body of GET /models: what's loaded per (symbol, slot), the schema
    fingerprint this service expects per symbol, and why a rejected slot was
    rejected. Training's auto-promote reads `expected_schema_version` and
    must not promote when a symbol is missing here (fail closed).
    """
    sets, errors = registry.snapshot()
    all_symbols = sorted(set(symbols) | {sym for sym, _ in sets} | {sym for sym, _ in errors})
    models: dict[str, dict] = {}
    expected: dict[str, str] = {}
    for symbol in all_symbols:
        expected[symbol] = expected_schema_version(symbol)
        slots = {}
        for slot in SLOTS:
            model_set = sets.get((symbol, slot))
            entry: dict = {"loaded": model_set is not None, "last_load_error": errors.get((symbol, slot))}
            if model_set is not None:
                md = model_set.metadata
                point_schema = schema_version(model_set.feature_columns)
                entry |= {
                    "model_version": md.get("model_version"),
                    "schema_version": point_schema,
                    "matches_expected": point_schema == expected[symbol],
                    "quantile": {
                        "status": model_set.quantile_status,
                        "quantile_model_version": md.get("quantile_model_version"),
                        "quantile_schema_version": md.get("quantile_schema_version"),
                    },
                }
            slots[slot] = entry
        models[symbol] = slots
    return {"expected_schema_version": expected, "models": models}


def poll_once(symbols: list[str]) -> None:
    for symbol in symbols:
        for slot in SLOTS:
            try:
                fetch_and_register(symbol, slot)
            except Exception:
                logger.exception("Failed to poll %s/%s", symbol, slot)


def poll_loop(stop_event: threading.Event, interval_seconds: int) -> None:
    """Poll immediately, then every `interval_seconds` until `stop_event` is
    set. Uses Event.wait rather than sleep so shutdown doesn't block for up
    to a full interval.
    """
    while not stop_event.is_set():
        try:
            poll_once(list_symbols())
        except Exception:
            logger.exception("Model poll cycle failed")
        stop_event.wait(interval_seconds)
