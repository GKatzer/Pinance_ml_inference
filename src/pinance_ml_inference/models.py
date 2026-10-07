import hashlib
from dataclasses import dataclass

import lightgbm as lgb


class SchemaMismatchError(ValueError):
    """Raised when a model's recorded feature list doesn't match what this
    service's vendored feature code currently computes.

    Means Pinance_ML's feature pipeline moved past the vendored commit
    without vendor/features.py being re-synced (see vendor/LOCK.md) -- or
    the pushed model is stale. Either way, predicting anyway would feed the
    model a feature vector shaped differently from what it was trained on.
    """


QUANTILE_LOADED = "loaded"
QUANTILE_ABSENT = "absent"
QUANTILE_IGNORED = "ignored_schema_mismatch"


def schema_version(feature_columns: list[str] | tuple[str, ...]) -> str:
    """Fingerprint of an ordered feature-name list. Must stay identical to
    Pinance_ML's scripts/export_models.py `_schema_version` (sha256 of the
    comma-joined names, first 12 hex chars) -- order matters, it's not a set
    hash. tests/test_models.py pins shared vectors against training.
    """
    return hashlib.sha256(",".join(feature_columns).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ModelSet:
    symbol: str
    feature_columns: list[str]
    boosters: dict[tuple[int, float], lgb.Booster]  # keyed by (horizon, quantile), e.g. (1, 0.5)
    metadata: dict
    quantiles: tuple[float, ...] = (0.5,)
    quantile_status: str = QUANTILE_ABSENT
    quantile_error: str | None = None  # why the corridor was ignored (hash or booster-name diff)


def quantile_status_from_metadata(metadata: dict) -> str:
    """Whether a slot's corridor (q10/q90) may be served alongside its point
    model. The corridor has no feature list of its own, only
    `quantile_schema_version`, so it's usable iff that equals the fingerprint
    of the point model's `feature_columns` -- i.e. it was trained on exactly
    the features we're about to feed. Point and corridor are pushed on
    independent schedules, so a corridor lagging a schema behind is a normal
    state: it gets ignored (point keeps serving), never a load failure. A
    missing `quantile_schema_version` can't be verified, so it's ignored too.
    """
    if not metadata.get("quantile_levels"):
        return QUANTILE_ABSENT
    if metadata.get("quantile_schema_version") == schema_version(metadata.get("feature_columns") or []):
        return QUANTILE_LOADED
    return QUANTILE_IGNORED


def quantiles_from_metadata(metadata: dict) -> tuple[float, ...]:
    """Which quantile *alphas* a slot serves, per the training-side
    contract (Pinance_ML/src/pinance_ml/model_storage.py docstring): the
    median is never a separate file -- h{h}.txt *is* the point model, and
    `objective="regression_l1"` already minimizes it, so it always doubles
    as q=0.5. Corridor tails are optional: `quantile_levels` absent or
    empty means point-only, same as the pre-corridor format -- not a
    schema mismatch -- but tails whose schema doesn't match are ignored, see
    quantile_status_from_metadata. This says nothing about *which horizons* the tails
    cover -- see booster_keys_from_metadata for that.
    """
    if quantile_status_from_metadata(metadata) != QUANTILE_LOADED:
        return (0.5,)
    return (0.5, *sorted(metadata["quantile_levels"]))


def feature_diff(booster_names: list[str], expected: list[str]) -> str:
    """Human-readable difference between a booster's embedded feature names
    and the expected list. Lists, not sets: LightGBM reads features by
    position, so the same names in another order is also a mismatch.
    """
    only_booster = [n for n in booster_names if n not in expected]
    only_expected = [n for n in expected if n not in booster_names]
    if only_booster or only_expected:
        return f"only in booster: {only_booster}, only in expected: {only_expected}"
    return f"same {len(expected)} names in a different order"


def booster_keys_from_metadata(metadata: dict) -> list[tuple[int, float]]:
    """Exact (horizon, quantile) booster files a slot's metadata implies --
    the single source of truth for both what to fetch from MinIO
    (model_registry.py) and what to assemble into a ModelSet (below), so
    the two can never desync.

    Point and corridor are independently trained pipelines with their own
    horizon lists by construction (model_storage.py's disjoint-keys
    design): the median is keyed by the point pipeline's own `horizons`,
    tails by the quantile pipeline's own `quantile_horizons` (falling back
    to `horizons` for metadata pushed before that key existed). In
    practice both pipelines currently always cover the same 1..12 range,
    but nothing guarantees that stays true.
    """
    keys = [(h, 0.5) for h in metadata["horizons"]]
    quantile_levels = sorted(metadata["quantile_levels"]) if quantile_status_from_metadata(metadata) == QUANTILE_LOADED else []
    if quantile_levels:
        quantile_horizons = metadata.get("quantile_horizons") or metadata["horizons"]
        keys += [(h, q) for h in quantile_horizons for q in quantile_levels]
    return keys


def build_model_set(
    symbol: str,
    metadata: dict,
    booster_texts: dict[tuple[int, float], str],
    expected_feature_columns: tuple[str, ...],
) -> ModelSet:
    """Build one symbol's ModelSet from already-downloaded MinIO content.

    Pure and I/O-free by design -- model_registry.py owns fetching bytes
    from MinIO, this just validates and assembles them -- so it's directly
    unit-testable and reusable for both the production and candidate slots.

    `expected_feature_columns` is passed in (rather than read from config
    here) so this module has no import-time dependency on the vendored
    feature code -- callers own the schema they expect, this module only
    checks it. There's no equivalent "expected quantiles" to check against:
    any `quantile_levels` shape in metadata is valid (see
    quantiles_from_metadata).
    """
    actual_features = metadata["feature_columns"]
    expected_features = list(expected_feature_columns)
    if actual_features != expected_features:
        raise SchemaMismatchError(
            f"{symbol}: model feature schema (source_commit={metadata.get('source_commit')}) "
            f"does not match what this service computes. Model expects {len(actual_features)} columns, "
            f"service computed {len(expected_features)}."
        )

    recorded = metadata.get("schema_version")
    if recorded is not None and recorded != schema_version(actual_features):
        raise SchemaMismatchError(
            f"{symbol}: metadata schema_version={recorded} does not match its own feature_columns "
            f"({schema_version(actual_features)})"
        )

    quantile_status = quantile_status_from_metadata(metadata)
    quantile_error = None
    if quantile_status == QUANTILE_IGNORED:
        quantile_error = (
            f"corridor ignored: quantile_schema_version={metadata.get('quantile_schema_version')} "
            f"!= schema_version={schema_version(actual_features)}"
        )

    boosters = {(h, q): lgb.Booster(model_str=booster_texts[(h, q)]) for h, q in booster_keys_from_metadata(metadata)}

    # metadata can disagree with the files actually pushed (mixed pushes into
    # one slot), and LightGBM embeds feature names in each file -- the files
    # are what really execute, so they're the final word.
    corridor_diff = None
    for (h, q), booster in boosters.items():
        names = booster.feature_name()
        if names == actual_features:
            continue
        diff = feature_diff(names, actual_features)
        if q == 0.5:
            raise SchemaMismatchError(
                f"{symbol}: point model h{h} was trained on different features than metadata.feature_columns "
                f"(schema_version={schema_version(names)} vs {schema_version(actual_features)}): {diff}"
            )
        corridor_diff = corridor_diff or (
            f"corridor ignored: h{h}_q{q} booster schema_version={schema_version(names)} "
            f"!= schema_version={schema_version(actual_features)}: {diff}"
        )
    if corridor_diff is not None:
        boosters = {k: b for k, b in boosters.items() if k[1] == 0.5}
        quantile_status, quantile_error = QUANTILE_IGNORED, corridor_diff

    return ModelSet(
        symbol=symbol,
        feature_columns=actual_features,
        boosters=boosters,
        metadata=metadata,
        quantiles=(0.5, *sorted({q for _, q in boosters if q != 0.5})),
        quantile_status=quantile_status,
        quantile_error=quantile_error,
    )
