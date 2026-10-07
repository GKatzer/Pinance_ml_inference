import json

import pytest
from minio.error import S3Error

from pinance_ml_inference import minio_client


class FakeResponse:
    def __init__(self, data: bytes):
        self.data = data

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinioClient:
    def __init__(self, objects: dict[str, bytes] | None = None, error: Exception | None = None):
        self.objects = objects or {}
        self.error = error

    def get_object(self, bucket_name, object_name):
        if self.error is not None:
            raise self.error
        if object_name not in self.objects:
            raise S3Error(
                response=None,
                code="NoSuchKey",
                message="not found",
                resource=object_name,
                request_id="req",
                host_id="host",
            )
        return FakeResponse(self.objects[object_name])


def test_fetch_metadata_returns_none_when_object_missing(monkeypatch):
    monkeypatch.setattr(minio_client, "_client", lambda: FakeMinioClient())

    assert minio_client.fetch_metadata("BTCUSDT", "candidate") is None


def test_fetch_metadata_parses_json_when_present(monkeypatch):
    metadata = {"model_version": "v1", "feature_columns": ["a"], "horizons": [1]}
    fake = FakeMinioClient({"BTCUSDT/production/metadata.json": json.dumps(metadata).encode()})
    monkeypatch.setattr(minio_client, "_client", lambda: fake)

    assert minio_client.fetch_metadata("BTCUSDT", "production") == metadata


def test_fetch_metadata_propagates_non_not_found_errors(monkeypatch):
    fake = FakeMinioClient(
        error=S3Error(
            response=None,
            code="AccessDenied",
            message="nope",
            resource="x",
            request_id="req",
            host_id="host",
        )
    )
    monkeypatch.setattr(minio_client, "_client", lambda: fake)

    with pytest.raises(S3Error):
        minio_client.fetch_metadata("BTCUSDT", "production")


def test_booster_key_median_has_no_suffix():
    assert minio_client._booster_key("BTCUSDT", "production", 1, 0.5) == "BTCUSDT/production/h1.txt"


def test_booster_key_suffixes_tail_quantiles():
    """Literal quantile value, matching export_models.py's own
    f"h{h}_q{q}.txt" -- not a rounded percentage."""
    assert minio_client._booster_key("BTCUSDT", "production", 1, 0.1) == "BTCUSDT/production/h1_q0.1.txt"
    assert minio_client._booster_key("BTCUSDT", "production", 1, 0.9) == "BTCUSDT/production/h1_q0.9.txt"


def test_fetch_booster_text_decodes_bytes(monkeypatch):
    fake = FakeMinioClient(
        {
            "BTCUSDT/production/h1.txt": b"median-content",
            "BTCUSDT/production/h1_q0.1.txt": b"q10-content",
            "BTCUSDT/production/h1_q0.9.txt": b"q90-content",
        }
    )
    monkeypatch.setattr(minio_client, "_client", lambda: fake)

    assert minio_client.fetch_booster_text("BTCUSDT", "production", 1, 0.5) == "median-content"
    assert minio_client.fetch_booster_text("BTCUSDT", "production", 1, 0.1) == "q10-content"
    assert minio_client.fetch_booster_text("BTCUSDT", "production", 1, 0.9) == "q90-content"
