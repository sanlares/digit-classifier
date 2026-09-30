"""Artifact resolution: cache, retries, and the bundled fallback.

The fallback test reproduces a real outage. Hugging Face returned 429 to the
Cloud Run egress IP, `fetch` raised, startup failed, and the container exited —
the service was down until the registry relented. These tests pin the behaviour
that makes that a log line instead.
"""

import urllib.error
from pathlib import Path

import pytest

import weights


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point weights at an http registry with a private cache directory."""
    monkeypatch.setattr(weights, "WEIGHTS_URI", "https://registry.example/models")
    monkeypatch.setattr(weights, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(weights, "REPO_DIR", tmp_path / "image")
    (tmp_path / "image").mkdir()
    monkeypatch.setattr(weights, "RETRIES", 3)
    # Keep the backoff from actually sleeping through the test run.
    monkeypatch.setattr(weights.time, "sleep", lambda _: None)
    return tmp_path


def http_error(code):
    return urllib.error.HTTPError("https://registry.example", code, "boom", {}, None)


def test_successful_download(registry, monkeypatch):
    def fake(_parsed, filename, destination):
        Path(destination).write_bytes(b"model-bytes")

    monkeypatch.setattr(weights, "_download_once", fake)
    path = weights.fetch("mlp.onnx")

    assert Path(path).read_bytes() == b"model-bytes"
    assert weights.describe() == "https://registry.example/models"


def test_cache_is_reused(registry, monkeypatch):
    calls = []

    def fake(_parsed, filename, destination):
        calls.append(filename)
        Path(destination).write_bytes(b"x")

    monkeypatch.setattr(weights, "_download_once", fake)
    weights.fetch("mlp.onnx")
    weights.fetch("mlp.onnx")

    assert len(calls) == 1, "second call should hit the cache"
    assert "cached" in weights.describe()


def test_retries_then_succeeds(registry, monkeypatch):
    attempts = []

    def flaky(_parsed, filename, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(429)
        Path(destination).write_bytes(b"ok")

    monkeypatch.setattr(weights, "_download_once", flaky)
    weights.fetch("cnn.onnx")

    assert len(attempts) == 3
    assert weights.describe() == "https://registry.example/models"


def test_429_falls_back_to_bundled_copy(registry, monkeypatch):
    """The outage case: the registry rate-limits us on every attempt."""
    (registry / "image" / "cnn.onnx").write_bytes(b"bundled")

    def always_429(_parsed, filename, destination):
        raise http_error(429)

    monkeypatch.setattr(weights, "_download_once", always_429)
    path = weights.fetch("cnn.onnx")

    assert Path(path).read_bytes() == b"bundled"
    assert "bundled in image" in weights.describe()


def test_fallback_when_no_bundled_copy_raises(registry, monkeypatch):
    """With nothing to fall back to, failing loudly is still correct."""
    monkeypatch.setattr(weights, "_download_once",
                        lambda *_: (_ for _ in ()).throw(http_error(503)))
    with pytest.raises(urllib.error.HTTPError):
        weights.fetch("missing.onnx")


def test_404_is_not_retried(registry, monkeypatch):
    """A missing artifact is permanent — retrying only delays the failure."""
    (registry / "image" / "typo.onnx").write_bytes(b"bundled")
    attempts = []

    def not_found(_parsed, filename, destination):
        attempts.append(1)
        raise http_error(404)

    monkeypatch.setattr(weights, "_download_once", not_found)
    weights.fetch("typo.onnx")

    assert len(attempts) == 1, "404 should fail fast, not retry"


def test_partial_download_is_cleaned_up(registry, monkeypatch):
    """A failed attempt must not leave a .part file that looks like a model."""
    (registry / "image" / "cnn.onnx").write_bytes(b"bundled")

    def write_then_fail(_parsed, filename, destination):
        Path(destination).write_bytes(b"half")
        raise http_error(500)

    monkeypatch.setattr(weights, "_download_once", write_then_fail)
    weights.fetch("cnn.onnx")

    leftovers = list((registry / "cache").glob("*.part"))
    assert leftovers == []


def test_no_weights_uri_uses_local_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "WEIGHTS_URI", "")
    monkeypatch.setattr(weights, "REPO_DIR", tmp_path)
    assert weights.fetch("mlp.onnx") == str(tmp_path / "mlp.onnx")
    assert weights.describe() == "local"
