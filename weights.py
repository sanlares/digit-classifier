"""Resolve model artifacts from a model registry, with a bundled fallback.

The registry is the source of truth: publishing a retrained model is an upload,
not a rebuild. But the registry is also a third party on the startup path, and a
scale-to-zero service re-fetches on every cold start. When Hugging Face
rate-limited the shared Cloud Run egress IP with a 429, startup raised, the
container exited, and the service went down — an outage caused by a dependency
that was never ours to control.

So the resolution order is:

1. Local cache, if this instance already downloaded the file.
2. The registry, retried with exponential backoff on transient failures.
3. The copy baked into the image, as a last resort.

Step 3 is the one that makes the difference: a registry hiccup becomes a log
line and a slightly stale model, not a 503. `describe()` reports which step won,
and /health surfaces it, so "is it serving the model I published?" is always
answerable.

Configure with environment variables:

    WEIGHTS_URI       Base location of the .onnx artifacts. Supported schemes:
                        s3://bucket/prefix     any S3-compatible store
                        https://host/prefix    plain HTTP (a public bucket, or a
                                               Hugging Face model repo)
                        file:///abs/path       local directory
                      Unset -> the repo directory (local development).

    S3_ENDPOINT_URL   Custom endpoint for non-AWS S3 stores, e.g.
                      Cloudflare R2 or a local MinIO. Ignored for AWS.

    WEIGHTS_CACHE     Where downloads are cached. Default: /tmp/mnist-weights

    WEIGHTS_RETRIES   Download attempts before falling back. Default: 3.
"""

from __future__ import annotations

import logging
import shutil
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent

WEIGHTS_URI = os.environ.get("WEIGHTS_URI", "").rstrip("/")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or None
CACHE_DIR = Path(os.environ.get("WEIGHTS_CACHE", "/tmp/mnist-weights"))
RETRIES = int(os.environ.get("WEIGHTS_RETRIES", 3))

# HTTP statuses worth retrying: rate limiting and transient server faults.
# A 404 means the artifact is not there and never will be, so it fails fast.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

# Set by fetch(), reported by /health.
last_source = "local"


def _s3_client():
    import boto3  # imported lazily: only needed when using s3://

    return boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)


def _download_once(parsed, filename: str, destination: Path) -> None:
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = "/".join(p for p in [parsed.path.strip("/"), filename] if p)
        _s3_client().download_file(bucket, key, str(destination))
    elif parsed.scheme in ("http", "https"):
        url = f"{WEIGHTS_URI}/{filename}"
        with urllib.request.urlopen(url, timeout=30) as response, \
                open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    else:
        raise ValueError(f"Unsupported WEIGHTS_URI scheme: {WEIGHTS_URI!r}")


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_STATUS
    # Timeouts, DNS failures, resets: all worth another go.
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


def _download_with_retry(parsed, filename: str, destination: Path) -> None:
    """Download with exponential backoff. Raises the last error if all fail."""
    for attempt in range(1, RETRIES + 1):
        try:
            _download_once(parsed, filename, destination)
            return
        except Exception as error:  # noqa: BLE001 - re-raised below
            if attempt == RETRIES or not _is_retryable(error):
                raise
            delay = 2 ** (attempt - 1)  # 1s, 2s, 4s...
            logger.warning(
                "fetching %s failed (attempt %d/%d): %s — retrying in %ds",
                filename, attempt, RETRIES, error, delay,
            )
            time.sleep(delay)


def fetch(filename: str, force: bool = False) -> str:
    """Return a local path to `filename`, downloading it if necessary."""
    global last_source

    bundled = REPO_DIR / filename

    if not WEIGHTS_URI:
        last_source = "local"
        return str(bundled)

    parsed = urlparse(WEIGHTS_URI)

    if parsed.scheme == "file":
        last_source = WEIGHTS_URI
        return str(Path(parsed.path) / filename)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / filename
    if target.exists() and not force:
        last_source = f"{WEIGHTS_URI} (cached)"
        return str(target)

    tmp = target.with_suffix(target.suffix + ".part")
    try:
        _download_with_retry(parsed, filename, tmp)
    except Exception as error:  # noqa: BLE001 - degraded, not fatal
        tmp.unlink(missing_ok=True)
        if bundled.exists():
            # The registry is unreachable but the image carries a known-good
            # copy. Serving a possibly stale model beats returning 503.
            logger.error(
                "could not fetch %s from %s (%s) — falling back to the copy "
                "bundled in the image", filename, WEIGHTS_URI, error,
            )
            last_source = f"bundled in image (registry unreachable: {error})"
            return str(bundled)
        raise

    tmp.replace(target)
    last_source = WEIGHTS_URI
    return str(target)


def describe() -> str:
    """Where the artifacts actually came from, for /health."""
    return last_source
