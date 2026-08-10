"""Resolve model weights from object storage, with a local fallback.

The container image ships *code only* — the ONNX artifacts live in object
storage and are pulled on startup into a local cache. That keeps the image
small, lets you ship new models without rebuilding, and is the usual split
between an artifact registry and a model registry.

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

Credentials follow the standard AWS chain (env vars, instance role, etc.).
"""

import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REPO_DIR = Path(__file__).resolve().parent

WEIGHTS_URI = os.environ.get("WEIGHTS_URI", "").rstrip("/")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL") or None
CACHE_DIR = Path(os.environ.get("WEIGHTS_CACHE", "/tmp/mnist-weights"))

# Filled in by fetch(), reported by the /health endpoint
last_source = "local"


def _s3_client():
    import boto3  # imported lazily: only needed when using s3://

    return boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)


def fetch(filename, force=False):
    """Return a local path to `filename`, downloading it if necessary."""
    global last_source

    if not WEIGHTS_URI:
        last_source = "local"
        return str(REPO_DIR / filename)

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

    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = "/".join(p for p in [parsed.path.strip("/"), filename] if p)
        _s3_client().download_file(bucket, key, str(tmp))
    elif parsed.scheme in ("http", "https"):
        url = f"{WEIGHTS_URI}/{filename}"
        with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        raise ValueError(f"Unsupported WEIGHTS_URI scheme: {WEIGHTS_URI!r}")

    tmp.replace(target)
    last_source = WEIGHTS_URI
    return str(target)


def describe():
    """Human-readable description of where the weights came from."""
    return last_source
