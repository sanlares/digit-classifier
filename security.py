"""Request-level safety controls for a public, unauthenticated endpoint.

The service accepts arbitrary user-supplied images and decodes them, which is
the entire attack surface. Three controls, each closing a hole verified against
the running service:

1. Request body size — Starlette buffers the whole body before a handler sees
   it, so a 100 MB payload is 100 MB of resident memory even when the handler
   ultimately rejects it.
2. Image dimensions — a 137 KB PNG can decode to 144 megapixels. PIL only warns
   above `MAX_IMAGE_PIXELS` and raises above twice that, and the raise happens
   *after* allocation is attempted. Checking the header before decoding avoids
   allocating at all.
3. Request rate — bounds how fast one client can spend the instance's CPU.

None of this replaces an edge WAF or API gateway. It is the layer that belongs
in the application, sized for a demo service running with a small instance cap.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# A 280x280 canvas PNG is a few KB; 2 MB leaves generous headroom for a photo
# uploaded to /predict/upload while staying far below the instance memory.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", 2 * 1024 * 1024))

# 4 megapixels ~ a 2000x2000 image. Anything larger is not a handwritten digit.
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 4_000_000))

# Requests per client per window. Generous for a human drawing digits.
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 60))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", 60))

# Belt and braces: PIL's own global ceiling, in case an image is opened
# somewhere that forgets to go through open_image_within_limits.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies before they are buffered into memory."""

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            raw_length = request.headers.get("content-length")
            if raw_length is None:
                # Every client this API serves sends Content-Length. Refusing
                # the alternative avoids having to stream-count an unbounded
                # chunked body just to enforce the same limit.
                return JSONResponse(
                    {"detail": "Content-Length required"}, status_code=411
                )
            try:
                length = int(raw_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Malformed Content-Length"}, status_code=400
                )
            if length > MAX_REQUEST_BYTES:
                logger.warning(
                    "rejected %d byte body from %s", length, client_key(request)
                )
                return JSONResponse(
                    {"detail": f"Request body exceeds {MAX_REQUEST_BYTES} bytes"},
                    status_code=413,
                )
        return await call_next(request)


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Behind Cloud Run (or any reverse proxy) the socket peer is the proxy, and
    the real client is the first entry of X-Forwarded-For. That header is
    spoofable in general; it is trustworthy only because the platform overwrites
    it. Do not reuse this for anything security-critical such as authorisation.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window request cap per client.

    In-process and therefore per-instance: with N instances a client gets N
    times the limit. That is an accepted trade for a demo — it turns a trivial
    flood into something that has to be deliberate, without adding a Redis
    dependency to a service whose whole point is that it starts in a second.
    """

    def __init__(self, app, limit: int = RATE_LIMIT_REQUESTS,
                 window: int = RATE_LIMIT_WINDOW_SECONDS):
        super().__init__(app)
        self.limit = limit
        self.window = window
        self._hits: dict[str, Deque[float]] = defaultdict(deque)

    def _prune(self, now: float) -> None:
        """Drop clients with no recent activity so the map cannot grow forever."""
        stale = [key for key, hits in self._hits.items()
                 if not hits or now - hits[-1] > self.window]
        for key in stale:
            del self._hits[key]

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        now = time.monotonic()
        key = client_key(request)
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = int(self.window - (now - hits[0])) + 1
            logger.warning("rate limited %s", key)
            return JSONResponse(
                {"detail": "Too many requests"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        if len(self._hits) > 1024:
            self._prune(now)
        return await call_next(request)


def open_image_within_limits(raw: bytes) -> Image.Image:
    """Decode `raw` into an RGB image composited over white.

    Raises HTTPException(400) for anything that is not a decodable image of a
    sane size. `Image.open` only parses the header, so the dimension check
    happens before any pixel buffer is allocated — which is the point.
    """
    try:
        image = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError:
        # Setting Image.MAX_IMAGE_PIXELS above makes PIL raise this from open()
        # itself, at header-parse time, for anything past twice the ceiling.
        # DecompressionBombError does not inherit from OSError or ValueError, so
        # it needs its own clause or it escapes as a 500.
        raise HTTPException(status_code=400, detail="Image too large")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Not a valid image")

    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {MAX_IMAGE_PIXELS} pixels ({width}x{height})",
        )

    try:
        image = image.convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError):
        # Truncated or malformed payloads only fail here, once decoding starts.
        raise HTTPException(status_code=400, detail="Corrupt image data")

    # The browser canvas is RGBA; anything undrawn is transparent, which would
    # read as black once converted to grayscale.
    flattened = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flattened.alpha_composite(image)
    return flattened.convert("RGB")
