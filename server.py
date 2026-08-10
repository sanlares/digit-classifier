"""Prediction service for the MNIST digit classifiers.

The same app runs locally and in the container:

    uvicorn server:app --reload --port 8008        # local
    docker compose up                              # containerised

Endpoints
    GET  /                drawing UI (static/index.html)
    GET  /health          runtime, loaded models, where the artifacts came from
    POST /predict         {"image": "data:image/png;base64,..."}  -> predictions
    POST /predict/upload  multipart file upload (field name: "file")

The service is public and unauthenticated, so every request-level limit lives in
`security.py` and is applied here as middleware.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

import inference
import preprocess
import security
import weights
from digits import draw_digit

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Comma-separated allowlist. "*" is safe here because the API carries no
# credentials and no cookies: there is nothing for a hostile origin to steal by
# calling it that it could not get by calling it directly. Set explicitly if the
# UI is hosted separately and you want to keep other origins out.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

PLACEHOLDER = ""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the models once, at startup.

    Doing it here rather than lazily on the first request means an unreachable
    artifact store fails the container immediately, instead of serving a 500 to
    whoever happens to arrive first.
    """
    global PLACEHOLDER
    inference.load_sessions()
    PLACEHOLDER = _placeholder_data_url()
    logger.info(
        "models loaded via %s (artifacts: %s)", inference.runtime(), weights.describe()
    )
    yield


app = FastAPI(title="MNIST digit classifier", version="2.1", lifespan=lifespan)

app.add_middleware(security.RateLimitMiddleware)
app.add_middleware(security.BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ImagePayload(BaseModel):
    """A PNG as a data URL or bare base64.

    `max_length` is a cheap second gate behind the body-size middleware: it
    bounds the decoded string even if the middleware is ever bypassed.
    """

    image: str = Field(..., max_length=security.MAX_REQUEST_BYTES)


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _placeholder_data_url(digit: int = 3) -> str:
    """Sample digit in 28x28 format, used to fill the thumbnail on first load."""
    arr = preprocess.to_mnist_array(draw_digit(digit), ink_is_dark=True)
    return _png_data_url(preprocess.array_to_image(arr))


def _run(image: Image.Image) -> dict:
    """Preprocess and score one image. Returns the JSON response body."""
    arr = preprocess.to_mnist_array(image, ink_is_dark=True)
    if arr.max() <= 0:
        return {"empty": True}

    x = np.ascontiguousarray(arr, dtype=np.float32).reshape(1, 1, 28, 28)
    out: dict = {
        "empty": False,
        "runtime": inference.runtime(),
        # Kept for backwards compatibility with a separately hosted UI; under
        # onnxruntime there is no torch device, so it carries the provider.
        "device": inference.provider(),
    }
    for name in inference.model_names():
        cls, probs, no_signal = inference.predict(name, x)
        out[name] = {
            "pred": cls,
            "probs": [round(float(p), 5) for p in probs],
            "no_signal": no_signal,
        }
    out["preview"] = _png_data_url(preprocess.array_to_image(arr))
    return out


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if PLACEHOLDER else "loading",
        "runtime": inference.runtime(),
        "provider": inference.provider(),
        "models": inference.model_names(),
        "weights_source": weights.describe(),
    }


@app.get("/placeholder")
def placeholder() -> dict:
    return {"image": PLACEHOLDER}


@app.post("/predict")
def predict_json(payload: ImagePayload) -> JSONResponse:
    data = payload.image.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Malformed base64 image")
    return JSONResponse(_run(security.open_image_within_limits(raw)))


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)) -> JSONResponse:
    # The body-size middleware has already capped how much can arrive, so
    # reading the upload whole is bounded.
    raw = await file.read()
    return JSONResponse(_run(security.open_image_within_limits(raw)))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
