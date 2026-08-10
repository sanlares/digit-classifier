"""HTTP contract of the service.

Covers the response shape the browser UI depends on, plus the request-level
limits from `security.py`. Each limit test corresponds to a hole that was
verified against a running instance before the control existed.
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import security
import server
from digits import draw_digit


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


def png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models"] == ["MLP", "CNN"]
    assert body["runtime"].startswith("onnxruntime")
    assert "weights_source" in body


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<canvas" in response.text


def test_placeholder_is_a_png_data_url(client):
    image = client.get("/placeholder").json()["image"]
    assert image.startswith("data:image/png;base64,")


def test_predict_response_shape(client):
    response = client.post("/predict", json={"image": png_b64(draw_digit(7))})
    assert response.status_code == 200
    body = response.json()

    assert body["empty"] is False
    assert body["preview"].startswith("data:image/png;base64,")
    for name in ("MLP", "CNN"):
        model = body[name]
        assert model["pred"] == 7
        assert len(model["probs"]) == 10
        assert sum(model["probs"]) == pytest.approx(1.0, abs=1e-3)
        assert model["no_signal"] is False


def test_predict_accepts_data_url_prefix(client):
    payload = "data:image/png;base64," + png_b64(draw_digit(1))
    assert client.post("/predict", json={"image": payload}).status_code == 200


def test_blank_canvas_reports_empty(client):
    blank = Image.new("RGB", (280, 280), "white")
    assert client.post("/predict", json={"image": png_b64(blank)}).json()["empty"] is True


def test_multipart_upload(client):
    buf = io.BytesIO()
    draw_digit(2).save(buf, format="PNG")
    buf.seek(0)
    response = client.post("/predict/upload", files={"file": ("d.png", buf, "image/png")})
    assert response.status_code == 200
    assert response.json()["CNN"]["pred"] == 2


# --- request-level limits ---------------------------------------------------


def test_malformed_base64_is_rejected(client):
    assert client.post("/predict", json={"image": "not base64!!"}).status_code == 400


def test_corrupt_png_is_rejected(client):
    truncated = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
    assert client.post("/predict", json={"image": truncated}).status_code == 400


def test_decompression_bomb_is_rejected(client):
    """A 137 KB PNG that decodes to 144 megapixels used to be served with a 200."""
    bomb = Image.new("L", (12000, 12000), 0)
    response = client.post("/predict", json={"image": png_b64(bomb)})
    assert response.status_code == 400
    assert "large" in response.json()["detail"].lower() or \
           "exceeds" in response.json()["detail"].lower()


def test_image_over_pixel_budget_is_rejected(client):
    oversized = Image.new("L", (2500, 2500), 0)  # 6.25 Mpx > the 4 Mpx budget
    assert client.post("/predict", json={"image": png_b64(oversized)}).status_code == 400


def test_oversized_body_is_rejected_before_buffering(client):
    payload = "A" * (security.MAX_REQUEST_BYTES + 1024)
    assert client.post("/predict", json={"image": payload}).status_code == 413


def test_rate_limit_returns_429():
    """Exercised on a throwaway app.

    Running this against the real service would burn its rate-limit budget and
    make every later test in the suite order-dependent.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(security.RateLimitMiddleware, limit=3, window=60)

    @app.post("/echo")
    def echo():
        return {"ok": True}

    with TestClient(app) as c:
        codes = [c.post("/echo").status_code for _ in range(5)]

    assert codes == [200, 200, 200, 429, 429]


def test_rate_limit_ignores_get_requests():
    """Only POSTs cost anything; the UI polls /health and /placeholder freely."""
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(security.RateLimitMiddleware, limit=2, window=60)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    with TestClient(app) as c:
        assert all(c.get("/ping").status_code == 200 for _ in range(10))
