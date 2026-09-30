# The service resolves its models from the registry named by WEIGHTS_URI at
# startup, so publishing a retrained model is an upload rather than a rebuild.
# A copy of the artifacts still ships here as a fallback: the registry is a
# third party on the startup path, and without a local copy one 429 from it
# takes the whole service down. See weights.py.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEIGHTS_CACHE=/tmp/mnist-weights \
    OMP_NUM_THREADS=1 \
    ORT_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependencies first so the layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Everything the service needs. The training and export scripts are excluded
# in .dockerignore rather than omitted here: an allowlist in this file means a
# newly added serving module is silently left out and the container dies on
# `import` at startup — which is exactly what happened with security.py.
COPY *.py ./
COPY static/ ./static/

# Fallback artifacts. The service resolves models from the registry named by
# WEIGHTS_URI; these are what it serves if that registry is unreachable, which
# is the difference between a stale model and a 503. ~300 KB.
COPY *.onnx ./

# Cloud Run, Hugging Face Spaces and most PaaS inject the port
ENV PORT=8080
EXPOSE 8080

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
