# Code-only image: the .onnx artifacts are NOT baked in, they are pulled from
# object storage at startup (see weights.py). Ship a new model without a rebuild.
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

# Explicit list, not `COPY *.py`: the training and export scripts import torch,
# which is not installed here. Copying them in would be dead weight at best and
# a confusing ImportError at worst.
COPY server.py inference.py preprocess.py weights.py artifacts.py digits.py ./
COPY static/ ./static/

# Cloud Run, Hugging Face Spaces and most PaaS inject the port
ENV PORT=8080
EXPOSE 8080

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
