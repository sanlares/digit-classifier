# Deployment

```
   Browser (static page)          Container (Cloud Run / Fly)                 HF model repo
  ┌──────────────────────┐       ┌──────────────────────────────────┐       ┌────────────────┐
  │ canvas 280×280       │ POST  │ FastAPI                          │ GET   │ mlp.onnx       │
  │ → PNG data URL       ├──────►│  /predict  normalise → 28×28     ├──────►│ cnn.onnx       │
  │ ← digit + 10 probs   │       │            ONNX Runtime          │ (once └────────────────┘
  └──────────────────────┘       └──────────────────────────────────┘  on start)
```

Artifacts are pulled on startup into `/tmp/mnist-weights`, so you can publish a
retrained model without rebuilding or redeploying the image. A copy also ships inside
the image as a fallback — see **Artifact resolution** below for why.

## Artifact resolution

On startup, each artifact is resolved in this order:

1. **Local cache** (`WEIGHTS_CACHE`) if this instance already downloaded it.
2. **The registry** at `WEIGHTS_URI`, retried with exponential backoff
   (`WEIGHTS_RETRIES`, default 3). Retries cover 408/425/429 and 5xx; a 404 fails
   fast because it will not fix itself.
3. **The copy bundled in the image**, if the registry cannot be reached.

Step 3 exists because of a real outage. Hugging Face returned `429 Too Many Requests`
to the Cloud Run egress IP — an address shared across Google's infrastructure, so the
limit can be reached by traffic that is not yours. `fetch` raised, startup failed, the
container exited, and every request got a 503 until the registry relented. A
scale-to-zero service re-fetches on every cold start, which makes that exposure
constant rather than rare.

`/health` always reports the winning source, including the failure that caused a
fallback, so "is this serving the model I published?" has an answer.

## Configuration

| Variable | Purpose | Example |
|---|---|---|
| `WEIGHTS_URI` | Where the `.onnx` files live. Unset → local directory. | `s3://my-models/mnist` |
| `S3_ENDPOINT_URL` | Endpoint for non-AWS S3 stores (R2, MinIO). | `https://<acct>.r2.cloudflarestorage.com` |
| `WEIGHTS_CACHE` | Download cache directory. | `/tmp/mnist-weights` |
| `ALLOWED_ORIGINS` | CORS allowlist. Defaults to `*`. | `https://user.github.io` |
| `PORT` | Listen port. Injected by most PaaS. | `8080` |

`WEIGHTS_URI` accepts `s3://`, `https://` and `file://`. Credentials use the standard
AWS chain (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or an attached identity).

## 1. Run locally

```bash
.venv/bin/python -m uvicorn server:app --reload --port 8008
```

Full stack with real object storage, exercising the production code path:

```bash
docker compose up --build
```

`http://localhost:8080` for the app, `http://localhost:9001` for the MinIO console
(`minioadmin` / `minioadmin`).

> Run `export_onnx.py` before the first `docker compose up`. The seed job bind-mounts
> `mlp.onnx` and `cnn.onnx` as **files**; if they do not exist, Docker silently creates
> directories with those names and the upload fails confusingly.

## 2. Publish the artifacts

**Hugging Face model repo** (no cloud account needed, no egress cost):

```bash
hf auth login
python publish_hf.py <your-hf-username>
```

The script prints the `WEIGHTS_URI` to set. Model repos are free and serve files
over plain HTTPS, so no bucket, no credentials and no egress bill.

> Use `publish_hf.py`, not the `hf` CLI directly: the CLI renamed flags between
> major versions (`--space_sdk` in 0.x, `--sdk` in 1.x) and which one you get
> depends on whether your virtualenv is active. The Python API it wraps is
> stable.

**Any S3-compatible bucket:**

```bash
python upload_weights.py --bucket my-models --prefix mnist
```

Add `--endpoint-url https://<account-id>.r2.cloudflarestorage.com` for Cloudflare R2.
The script prints the `WEIGHTS_URI` to set. R2 is the cheapest option here: no egress
fees, and 300 KB of artifacts sits well inside the free tier.

## 3. Deploy the container

> **Build for the right architecture.** On an Apple Silicon Mac `docker build` produces
> an **arm64** image; most hosts run **amd64** and will fail with an exec format error.
> Either let the platform build it (`gcloud run deploy --source .` does), or build
> explicitly:
>
> ```bash
> docker buildx build --platform linux/amd64 -t <registry>/mnist-classifier:1.0 --push .
> ```

### Google Cloud Run (recommended)

Scales to zero, so an idle demo costs nothing, and the free tier covers a portfolio
link comfortably.

```bash
./deploy_cloudrun.sh <personal-gcp-project-id>
```

The script requires the project id as an argument instead of reading it from
`gcloud config`, and refuses to deploy to a hardcoded blocklist. On a work laptop
the active gcloud configuration usually points at an employer project, and a
personal demo does not belong there.

It deploys with `--source`, so Cloud Build produces the image — which also sidesteps
the architecture trap above. 512 MiB is enough: ONNX Runtime plus two small graphs
is a fraction of what a torch-based image needed.

### Hugging Face Spaces — needs a paid plan

Docker Spaces are **not** free: `cpu-basic` hosting for Docker and Gradio Spaces
requires a PRO subscription (~$9/month), and the API returns `402 Payment Required`
without it. Only *Static* Spaces are free, and a static Space cannot run this
container. Model repos, however, are free — which is why the artifacts live there
while the service runs elsewhere.

### Fly.io

```bash
fly launch --no-deploy
fly deploy
```

## 4. Hosting the UI separately (optional)

The container serves the page at `/`, which is usually enough. To host the frontend on
GitHub Pages against a remote API instead:

1. Publish `static/index.html`.
2. Point it at the API: `https://user.github.io/repo/?api=https://<host>`
   (or set `window.API_BASE` in the HTML).
3. Lock down CORS: `ALLOWED_ORIGINS=https://user.github.io`.

## Live deployment

  Service:   https://mnist-digit-classifier-723611762821.us-central1.run.app
  Artifacts: https://huggingface.co/dakinga/mnist-digit-classifier

## Threat model

The endpoint is public, unauthenticated, and decodes attacker-supplied images —
that decode is the whole attack surface. What is defended, and how:

| Risk | Control | Where |
|---|---|---|
| Decompression bomb (a 137 KB PNG that expands to 144 Mpx) | Dimensions checked from the header before any pixel buffer is allocated; PIL's global ceiling as a backstop | `security.open_image_within_limits` |
| Memory exhaustion via a large body (Starlette buffers the whole request before a handler runs) | `Content-Length` rejected above 2 MB, before buffering | `security.BodySizeLimitMiddleware` |
| Request flooding / cost amplification | 60 POSTs per client per minute, plus `--max-instances 3` and a 30 s request timeout | `security.RateLimitMiddleware`, `deploy_cloudrun.sh` |
| Malformed or truncated images | Every PIL failure path mapped to `400`, so nothing escapes as a `500` with a stack trace | `security.open_image_within_limits` |

Every limit is tunable through environment variables (`MAX_REQUEST_BYTES`,
`MAX_IMAGE_PIXELS`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`) and is
covered by tests in `tests/test_api.py`.

**Accepted risks.** The rate limiter is in-process, so a client gets the limit
once per running instance — with `--max-instances 3` that is a bounded 3x, and
the alternative is a Redis dependency on a service whose selling point is a
one-second cold start. `X-Forwarded-For` is trusted for client identity because
Cloud Run overwrites it; behind a different proxy that assumption needs
rechecking. CORS defaults to `*`, which is safe only because the API carries no
credentials or cookies — there is nothing a hostile origin can obtain by calling
it that it could not obtain by calling it directly. The models themselves are
public artifacts, so model extraction is not a concern here.

## Operational notes

- **The artifact cache survives a restart.** `WEIGHTS_CACHE` lives in the container
  filesystem, so after uploading new artifacts a `docker compose restart` will keep
  serving the cached copies. You need a fresh container to pick them up.
- **`/health`** reports the runtime, the execution provider, the loaded models and the
  artifact source — including whether they came from the cache. It is the fastest way
  to confirm a deploy is running the model you think it is.
