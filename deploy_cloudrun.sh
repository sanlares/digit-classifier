#!/usr/bin/env bash
# Deploy the service to Google Cloud Run.
#
#   ./deploy_cloudrun.sh <personal-gcp-project-id> [hf-username]
#
# The project id is required rather than inherited from `gcloud config`, on
# purpose: the active gcloud configuration on a work machine usually points at
# an employer project, and a personal portfolio demo does not belong there.
# BLOCKED_PROJECTS below is a second line of defence.

set -euo pipefail

PROJECT="${1:-}"
HF_USER="${2:-dakinga}"
HF_REPO="${3:-mnist-digit-classifier}"

SERVICE="mnist-digit-classifier"
REGION="${REGION:-us-central1}"

# Projects this script must never deploy to, as a space-separated list.
# Kept out of the repo on purpose — a blocklist of employer project ids is not
# something to publish. Set it in your shell profile:
#
#   export BLOCKED_GCP_PROJECTS="some-work-project another-one"
#
# The real protection is the required argument above: this script never inherits
# a project from `gcloud config`, so a work project can only be hit by typing it.
read -r -a BLOCKED_PROJECTS <<< "${BLOCKED_GCP_PROJECTS:-}"

if [[ -z "$PROJECT" ]]; then
  cat >&2 <<EOF
usage: $0 <personal-gcp-project-id> [hf-username] [hf-repo]

Current gcloud context (NOT used unless you pass it explicitly):
  account: $(gcloud config get-value account 2>/dev/null || echo unknown)
  project: $(gcloud config get-value project 2>/dev/null || echo unknown)
EOF
  exit 1
fi

for blocked in "${BLOCKED_PROJECTS[@]+"${BLOCKED_PROJECTS[@]}"}"; do
  if [[ "$PROJECT" == "$blocked" ]]; then
    echo "refusing to deploy to '$PROJECT' — that is a work project." >&2
    echo "Create a personal project and pass its id instead." >&2
    exit 1
  fi
done

ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "account : $ACCOUNT"
echo "project : $PROJECT"
echo "region  : $REGION"
echo

# ASSUME_YES skips the prompt for scripted runs. The blocklist check above is
# not skippable — that is the guard that actually matters.
if [[ "${ASSUME_YES:-}" != "1" ]]; then
  read -r -p "Deploy '$SERVICE' to this project? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted."; exit 1; }
fi

WEIGHTS_URI="https://huggingface.co/$HF_USER/$HF_REPO/resolve/main"

# --source builds on Cloud Build, which produces a linux/amd64 image. Building
# locally on an Apple Silicon Mac would produce arm64 and fail to start.
# --allow-unauthenticated is intentional: this is a public demo. Abuse is
# bounded by max-instances, the request timeout, and the limits in security.py.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --concurrency 40 \
  --timeout 30s \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8080 \
  --set-env-vars "WEIGHTS_URI=$WEIGHTS_URI"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
        --region "$REGION" --format 'value(status.url)')"

echo
echo "Deployed: $URL"
echo
echo "Checks:"
echo "  curl $URL/health"
echo "  open $URL"
