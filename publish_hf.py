"""Publish the ONNX artifacts to a Hugging Face model repo.

    hf auth login                      # once — it needs your token
    python publish_hf.py <username>

The model repo is free and serves the artifacts over plain HTTPS, which is what
`WEIGHTS_URI` points at. The service itself runs on Cloud Run — see DEPLOY.md.
Hugging Face requires a PRO plan to host Docker Spaces, so the Space route is
not used here.

Re-runnable: every step is idempotent.

Uses the huggingface_hub Python API rather than the `hf` CLI on purpose. The CLI
renamed flags between major versions (`--space_sdk` in 0.x, `--sdk` in 1.x), and
which one you get depends on whether a virtualenv is active. The Python API is
stable across both.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo, upload_file

from artifacts import MODEL_FILES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("username", help="your Hugging Face username")
    p.add_argument("--repo", default="mnist-digit-classifier")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    try:
        whoami = HfApi().whoami()["name"]
    except Exception:
        sys.exit("not logged in — run 'hf auth login' first")
    print(f"logged in as {whoami}")

    for filename in MODEL_FILES.values():
        if not os.path.exists(filename):
            sys.exit(f"missing {filename} — run export_onnx.py first")

    model_id = f"{args.username}/{args.repo}"
    weights_uri = f"https://huggingface.co/{model_id}/resolve/main"

    print(f"\n==> model repo: {model_id}")
    create_repo(model_id, repo_type="model", exist_ok=True, private=args.private)
    for filename in MODEL_FILES.values():
        upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=model_id,
            repo_type="model",
            commit_message=f"Add {filename}",
        )
        print(f"    uploaded {filename}")

    print(f"""
Done.  https://huggingface.co/{model_id}

Set this on the service (deploy_cloudrun.sh passes it for you):

    WEIGHTS_URI={weights_uri}
""")


if __name__ == "__main__":
    main()
