"""Upload the exported ONNX artifacts to an S3-compatible bucket.

Works with AWS S3, Cloudflare R2, MinIO, or GCS (via its S3 interoperability
endpoint) — the only difference is --endpoint-url and the credentials in your
environment.

    # AWS S3
    python upload_weights.py --bucket my-models --prefix mnist

    # Cloudflare R2
    python upload_weights.py --bucket my-models --prefix mnist \
        --endpoint-url https://<account-id>.r2.cloudflarestorage.com

    # Local MinIO (docker compose does this for you)
    python upload_weights.py --bucket models --prefix mnist \
        --endpoint-url http://localhost:9000

Prints the WEIGHTS_URI to set on the service.
"""

import argparse
import os

import boto3

from artifacts import MODEL_FILES

FILES = list(MODEL_FILES.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="mnist", help="key prefix inside the bucket")
    p.add_argument("--endpoint-url", default=os.environ.get("S3_ENDPOINT_URL"))
    p.add_argument("--create", action="store_true", help="create the bucket if missing")
    args = p.parse_args()

    s3 = boto3.client("s3", endpoint_url=args.endpoint_url)

    if args.create:
        try:
            s3.create_bucket(Bucket=args.bucket)
            print(f"created bucket {args.bucket}")
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass

    prefix = args.prefix.strip("/")
    for f in FILES:
        if not os.path.exists(f):
            raise SystemExit(f"missing {f} — run export_onnx.py first")
        key = f"{prefix}/{f}" if prefix else f
        s3.upload_file(f, args.bucket, key)
        print(f"uploaded {f} -> s3://{args.bucket}/{key}")

    uri = f"s3://{args.bucket}/{prefix}" if prefix else f"s3://{args.bucket}"
    print(f"\nSet on the service:  WEIGHTS_URI={uri}")
    if args.endpoint_url:
        print(f"                     S3_ENDPOINT_URL={args.endpoint_url}")


if __name__ == "__main__":
    main()
