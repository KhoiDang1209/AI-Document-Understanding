"""MinIO object storage for uploaded images via a boto3 S3 client."""

from __future__ import annotations

from typing import Any

from docintel.config import Settings


def make_s3_client(settings: Settings) -> Any:
    """Build a boto3 S3 client pointed at the MinIO endpoint."""
    import boto3

    scheme = "https" if settings.minio_secure else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """Create ``bucket`` if it does not already exist.

    ``head_bucket`` raises ``ClientError`` with a 404/NoSuchBucket code when the
    bucket is absent; only that case is recovered by creating it. Other errors
    (forbidden, endpoint unreachable, bad credentials) propagate so the real
    cause surfaces instead of a confusing follow-on ``create_bucket`` failure.
    """
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchBucket", "NotFound"}:
            client.create_bucket(Bucket=bucket)
        else:
            raise


def put_image(client: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
    """Store image bytes under ``key``."""
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_image(client: Any, bucket: str, key: str) -> bytes | None:
    """Fetch image bytes for ``key``; return None if the object is absent."""
    from botocore.exceptions import ClientError

    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return None
    data: bytes = response["Body"].read()
    return data
