from __future__ import annotations

from collections.abc import Iterable

from google.cloud import storage


def gs_uri(bucket: str, name: str) -> str:
    return f"gs://{bucket}/{name}"


def list_tenant_prefixes(client: storage.Client, bucket: str) -> list[str]:
    """
    Returns top-level prefixes as tenant IDs by listing with delimiter='/'
    """
    b = client.bucket(bucket)
    it = client.list_blobs(b, delimiter="/", prefix="")
    # Force iteration to populate prefixes
    for _ in it:
        pass
    prefixes = getattr(it, "prefixes", set()) or set()
    out: list[str] = []
    for p in prefixes:
        p = p.strip("/")
        if p:
            out.append(p)
    out.sort()
    return out


def list_objects(client: storage.Client, bucket: str, prefix: str) -> Iterable[storage.Blob]:
    b = client.bucket(bucket)
    return client.list_blobs(b, prefix=prefix)


def download_bytes(client: storage.Client, bucket: str, name: str) -> bytes:
    b = client.bucket(bucket)
    blob = b.blob(name)
    return blob.download_as_bytes()


def upload_text(client: storage.Client, bucket: str, name: str, text: str, *, content_type: str = "text/plain") -> None:
    b = client.bucket(bucket)
    blob = b.blob(name)
    blob.upload_from_string(text, content_type=content_type)
