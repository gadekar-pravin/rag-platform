from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

from google.cloud import documentai_v1 as documentai
from google.cloud.storage import Client


class _DoneOperation(Protocol):
    def done(self) -> bool:
        ...


@dataclass(frozen=True)
class DocAIConfig:
    project: str
    location: str
    processor_id: str

    @property
    def processor_name(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/processors/{self.processor_id}"


class DocumentAIClient:
    """
    Minimal Document AI helper for:
    - Online OCR for images
    - Batch OCR for PDFs (GCS in → GCS out)
    """

    def __init__(self, *, cfg: DocAIConfig, storage_client: Client) -> None:
        self._cfg = cfg
        self._doc_client = documentai.DocumentProcessorServiceClient()
        self._storage = storage_client

    def ocr_online(self, *, content: bytes, mime_type: str) -> tuple[str, dict[str, Any]]:
        req = documentai.ProcessRequest(
            name=self._cfg.processor_name,
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
        )
        resp = self._doc_client.process_document(request=req)
        text = resp.document.text or ""
        meta = {
            "provider": "documentai",
            "mode": "online",
            "mime_type": mime_type,
            "pages": len(resp.document.pages) if resp.document.pages else None,
        }
        return text, meta

    def ocr_pdf_batch(
        self,
        *,
        input_gcs_uri: str,
        output_gcs_prefix: str,  # gs://bucket/prefix/...
        timeout_s: int = 1800,
        poll_s: float = 5.0,
    ) -> tuple[str, dict[str, Any]]:
        # BatchProcessRequest wants output_config.gcs_destination.uri = "gs://bucket/prefix/"
        if not output_gcs_prefix.endswith("/"):
            output_gcs_prefix += "/"

        gcs_doc = documentai.GcsDocument(gcs_uri=input_gcs_uri, mime_type="application/pdf")
        input_docs = documentai.BatchDocumentsInputConfig(gcs_documents=documentai.GcsDocuments(documents=[gcs_doc]))
        output_cfg = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=output_gcs_prefix)
        )

        req = documentai.BatchProcessRequest(
            name=self._cfg.processor_name,
            input_documents=input_docs,
            document_output_config=output_cfg,
        )
        op = cast(_DoneOperation, self._doc_client.batch_process_documents(request=req))

        start = time.time()
        while not op.done():
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Document AI batch OCR timed out after {timeout_s}s")
            time.sleep(poll_s)

        # Parse output JSON written by DocAI
        text = self._read_docai_output_text(output_gcs_prefix)

        meta = {
            "provider": "documentai",
            "mode": "batch",
            "input": input_gcs_uri,
            "output_prefix": output_gcs_prefix,
        }
        return text, meta

    def _read_docai_output_text(self, output_gcs_prefix: str) -> str:
        # output_gcs_prefix: gs://bucket/prefix/
        assert output_gcs_prefix.startswith("gs://")
        without = output_gcs_prefix[len("gs://") :]
        bucket, _, prefix = without.partition("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        blobs = list(self._storage.list_blobs(bucket, prefix=prefix))
        # DocAI writes multiple json shards; keep stable order by name
        json_blobs = sorted([b for b in blobs if b.name.lower().endswith(".json")], key=lambda b: b.name)

        parts: list[str] = []
        for b in json_blobs:
            raw = b.download_as_text(encoding="utf-8", errors="ignore")
            # Some outputs have {"document": {...}}; others may be direct Document JSON.
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            doc_obj = obj.get("document") if isinstance(obj, dict) else None
            if doc_obj is None:
                doc_obj = obj

            try:
                doc = documentai.Document.from_json(json.dumps(doc_obj))
                parts.append(doc.text or "")
            except Exception:
                # Fallback: best-effort if JSON shape differs
                t = ""
                if isinstance(doc_obj, dict):
                    t = str(doc_obj.get("text") or "")
                if t:
                    parts.append(t)

        return "\n".join([p for p in parts if p])
