from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_service.ingestion.types import ExtractResult, WorkItem


class Extractor(ABC):
    @abstractmethod
    def can_handle(self, item: WorkItem) -> bool: ...

    @abstractmethod
    def extract(self, *, item: WorkItem, data: bytes) -> ExtractResult: ...


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Remove null bytes, normalize whitespace a bit
    text = text.replace("\x00", "")
    # Collapse very long runs of blank lines
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.strip()