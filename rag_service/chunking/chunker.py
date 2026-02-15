"""Document chunking for RAG (Retrieval Augmented Generation).

Ported from ApexFlow's core/rag/chunker.py with the settings_loader dependency
removed — all config is accepted as function parameters.

Two strategies:
1) Rule-Based (Recursive): Fast, dependency-free hierarchical splitting.
2) Semantic (Gemini): LLM-driven topic-shift detection.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 2000
_DEFAULT_CHUNK_OVERLAP = 200
_DEFAULT_SEMANTIC_BLOCK_WORDS = 1024
_DEFAULT_SEMANTIC_MODEL = "gemini-2.5-flash-lite"

_SEPARATORS: list[str] = ["\n\n", "\n", ". ", ".\n", "? ", "! ", " ", ""]


async def chunk_document(
    text: str,
    method: str = "rule_based",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    *,
    semantic_model: str = _DEFAULT_SEMANTIC_MODEL,
    semantic_block_words: int = _DEFAULT_SEMANTIC_BLOCK_WORDS,
) -> list[str]:
    """Chunk a document into segments for embedding.

    Args:
        text: Full document text.
        method: "rule_based" (default) or "semantic" (LLM-driven).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap in characters carried between adjacent chunks.
        semantic_model: Gemini model for semantic chunking.
        semantic_block_words: Word block size for semantic chunking.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    chunk_size, chunk_overlap = _validate_chunk_params(chunk_size, chunk_overlap)

    if method == "semantic":
        try:
            chunks = await _chunk_semantic(text, semantic_model, semantic_block_words)
            return _enforce_max_chunk_size(chunks, chunk_size, chunk_overlap)
        except Exception:
            logger.exception("Semantic chunking failed; falling back to rule_based")
            return _chunk_recursive(text, chunk_size, chunk_overlap)

    return _chunk_recursive(text, chunk_size, chunk_overlap)


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
    """Sanitize chunk parameters."""
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if chunk_overlap < 0:
        logger.warning("chunk_overlap < 0 (%d); clamping to 0", chunk_overlap)
        chunk_overlap = 0

    if chunk_overlap >= chunk_size:
        new_overlap = max(min(chunk_size // 2, chunk_size - 1), 0)
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d); clamping overlap to %d",
            chunk_overlap,
            chunk_size,
            new_overlap,
        )
        chunk_overlap = new_overlap

    return chunk_size, chunk_overlap


# ---------------------------------------------------------------------------
# Rule-based recursive splitter
# ---------------------------------------------------------------------------


def _chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Hierarchical recursive splitting with inter-chunk overlap."""
    raw = _split_text_recursive(text, _SEPARATORS, chunk_size, chunk_overlap)

    if chunk_overlap <= 0 or len(raw) <= 1:
        return raw

    overlapped: list[str] = [raw[0]]
    for i in range(1, len(raw)):
        prev = raw[i - 1]
        tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        if tail and not raw[i].startswith(tail):
            overlapped.append(tail + raw[i])
        else:
            overlapped.append(raw[i])

    return overlapped


def _split_text_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text, guaranteeing each chunk <= chunk_size."""
    if len(text) <= chunk_size or not separators:
        return [text]

    separator = separators[0]
    next_separators = separators[1:]

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if separator not in text:
        return _split_text_recursive(text, next_separators, chunk_size, chunk_overlap)

    splits = text.split(separator)
    parts: list[str] = []
    for i, s in enumerate(splits):
        if i < len(splits) - 1:
            parts.append(s + separator)
        else:
            parts.append(s)

    chunks: list[str] = []
    current = ""

    def commit(s: str) -> None:
        if s and s.strip():
            chunks.append(s)

    for part in parts:
        if not part:
            continue

        if len(part) > chunk_size:
            if current:
                commit(current)
                current = ""
            if not next_separators:
                chunks.extend(_split_text_recursive(part, [""], chunk_size, chunk_overlap))
            else:
                chunks.extend(_split_text_recursive(part, next_separators, chunk_size, chunk_overlap))
            continue

        if not current:
            current = part
            continue

        candidate = current + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            commit(current)
            current = part

    if current:
        commit(current)

    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
        else:
            final.extend(_split_text_recursive(c, next_separators or [""], chunk_size, chunk_overlap))

    return final


def _enforce_max_chunk_size(chunks: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Ensure no chunk exceeds chunk_size by splitting oversized items."""
    out: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            out.append(c)
        else:
            out.extend(_chunk_recursive(c, chunk_size, chunk_overlap))
    return out


# ---------------------------------------------------------------------------
# Semantic chunker (Gemini)
# ---------------------------------------------------------------------------


async def _chunk_semantic(text: str, model: str, block_words: int) -> list[str]:
    """Semantic chunking via LLM topic-shift detection."""
    from rag_service.embedding import _get_gemini_client

    client = _get_gemini_client()
    words = text.split()
    final_chunks: list[str] = []
    current_index = 0
    buffer_words: list[str] = []
    min_words_before_split = max(30, block_words // 32)

    while current_index < len(words) or buffer_words:
        needed = block_words - len(buffer_words)
        if needed > 0 and current_index < len(words):
            take = min(needed, len(words) - current_index)
            buffer_words.extend(words[current_index : current_index + take])
            current_index += take

        if not buffer_words:
            break

        if current_index >= len(words) and len(buffer_words) < (block_words // 2):
            final_chunks.append(" ".join(buffer_words))
            break

        block_text = " ".join(buffer_words)

        try:
            split_word_index = await _detect_topic_shift(client, block_text, model)
        except Exception as e:
            logger.warning("LLM topic detection failed: %s. Committing block as-is.", e)
            split_word_index = None

        if split_word_index is not None and split_word_index >= min_words_before_split:
            left = buffer_words[:split_word_index]
            right = buffer_words[split_word_index:]
            if left:
                final_chunks.append(" ".join(left))
            buffer_words = right
            if not buffer_words:
                continue
        else:
            final_chunks.append(block_text)
            buffer_words = []

    return final_chunks


async def _detect_topic_shift(client: Any, text: str, model: str) -> int | None:
    """Query Gemini for a topic boundary within text."""
    from google.genai import types

    prompt = (
        "You are detecting topic boundaries for semantic chunking.\n"
        "Analyze the following text block and decide if there is a clear shift to a new,"
        " unrelated topic or section.\n\n"
        "Return ONLY valid JSON in one of these forms:\n"
        '  {"shift": false}\n'
        '  {"shift": true, "start_word_index": <int>}\n\n'
        "Rules:\n"
        "- Be conservative: only mark shift=true if topics are clearly distinct.\n"
        "- start_word_index is 0-based within this block.\n"
        "- If no shift, use shift=false.\n\n"
        f"Text Block:\n{text}\n"
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=80),
    )

    raw = (getattr(response, "text", "") or "").strip()
    if not raw:
        return None

    if "NO_SHIFT" in raw.upper():
        return None

    data = _parse_shift_json(raw)
    if not data or not data.get("shift"):
        return None

    idx = data.get("start_word_index")
    if isinstance(idx, bool) or idx is None:
        return None

    try:
        idx_int = int(idx)
    except Exception:
        return None

    n_words = len(text.split())
    if idx_int <= 0 or idx_int >= n_words:
        return None

    return idx_int


def _parse_shift_json(raw: str) -> dict[str, Any] | None:
    """Parse model response as JSON with tolerant fallback."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    shift_match = re.search(r"\bshift\b\s*[:=]\s*(true|false)", raw, flags=re.IGNORECASE)
    idx_match = re.search(r"\bstart_word_index\b\s*[:=]\s*(\d+)", raw)

    if shift_match:
        shift_val = shift_match.group(1).lower() == "true"
        idx_val = int(idx_match.group(1)) if idx_match else None
        return {"shift": shift_val, "start_word_index": idx_val}

    return None
