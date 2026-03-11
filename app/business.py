from __future__ import annotations

import time
from typing import Any

from app import llm, vector
from app.config import get_settings

NO_INFO_RESPONSE = "I could not find the answer in the provided documents."
MAX_MATCH_PREVIEW_CHARS = 240


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_document(document_id: str, source: str, text: str, owner_id: int) -> dict[str, Any]:
    chunks = chunk_text(text)
    upserted = vector.upsert_chunks(document_id=document_id, chunks=chunks, owner_id=owner_id)
    return {
        "document_id": document_id,
        "source": source,
        "owner_id": owner_id,
        "chunks": len(chunks),
        "vectors_upserted": upserted,
    }


def run_vector_search(query: str, owner_id: int, limit: int = 3) -> dict[str, Any]:
    results = vector.search(query=query, owner_id=owner_id, limit=limit)
    return {
        "query": query,
        "owner_id": owner_id,
        "matches": results,
    }


def run_long_llm_task(query: str, owner_id: int) -> dict[str, Any]:
    # Simulate heavy business process.
    time.sleep(2)
    retrieval = run_vector_search(query=query, owner_id=owner_id, limit=3)
    settings = get_settings()
    matches = retrieval["matches"]
    context = [m["text"] for m in matches[: settings.rag_max_context_chunks]]
    if not context:
        answer = NO_INFO_RESPONSE
    else:
        answer = llm.answer_query(query=query, context_chunks=context)

    top_matches_preview: list[dict[str, Any]] = []
    for match in matches[: settings.rag_max_context_chunks]:
        snippet = (match.get("text") or "").strip()
        if len(snippet) > MAX_MATCH_PREVIEW_CHARS:
            snippet = f"{snippet[:MAX_MATCH_PREVIEW_CHARS]}..."
        top_matches_preview.append(
            {
                "document_id": match.get("document_id"),
                "chunk_idx": match.get("chunk_idx"),
                "score": match.get("score"),
                "text_snippet": snippet,
            }
        )

    return {
        "query": query,
        "owner_id": owner_id,
        "answer": answer,
        "retrieval_count": len(context),
        "top_matches_preview": top_matches_preview,
    }
