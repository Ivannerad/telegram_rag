from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings

COLLECTION_NAME = "documents"
FALLBACK_VECTOR_SIZE = 8


def get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=1)
def get_embeddings_client() -> OpenAIEmbeddings | None:
    settings = get_settings()
    api_key = settings.openai_api_key or settings.llm_api_key
    if not api_key:
        return None
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key,
        base_url=settings.openai_base_url or None,
    )


def fallback_embed(text: str, dim: int = FALLBACK_VECTOR_SIZE) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for i in range(dim):
        raw = digest[i] / 255.0
        values.append(raw)
    return values


def embed_text(text: str) -> list[float]:
    settings = get_settings()
    client = get_embeddings_client()
    if client is None or settings.llm_provider != "openai":
        return fallback_embed(text)

    try:
        return client.embed_query(text)
    except Exception:
        # Keep ingestion available even if external embedding API is temporarily unreachable.
        return fallback_embed(text)


def embedding_dimension() -> int:
    settings = get_settings()
    if settings.llm_provider != "openai" or get_embeddings_client() is None:
        return FALLBACK_VECTOR_SIZE

    if settings.openai_embedding_model == "text-embedding-3-small":
        return 1536
    return 3072


def ensure_collection() -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    vector_size = embedding_dimension()
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        return

    collection = client.get_collection(collection_name=COLLECTION_NAME)
    current_size = collection.config.params.vectors.size
    if current_size != vector_size:
        # Recreate if vector size changed after switching embedding models.
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def _owner_filter(owner_id: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="owner_id",
                match=models.MatchValue(value=owner_id),
            )
        ]
    )


def upsert_chunks(document_id: str, chunks: list[str], owner_id: int) -> int:
    ensure_collection()
    client = get_client()

    points: list[models.PointStruct] = []
    for idx, chunk in enumerate(chunks):
        point_id = int(hashlib.sha256(f"{document_id}:{idx}".encode("utf-8")).hexdigest()[:15], 16)
        payload: dict[str, Any] = {
            "document_id": document_id,
            "chunk_idx": idx,
            "text": chunk,
            "owner_id": owner_id,
        }
        points.append(models.PointStruct(id=point_id, vector=embed_text(chunk), payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(query: str, owner_id: int, limit: int = 3, min_score: float | None = None) -> list[dict[str, Any]]:
    ensure_collection()
    client = get_client()
    settings = get_settings()
    vector = embed_text(query)
    owner_filter = _owner_filter(owner_id)
    score_threshold = settings.vector_score_threshold if min_score is None else min_score

    if hasattr(client, "search"):
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=owner_filter,
            score_threshold=score_threshold,
            limit=limit,
        )
    else:
        query_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=owner_filter,
            score_threshold=score_threshold,
            limit=limit,
        )
        hits = query_response.points

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        results.append(
            {
                "score": getattr(hit, "score", 0.0),
                "document_id": payload.get("document_id"),
                "chunk_idx": payload.get("chunk_idx"),
                "text": payload.get("text", ""),
            }
        )
    return [r for r in results if float(r.get("score", 0.0)) >= score_threshold]
