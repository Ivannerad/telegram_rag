from __future__ import annotations

from celery import shared_task

from app import business, db


@shared_task(name="tasks.ingest_document", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def ingest_document(self, job_id: str, document_id: str, source: str, text: str, owner_id: int) -> dict:
    db.update_job(job_id, "running")
    try:
        result = business.ingest_document(document_id=document_id, source=source, text=text, owner_id=owner_id)
        db.update_job(job_id, "done", result=result)
        return result
    except Exception as exc:
        db.update_job(job_id, "failed", error=str(exc))
        raise


@shared_task(name="tasks.long_llm_task", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def long_llm_task(self, job_id: str, query: str, owner_id: int) -> dict:
    db.update_job(job_id, "running")
    try:
        result = business.run_long_llm_task(query=query, owner_id=owner_id)
        db.update_job(job_id, "done", result=result)
        return result
    except Exception as exc:
        db.update_job(job_id, "failed", error=str(exc))
        raise


@shared_task(name="tasks.vector_search_task", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def vector_search_task(self, job_id: str, query: str, owner_id: int, limit: int = 5) -> dict:
    db.update_job(job_id, "running")
    try:
        result = business.run_vector_search(query=query, owner_id=owner_id, limit=limit)
        db.update_job(job_id, "done", result=result)
        return result
    except Exception as exc:
        db.update_job(job_id, "failed", error=str(exc))
        raise
