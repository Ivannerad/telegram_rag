from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import business, db
from app.config import get_settings
from app.document_parser import UnsupportedDocumentTypeError, extract_text_from_document
from app.logging_setup import configure_logging
from worker.celery_app import celery_app

configure_logging()
app = FastAPI(title="Telegram RAG API")


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    owner_id: int = Field(gt=0)


class IngestRequest(BaseModel):
    source: str = Field(default="telegram")
    text: str = Field(min_length=1)
    owner_id: int = Field(gt=0)
    document_id: str | None = None


class LongLlmRequest(BaseModel):
    query: str = Field(min_length=1)
    owner_id: int = Field(gt=0)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    owner_id: int = Field(gt=0)
    limit: int = 5


class TaskListRequest(BaseModel):
    owner_id: int = Field(gt=0)
    status: Literal["pending", "running", "done", "failed"] | None = Field(default=None)
    limit: int = Field(default=10, ge=1, le=50)


def verify_internal_token(x_internal_token: Annotated[str | None, Header()] = None) -> None:
    settings = get_settings()
    if x_internal_token != settings.api_internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")


@app.on_event("startup")
def startup() -> None:
    db.ensure_schema()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", dependencies=[Depends(verify_internal_token)])
def ask(payload: AskRequest) -> dict:
    retrieval = business.run_vector_search(query=payload.query, owner_id=payload.owner_id, limit=3)
    settings = get_settings()
    chunks = [m["text"] for m in retrieval["matches"][: settings.rag_max_context_chunks]]
    answer = (
        business.NO_INFO_RESPONSE
        if not chunks
        else business.llm.answer_query(query=payload.query, context_chunks=chunks)
    )
    return {
        "query": payload.query,
        "owner_id": payload.owner_id,
        "answer": answer,
        "matches": retrieval["matches"],
    }


@app.post("/tasks/ingest", dependencies=[Depends(verify_internal_token)])
def enqueue_ingest(payload: IngestRequest) -> dict:
    job_id = str(uuid.uuid4())
    document_id = payload.document_id or job_id
    task_name = "tasks.ingest_document"
    job_payload = {"document_id": document_id, "source": payload.source, "owner_id": payload.owner_id}
    db.create_job(job_id=job_id, task_name=task_name, payload=job_payload)

    celery_app.send_task(
        task_name,
        kwargs={
            "job_id": job_id,
            "document_id": document_id,
            "source": payload.source,
            "text": payload.text,
            "owner_id": payload.owner_id,
        },
    )

    return {"job_id": job_id, "status": "pending", "task": task_name}


@app.post("/tasks/ingest-file", dependencies=[Depends(verify_internal_token)])
async def enqueue_ingest_file(
    owner_id: int = Form(gt=0),
    source: str = Form(default="telegram"),
    document_id: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> dict:
    raw = await file.read()
    try:
        text = extract_text_from_document(filename=file.filename, content_type=file.content_type, raw=raw)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    resolved_document_id = document_id or job_id
    task_name = "tasks.ingest_document"
    job_payload = {
        "document_id": resolved_document_id,
        "source": source,
        "owner_id": owner_id,
        "filename": file.filename,
    }
    db.create_job(job_id=job_id, task_name=task_name, payload=job_payload)

    celery_app.send_task(
        task_name,
        kwargs={
            "job_id": job_id,
            "document_id": resolved_document_id,
            "source": source,
            "text": text,
            "owner_id": owner_id,
        },
    )

    return {"job_id": job_id, "status": "pending", "task": task_name}


@app.post("/tasks/llm", dependencies=[Depends(verify_internal_token)])
def enqueue_llm(payload: LongLlmRequest) -> dict:
    job_id = str(uuid.uuid4())
    task_name = "tasks.long_llm_task"
    db.create_job(job_id=job_id, task_name=task_name, payload={"query": payload.query, "owner_id": payload.owner_id})

    celery_app.send_task(task_name, kwargs={"job_id": job_id, "query": payload.query, "owner_id": payload.owner_id})
    return {"job_id": job_id, "status": "pending", "task": task_name}


@app.post("/tasks/search", dependencies=[Depends(verify_internal_token)])
def enqueue_search(payload: SearchRequest) -> dict:
    job_id = str(uuid.uuid4())
    task_name = "tasks.vector_search_task"
    db.create_job(
        job_id=job_id,
        task_name=task_name,
        payload={"query": payload.query, "owner_id": payload.owner_id, "limit": payload.limit},
    )

    celery_app.send_task(
        task_name,
        kwargs={"job_id": job_id, "query": payload.query, "owner_id": payload.owner_id, "limit": payload.limit},
    )
    return {"job_id": job_id, "status": "pending", "task": task_name}


@app.get("/tasks/{job_id}", dependencies=[Depends(verify_internal_token)])
def task_status(job_id: str, owner_id: int | None = None) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if owner_id is not None:
        payload = job.get("payload") or {}
        job_owner_id = payload.get("owner_id")
        if str(job_owner_id) != str(owner_id):
            raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/tasks/list", dependencies=[Depends(verify_internal_token)])
def list_tasks(payload: TaskListRequest) -> dict:
    jobs = db.list_jobs(owner_id=payload.owner_id, status=payload.status, limit=payload.limit)
    return {
        "owner_id": payload.owner_id,
        "status": payload.status,
        "limit": payload.limit,
        "jobs": jobs,
    }
