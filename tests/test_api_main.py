from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import main
from app.document_parser import ParserDependencyError, UnsupportedDocumentTypeError


def auth_headers(token: str = "test-token") -> dict[str, str]:
    return {"x-internal-token": token}


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(api_internal_token="test-token", rag_max_context_chunks=2),
    )


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(main.db, "ensure_schema", lambda: None)
    return TestClient(main.app)


def test_health(monkeypatch) -> None:
    with make_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_requires_auth(monkeypatch) -> None:
    with make_client(monkeypatch) as client:
        response = client.post("/ask", json={"query": "What is this?", "owner_id": 123})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal token"}


def test_ask_with_matches(monkeypatch) -> None:
    run_vector_search = Mock(
        return_value={
            "query": "What is this?",
            "owner_id": 123,
            "matches": [
                {"text": "chunk-1", "score": 0.9},
                {"text": "chunk-2", "score": 0.8},
                {"text": "chunk-3", "score": 0.7},
            ],
        }
    )
    answer_query = Mock(return_value="answer from llm")

    monkeypatch.setattr(main.business, "run_vector_search", run_vector_search)
    monkeypatch.setattr(main.business.llm, "answer_query", answer_query)

    with make_client(monkeypatch) as client:
        response = client.post("/ask", headers=auth_headers(), json={"query": "What is this?", "owner_id": 123})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What is this?"
    assert payload["owner_id"] == 123
    assert payload["answer"] == "answer from llm"
    assert len(payload["matches"]) == 3
    run_vector_search.assert_called_once_with(query="What is this?", owner_id=123, limit=3)
    answer_query.assert_called_once_with(query="What is this?", context_chunks=["chunk-1", "chunk-2"])


def test_ask_without_matches(monkeypatch) -> None:
    run_vector_search = Mock(return_value={"query": "No match", "owner_id": 55, "matches": []})
    answer_query = Mock()

    monkeypatch.setattr(main.business, "run_vector_search", run_vector_search)
    monkeypatch.setattr(main.business.llm, "answer_query", answer_query)

    with make_client(monkeypatch) as client:
        response = client.post("/ask", headers=auth_headers(), json={"query": "No match", "owner_id": 55})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == main.business.NO_INFO_RESPONSE
    assert payload["matches"] == []
    answer_query.assert_not_called()


def test_tasks_ingest(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()

    monkeypatch.setattr(main.uuid, "uuid4", lambda: "job-123")
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest",
            headers=auth_headers(),
            json={"source": "telegram", "text": "hello world", "owner_id": 7},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-123", "status": "pending", "task": "tasks.ingest_document"}
    create_job.assert_called_once_with(
        job_id="job-123",
        task_name="tasks.ingest_document",
        payload={"document_id": "job-123", "source": "telegram", "owner_id": 7},
    )
    send_task.assert_called_once_with(
        "tasks.ingest_document",
        kwargs={
            "job_id": "job-123",
            "document_id": "job-123",
            "source": "telegram",
            "text": "hello world",
            "owner_id": 7,
        },
    )


def test_tasks_ingest_requires_auth(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest",
            json={"source": "telegram", "text": "hello world", "owner_id": 7},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal token"}
    create_job.assert_not_called()
    send_task.assert_not_called()


def test_tasks_ingest_file_success(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()
    extract_text = Mock(return_value="parsed text")

    monkeypatch.setattr(main.uuid, "uuid4", lambda: "file-job-1")
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)
    monkeypatch.setattr(main, "extract_text_from_document", extract_text)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest-file",
            headers=auth_headers(),
            data={"owner_id": "11", "source": "telegram"},
            files={"file": ("notes.txt", b"raw-bytes", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "file-job-1", "status": "pending", "task": "tasks.ingest_document"}
    extract_text.assert_called_once_with(filename="notes.txt", content_type="text/plain", raw=b"raw-bytes")
    create_job.assert_called_once_with(
        job_id="file-job-1",
        task_name="tasks.ingest_document",
        payload={
            "document_id": "file-job-1",
            "source": "telegram",
            "owner_id": 11,
            "filename": "notes.txt",
        },
    )
    send_task.assert_called_once_with(
        "tasks.ingest_document",
        kwargs={
            "job_id": "file-job-1",
            "document_id": "file-job-1",
            "source": "telegram",
            "text": "parsed text",
            "owner_id": 11,
        },
    )


def test_tasks_ingest_file_requires_auth(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()
    extract_text = Mock(return_value="parsed text")
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)
    monkeypatch.setattr(main, "extract_text_from_document", extract_text)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest-file",
            data={"owner_id": "11", "source": "telegram"},
            files={"file": ("notes.txt", b"raw-bytes", "text/plain")},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal token"}
    create_job.assert_not_called()
    send_task.assert_not_called()
    extract_text.assert_not_called()


def test_tasks_ingest_file_unsupported_type(monkeypatch) -> None:
    def _raise_unsupported(**_kwargs):
        raise UnsupportedDocumentTypeError("Unsupported file type")

    monkeypatch.setattr(main, "extract_text_from_document", _raise_unsupported)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest-file",
            headers=auth_headers(),
            data={"owner_id": "9", "source": "telegram"},
            files={"file": ("archive.bin", b"\x00\x01", "application/octet-stream")},
        )

    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported file type"}


def test_tasks_ingest_file_value_error(monkeypatch) -> None:
    def _raise_value_error(**_kwargs):
        raise ValueError("Could not extract readable text from the uploaded file")

    monkeypatch.setattr(main, "extract_text_from_document", _raise_value_error)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest-file",
            headers=auth_headers(),
            data={"owner_id": "9", "source": "telegram"},
            files={"file": ("empty.txt", b"", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Could not extract readable text from the uploaded file"}


def test_tasks_ingest_file_parser_dependency_error(monkeypatch) -> None:
    def _raise_dependency_error(**_kwargs):
        raise ParserDependencyError("PDF parsing requires dependency 'pypdf'")

    monkeypatch.setattr(main, "extract_text_from_document", _raise_dependency_error)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/ingest-file",
            headers=auth_headers(),
            data={"owner_id": "9", "source": "telegram"},
            files={"file": ("sample.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "File parsing is temporarily unavailable"}


def test_tasks_llm(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()

    monkeypatch.setattr(main.uuid, "uuid4", lambda: "llm-job-1")
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/llm",
            headers=auth_headers(),
            json={"query": "Summarize this", "owner_id": 3},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "llm-job-1", "status": "pending", "task": "tasks.long_llm_task"}
    create_job.assert_called_once_with(
        job_id="llm-job-1",
        task_name="tasks.long_llm_task",
        payload={"query": "Summarize this", "owner_id": 3},
    )
    send_task.assert_called_once_with(
        "tasks.long_llm_task",
        kwargs={"job_id": "llm-job-1", "query": "Summarize this", "owner_id": 3},
    )


def test_tasks_search(monkeypatch) -> None:
    create_job = Mock()
    send_task = Mock()

    monkeypatch.setattr(main.uuid, "uuid4", lambda: "search-job-1")
    monkeypatch.setattr(main.db, "create_job", create_job)
    monkeypatch.setattr(main.celery_app, "send_task", send_task)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/search",
            headers=auth_headers(),
            json={"query": "find me", "owner_id": 8, "limit": 7},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "search-job-1", "status": "pending", "task": "tasks.vector_search_task"}
    create_job.assert_called_once_with(
        job_id="search-job-1",
        task_name="tasks.vector_search_task",
        payload={"query": "find me", "owner_id": 8, "limit": 7},
    )
    send_task.assert_called_once_with(
        "tasks.vector_search_task",
        kwargs={"job_id": "search-job-1", "query": "find me", "owner_id": 8, "limit": 7},
    )


def test_task_status_not_found(monkeypatch) -> None:
    monkeypatch.setattr(main.db, "get_job", lambda _job_id: None)

    with make_client(monkeypatch) as client:
        response = client.get("/tasks/missing", headers=auth_headers(), params={"owner_id": 1})

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_task_status_owner_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        main.db,
        "get_job",
        lambda _job_id: {
            "job_id": "abc",
            "status": "pending",
            "payload": {"owner_id": 999},
        },
    )

    with make_client(monkeypatch) as client:
        response = client.get("/tasks/abc", headers=auth_headers(), params={"owner_id": 1})

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_task_status_success_with_owner_id_match(monkeypatch) -> None:
    job = {
        "job_id": "abc",
        "task_name": "tasks.vector_search_task",
        "status": "done",
        "payload": {"owner_id": 77, "query": "hello"},
        "result": {"matches": []},
    }
    monkeypatch.setattr(main.db, "get_job", lambda _job_id: job)

    with make_client(monkeypatch) as client:
        response = client.get("/tasks/abc", headers=auth_headers(), params={"owner_id": 77})

    assert response.status_code == 200
    assert response.json() == job


def test_task_status_requires_owner_id(monkeypatch) -> None:
    get_job = Mock()
    monkeypatch.setattr(main.db, "get_job", get_job)

    with make_client(monkeypatch) as client:
        response = client.get("/tasks/abc", headers=auth_headers())

    assert response.status_code == 422
    get_job.assert_not_called()


def test_tasks_list(monkeypatch) -> None:
    list_jobs = Mock(return_value=[{"job_id": "j-1", "status": "pending"}])
    monkeypatch.setattr(main.db, "list_jobs", list_jobs)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/list",
            headers=auth_headers(),
            json={"owner_id": 42, "status": "pending", "limit": 5},
        )

    assert response.status_code == 200
    assert response.json() == {
        "owner_id": 42,
        "status": "pending",
        "limit": 5,
        "jobs": [{"job_id": "j-1", "status": "pending"}],
    }
    list_jobs.assert_called_once_with(owner_id=42, status="pending", limit=5)


def test_tasks_list_limit_too_large(monkeypatch) -> None:
    with make_client(monkeypatch) as client:
        response = client.post(
            "/tasks/list",
            headers=auth_headers(),
            json={"owner_id": 42, "limit": 51},
        )

    assert response.status_code == 422


def test_groups_bind_success(monkeypatch) -> None:
    upsert_group_binding = Mock(return_value={"group_id": -1001234, "owner_id": 42, "group_label": "Team"})
    monkeypatch.setattr(main.db, "upsert_group_binding", upsert_group_binding)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/groups/bind",
            headers=auth_headers(),
            json={"owner_id": 42, "group_id": -1001234, "group_label": "Team"},
        )

    assert response.status_code == 200
    assert response.json() == {"binding": {"group_id": -1001234, "owner_id": 42, "group_label": "Team"}}
    upsert_group_binding.assert_called_once_with(group_id=-1001234, owner_id=42, group_label="Team")


def test_groups_bind_conflict(monkeypatch) -> None:
    monkeypatch.setattr(main.db, "upsert_group_binding", lambda **_kwargs: None)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/groups/bind",
            headers=auth_headers(),
            json={"owner_id": 42, "group_id": -1001234, "group_label": "Team"},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "Group is already bound to another owner"}


def test_groups_unbind(monkeypatch) -> None:
    delete_group_binding = Mock(return_value=True)
    monkeypatch.setattr(main.db, "delete_group_binding", delete_group_binding)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/groups/unbind",
            headers=auth_headers(),
            json={"owner_id": 42, "group_id": -1001234},
        )

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    delete_group_binding.assert_called_once_with(group_id=-1001234, owner_id=42)


def test_groups_list(monkeypatch) -> None:
    list_group_bindings = Mock(return_value=[{"group_id": -1001234, "owner_id": 42, "group_label": "Team"}])
    monkeypatch.setattr(main.db, "list_group_bindings", list_group_bindings)

    with make_client(monkeypatch) as client:
        response = client.post("/groups/list", headers=auth_headers(), json={"owner_id": 42, "limit": 10})

    assert response.status_code == 200
    assert response.json() == {
        "owner_id": 42,
        "limit": 10,
        "groups": [{"group_id": -1001234, "owner_id": 42, "group_label": "Team"}],
    }
    list_group_bindings.assert_called_once_with(owner_id=42, limit=10)


def test_groups_resolve_success(monkeypatch) -> None:
    binding = {"group_id": -100999, "owner_id": 42, "group_label": "Ops"}
    monkeypatch.setattr(main.db, "get_group_binding", lambda group_id: binding if group_id == -100999 else None)

    with make_client(monkeypatch) as client:
        response = client.get("/groups/resolve", headers=auth_headers(), params={"group_id": -100999})

    assert response.status_code == 200
    assert response.json() == {"binding": binding}


def test_groups_resolve_not_found(monkeypatch) -> None:
    monkeypatch.setattr(main.db, "get_group_binding", lambda group_id: None)

    with make_client(monkeypatch) as client:
        response = client.get("/groups/resolve", headers=auth_headers(), params={"group_id": -100999})

    assert response.status_code == 404
    assert response.json() == {"detail": "Group is not bound to any owner"}


def test_groups_resolve_requires_auth(monkeypatch) -> None:
    get_group_binding = Mock()
    monkeypatch.setattr(main.db, "get_group_binding", get_group_binding)

    with make_client(monkeypatch) as client:
        response = client.get("/groups/resolve", params={"group_id": -100999})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid internal token"}
    get_group_binding.assert_not_called()
