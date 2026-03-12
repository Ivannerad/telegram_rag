from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import psycopg
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
API_BASE_URL = "http://127.0.0.1:8000"
POSTGRES_DSN = "host=127.0.0.1 port=5432 dbname=telegram_rag user=telegram password=telegram"
QDRANT_URL = "http://127.0.0.1:6333"
INTERNAL_TOKEN = "functional-test-token"
COLLECTION_NAME = "documents"
RUN_DOCKER_TESTS = os.getenv("RUN_DOCKER_TESTS") == "1"


def _run_compose(project: str, env_file: Path, compose_files: list[Path], args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-p", project, "--env-file", str(env_file)]
    for compose_file in compose_files:
        cmd.extend(["-f", str(compose_file)])
    cmd.extend(args)
    return subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "DOCKER_BUILDKIT": "1", "COMPOSE_DOCKER_CLI_BUILD": "1"},
    )


def _write_env_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "POSTGRES_USER=telegram",
                "POSTGRES_PASSWORD=telegram",
                "POSTGRES_DB=telegram_rag",
                "POSTGRES_HOST=postgres",
                "POSTGRES_PORT=5432",
                "RABBITMQ_USER=telegram",
                "RABBITMQ_PASSWORD=telegram",
                "RABBITMQ_HOST=rabbitmq",
                "RABBITMQ_PORT=5672",
                "RABBITMQ_VHOST=/",
                "QDRANT_HOST=qdrant",
                "QDRANT_PORT=6333",
                "API_HOST=0.0.0.0",
                "API_PORT=8000",
                f"API_INTERNAL_TOKEN={INTERNAL_TOKEN}",
                "TELEGRAM_API_ID=1",
                "TELEGRAM_API_HASH=dummy",
                "TELEGRAM_BOT_TOKEN=dummy",
                "TELEGRAM_SESSION_NAME=telegram_bot",
                "LLM_PROVIDER=dummy",
                "LLM_API_KEY=",
                "OPENAI_API_KEY=",
                "OPENAI_BASE_URL=",
                "OPENAI_MODEL=gpt-5",
                "OPENAI_EMBEDDING_MODEL=text-embedding-3-large",
                "VECTOR_SCORE_THRESHOLD=0.0",
                "LLM_TEMPERATURE=0.0",
                "LLM_MAX_TOKENS=120",
                "LLM_MAX_ANSWER_WORDS=40",
                "RAG_MAX_CONTEXT_CHUNKS=2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_compose_override(path: Path, env_file: Path) -> None:
    env_file_text = str(env_file).replace("\\", "\\\\").replace('"', '\\"')
    path.write_text(
        "\n".join(
            [
                "services:",
                "  api:",
                "    env_file:",
                f'      - "{env_file_text}"',
                "  worker:",
                "    env_file:",
                f'      - "{env_file_text}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _wait_for_api_health(timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{API_BASE_URL}/health", timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - integration retry path
            last_error = exc
        time.sleep(1)
    raise AssertionError(f"API health check did not pass in {timeout_seconds}s. Last error: {last_error!r}")


def _wait_for_job_terminal(job_id: str, owner_id: int, timeout_seconds: int = 90) -> dict:
    deadline = time.time() + timeout_seconds
    headers = {"x-internal-token": INTERNAL_TOKEN}
    last_payload: dict | None = None
    while time.time() < deadline:
        response = httpx.get(
            f"{API_BASE_URL}/tasks/{job_id}",
            params={"owner_id": owner_id},
            headers=headers,
            timeout=5.0,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        last_payload = payload
        if payload.get("status") in {"done", "failed"}:
            return payload
        time.sleep(1)
    raise AssertionError(f"Job did not complete in {timeout_seconds}s. Last payload: {last_payload}")


@pytest.mark.skipif(not RUN_DOCKER_TESTS, reason="Set RUN_DOCKER_TESTS=1 to run Docker functional tests")
def test_compose_ingest_pipeline() -> None:
    if subprocess.run(["docker", "compose", "version"], capture_output=True, text=True).returncode != 0:
        pytest.skip("docker compose is not available")

    project = f"telegram-rag-func-{uuid.uuid4().hex[:8]}"
    env_file = ROOT_DIR / f".env.functional.{project}"
    compose_override = ROOT_DIR / f"docker-compose.functional.{project}.yml"
    _write_env_file(env_file)
    _write_compose_override(compose_override, env_file)

    try:
        _run_compose(
            project,
            env_file,
            [COMPOSE_FILE, compose_override],
            ["up", "-d", "--build", "api", "worker", "postgres", "rabbitmq", "qdrant"],
        )
        _wait_for_api_health()

        owner_id = 987654
        document_id = f"functional-doc-{uuid.uuid4().hex[:8]}"
        ingest_text = (
            "Docker functional integration payload. "
            "This line should be persisted through RabbitMQ worker processing into Qdrant."
        )
        ingest_response = httpx.post(
            f"{API_BASE_URL}/tasks/ingest",
            headers={"x-internal-token": INTERNAL_TOKEN},
            json={
                "source": "functional-test",
                "text": ingest_text,
                "owner_id": owner_id,
                "document_id": document_id,
            },
            timeout=10.0,
        )
        assert ingest_response.status_code == 200, ingest_response.text
        job_id = ingest_response.json()["job_id"]

        final_job = _wait_for_job_terminal(job_id=job_id, owner_id=owner_id)
        assert final_job["status"] == "done"
        assert final_job["task_name"] == "tasks.ingest_document"
        assert final_job["result"]["document_id"] == document_id
        assert final_job["result"]["owner_id"] == owner_id
        assert final_job["result"]["vectors_upserted"] >= 1
        assert final_job["error"] is None

        with psycopg.connect(POSTGRES_DSN, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status, task_name, payload, result, error FROM bg_jobs WHERE job_id = %s", (job_id,))
                row = cur.fetchone()
        assert row is not None
        assert row["status"] == "done"
        assert row["task_name"] == "tasks.ingest_document"
        assert row["payload"]["owner_id"] == owner_id
        assert row["payload"]["document_id"] == document_id
        assert row["result"]["vectors_upserted"] >= 1
        assert row["error"] is None

        qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        query_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.0] * 8,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    ),
                    qdrant_models.FieldCondition(
                        key="owner_id",
                        match=qdrant_models.MatchValue(value=owner_id),
                    ),
                ]
            ),
            limit=1,
        )
        assert len(query_response.points) >= 1

        search_response = httpx.post(
            f"{API_BASE_URL}/tasks/search",
            headers={"x-internal-token": INTERNAL_TOKEN},
            json={
                "query": "RabbitMQ worker processing into Qdrant",
                "owner_id": owner_id,
                "limit": 5,
            },
            timeout=10.0,
        )
        assert search_response.status_code == 200, search_response.text
        search_job_id = search_response.json()["job_id"]
        search_job = _wait_for_job_terminal(job_id=search_job_id, owner_id=owner_id)
        assert search_job["status"] == "done"
        assert search_job["task_name"] == "tasks.vector_search_task"
        matches = (search_job.get("result") or {}).get("matches") or []
        assert matches
        assert any((match or {}).get("document_id") == document_id for match in matches)
    finally:
        subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                project,
                "--env-file",
                str(env_file),
                "-f",
                str(COMPOSE_FILE),
                "-f",
                str(compose_override),
                "down",
                "-v",
                "--remove-orphans",
            ],
            cwd=ROOT_DIR,
            text=True,
            capture_output=True,
        )
        env_file.unlink(missing_ok=True)
        compose_override.unlink(missing_ok=True)
