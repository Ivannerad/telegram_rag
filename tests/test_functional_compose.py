from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import psycopg
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"
INTERNAL_TOKEN = "functional-test-token"
COLLECTION_NAME = "documents"
RUN_DOCKER_TESTS = os.getenv("RUN_DOCKER_TESTS") == "1"
SERVICES = ("api", "worker", "postgres", "rabbitmq", "qdrant")

pytestmark = pytest.mark.integration


@dataclass
class ComposeStack:
    project: str
    env_file: Path
    compose_override: Path
    api_base_url: str = ""
    postgres_dsn: str = ""
    qdrant_url: str = ""

    @property
    def compose_files(self) -> list[Path]:
        return [COMPOSE_FILE, self.compose_override]


def _docker_compose_available() -> bool:
    return subprocess.run(["docker", "compose", "version"], capture_output=True, text=True).returncode == 0


def _run_compose(stack: ComposeStack, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-p", stack.project, "--env-file", str(stack.env_file)]
    for compose_file in stack.compose_files:
        cmd.extend(["-f", str(compose_file)])
    cmd.extend(args)
    return subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        check=check,
        text=True,
        capture_output=True,
        env={**os.environ, "DOCKER_BUILDKIT": "1", "COMPOSE_DOCKER_CLI_BUILD": "1"},
    )


def _compose_logs(stack: ComposeStack, services: tuple[str, ...] = SERVICES, tail: int = 120) -> str:
    try:
        result = _run_compose(stack, ["logs", "--no-color", "--tail", str(tail), *services], check=False)
    except Exception as exc:  # pragma: no cover - integration diagnostics path
        return f"Failed to collect compose logs: {exc!r}"
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return output.strip()


def _resolve_host_port(stack: ComposeStack, service: str, container_port: int) -> int:
    result = _run_compose(stack, ["port", service, str(container_port)])
    line = (result.stdout or "").strip().splitlines()
    if not line:
        raise AssertionError(f"No mapped host port for {service}:{container_port}")
    host_port = line[-1].rsplit(":", 1)[-1].strip()
    return int(host_port)


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
                "    ports:",
                '      - "127.0.0.1::8000"',
                "  worker:",
                "    env_file:",
                f'      - "{env_file_text}"',
                "  postgres:",
                "    ports:",
                '      - "127.0.0.1::5432"',
                "  qdrant:",
                "    ports:",
                '      - "127.0.0.1::6333"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _wait_for_api_health(stack: ComposeStack, timeout_seconds: int = 150) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{stack.api_base_url}/health", timeout=2.5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - integration retry path
            last_error = exc
        time.sleep(1)

    logs = _compose_logs(stack, services=("api", "worker", "postgres", "rabbitmq", "qdrant"), tail=160)
    raise AssertionError(
        f"API health check did not pass in {timeout_seconds}s. Last error: {last_error!r}\nRecent logs:\n{logs}"
    )


def _wait_for_job_terminal(stack: ComposeStack, job_id: str, owner_id: int, timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    headers = {"x-internal-token": INTERNAL_TOKEN}
    last_payload: dict | None = None
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(
                f"{stack.api_base_url}/tasks/{job_id}",
                params={"owner_id": owner_id},
                headers=headers,
                timeout=5.0,
            )
            if response.status_code == 200:
                payload = response.json()
                last_payload = payload
                if payload.get("status") in {"done", "failed"}:
                    return payload
        except Exception as exc:  # pragma: no cover - integration retry path
            last_error = exc
        time.sleep(1)

    raise AssertionError(
        f"Job did not complete in {timeout_seconds}s. Last payload: {last_payload}, last error: {last_error!r}"
    )


def _cleanup_owner(stack: ComposeStack, owner_id: int) -> None:
    with psycopg.connect(stack.postgres_dsn, row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM bg_jobs WHERE payload->>'owner_id' = %s", (str(owner_id),))

    qdrant = QdrantClient(url=stack.qdrant_url, check_compatibility=False)
    filter_by_owner = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="owner_id",
                match=qdrant_models.MatchValue(value=owner_id),
            )
        ]
    )
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=filter_by_owner),
            wait=True,
        )
    except Exception:
        # Collection may not exist if ingestion did not happen.
        pass


def _enqueue_ingest(stack: ComposeStack, owner_id: int, document_id: str, text: str) -> str:
    response = httpx.post(
        f"{stack.api_base_url}/tasks/ingest",
        headers={"x-internal-token": INTERNAL_TOKEN},
        json={
            "source": "functional-test",
            "text": text,
            "owner_id": owner_id,
            "document_id": document_id,
        },
        timeout=10.0,
    )
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _enqueue_search(stack: ComposeStack, owner_id: int, query: str, limit: int = 5) -> str:
    response = httpx.post(
        f"{stack.api_base_url}/tasks/search",
        headers={"x-internal-token": INTERNAL_TOKEN},
        json={
            "query": query,
            "owner_id": owner_id,
            "limit": limit,
        },
        timeout=10.0,
    )
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _wait_for_worker_readiness(stack: ComposeStack) -> None:
    owner_id = int(uuid.uuid4().int % 2_000_000_000) + 1
    document_id = f"warmup-doc-{uuid.uuid4().hex[:8]}"
    try:
        job_id = _enqueue_ingest(stack, owner_id=owner_id, document_id=document_id, text="worker readiness smoke test")
        job = _wait_for_job_terminal(stack, job_id=job_id, owner_id=owner_id, timeout_seconds=150)
        if job.get("status") != "done":
            raise AssertionError(f"Worker readiness smoke job failed: {job}")
    finally:
        _cleanup_owner(stack, owner_id)


@pytest.fixture(scope="session")
def compose_stack() -> ComposeStack:
    if not RUN_DOCKER_TESTS:
        pytest.skip("Set RUN_DOCKER_TESTS=1 to run Docker integration tests")
    if not _docker_compose_available():
        pytest.skip("docker compose is not available")

    project = f"telegram-rag-it-{uuid.uuid4().hex[:8]}"
    env_file = ROOT_DIR / f".env.integration.{project}"
    compose_override = ROOT_DIR / f"docker-compose.integration.{project}.yml"
    stack = ComposeStack(project=project, env_file=env_file, compose_override=compose_override)
    _write_env_file(env_file)
    _write_compose_override(compose_override, env_file)

    try:
        _run_compose(
            stack,
            ["up", "-d", "--build", "api", "worker", "postgres", "rabbitmq", "qdrant"],
        )
        api_port = _resolve_host_port(stack, "api", 8000)
        pg_port = _resolve_host_port(stack, "postgres", 5432)
        qdrant_port = _resolve_host_port(stack, "qdrant", 6333)
        stack.api_base_url = f"http://127.0.0.1:{api_port}"
        stack.postgres_dsn = f"host=127.0.0.1 port={pg_port} dbname=telegram_rag user=telegram password=telegram"
        stack.qdrant_url = f"http://127.0.0.1:{qdrant_port}"
        _wait_for_api_health(stack)
        _wait_for_worker_readiness(stack)
        yield stack
    finally:
        _run_compose(stack, ["down", "-v", "--remove-orphans"], check=False)
        env_file.unlink(missing_ok=True)
        compose_override.unlink(missing_ok=True)


@pytest.fixture()
def owner_id(compose_stack: ComposeStack) -> int:
    owner = int(uuid.uuid4().int % 2_000_000_000) + 1
    yield owner
    _cleanup_owner(compose_stack, owner)


@pytest.mark.integration
def test_compose_ingest_pipeline(compose_stack: ComposeStack, owner_id: int) -> None:
    document_id = f"it-doc-{uuid.uuid4().hex[:10]}"
    ingest_text = (
        "Docker integration ingest payload. "
        "The worker should persist vectors and complete the job successfully."
    )

    job_id = _enqueue_ingest(compose_stack, owner_id=owner_id, document_id=document_id, text=ingest_text)
    final_job = _wait_for_job_terminal(compose_stack, job_id=job_id, owner_id=owner_id)

    assert final_job["status"] == "done"
    assert final_job["task_name"] == "tasks.ingest_document"
    assert final_job["result"]["document_id"] == document_id
    assert final_job["result"]["owner_id"] == owner_id
    assert final_job["result"]["vectors_upserted"] >= 1
    assert final_job["error"] is None

    with psycopg.connect(compose_stack.postgres_dsn, row_factory=psycopg.rows.dict_row) as conn:
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

    qdrant = QdrantClient(url=compose_stack.qdrant_url, check_compatibility=False)
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


@pytest.mark.integration
def test_compose_search_pipeline(compose_stack: ComposeStack, owner_id: int) -> None:
    needle = f"E2E_NEEDLE_{uuid.uuid4().hex[:10]}"
    document_id = f"search-doc-{uuid.uuid4().hex[:10]}"
    ingest_text = f"Search integration fixture text with unique marker: {needle}."

    ingest_job_id = _enqueue_ingest(compose_stack, owner_id=owner_id, document_id=document_id, text=ingest_text)
    ingest_job = _wait_for_job_terminal(compose_stack, job_id=ingest_job_id, owner_id=owner_id)
    assert ingest_job["status"] == "done"

    search_job_id = _enqueue_search(compose_stack, owner_id=owner_id, query=needle, limit=5)
    search_job = _wait_for_job_terminal(compose_stack, job_id=search_job_id, owner_id=owner_id)

    assert search_job["status"] == "done"
    assert search_job["task_name"] == "tasks.vector_search_task"
    matches = (search_job.get("result") or {}).get("matches") or []
    assert matches
    assert any((match or {}).get("document_id") == document_id for match in matches)
