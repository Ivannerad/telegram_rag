# telegram_rag

Docker-first Telegram RAG bot scaffold with:
- `api` (FastAPI)
- `telegram-bot` (Telethon)
- `worker` (Celery)
- `rabbitmq` (queue broker)
- `postgres` (job status + metadata)
- `qdrant` (vector database)

## Quick start

1. Create env file:
```bash
cp .env.example .env
```

2. Fill Telegram and token values in `.env`:
- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELEGRAM_BOT_TOKEN`
- `API_INTERNAL_TOKEN`
- `OPENAI_API_KEY`

3. Start stack:
```bash
docker compose up --build
```

## Dependency management (Poetry)
- Dependencies are defined in `pyproject.toml`.
- Docker images install dependencies with Poetry.
- Local development uses project virtualenv: `.venv` (`poetry.toml`).
- For local dev:
```bash
poetry config --local virtualenvs.in-project true
poetry install
poetry run uvicorn api.main:app --reload
```

## OpenAI connectivity check (local)
```bash
poetry run python scripts/check_openai_connection.py
poetry run python scripts/check_openai_connection.py --chat
```

## Telegram commands
- `/ingest <text>`: enqueue document ingestion
- send supported file attachment (PDF/TXT/MD/CSV/JSON/XML/code): extract text then enqueue ingestion
- `/long <query>`: enqueue long LLM task
- `/status <job_id>`: check background task status
- `/jobs`: open inline buttons to browse your jobs by status and open result details
- any plain text message: synchronous `/ask` call

## API endpoints
- `GET /health`
- `POST /ask`
- `POST /tasks/ingest`
- `POST /tasks/ingest-file` (multipart file upload)
- `POST /tasks/llm`
- `POST /tasks/search`
- `POST /tasks/list`
- `GET /tasks/{job_id}`

All non-health endpoints require header:
- `X-Internal-Token: <API_INTERNAL_TOKEN>`

## Queue topology
- Exchange: `tasks.direct`
- Queues:
  - `ingest_queue` (`tasks.ingest_document`)
  - `llm_queue` (`tasks.long_llm_task`)
  - `maintenance_queue` (`tasks.vector_search_task`)
  - `dlq`

## Notes
- OpenAI integration is implemented through LangChain (`langchain-openai`).
- Chat model: `OPENAI_MODEL` (default `gpt-5`).
- Embedding model: `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-large`).
- Prompt: `LLM_SYSTEM_PROMPT` (enforces answer only from vector context).
- Retrieval cutoff: `VECTOR_SCORE_THRESHOLD` (minimum similarity score).
- If `LLM_PROVIDER` is not `openai` or no API key is set, code falls back to dummy local behavior.
- Worker owns long-running business logic and consumes RabbitMQ tasks.
