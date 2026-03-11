from __future__ import annotations

import asyncio
import json

import httpx
from telethon import Button
from telethon import TelegramClient, events

from app.config import get_settings


settings = get_settings()
BASE_URL = f"http://api:{settings.api_port}"
HEADERS = {"X-Internal-Token": settings.api_internal_token}


async def post_json(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}{path}", json=payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()


async def post_multipart(path: str, data: dict[str, str], files: dict) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{BASE_URL}{path}", data=data, files=files, headers=HEADERS)
        response.raise_for_status()
        return response.json()


async def get_json(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{BASE_URL}{path}", params=params, headers=HEADERS)
        response.raise_for_status()
        return response.json()


client = TelegramClient(settings.telegram_session_name, settings.telegram_api_id, settings.telegram_api_hash)

HELP_TEXT = (
    "Telegram RAG Bot\n\n"
    "How to use:\n"
    "1. Send any text message to get an answer only from documents uploaded by you.\n"
    "2. Use /ingest <text> to add raw text into the vector database.\n"
    "3. Send a file (PDF, TXT, MD, CSV, JSON, XML, source code, etc.) to ingest it.\n"
    "4. Use /long <query> to run a long background LLM task.\n"
    "5. Use /status <job_id> to check async task progress.\n"
    "6. Use /jobs to browse your jobs by status and open results.\n\n"
    "If no relevant context is found, bot replies: Sorry! Do not have information.\n\n"
    "Commands:\n"
    "/start - show this guide\n"
    "/help - show this guide\n"
    "/ingest <text>\n"
    "<send supported file>\n"
    "/long <query>\n"
    "/status <job_id>\n"
    "/jobs"
)

JOB_STATUSES = ("pending", "running", "done", "failed")
MAX_TELEGRAM_MESSAGE_LEN = 3800


@client.on(events.NewMessage(pattern=r"/start$"))
async def start_handler(event: events.NewMessage.Event) -> None:
    await event.reply(HELP_TEXT)


@client.on(events.NewMessage(pattern=r"/help$"))
async def help_handler(event: events.NewMessage.Event) -> None:
    await event.reply(HELP_TEXT)


@client.on(events.NewMessage(pattern=r"/ingest(?:\s+([\s\S]+))?"))
async def ingest_handler(event: events.NewMessage.Event) -> None:
    text = (event.pattern_match.group(1) or "").strip()
    if not text:
        await event.reply("Usage: /ingest <document text>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await post_json("/tasks/ingest", {"source": "telegram", "text": text, "owner_id": event.sender_id})
    await event.reply(f"Ingestion queued. job_id={data['job_id']}")


@client.on(events.NewMessage(func=lambda e: bool(getattr(e.message, "file", None))))
async def ingest_file_handler(event: events.NewMessage.Event) -> None:
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return
    if not event.message or not event.message.file:
        return

    file_name = event.file.name if event.file else None
    mime_type = event.file.mime_type if event.file else "application/octet-stream"
    payload = await event.download_media(file=bytes)
    if not payload:
        await event.reply("Could not download file content.")
        return

    try:
        data = await post_multipart(
            "/tasks/ingest-file",
            data={"source": "telegram", "owner_id": str(event.sender_id)},
            files={"file": (file_name or "upload.bin", payload, mime_type)},
        )
        await event.reply(f"File ingestion queued. job_id={data['job_id']}")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        await event.reply(f"File ingestion failed: {detail}")


@client.on(events.NewMessage(pattern=r"/long(?:\s+([\s\S]+))?"))
async def long_task_handler(event: events.NewMessage.Event) -> None:
    query = (event.pattern_match.group(1) or "").strip()
    if not query:
        await event.reply("Usage: /long <query>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await post_json("/tasks/llm", {"query": query, "owner_id": event.sender_id})
    await event.reply(f"Long LLM task queued. job_id={data['job_id']}")


@client.on(events.NewMessage(pattern=r"/status(?:\s+([a-zA-Z0-9\-]+))?"))
async def status_handler(event: events.NewMessage.Event) -> None:
    job_id = (event.pattern_match.group(1) or "").strip()
    if not job_id:
        await event.reply("Usage: /status <job_id>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    try:
        payload = await get_json(f"/tasks/{job_id}", params={"owner_id": event.sender_id})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            await event.reply("Job not found")
            return
        raise

    await event.reply(f"job_id={payload['job_id']} status={payload['status']}")


def _status_keyboard() -> list[list[Button]]:
    return [
        [
            Button.inline(JOB_STATUSES[0].capitalize(), f"jobs:{JOB_STATUSES[0]}".encode("utf-8")),
            Button.inline(JOB_STATUSES[1].capitalize(), f"jobs:{JOB_STATUSES[1]}".encode("utf-8")),
        ],
        [
            Button.inline(JOB_STATUSES[2].capitalize(), f"jobs:{JOB_STATUSES[2]}".encode("utf-8")),
            Button.inline(JOB_STATUSES[3].capitalize(), f"jobs:{JOB_STATUSES[3]}".encode("utf-8")),
        ],
        [Button.inline("All", b"jobs:all")],
    ]


def _short_job_line(job: dict) -> str:
    task = (job.get("task_name") or "").replace("tasks.", "")
    return f"{job.get('job_id')} | {job.get('status')} | {task}"


def _job_result_text(job: dict) -> str:
    parts: list[str] = [
        f"job_id={job.get('job_id')}",
        f"status={job.get('status')}",
        f"task={job.get('task_name')}",
    ]
    payload = job.get("payload")
    if payload:
        parts.append(f"payload={json.dumps(payload, ensure_ascii=True)}")
    result = job.get("result")
    if result is not None:
        parts.append(f"result={json.dumps(result, ensure_ascii=True)}")
    error = job.get("error")
    if error:
        parts.append(f"error={error}")
    text = "\n".join(parts)
    if len(text) > MAX_TELEGRAM_MESSAGE_LEN:
        return f"{text[:MAX_TELEGRAM_MESSAGE_LEN]}...\n(truncated)"
    return text


@client.on(events.NewMessage(pattern=r"/jobs$"))
async def jobs_handler(event: events.NewMessage.Event) -> None:
    await event.reply("Select status to view your jobs:", buttons=_status_keyboard())


@client.on(events.CallbackQuery(pattern=rb"jobs:(pending|running|done|failed|all)"))
async def jobs_list_callback(event: events.CallbackQuery.Event) -> None:
    if not event.sender_id:
        await event.answer("Cannot identify your user id.", alert=True)
        return
    status = event.pattern_match.group(1).decode("utf-8")
    selected_status = None if status == "all" else status
    data = await post_json(
        "/tasks/list",
        {"owner_id": event.sender_id, "status": selected_status, "limit": 10},
    )
    jobs = data.get("jobs", [])
    if not jobs:
        await event.edit(
            f"No jobs found for status={status}.",
            buttons=_status_keyboard(),
        )
        return

    lines = [f"Your jobs (status={status}, latest 10):"]
    buttons: list[list[Button]] = []
    for job in jobs:
        lines.append(_short_job_line(job))
        job_id = str(job.get("job_id") or "")
        if job_id:
            buttons.append([Button.inline(f"Open {job_id[:8]}...", f"job:{job_id}".encode("utf-8"))])
    buttons.append([Button.inline("Back to statuses", b"jobs:all")])
    await event.edit("\n".join(lines), buttons=buttons)


@client.on(events.CallbackQuery(pattern=rb"job:([a-zA-Z0-9\-]+)"))
async def job_result_callback(event: events.CallbackQuery.Event) -> None:
    if not event.sender_id:
        await event.answer("Cannot identify your user id.", alert=True)
        return

    job_id = event.pattern_match.group(1).decode("utf-8")
    try:
        job = await get_json(f"/tasks/{job_id}", params={"owner_id": event.sender_id})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            await event.answer("Job not found", alert=True)
            return
        raise

    await event.edit(_job_result_text(job), buttons=[[Button.inline("Back to statuses", b"jobs:all")]])


@client.on(events.NewMessage)
async def ask_handler(event: events.NewMessage.Event) -> None:
    if event.message and event.message.file:
        return
    text = (event.raw_text or "").strip()
    if not text or text.startswith("/"):
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await post_json("/ask", {"query": text, "owner_id": event.sender_id})
    await event.reply(data["answer"])


async def main() -> None:
    await client.start(bot_token=settings.telegram_bot_token)
    print("Telethon bot started")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
