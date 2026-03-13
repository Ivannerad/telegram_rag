from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Awaitable

import httpx
from telethon import Button
from telethon import TelegramClient, events

from app.business import NO_INFO_RESPONSE
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
    "1. In private chat: send text directly or use /ask <question>.\n"
    "2. Use /ingest <text> to add raw text into the vector database (private or group chat).\n"
    "3. Send a file (PDF, TXT, MD, CSV, JSON, XML, source code, etc.) to ingest it.\n"
    "4. Bind groups via /group_add <group_id> [label] in private chat or /group_add [label] inside a group.\n"
    "5. In a bound group bot answers only to /ask <question>.\n"
    "6. Use /long <query>, /status <job_id>, /jobs for background tasks.\n\n"
    f"If no relevant context is found, bot replies: {NO_INFO_RESPONSE}\n\n"
    "Commands:\n"
    "/start - show this guide\n"
    "/help - show this guide\n"
    "/ingest <text>\n"
    "<send supported file>\n"
    "/ask <question>\n"
    "/group_add <group_id> [group_label]\n"
    "/group_remove <group_id> (or /group_remove in a group chat)\n"
    "/groups\n"
    "/long <query>\n"
    "/status <job_id>\n"
    "/jobs"
)

JOB_STATUSES = ("pending", "running", "done", "failed")
MAX_TELEGRAM_MESSAGE_LEN = 3800
GROUP_LIMIT_WINDOW_SECONDS = 60
GROUP_LIMIT_MAX_REQUESTS = 20
USER_LIMIT_WINDOW_SECONDS = 60
USER_LIMIT_MAX_REQUESTS = 5
USER_COOLDOWN_SECONDS = 3.0

GROUP_USAGE_BUCKETS: dict[int, deque[float]] = {}
USER_USAGE_BUCKETS: dict[tuple[int, int], deque[float]] = {}
USER_LAST_REQUEST_TS: dict[tuple[int, int], float] = {}

CMD_SUFFIX = r"(?:@[\w_]+)?"
API_UNAVAILABLE_MESSAGE = "Backend service is temporarily unavailable. Please try again in a moment."
API_ERROR_MESSAGE = "Backend request failed. Please try again in a moment."
MAX_CALLBACK_ALERT_LEN = 200


def _parse_group_id(value: str) -> int:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Group id is required")
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError("Group id must be numeric (example: -1001234567890)") from exc


def _http_error_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    raw = (response.text or "").strip()
    return raw or None


async def _safe_api_call(
    event: Any,
    api_call: Awaitable[dict],
    *,
    status_messages: dict[int, str] | None = None,
) -> dict | None:
    async def send_message(text: str) -> None:
        if hasattr(event, "answer"):
            alert_text = text[:MAX_CALLBACK_ALERT_LEN]
            await event.answer(alert_text, alert=True)
            return
        if hasattr(event, "reply"):
            await event.reply(text)

    try:
        return await api_call
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_messages and status_code in status_messages:
            await send_message(status_messages[status_code])
            return None
        detail = _http_error_detail(exc.response)
        if detail:
            detail = detail[:200]
            await send_message(f"{API_ERROR_MESSAGE} (HTTP {status_code}: {detail})")
        else:
            await send_message(f"{API_ERROR_MESSAGE} (HTTP {status_code}).")
        return None
    except httpx.RequestError:
        await send_message(API_UNAVAILABLE_MESSAGE)
        return None


def _evict_old(bucket: deque[float], window_seconds: int, now_ts: float) -> None:
    while bucket and now_ts - bucket[0] > window_seconds:
        bucket.popleft()


def _group_limit_exceeded(group_id: int) -> bool:
    now_ts = time.monotonic()
    group_bucket = GROUP_USAGE_BUCKETS.setdefault(group_id, deque())
    _evict_old(group_bucket, GROUP_LIMIT_WINDOW_SECONDS, now_ts)
    return len(group_bucket) >= GROUP_LIMIT_MAX_REQUESTS


def _consume_group_limit(group_id: int) -> None:
    now_ts = time.monotonic()
    group_bucket = GROUP_USAGE_BUCKETS.setdefault(group_id, deque())
    _evict_old(group_bucket, GROUP_LIMIT_WINDOW_SECONDS, now_ts)
    group_bucket.append(now_ts)


def _user_limits_exceeded(group_id: int, sender_id: int) -> str | None:
    now_ts = time.monotonic()
    user_key = (group_id, sender_id)
    user_bucket = USER_USAGE_BUCKETS.setdefault(user_key, deque())
    _evict_old(user_bucket, USER_LIMIT_WINDOW_SECONDS, now_ts)
    if len(user_bucket) >= USER_LIMIT_MAX_REQUESTS:
        return "Too many /ask requests from your account in this group. Please wait a minute."

    last_ts = USER_LAST_REQUEST_TS.get(user_key)
    if last_ts is not None and now_ts - last_ts < USER_COOLDOWN_SECONDS:
        return "Please wait a few seconds before sending another /ask request."

    user_bucket.append(now_ts)
    USER_LAST_REQUEST_TS[user_key] = now_ts
    return None


@client.on(events.NewMessage(pattern=rf"/start{CMD_SUFFIX}$"))
async def start_handler(event: events.NewMessage.Event) -> None:
    await event.reply(HELP_TEXT)


@client.on(events.NewMessage(pattern=rf"/help{CMD_SUFFIX}$"))
async def help_handler(event: events.NewMessage.Event) -> None:
    await event.reply(HELP_TEXT)


@client.on(events.NewMessage(pattern=rf"/ingest{CMD_SUFFIX}(?:\s+([\s\S]+))?"))
async def ingest_handler(event: events.NewMessage.Event) -> None:
    text = (event.pattern_match.group(1) or "").strip()
    if not text:
        await event.reply("Usage: /ingest <document text>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await _safe_api_call(
        event,
        post_json("/tasks/ingest", {"source": "telegram", "text": text, "owner_id": event.sender_id}),
    )
    if not data:
        return
    job_id = data.get("job_id")
    await event.reply(f"Ingestion queued. job_id={job_id}\nNext: /status {job_id} or /jobs")


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

    data = await _safe_api_call(
        event,
        post_multipart(
            "/tasks/ingest-file",
            data={"source": "telegram", "owner_id": str(event.sender_id)},
            files={"file": (file_name or "upload.bin", payload, mime_type)},
        ),
    )
    if not data:
        return
    job_id = data.get("job_id")
    await event.reply(f"File ingestion queued. job_id={job_id}\nNext: /status {job_id} or /jobs")


@client.on(events.NewMessage(pattern=rf"/long{CMD_SUFFIX}(?:\s+([\s\S]+))?"))
async def long_task_handler(event: events.NewMessage.Event) -> None:
    query = (event.pattern_match.group(1) or "").strip()
    if not query:
        await event.reply("Usage: /long <query>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await _safe_api_call(event, post_json("/tasks/llm", {"query": query, "owner_id": event.sender_id}))
    if not data:
        return
    await event.reply(f"Long LLM task queued. job_id={data['job_id']}")


@client.on(events.NewMessage(pattern=rf"/status{CMD_SUFFIX}(?:\s+([a-zA-Z0-9\-]+))?"))
async def status_handler(event: events.NewMessage.Event) -> None:
    job_id = (event.pattern_match.group(1) or "").strip()
    if not job_id:
        await event.reply("Usage: /status <job_id>")
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    payload = await _safe_api_call(
        event,
        get_json(f"/tasks/{job_id}", params={"owner_id": event.sender_id}),
        status_messages={404: "Job not found"},
    )
    if not payload:
        return

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
    task_name = str(job.get("task_name") or "")
    task_short = task_name.replace("tasks.", "") if task_name else "unknown"
    parts: list[str] = [
        "Job details",
        f"- ID: {job.get('job_id')}",
        f"- Status: {job.get('status')}",
        f"- Task: {task_short}",
    ]

    summary_lines: list[str] = []
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    result = job.get("result") if isinstance(job.get("result"), dict) else None
    if task_name == "tasks.ingest_document":
        source = payload.get("source")
        document_id = (result or {}).get("document_id") or payload.get("document_id")
        chunks = (result or {}).get("chunks")
        vectors = (result or {}).get("vectors_upserted")
        if document_id:
            summary_lines.append(f"document_id={document_id}")
        if source:
            summary_lines.append(f"source={source}")
        if chunks is not None:
            summary_lines.append(f"chunks={chunks}")
        if vectors is not None:
            summary_lines.append(f"vectors_upserted={vectors}")
    elif task_name == "tasks.vector_search_task":
        match_count = len(result.get("matches", [])) if result else 0
        if payload.get("query"):
            summary_lines.append(f"query={payload['query']}")
        summary_lines.append(f"matches={match_count}")
    elif task_name == "tasks.long_llm_task" and result:
        if result.get("query"):
            summary_lines.append(f"query={result['query']}")
        if result.get("retrieval_count") is not None:
            summary_lines.append(f"retrieval_count={result['retrieval_count']}")
        if result.get("answer"):
            answer = str(result["answer"]).strip()
            summary_lines.append(f"answer={answer[:120]}{'...' if len(answer) > 120 else ''}")

    if summary_lines:
        parts.append("")
        parts.append("Summary")
        parts.extend(f"- {line}" for line in summary_lines)

    if result and not summary_lines:
        compact = json.dumps(result, ensure_ascii=True)
        parts.append("")
        parts.append("Result")
        parts.append(compact[:1000] + ("..." if len(compact) > 1000 else ""))

    error = job.get("error")
    if error:
        parts.append("")
        parts.append("Error")
        parts.append(str(error))

    text = "\n".join(parts)
    if len(text) > MAX_TELEGRAM_MESSAGE_LEN:
        return f"{text[:MAX_TELEGRAM_MESSAGE_LEN]}...\n(truncated)"
    return text


@client.on(events.NewMessage(pattern=rf"/jobs{CMD_SUFFIX}$"))
async def jobs_handler(event: events.NewMessage.Event) -> None:
    await event.reply("Select status to view your jobs:", buttons=_status_keyboard())


@client.on(events.NewMessage(pattern=rf"/group_add{CMD_SUFFIX}(?:\s+([^\s]+)(?:\s+([\s\S]+))?)?$"))
async def group_add_handler(event: events.NewMessage.Event) -> None:
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return
    group_id_raw = (event.pattern_match.group(1) or "").strip()
    group_label = (event.pattern_match.group(2) or "").strip() or None
    if event.is_group or event.is_channel:
        if group_id_raw:
            try:
                group_id = _parse_group_id(group_id_raw)
            except ValueError as exc:
                await event.reply(str(exc))
                return
        else:
            group_id = event.chat_id
            if group_id is None:
                await event.reply("Cannot detect this group id.")
                return
    else:
        if not group_id_raw:
            await event.reply("Usage: /group_add <group_id> [group_label]")
            return
        try:
            group_id = _parse_group_id(group_id_raw)
        except ValueError as exc:
            await event.reply(str(exc))
            return

    payload = {"group_id": group_id, "owner_id": event.sender_id, "group_label": group_label}
    data = await _safe_api_call(
        event,
        post_json("/groups/bind", payload),
        status_messages={409: "This group is already bound to another account."},
    )
    if not data:
        return
    binding = data.get("binding") or {}
    await event.reply(
        f"Group bound: group_id={binding.get('group_id', group_id)} owner_id={binding.get('owner_id', event.sender_id)}"
    )


@client.on(events.NewMessage(pattern=rf"/group_remove{CMD_SUFFIX}(?:\s+([^\s]+))?$"))
async def group_remove_handler(event: events.NewMessage.Event) -> None:
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return
    group_id_raw = (event.pattern_match.group(1) or "").strip()
    if event.is_group or event.is_channel:
        if group_id_raw:
            try:
                group_id = _parse_group_id(group_id_raw)
            except ValueError as exc:
                await event.reply(str(exc))
                return
        else:
            group_id = event.chat_id
            if group_id is None:
                await event.reply("Cannot detect this group id.")
                return
    else:
        if not group_id_raw:
            await event.reply("Usage: /group_remove <group_id>")
            return
        try:
            group_id = _parse_group_id(group_id_raw)
        except ValueError as exc:
            await event.reply(str(exc))
            return

    data = await _safe_api_call(event, post_json("/groups/unbind", {"group_id": group_id, "owner_id": event.sender_id}))
    if not data:
        return
    if data.get("deleted"):
        await event.reply(f"Group removed: {group_id}")
        return
    await event.reply("Group is not bound to your account.")


@client.on(events.NewMessage(pattern=rf"/groups{CMD_SUFFIX}$"))
async def groups_handler(event: events.NewMessage.Event) -> None:
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return
    data = await _safe_api_call(event, post_json("/groups/list", {"owner_id": event.sender_id, "limit": 100}))
    if not data:
        return
    groups = data.get("groups", [])
    if not groups:
        await event.reply("No bound groups yet. Use /group_add <group_id> [group_label].")
        return

    lines = ["Your bound groups:"]
    for item in groups:
        label = item.get("group_label")
        suffix = f" ({label})" if label else ""
        lines.append(f"- {item.get('group_id')}{suffix}")
    await event.reply("\n".join(lines))


@client.on(events.NewMessage(pattern=rf"/ask{CMD_SUFFIX}(?:\s+([\s\S]+))?"))
async def ask_command_handler(event: events.NewMessage.Event) -> None:
    query = (event.pattern_match.group(1) or "").strip()
    if not query:
        await event.reply("Usage: /ask <question>")
        return
    if len(query) > 1000:
        await event.reply("Question is too long. Please keep /ask under 1000 characters.")
        return

    is_group_chat = bool(event.is_group or event.is_channel)
    if is_group_chat:
        group_id = event.chat_id
        if group_id is None:
            await event.reply("Cannot detect group id for this message.")
            return
        if not event.sender_id:
            await event.reply("Cannot identify your user id.")
            return

        resolved = await _safe_api_call(
            event,
            get_json("/groups/resolve", params={"group_id": group_id}),
            status_messages={
                404: "This group is not bound to any account. Add it in private chat with /group_add <group_id>."
            },
        )
        if not resolved:
            return

        binding = resolved.get("binding") or {}
        owner_id = binding.get("owner_id")
        if not owner_id:
            await event.reply("Group binding is invalid. Please re-bind the group.")
            return
        if _group_limit_exceeded(group_id):
            await event.reply("Group request limit reached. Please wait a minute before more /ask requests.")
            return
        user_limit_error = _user_limits_exceeded(group_id, event.sender_id)
        if user_limit_error:
            await event.reply(user_limit_error)
            return
        _consume_group_limit(group_id)

        data = await _safe_api_call(event, post_json("/ask", {"query": query, "owner_id": owner_id}))
        if not data:
            return
        await event.reply(data["answer"])
        return

    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return
    data = await _safe_api_call(event, post_json("/ask", {"query": query, "owner_id": event.sender_id}))
    if not data:
        return
    await event.reply(data["answer"])


@client.on(events.CallbackQuery(pattern=rb"jobs:(pending|running|done|failed|all)"))
async def jobs_list_callback(event: events.CallbackQuery.Event) -> None:
    if not event.sender_id:
        await event.answer("Cannot identify your user id.", alert=True)
        return
    status = event.pattern_match.group(1).decode("utf-8")
    selected_status = None if status == "all" else status
    data = await _safe_api_call(
        event,
        post_json(
            "/tasks/list",
            {"owner_id": event.sender_id, "status": selected_status, "limit": 10},
        ),
    )
    if not data:
        return
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
    job = await _safe_api_call(
        event,
        get_json(f"/tasks/{job_id}", params={"owner_id": event.sender_id}),
        status_messages={404: "Job not found"},
    )
    if not job:
        return

    await event.edit(_job_result_text(job), buttons=[[Button.inline("Back to statuses", b"jobs:all")]])


@client.on(events.NewMessage)
async def ask_handler(event: events.NewMessage.Event) -> None:
    if event.is_group or event.is_channel:
        return
    if event.message and event.message.file:
        return
    text = (event.raw_text or "").strip()
    if not text or text.startswith("/"):
        return
    if not event.sender_id:
        await event.reply("Cannot identify your user id.")
        return

    data = await _safe_api_call(event, post_json("/ask", {"query": text, "owner_id": event.sender_id}))
    if not data:
        return
    await event.reply(data["answer"])


async def main() -> None:
    await client.start(bot_token=settings.telegram_bot_token)
    print("Telethon bot started")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
