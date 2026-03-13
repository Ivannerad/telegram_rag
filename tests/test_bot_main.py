from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable

import httpx

from app.config import get_settings
from app.business import NO_INFO_RESPONSE

if not (os.getenv("TELEGRAM_API_ID") or "").strip():
    os.environ["TELEGRAM_API_ID"] = "1"
if not (os.getenv("TELEGRAM_API_HASH") or "").strip():
    os.environ["TELEGRAM_API_HASH"] = "test-hash"
get_settings.cache_clear()

from bot import main as bot_main


class _PatternMatch:
    def __init__(self, values: dict[int, str | None]) -> None:
        self._values = values

    def group(self, index: int) -> str | None:
        return self._values.get(index)


class _Event:
    def __init__(
        self,
        *,
        sender_id: int = 123,
        is_group: bool = False,
        is_channel: bool = False,
        chat_id: int | None = None,
        pattern_group_1: str | None = None,
    ) -> None:
        self.sender_id = sender_id
        self.is_group = is_group
        self.is_channel = is_channel
        self.chat_id = chat_id
        self.pattern_match = _PatternMatch({1: pattern_group_1})
        self.replies: list[str] = []

    async def reply(self, text: str, **_: object) -> None:
        self.replies.append(text)


class _CallbackEvent:
    def __init__(self, *, sender_id: int = 123, status: str = "all", job_id: str = "job-1") -> None:
        self.sender_id = sender_id
        self.pattern_match = _PatternMatch({1: status.encode("utf-8")})
        self.answers: list[tuple[str, bool]] = []
        self.edits: list[str] = []
        self._job_id = job_id

    async def answer(self, text: str, *, alert: bool = False) -> None:
        self.answers.append((text, alert))

    async def edit(self, text: str, **_: object) -> None:
        self.edits.append(text)


def _run(coro: Awaitable[object]) -> None:
    asyncio.run(coro)


def test_ask_command_handles_request_error(monkeypatch) -> None:
    async def fake_post_json(path: str, payload: dict) -> dict:
        raise httpx.RequestError("network down", request=httpx.Request("POST", f"http://test{path}"))

    monkeypatch.setattr(bot_main, "post_json", fake_post_json)
    event = _Event(pattern_group_1="what is this?")

    _run(bot_main.ask_command_handler(event))

    assert event.replies == [bot_main.API_UNAVAILABLE_MESSAGE]


def test_ask_command_handles_http_status_error(monkeypatch) -> None:
    async def fake_post_json(path: str, payload: dict) -> dict:
        request = httpx.Request("POST", f"http://test{path}")
        response = httpx.Response(500, request=request, json={"detail": "backend exploded"})
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr(bot_main, "post_json", fake_post_json)
    event = _Event(pattern_group_1="what is this?")

    _run(bot_main.ask_command_handler(event))

    assert len(event.replies) == 1
    assert event.replies[0] == f"{bot_main.API_ERROR_MESSAGE} (HTTP 500: backend exploded)"


def test_no_info_text_consistency() -> None:
    assert bot_main.NO_INFO_RESPONSE == NO_INFO_RESPONSE
    assert NO_INFO_RESPONSE in bot_main.HELP_TEXT
    assert "Sorry! Do not have information." not in bot_main.HELP_TEXT


def test_job_result_text_uses_human_readable_summary() -> None:
    job = {
        "job_id": "job-123",
        "status": "done",
        "task_name": "tasks.ingest_document",
        "payload": {"document_id": "doc-1", "source": "telegram", "owner_id": 123},
        "result": {"document_id": "doc-1", "source": "telegram", "owner_id": 123, "chunks": 4, "vectors_upserted": 4},
        "error": None,
    }

    text = bot_main._job_result_text(job)

    assert "Job details" in text
    assert "- Task: ingest_document" in text
    assert "Summary" in text
    assert "- document_id=doc-1" in text
    assert "- chunks=4" in text
    assert "payload=" not in text
    assert "result={" not in text


def test_group_remove_uses_current_group_id_when_missing_argument(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, payload: dict) -> dict:
        captured["path"] = path
        captured["payload"] = payload
        return {"deleted": True}

    monkeypatch.setattr(bot_main, "post_json", fake_post_json)
    event = _Event(is_group=True, chat_id=-1001234567890, pattern_group_1=None)

    _run(bot_main.group_remove_handler(event))

    assert captured["path"] == "/groups/unbind"
    assert captured["payload"] == {"group_id": -1001234567890, "owner_id": 123}
    assert event.replies == ["Group removed: -1001234567890"]


def test_jobs_callback_request_error_uses_alert(monkeypatch) -> None:
    async def fake_post_json(path: str, payload: dict) -> dict:
        raise httpx.RequestError("network down", request=httpx.Request("POST", f"http://test{path}"))

    monkeypatch.setattr(bot_main, "post_json", fake_post_json)
    event = _CallbackEvent(status="all")

    _run(bot_main.jobs_list_callback(event))

    assert event.answers == [(bot_main.API_UNAVAILABLE_MESSAGE, True)]
    assert not event.edits


def test_jobs_callback_http_error_is_capped_for_telegram_alert(monkeypatch) -> None:
    long_detail = "x" * 500

    async def fake_post_json(path: str, payload: dict) -> dict:
        request = httpx.Request("POST", f"http://test{path}")
        response = httpx.Response(500, request=request, json={"detail": long_detail})
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    monkeypatch.setattr(bot_main, "post_json", fake_post_json)
    event = _CallbackEvent(status="all")

    _run(bot_main.jobs_list_callback(event))

    assert len(event.answers) == 1
    alert_text, is_alert = event.answers[0]
    assert is_alert is True
    assert len(alert_text) <= bot_main.MAX_CALLBACK_ALERT_LEN
