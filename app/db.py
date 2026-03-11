from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.config import get_settings


def get_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.postgres_dsn, autocommit=True, row_factory=dict_row)


def ensure_schema() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bg_jobs (
                    job_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    result JSONB,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_group_bindings (
                    group_id BIGINT PRIMARY KEY,
                    owner_id BIGINT NOT NULL,
                    group_label TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )


def create_job(job_id: str, task_name: str, payload: dict[str, Any]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bg_jobs (job_id, task_name, status, payload)
                VALUES (%s, %s, 'pending', %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    task_name = EXCLUDED.task_name,
                    status = 'pending',
                    payload = EXCLUDED.payload,
                    updated_at = NOW();
                """,
                (job_id, task_name, Json(payload)),
            )


def update_job(
    job_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bg_jobs
                SET status = %s,
                    result = %s,
                    error = %s,
                    updated_at = NOW()
                WHERE job_id = %s;
                """,
                (status, Json(result) if result is not None else None, error, job_id),
            )


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM bg_jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()

    if row is None:
        return None

    for key in ("payload", "result"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                row[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return row


def list_jobs(owner_id: int, status: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 50))
    with get_conn() as conn:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT *
                    FROM bg_jobs
                    WHERE payload->>'owner_id' = %s
                      AND status = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (str(owner_id), status, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM bg_jobs
                    WHERE payload->>'owner_id' = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (str(owner_id), limit),
                )
            rows = cur.fetchall()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        for key in ("payload", "result"):
            value = row.get(key)
            if isinstance(value, str):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        normalized.append(row)
    return normalized


def upsert_group_binding(group_id: int, owner_id: int, group_label: str | None = None) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telegram_group_bindings (group_id, owner_id, group_label)
                VALUES (%s, %s, %s)
                ON CONFLICT (group_id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id,
                    group_label = EXCLUDED.group_label,
                    updated_at = NOW()
                RETURNING *;
                """,
                (group_id, owner_id, group_label),
            )
            row = cur.fetchone()
    return row or {}


def delete_group_binding(group_id: int, owner_id: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM telegram_group_bindings
                WHERE group_id = %s AND owner_id = %s;
                """,
                (group_id, owner_id),
            )
            return cur.rowcount > 0


def list_group_bindings(owner_id: int, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 500))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM telegram_group_bindings
                WHERE owner_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (owner_id, limit),
            )
            rows = cur.fetchall()
    return rows


def get_group_binding(group_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM telegram_group_bindings
                WHERE group_id = %s;
                """,
                (group_id,),
            )
            row = cur.fetchone()
    return row
