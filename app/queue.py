from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from app.config import get_settings


def create_celery() -> Celery:
    settings = get_settings()

    app = Celery("telegram_rag", broker=settings.rabbitmq_url)
    app.conf.update(
        task_default_exchange="tasks.direct",
        task_default_exchange_type="direct",
        task_default_routing_key="task.default",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_queues=(
            Queue("ingest_queue", Exchange("tasks.direct", type="direct"), routing_key="task.ingest.document"),
            Queue("llm_queue", Exchange("tasks.direct", type="direct"), routing_key="task.llm.long"),
            Queue("maintenance_queue", Exchange("tasks.direct", type="direct"), routing_key="task.maintenance.search"),
            Queue("dlq", Exchange("tasks.dlq", type="direct"), routing_key="task.dlq"),
        ),
        task_routes={
            "tasks.ingest_document": {"queue": "ingest_queue", "routing_key": "task.ingest.document"},
            "tasks.long_llm_task": {"queue": "llm_queue", "routing_key": "task.llm.long"},
            "tasks.vector_search_task": {"queue": "maintenance_queue", "routing_key": "task.maintenance.search"},
        },
    )
    return app
