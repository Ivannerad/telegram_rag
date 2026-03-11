from app.queue import create_celery
from app.logging_setup import configure_logging

configure_logging()
celery_app = create_celery()
celery_app.conf.imports = ("worker.tasks",)
