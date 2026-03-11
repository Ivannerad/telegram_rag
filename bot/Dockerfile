FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md /app/
RUN poetry install --no-root

COPY . /app
