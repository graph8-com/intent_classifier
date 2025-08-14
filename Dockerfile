# syntax=docker/dockerfile:1.7-labs
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY main.py worker.py clickhouse_check.py ./
# Optional default topics bundled; can be overridden by mounting
COPY topics.csv ./
COPY .env ./

# Use uv for fast/locked installs
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

# Add a non-root user and fix permissions on app dir and venv
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default topics path; can be set/overridden via env or volume
ENV TOPICS_PATH=/app/topics.csv
ENV PATH=/app/.venv/bin:$PATH

# Run the installed console script directly (no build at runtime)
ENTRYPOINT ["intent-worker"]


