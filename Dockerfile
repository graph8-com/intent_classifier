# syntax=docker/dockerfile:1.7-labs
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY main.py ./

# Use uv for fast/locked installs
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

# Add a non-root user
RUN useradd -m appuser
USER appuser

# Default topics path mounted at runtime; override with --topics
ENV TOPICS_PATH=/app/topics.csv

ENTRYPOINT ["uv", "run", "python", "main.py"]
CMD ["--topk", "5", "--topics", "/app/topics.csv"]


