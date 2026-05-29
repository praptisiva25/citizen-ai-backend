# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.12.0 \
    && python -m pip install -r requirements.txt


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers \
    TORCH_HOME=/home/appuser/.cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence-transformers \
    XDG_CACHE_HOME=/home/appuser/.cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app/uploads /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["python", "app.py"]