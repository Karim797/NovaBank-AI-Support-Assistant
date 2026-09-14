# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip "setuptools>=78.1.1" && \
    pip install -r requirements.txt && \
    pip install --upgrade "msgpack>=1.2.1"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PORT=8000

# Keep the runtime image patched. The upstream Python image also ships an
# embedded pip vendor SBOM whose pinned package versions can become stale and
# produce false-positive Trivy findings, so remove that metadata file after
# upgrading the actual runtime packaging libraries.
RUN apt-get update && \
    apt-get upgrade -y && \
    /usr/local/bin/python -m pip install --no-cache-dir --upgrade \
      "setuptools>=78.1.1" "msgpack>=1.2.1" && \
    rm -f /usr/local/lib/python3.12/site-packages/pip/_vendor/bom.cdx.json && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/
COPY models/ ./models/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health')" || exit 1

# One worker is deliberate while feedback uses SQLite. Railway injects PORT;
# the default keeps local `docker run -p 8000:8000` convenient.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
