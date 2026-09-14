# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build.
#   builder: installs dependencies into a virtualenv (needs no compiler in the
#            final image, and the wheel cache layer survives code edits)
#   runtime: slim base + venv + source + artifacts, running as a non-root user
#
# Layer order is deliberate: requirements.txt is copied and installed BEFORE the
# source, so editing a .py file re-uses the cached dependency layer. Copying the
# whole context first is the classic mistake that makes every build a full
# reinstall.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED: logs reach the collector immediately instead of sitting in
# a 4 KB buffer, which is what makes a crashed container look silent.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/
# Artifacts are baked in so the image is self-contained and immutable: the image
# tag then identifies the model, not just the code. The alternative (mount them
# from Blob Storage at boot) is better when models are large or rotate faster
# than code; at 23 MB, baking wins on simplicity and start-up determinism.
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level liveness. Kubernetes ignores this and uses its own probes
# (deploy/k8s/deployment.yaml); Docker Compose and Azure Container Apps use it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

# 0.0.0.0, not 127.0.0.1: binding to loopback inside a container makes the port
# unreachable from outside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
