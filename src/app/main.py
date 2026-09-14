"""FastAPI application.

Two things here are load-bearing and are the usual interview questions:

1. **Artifacts load once, in the lifespan handler.** Loading a joblib pipeline is
   ~200 ms; doing it per request would put that on every call and multiply
   memory by the worker count. The state lives on `app.state`, not in a module
   global, so tests can build an app with a stubbed state.

2. **/health and /ready are different things.**
   - `/health` = this process is alive and serving. Kubernetes maps it to
     livenessProbe; failing it *restarts the container*.
   - `/ready` = the model, the index and the LLM client are loaded and the
     instance can take traffic. Mapped to readinessProbe; failing it only
     *removes the pod from the load balancer*.
   Wiring liveness to a dependency check is a classic outage generator: a slow
   dependency then kills every pod in a rolling restart loop.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.classifier import IntentClassifier
from app.config import settings
from app.feedback import FeedbackStore, ensure_parent
from app.kb import corpus_sha256
from app.llm import build_llm
from app.logging_config import configure_logging, request_id_var, text_fingerprint
from app.observability import RateLimiter, metrics
from app.retrieval import KnowledgeIndex  # noqa: F401  (needed for joblib unpickling)
from app.router import SupportRouter
from app.schemas import (
    BatchChatRequest,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ReadyResponse,
)

logger = logging.getLogger(__name__)
limiter = RateLimiter(settings.rate_limit_per_minute)


def _load_state(app: FastAPI) -> None:
    import joblib

    app.state.errors = {}
    try:
        app.state.classifier = IntentClassifier.load(settings.resolve(settings.model_path))
    except Exception as exc:  # noqa: BLE001 - startup must report, not crash silently
        app.state.classifier = None
        app.state.errors["classifier"] = str(exc)
        logger.error("classifier_load_failed", exc_info=exc)

    app.state.corpus_sha256 = None
    app.state.corpus_fresh = False
    try:
        app.state.index = joblib.load(settings.resolve(settings.index_path))
        current_corpus = corpus_sha256(settings.resolve(settings.knowledge_base_dir))
        artifact_corpus = str(app.state.index.metadata.get("corpus_sha256", ""))
        app.state.corpus_sha256 = current_corpus
        app.state.corpus_fresh = bool(artifact_corpus) and artifact_corpus == current_corpus
        if not app.state.corpus_fresh:
            app.state.errors["corpus"] = (
                "knowledge-base content does not match the retrieval index; rebuild the index"
            )
            logger.error(
                "corpus_index_mismatch",
                extra={"artifact_corpus_sha256": artifact_corpus, "corpus_sha256": current_corpus},
            )
        logger.info(
            "index_loaded",
            extra={
                "index_version": app.state.index.version,
                "n_chunks": app.state.index.n_chunks,
                "corpus_sha256": current_corpus,
                "corpus_fresh": app.state.corpus_fresh,
            },
        )
    except Exception as exc:  # noqa: BLE001
        app.state.index = None
        app.state.errors["index"] = str(exc)
        logger.error("index_load_failed", exc_info=exc)

    try:
        app.state.llm = build_llm(settings)
    except Exception as exc:  # noqa: BLE001
        app.state.llm = None
        app.state.errors["llm"] = str(exc)
        logger.error("llm_init_failed", exc_info=exc)

    if app.state.classifier and app.state.index and app.state.llm and app.state.corpus_fresh:
        app.state.router = SupportRouter(
            app.state.classifier, app.state.index, app.state.llm, settings
        )
    else:
        app.state.router = None

    db_path = FeedbackStore._path_from_url(settings.feedback_db_url)
    ensure_parent(db_path)
    app.state.store = FeedbackStore(settings.feedback_db_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("startup", extra={"app_version": settings.app_version, "env": settings.app_env})
    _load_state(app)
    yield
    store = getattr(app.state, "store", None)
    if store:
        store.close()
    logger.info("shutdown")


app = FastAPI(
    title="NovaBank AI Support Assistant",
    version=settings.app_version,
    description="Intent classification + grounded RAG over NovaBank's (fictional) policy base.",
    lifespan=lifespan,
)


# ----------------------------------------------------------------- middleware
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request_id_var.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()

    client = request.headers.get("x-api-key") or (request.client.host if request.client else "-")
    if request.url.path in {"/chat", "/chat/batch"} and not limiter.allow(client, time.time()):
        metrics.observe_request(request.url.path, 429, 0.0)
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded", "request_id": request_id},
            headers={"x-request-id": request_id, "retry-after": "60"},
        )

    try:
        response = await call_next(request)
    except Exception:
        latency = (time.perf_counter() - started) * 1000
        metrics.observe_request(request.url.path, 500, latency)
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    latency = (time.perf_counter() - started) * 1000
    metrics.observe_request(request.url.path, response.status_code, latency)
    response.headers["x-request-id"] = request_id
    logger.info(
        "http_request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(latency, 2),
        },
    )
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No key configured => open (local dev). Configured => must match.
    Constant-time comparison avoids leaking the key through timing."""
    if not settings.api_key:
        return
    import hmac

    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


def get_router(request: Request) -> SupportRouter:
    router = getattr(request.app.state, "router", None)
    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"service not ready: {request.app.state.errors}",
        )
    return router


# ------------------------------------------------------------------ endpoints
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness. Deliberately checks nothing external."""
    return HealthResponse(status="ok", app_version=settings.app_version)


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(request: Request) -> JSONResponse:
    state = request.app.state
    checks = {
        "classifier": state.classifier is not None,
        "index": state.index is not None,
        "corpus_fresh": bool(getattr(state, "corpus_fresh", False)),
        "llm": state.llm is not None,
        "router": state.router is not None,
        "store": getattr(state, "store", None) is not None,
    }
    body = ReadyResponse(
        status="ready" if all(checks.values()) else "not_ready",
        checks=checks,
        model_version=state.classifier.model_version if state.classifier else None,
        index_version=state.index.version if state.index else None,
        corpus_sha256=getattr(state, "corpus_sha256", None),
        llm_provider=settings.llm_provider,
        confidence_threshold=(
            settings.confidence_threshold
            if settings.confidence_threshold is not None
            else (state.classifier.threshold if state.classifier else None)
        ),
    )
    code = 200 if body.status == "ready" else 503
    return JSONResponse(status_code=code, content=body.model_dump())


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["assistant"],
    dependencies=[Depends(require_api_key)],
)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    router = get_router(request)
    started = time.perf_counter()
    result = router.answer(payload.question)
    latency_ms = int((time.perf_counter() - started) * 1000)
    response = ChatResponse(
        request_id=request.state.request_id,
        answer=result.answer,
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        intent_is_confident=result.intent_is_confident,
        route=result.route,
        sources=result.sources,
        needs_human_escalation=result.needs_human_escalation,
        model_version=router.classifier.model_version,
        llm_model=result.llm_model,
        prompt_version=result.prompt_version,
        latency_ms=latency_ms,
    )
    _record(request, payload.question, response, result)
    return response


@app.post(
    "/chat/batch",
    response_model=list[ChatResponse],
    tags=["assistant"],
    dependencies=[Depends(require_api_key)],
)
def chat_batch(payload: BatchChatRequest, request: Request) -> list[ChatResponse]:
    router = get_router(request)
    out: list[ChatResponse] = []
    for item in payload.items:
        started = time.perf_counter()
        result = router.answer(item.question)
        latency_ms = int((time.perf_counter() - started) * 1000)
        response = ChatResponse(
            request_id=f"{request.state.request_id}-{len(out)}",
            answer=result.answer,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            intent_is_confident=result.intent_is_confident,
            route=result.route,
            sources=result.sources,
            needs_human_escalation=result.needs_human_escalation,
            model_version=router.classifier.model_version,
            llm_model=result.llm_model,
            prompt_version=result.prompt_version,
            latency_ms=latency_ms,
        )
        _record(request, item.question, response, result)
        out.append(response)
    return out


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["assistant"],
    dependencies=[Depends(require_api_key)],
)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    known = request.app.state.store.log_feedback(
        payload.request_id, payload.helpful, payload.comment
    )
    if not known:
        # Recorded anyway (the prediction row may have been pruned), but flag it:
        # feedback that cannot be joined to a prediction is much less useful.
        logger.warning("feedback_for_unknown_request", extra={"target_request_id": payload.request_id})
    return FeedbackResponse(status="recorded", request_id=payload.request_id)


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["ops"],
    dependencies=[Depends(require_api_key)],
)
def prometheus_metrics() -> str:
    return metrics.prometheus()


@app.get("/stats", tags=["ops"], dependencies=[Depends(require_api_key)])
def stats(request: Request) -> dict:
    n, rate = request.app.state.store.helpfulness_rate()
    return {**metrics.snapshot(), "feedback_count": n, "helpful_rate": rate}


def _record(request: Request, question: str, response: ChatResponse, result) -> None:
    metrics.observe_answer(
        route=response.route.value,
        intent=response.intent,
        confidence=response.intent_confidence,
        escalated=response.needs_human_escalation,
    )
    if response.route.value == "llm_unavailable":
        metrics.note_llm_failure()
    logger.info(
        "answer",
        extra={
            "intent": response.intent,
            "confidence": response.intent_confidence,
            "route": response.route.value,
            "n_sources": len(response.sources),
            "retrieval_top_score": result.retrieval_top_score,
            "escalated": response.needs_human_escalation,
            "model_version": response.model_version,
            "llm_model": response.llm_model,
            "latency_ms": response.latency_ms,
            **result.timings_ms,
        },
    )
    try:
        request.app.state.store.log_prediction(
            request_id=response.request_id,
            question_hash=text_fingerprint(question),
            question_chars=len(question),
            intent=response.intent,
            intent_confidence=response.intent_confidence,
            route=response.route.value,
            source_ids=",".join(s.chunk_id for s in response.sources),
            retrieval_top_score=result.retrieval_top_score,
            answer_chars=len(response.answer),
            needs_escalation=int(response.needs_human_escalation),
            model_version=response.model_version,
            llm_model=response.llm_model,
            prompt_version=response.prompt_version,
            latency_ms=response.latency_ms,
        )
    except Exception:  # noqa: BLE001 - never fail a customer answer on a logging write
        logger.exception("prediction_log_failed")
