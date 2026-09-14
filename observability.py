"""Metrics.

Decision -> Reason -> Alternative -> Trade-off
- Decision: a tiny in-process registry that emits Prometheus text format at
  `/metrics`, with a bounded reservoir of recent latencies for percentiles.
- Reason: no extra dependency, and the exposition format is what every scraper
  (Prometheus, Grafana Agent, Azure Monitor's OpenTelemetry collector) already
  speaks. Percentiles are computed from raw samples so p95/p99 are exact for the
  window rather than approximated from buckets.
- Alternative: `prometheus_client` (histograms, multiprocess mode) or pushing
  OpenTelemetry spans.
- Trade-off: this registry is per-process and resets on restart, so with several
  replicas each exposes its own view — fine, because the scraper aggregates by
  instance label. It is *not* suitable for exact long-window quantiles; for that
  the scraped histogram approach is correct. Swap when there is a real SLO.

Cardinality warning: labels here are bounded (route, intent, status). Never add
`request_id` or the question text as a label — that is the classic way to melt a
Prometheus server.
"""

from __future__ import annotations

import threading
from collections import Counter, deque

MAX_SAMPLES = 2000


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = Counter()          # (endpoint, status_class) -> n
        self.routes = Counter()            # route decision -> n
        self.intents = Counter()           # predicted intent -> n
        self.escalations = 0
        self.llm_failures = 0
        self._latency: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._confidence: deque[float] = deque(maxlen=MAX_SAMPLES)

    def observe_request(self, endpoint: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests[(endpoint, f"{status // 100}xx")] += 1
            self._latency.append(latency_ms)

    def observe_answer(self, route: str, intent: str, confidence: float, escalated: bool) -> None:
        with self._lock:
            self.routes[route] += 1
            self.intents[intent] += 1
            self._confidence.append(confidence)
            if escalated:
                self.escalations += 1

    def note_llm_failure(self) -> None:
        with self._lock:
            self.llm_failures += 1

    def percentile(self, p: float) -> float:
        with self._lock:
            data = sorted(self._latency)
        if not data:
            return 0.0
        k = max(0, min(len(data) - 1, int(round(p / 100.0 * (len(data) - 1)))))
        return round(data[k], 2)

    def snapshot(self) -> dict:
        with self._lock:
            conf = list(self._confidence)
            n = len(self._latency)
        return {
            "requests_total": sum(self.requests.values()),
            "answers_total": sum(self.routes.values()),
            "latency_samples": n,
            "latency_p50_ms": self.percentile(50),
            "latency_p95_ms": self.percentile(95),
            "latency_p99_ms": self.percentile(99),
            "mean_confidence": round(sum(conf) / len(conf), 4) if conf else None,
            "low_confidence_share": (
                round(sum(1 for c in conf if c < 0.4) / len(conf), 4) if conf else None
            ),
            "escalations_total": self.escalations,
            "llm_failures_total": self.llm_failures,
            "routes": dict(self.routes),
            "top_intents": dict(Counter(self.intents).most_common(10)),
        }

    def prometheus(self) -> str:
        lines = [
            "# HELP nb_requests_total HTTP requests by endpoint and status class",
            "# TYPE nb_requests_total counter",
        ]
        for (endpoint, status), value in sorted(self.requests.items()):
            lines.append(f'nb_requests_total{{endpoint="{endpoint}",status="{status}"}} {value}')
        lines += [
            "# HELP nb_request_latency_ms Request latency percentiles over the recent window",
            "# TYPE nb_request_latency_ms gauge",
            f'nb_request_latency_ms{{quantile="0.5"}} {self.percentile(50)}',
            f'nb_request_latency_ms{{quantile="0.95"}} {self.percentile(95)}',
            f'nb_request_latency_ms{{quantile="0.99"}} {self.percentile(99)}',
            "# HELP nb_route_total Answers by route decision",
            "# TYPE nb_route_total counter",
        ]
        for route, value in sorted(self.routes.items()):
            lines.append(f'nb_route_total{{route="{route}"}} {value}')
        lines += [
            "# HELP nb_intent_total Answers by predicted intent",
            "# TYPE nb_intent_total counter",
        ]
        for intent, value in sorted(self.intents.items()):
            lines.append(f'nb_intent_total{{intent="{intent}"}} {value}')
        lines += [
            "# HELP nb_escalations_total Answers flagged for a human",
            "# TYPE nb_escalations_total counter",
            f"nb_escalations_total {self.escalations}",
            "# HELP nb_llm_failures_total LLM provider failures",
            "# TYPE nb_llm_failures_total counter",
            f"nb_llm_failures_total {self.llm_failures}",
        ]
        return "\n".join(lines) + "\n"


metrics = Metrics()


class RateLimiter:
    """Fixed-window per-client limiter. Enough to stop a runaway script; it is
    not a defence against a distributed attack — that belongs at the gateway
    (Azure Front Door / APIM), which is where it goes in production."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client: str, now: float) -> bool:
        if self.per_minute <= 0:
            return True
        with self._lock:
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True
