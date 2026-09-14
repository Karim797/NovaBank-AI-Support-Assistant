# NovaBank AI Support Assistant

An end-to-end AI support system for a **fictional** UK digital bank: an intent
classifier, a deterministic router, grounded RAG over a policy knowledge base,
a FastAPI service, tests with real quality gates, Docker, CI/CD, MLflow, drift
monitoring, and a cloud deployment design.

The point of the project is not the classifier. It is what has to exist around a
model before it can answer a customer: **routing, grounding, refusal, escalation,
observability, and a release process that can block a bad model.**

Licensed under the [MIT License](LICENSE).

> NovaBank is invented. Every fee, limit and SLA in `knowledge_base/` is
> synthetic and describes no real institution.

---

## 1. The problem

A customer asks something like *"my card was charged twice"* or *"why is my cash
withdrawal still pending?"*. A useful answer must be specific (the actual SLA,
the actual fee), traceable to a policy document, and safe: it must refuse rather
than invent a number, and it must hand off to a human when the situation calls
for one.

An LLM alone fails all three. A classifier alone can only label. This system
composes them, and puts the decisions in code.

## 2. Architecture

```
                          ┌──────────────────────────────────────────┐
   POST /chat ──────────▶ │ FastAPI  (middleware: request_id, rate    │
                          │ limit, api key, metrics, JSON logging)    │
                          └──────────────────┬───────────────────────┘
                                             ▼
                          ┌──────────────────────────────────────────┐
                          │ Intent classifier                        │
                          │ TF-IDF + LogisticRegression, 77 intents  │
                          │ → intent, probability                    │
                          └──────────────────┬───────────────────────┘
                                             ▼
                          ┌──────────────────────────────────────────┐
                          │ Router  (deterministic Python)           │
                          │ confidence ≥ 0.45 ?   policy or RAG ?    │
                          └───┬──────────────────────────┬───────────┘
       confident + policy     │                          │  everything else
                              ▼                          ▼
              ┌───────────────────────────┐   ┌──────────────────────────────┐
              │ Fixed compliance answer   │   │ Hybrid retriever             │
              │ (lost card, fraud, …)     │   │ BM25 ⊕ TF-IDF cosine, RRF    │
              │ no LLM, no paraphrase     │   │ topic filter from the intent │
              └───────────────┬───────────┘   └──────────────┬───────────────┘
                              │                     top-k ≥ score floor?
                              │                    no │            │ yes
                              │                       ▼            ▼
                              │            ┌──────────────┐  ┌─────────────────┐
                              │            │ Refuse +     │  │ LLM, context-   │
                              │            │ escalate     │  │ only, JSON out  │
                              │            └──────┬───────┘  └────────┬────────┘
                              │                   │                   ▼
                              │                   │      ┌──────────────────────────┐
                              │                   │      │ Grounding check:         │
                              │                   │      │ citations ⊆ retrieved?   │
                              │                   │      │ no → discard & refuse    │
                              │                   │      └────────────┬─────────────┘
                              └───────────────────┴───────────────────┘
                                                  ▼
                          ┌──────────────────────────────────────────┐
                          │ ChatResponse: answer, intent, confidence,│
                          │ route, sources, escalation, versions     │
                          └───────┬────────────────────────┬─────────┘
                                  ▼                        ▼
                        prediction log (SQLite)     metrics + JSON logs
                                  ▲                        │
                        /feedback 👍/👎 joins on request_id │
                                                           ▼
                                              monitoring/drift.py (PSI, KS)
```

**The LLM decides nothing.** Which path a question takes, which documents it may
see, whether the answer may be returned, and whether a human is needed are all
decided in Python from a probability, a retrieval score, and a lookup table.
Every branch returns a `route` enum that appears in the response, the log line
and the prediction table.

### The routing table

| Condition | Route | LLM? |
|---|---|---|
| confident intent + safety policy (lost/stolen/compromised card, lost phone, swallowed card) | `deterministic_policy` | no |
| confident intent (`p ≥ 0.45`) | `rag_topic_filtered` — retrieval restricted to that intent's topics | yes |
| low confidence | `rag_unfiltered` — whole knowledge base | yes |
| best retrieval score < 0.15 | `no_relevant_context` — refuse, escalate | no |
| LLM output unparseable or cites nothing retrieved (after one repair attempt) | `ungrounded_answer` — discard, refuse, escalate | attempted |
| provider fails after retries | `llm_unavailable` — degraded refusal, escalate | attempted |

## 3. Measured results

Everything below was produced by the pipeline in this repo, not quoted from a
paper. Reproduce with `bash scripts/run_local.sh`; outputs land in `reports/`.

**Intent classification** — official Banking77 split (10,003 train / 3,080 test,
77 intents, 5.34× class imbalance):

| Metric | Value |
|---|---|
| Test macro F1 | **0.9125** |
| Test macro F1, train/test duplicates removed | 0.9123 |
| Test accuracy | 0.9123 |
| Confidence threshold (selected on validation) | 0.45 |
| Coverage at threshold | 90.6% of traffic |
| Accuracy on confident traffic | **95.5%** |
| Accuracy on low-confidence traffic | 50.7% |
| Artifact size / inference | 23 MB / ~3 ms |

Macro F1, not accuracy: 77 classes with a 5.34× imbalance, and the rarest
intents (`card_swallowed`, `compromised_card`) are the ones with the highest cost
of being wrong.

The confidence column is the load-bearing one. The threshold is not a taste
judgement — it comes from the coverage/accuracy sweep in
`reports/train_report.json`, and a test asserts that confident traffic really is
more accurate than the rest. If that stops being true, the routing gate is
decoration and CI says so.

**Retrieval** — 34 labelled queries (30 in-scope, 4 out-of-scope) in
`training/data/retrieval_queries.jsonl`:

| Metric | Value |
|---|---|
| recall@1 | 0.833 |
| recall@4 (no filter) | 0.900 |
| **recall@4 with intent topic filter** | **1.000** |
| MRR@4 (no filter → filtered) | 0.861 → 0.978 |
| Out-of-scope questions refused | 4/4 |

That filtered-vs-unfiltered gap is the empirical justification for the whole
architecture: the classifier is not decoration, it measurably improves
retrieval. A quality gate asserts the filter never makes recall worse — if a
future change breaks that, the filter should be deleted, and the test will say
so.

**Data audit** (`reports/data_audit.json`): 6 test rows (0.195% of test) are
verbatim duplicates of training rows. Reported, not hidden — and macro F1 is
also reported on the de-duplicated set, with a gate that fails if the two ever
diverge by more than 0.01. A headline metric that leans on memorisation is worse
than a lower honest one.

**Knowledge base**: 19 documents → 107 chunks, mean 339 characters.

## 4. Repository layout

```
├── src/app/                 # everything that ships to production
│   ├── main.py              # FastAPI: lifespan loading, endpoints, middleware
│   ├── router.py            # ★ the decision logic — read this first
│   ├── classifier.py        # artifact bundle loader + inference
│   ├── retrieval.py         # BM25 ⊕ TF-IDF hybrid retriever, RRF fusion
│   ├── kb.py                # heading-aware chunker
│   ├── llm.py               # provider abstraction: Anthropic | extractive
│   ├── prompts.py           # versioned prompts + refusal text
│   ├── schemas.py           # Pydantic contracts (HTTP and LLM output)
│   ├── feedback.py          # prediction log + 👍/👎 store
│   ├── observability.py     # metrics registry, rate limiter
│   ├── logging_config.py    # JSON logs, request-id contextvar, PII hashing
│   └── resources/           # intent→topic routes, deterministic policies
├── training/                # never imported by the app
│   ├── prepare_data.py      # fetch, integrity + leakage audit, split
│   ├── train_intent.py      # pipeline, threshold sweep, artifact, MLflow
│   ├── evaluate.py          # test-set eval, per-class report, confusion pairs
│   ├── build_index.py       # KB → chunks → retriever artifact
│   └── eval_retrieval.py    # recall@k, MRR, out-of-scope refusal rate
├── knowledge_base/          # 19 fictional NovaBank policy documents
├── monitoring/              # PSI/KS drift job + alert rule definitions
├── tests/                   # 75 tests, incl. 7 marked `quality_gate`
├── deploy/azure|k8s/        # deployment configuration (not executed here)
├── ui/streamlit_app.py      # demo client — zero business logic
├── .github/workflows/       # ci.yml (correctness) · cd.yml (release)
└── Dockerfile · docker-compose.yml · Makefile
```

## 5. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

make data      # download Banking77, audit, split         → data/processed, reports/
make train     # fit, sweep threshold, write the bundle    → models/intent_clf_v1.joblib
make index     # chunk the KB, build the retriever         → models/kb_index_v1.joblib
make eval      # test-set evaluation                       → reports/eval_report.json
make test      # 75 tests including the quality gates
make serve     # http://localhost:8000/docs
```

`LLM_PROVIDER=extractive` (the default) runs the entire path with no API key and
no network — that is how CI exercises it. Set `LLM_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY` for fluent generation. The `llm_model` field in every
response says which one answered.

**Docker**

```bash
docker build -t novabank-assistant:local .        # builds; artifacts baked in
docker run -p 8000:8000 --env-file .env novabank-assistant:local
docker compose up --build                          # api + Streamlit UI + MLflow
docker logs -f <container>                         # JSON logs on stdout
docker exec -it <container> sh                     # poke inside
bash scripts/smoke_test.sh http://localhost:8000
```

Docker terms as used here: the **Dockerfile** is the recipe; a **layer** is one
cached instruction result (which is why `requirements.txt` is copied before the
source); the **image** is the built filesystem; a **container** is a running
instance; a **tag** names an image version (here, the 12-char commit SHA, never
`latest`); a **registry** stores images (Azure Container Registry); a **volume**
persists data past the container's life (the SQLite feedback file).

## 6. API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness. Checks nothing external → Kubernetes `livenessProbe`. |
| `GET /ready` | Readiness: model, index, LLM, store loaded; echoes versions and the active threshold → `readinessProbe`. |
| `POST /chat` | The assistant. |
| `POST /chat/batch` | Up to 20 questions, one vectorisation pass. |
| `POST /feedback` | 👍/👎 joined to the prediction by `request_id`. |
| `GET /metrics` | Prometheus text format. |
| `GET /stats` | Human-readable snapshot: p50/p95/p99, route mix, helpfulness. |

Wiring liveness to a dependency check is a classic self-inflicted outage: a slow
dependency then restarts every replica in a loop. `/health` stays dumb on
purpose.

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question":"Why is my cash withdrawal still pending?"}' | jq
```

Actual response from a local run (`LLM_PROVIDER=extractive`, so the prose is
selected policy text rather than generated):

```json
{
  "request_id": "b7c1f0a2d95e4c18",
  "answer": "An ATM withdrawal shows as pending until the ATM operator settles it, normally within 3 business days and always within 7 days. If a pending withdrawal has not settled after 7 days it expires automatically and the reserved money returns to your balance.",
  "intent": "pending_cash_withdrawal",
  "intent_confidence": 0.9987,
  "intent_is_confident": true,
  "route": "rag_topic_filtered",
  "sources": [
    {"chunk_id": "cash-withdrawals-and-atms#withdrawal-still-pending",
     "doc_id": "cash-withdrawals-and-atms", "title": "Cash Withdrawals and ATMs",
     "section": "Withdrawal still pending", "score": 0.3177}
  ],
  "needs_human_escalation": false,
  "model_version": "intent-clf-v1",
  "llm_model": "extractive-v1",
  "prompt_version": "grounded-v1.2",
  "latency_ms": 88
}
```

Ask it something out of scope (*"what is the capital of France?"*) and it returns
`route: "no_relevant_context"`, no sources, and `needs_human_escalation: true`.

## 7. Testing

`pytest` — 75 tests. Two categories, deliberately mixed in one suite because
both can break a release:

*Software*: schema validation and 422s on bad input, `/health` vs `/ready`
semantics, 503 when artifacts failed to load, rate limiting, API-key auth,
request-id propagation, feedback persistence, PII redaction, chunker invariants,
BM25 IDF formula, LLM retry/backoff on 5xx and no retry on 4xx.

*ML and behaviour* (`-m quality_gate`, the ones that block deployment):

| Gate | Threshold | Why this number |
|---|---|---|
| Test macro F1 | ≥ 0.85 | ~6 points below the measured 0.9125. Wide enough to absorb data refreshes and library churn, tight enough that a real regression (a broken preprocessing step, a truncated training set) trips it. A gate at 0.91 would fail on noise and get disabled — a disabled gate protects nothing. |
| macro F1 − dedup macro F1 | < 0.01 | Catches a headline number propped up by train/test duplicates. |
| Accuracy(confident) > accuracy(all) > accuracy(low-conf) | strict | The routing threshold is only meaningful if confidence separates right from wrong. |
| Coverage at threshold | ≥ 0.85 | An abstain-happy model is useless even at high accuracy. |
| Retrieval recall@4 | ≥ 0.85 | Generation cannot be grounded in a document that was never retrieved. |
| Out-of-scope refusal rate | ≥ 0.75 | Refusing is a feature; a system that always answers is the dangerous one. |
| recall@4 filtered ≥ recall@4 unfiltered | strict | If the intent filter ever hurts retrieval, delete the filter. |

Behavioural tests assert the router's contract directly: a lost-card question
takes the deterministic path and never reaches the LLM; a fabricated citation is
stripped and, if nothing valid remains, the answer is discarded; an LLM outage
degrades to a refusal rather than a 500.

## 8. MLflow and the model lifecycle

`training/train_intent.py` logs to a local file store (`mlruns/`), or to a
tracking server via `MLFLOW_TRACKING_URI`. MLflow is imported lazily and failures
are non-fatal: training must work on a laptop with no server, and MLflow is not
installed in the serving image.

**Experiment** = the question being studied (`banking-intent-classification`).
**Run** = one attempt at it. Within a run: **parameters** are inputs you chose
(C, n-gram range, threshold); **metrics** are numbers you measured (macro F1,
artifact size); **artifacts** are files produced (the bundle, the confusion
matrix, the audit report); **tags** are searchable labels (git SHA, dataset
hash, stage).

Promotion path — **implemented here**: experiment → validation → artifact with
full provenance → CI quality gate → SHA-tagged image → staging → smoke test →
production behind approval, with a 20% canary and one-command rollback.

**Not implemented, and not pretended otherwise**: a hosted Model Registry with
stage transitions, shadow mode, and automated champion/challenger promotion.
What they mean, and what each would take here:

- **Champion / challenger** — the model serving traffic vs a candidate scored on
  the same inputs. Would need the router to call both classifiers and log both
  predictions against one `request_id`.
- **Shadow mode** — the challenger sees production traffic but its output never
  reaches a customer. Cheap to add for the classifier (a second `predict_proba`
  and an extra log column); expensive for the LLM path, because it doubles
  generation cost.
- **Rollback** — already real: images are immutable and SHA-tagged, and the old
  revision stays warm, so rollback is a traffic-weight change (`cd.yml`,
  `rollback` job), not a rebuild.

## 9. Monitoring

`python -m monitoring.drift --build-reference` snapshots the evaluation
distribution; `python -m monitoring.drift` runs in a scheduled job and compares
recent production predictions against it.

*Service*: p50/p95/p99 latency, throughput, 4xx/5xx rate, LLM failure count,
readiness flapping.
*Model and data*: intent mix, confidence distribution (PSI + KS), share of
traffic below the routing threshold, refusal rate, escalation rate, helpfulness.

**Data drift is not model degradation.** Data drift means P(X) changed — the
questions look different (a campaign, a card-delivery backlog, a new product).
Concept drift means P(y|X) changed — the same question now has a different right
answer. In this system concept drift almost always means *a policy document
changed*, which no input-distribution test can detect; it is caught by the
corpus hash in `reports/index_report.json` and by the feedback signal. A PSI
alert is a "go and look", never an automatic retrain trigger.

**Getting labels late.** Ground truth arrives after the fact: a 👎, an agent
re-tagging a conversation, a customer re-asking. All of it carries a
`request_id`, which is exactly why every prediction is written to the
`predictions` table at answer time. Join on `request_id` and you have a delayed,
biased, but real evaluation set — biased because people who leave feedback are
not a random sample, so track the rate as a trend, not as accuracy.

Alert conditions with severities and runbook actions: `monitoring/alerts.yaml`.

## 10. Security

Implemented: no secrets in the image or in git (`.env` gitignored,
`.env.example` documents the surface); constant-time API-key comparison; strict
Pydantic validation with a 500-character question cap and a 20-item batch cap;
fixed-window rate limiting; card-number / CVV / PIN redaction before an answer or
a log line is written; raw questions never persisted (length + SHA-256
fingerprint only); non-root container user with a read-only root filesystem in
the Kubernetes manifest; error responses that return a request id rather than a
stack trace; dependency scanning (`pip-audit`), image scanning (Trivy) and
secret scanning (gitleaks) in CI.

Designed, not implemented: OAuth2/JWT with per-tenant scopes in front of
`/chat`, WAF and distributed rate limiting at the gateway, Key Vault–backed
secret rotation with managed identity (the Bicep uses it), network egress
restricted to the LLM provider, and a signed audit log of escalations.

## 11. Cloud deployment (Azure)

| Component | Service | Why |
|---|---|---|
| Images | Container Registry | SHA-tagged, scanned, geo-replicable |
| API | **Container Apps** | Revisions, traffic splitting, autoscale, managed TLS — without a cluster to operate |
| Secrets | Key Vault + managed identity | No credentials in config or CI |
| Data/artifacts | Blob Storage | Datasets, model bundles, MLflow artifact store |
| Observability | Log Analytics + Application Insights | JSON logs are already structured for it |
| Feedback store | PostgreSQL Flexible Server | Replaces SQLite the moment there is more than one replica |

**AKS is not used, on purpose.** This is one stateless HTTP service. Kubernetes
would add a control plane, node pools, an ingress controller and upgrade cycles
to maintain, in exchange for capabilities this workload does not need.
`deploy/k8s/` shows the manifests and the concept mapping (Pod, Deployment,
Service, ConfigMap, Secret, HPA, `readinessProbe` → `/ready`,
`livenessProbe` → `/health`, `kubectl rollout undo`) for when it does — a GPU
model, several services, or a mesh.

## 12. Engineering decisions and trade-offs

Full table with alternatives in [`docs/DECISIONS.md`](docs/DECISIONS.md). The
five that shaped the system:

1. **The router is deterministic Python, not an agent.** An LLM asked "should
   this go to a human?" gives a plausible answer with no calibration and no
   audit trail. A threshold on a measured probability gives a number you can
   tune, a curve you can plot, and a reason code you can log. An agent with
   dynamic tool selection would be justified if tools were numerous and
   composable; here there are two paths, and a lookup table beats a language
   model at picking between two paths.
2. **Grounding is enforced in code, not by the prompt.** The prompt asks for
   citations; the *schema* requires them; the *router* verifies that every cited
   id was actually retrieved and discards the answer if none survives. Only the
   third layer is a control — the first two are requests.
3. **Lexical hybrid retrieval before embeddings.** ~100 chunks of jargon-dense
   policy text where the discriminating tokens are literal (`chargeback`,
   `SEPA`, `GBP 200`). BM25 ⊕ TF-IDF, fused by rank, costs ~2 ms, no model
   download, fully deterministic — and measures recall@4 = 1.00 with the topic
   filter. Its weakness is paraphrase ("money didn't land" vs "balance not
   updated"), which is exactly why `EmbeddingBackend` is a real seam and why the
   recall harness exists: add the dense arm when the measurement says lexical is
   the bottleneck, not because embeddings are fashionable.
4. **One artifact bundle, never a bare estimator.** The pickle carries the
   fitted pipeline, the label set, the threshold, the training data hashes, the
   git SHA, and the library versions. `joblib.dump(model)` on its own leaves
   serving with no idea which threshold was evaluated or which data it saw.
5. **Two LLM providers behind one interface, one of which needs no key.** The
   offline `extractive` provider is a first-class implementation, not a test
   mock, so CI runs the real request path end to end — and production gets a
   sane degraded mode when the API is down. It is *not* a language model, and
   the response says so.

## 13. Limitations

- **The extractive provider is not generative.** It stitches sentences from the
  retrieved passages. The demo's answer quality with `LLM_PROVIDER=extractive`
  understates the system; grounding, routing and refusal behave identically
  either way.
- **The retrieval eval set is 34 hand-written queries.** Enough to catch a
  regression, too small for a confident recall estimate. Real production queries
  should replace it as soon as there are any.
- **No conversation memory.** Each request is independent; "and what about
  abroad?" has no antecedent. Multi-turn needs a session store and query
  rewriting, both of which change the retrieval contract.
- **SQLite dies with the container** unless the volume is mounted, and supports
  one writer. It is an MVP choice with a documented exit.
- **In-process metrics reset on restart** and are per-replica.
- **Chunking assumes well-formed Markdown headings.** Point this at scraped PDFs
  and the structural chunker degrades; use a sliding window there.
- **No load testing.** The latency numbers are single-request measurements on
  one core, not a throughput profile.
- **Docker build, Azure deployment and the CD workflow have not been executed**
  in this reference implementation — they require a daemon and a subscription.
  The training pipeline, the index build, the evaluations, the test suite and
  the API were all run, and every number in section 3 comes from those runs.

## 14. If this went to production next

In priority order, with the trigger for each:

1. Replace the 34-query retrieval set with sampled production questions
   (trigger: first week of real traffic).
2. Add the dense retrieval arm and re-measure recall@4 (trigger: refusal rate
   above 15%, or recall plateauing on paraphrase-heavy queries).
3. Move the feedback store to Postgres (trigger: the second replica).
4. Shadow-mode a challenger classifier (trigger: any proposed model change).
5. Multi-turn context with query rewriting (trigger: a measurable share of
   follow-up questions in the logs).
6. Human-handoff integration — the escalation flag currently sets a boolean;
   production needs it to create a ticket with the conversation attached.
7. Per-intent answer quality review with agents, which is the only way to get
   strong labels rather than the weak 👍/👎 signal.
