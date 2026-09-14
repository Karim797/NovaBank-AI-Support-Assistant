# Engineering decisions

Every entry: **Decision → Reason → Alternative → Trade-off**, plus the mistake
that is easy to make at that spot. Read this next to your own implementation —
the disagreements are where the learning is.

---

## D1 — The router is deterministic code, the LLM is a text generator

**Decision.** Path selection, document scoping, answer admission and escalation
are `if` statements over a probability, a retrieval score and a JSON lookup
table (`src/app/router.py`). The LLM's only job is prose from supplied passages.

**Reason.** Every one of those decisions needs to be auditable, tunable and
testable. A probability threshold has a coverage/accuracy curve; a lookup table
has a diff; an LLM judgement has neither.

**Alternative.** An agent with tool selection (`search_kb`, `escalate`,
`answer_directly`).

**Trade-off.** The agent handles novel compositions the table has no row for.
It also costs an extra round trip per decision, makes latency unpredictable,
and turns "why did it escalate?" into an unanswerable question. Agents earn
their keep when tools are numerous and composable. Two paths is not that.

**Easy mistake.** Letting the LLM emit `"needs_human": true` and trusting it.
It will be plausible and uncalibrated, and you will never be able to tune it.

---

## D2 — Grounding enforced in code, not in the prompt

**Decision.** Three layers: the prompt asks for citations; `GroundedAnswer`
requires the JSON shape; the router checks every cited id against the ids
actually retrieved, drops unknown ones, and discards the whole answer if none
survives.

**Reason.** A prompt instruction is a request. Only layer three is a control.

**Alternative.** Trust the system prompt; or add an LLM-as-judge grader.

**Trade-off.** Set-membership checking is free and deterministic but only
verifies that citations *exist*, not that the answer faithfully reflects them —
a model can cite the right chunk and still misstate the fee. An entailment check
(NLI model or judge LLM) would catch that, at the cost of a second inference on
every request. The right next step once there is a faithfulness measurement to
justify it.

**Easy mistake.** Returning the model's `sources` array to the customer without
intersecting it with the retrieved set — a fabricated citation looks exactly as
authoritative as a real one.

---

## D3 — Hybrid lexical retrieval before dense embeddings

**Decision.** BM25 (Okapi, on a sklearn count matrix) fused with TF-IDF cosine
by Reciprocal Rank Fusion, with an intent-derived topic pre-filter. Ranking by
RRF, thresholding on raw cosine.

**Reason.** ~100 chunks of policy prose whose discriminating tokens are literal.
Zero model download, ~2 ms, deterministic, and measured recall@4 = 1.00 with the
topic filter (0.90 without).

**Alternative.** sentence-transformers or a hosted embedding API + FAISS/pgvector.

**Trade-off.** Dense retrieval wins on vocabulary mismatch — "money didn't land"
vs "balance not updated" — which is this retriever's known weak spot.
`EmbeddingBackend` in `retrieval.py` is the seam: a dense arm is a third rank
list into the same RRF, not a rewrite. Adding it before measuring would be
paying latency, cost and an inference dependency for an unquantified gain.

**Why threshold on cosine, not RRF.** An RRF score is only meaningful relative to
the other candidates for that query, so a fixed floor on it means nothing.
Cosine is on a stable 0–1 scale across queries, so 0.15 means the same thing
every time.

**Easy mistake.** Min-max normalising the two score arrays and averaging. The
top hit then always scores 1.0, and any relevance floor becomes unreachable.

---

## D4 — Structural chunking, not fixed windows

**Decision.** Split on `##` headings; split oversized sections on paragraph
boundaries with one paragraph of overlap; merge stubs into their neighbour;
prepend document title + section heading to the indexed text.

**Reason.** The corpus was authored so one section = one self-contained answer.
A 512-character window would cut the fee table in half.

**Alternative.** Fixed-size sliding window, corpus-agnostic.

**Trade-off.** Structural chunking depends on well-formed headings. For scraped
HTML or PDFs there are none, and a window wins. Know which world you are in.

**Easy mistake.** Indexing the chunk body only. A query like "atm fee" often
matches the *heading* more strongly than the prose, so the heading must be in
the indexed text.

---

## D5 — One artifact bundle with provenance

**Decision.** The pickle is a dict: fitted pipeline, label list, threshold,
`model_version`, git SHA, dataset SHA-256, train/val metrics, threshold sweep,
sklearn/numpy/python versions, schema version. `IntentClassifier` refuses to
load a bundle with the wrong schema version or a missing key.

**Reason.** At 3am the only question that matters is "what exactly is running?".
The artifact answers it, and `/ready` echoes it.

**Alternative.** `joblib.dump(pipeline)` + a README.

**Trade-off.** Slightly more code and a schema to migrate. Nothing else.

**Easy mistake.** Saving the estimator but not the vectorizer, or saving both
separately and letting them drift out of sync. One Pipeline, one file.

---

## D6 — Threshold chosen from a curve, not by feel

**Decision.** Sweep 0.20–0.60 on validation, take the highest threshold that
still leaves ≥ 90% coverage. Selected: **0.45**.

**Reason.** Below the threshold nothing breaks — the router searches the whole
KB instead of a topic slice — so abstaining is cheap and a mild bias toward it
is safe. Measured effect: 90.6% coverage, 95.5% accuracy on confident traffic
vs 91.2% overall.

**Alternative.** A fixed 0.5, or per-class thresholds.

**Trade-off.** Per-class thresholds would squeeze more out of the rare intents,
at the cost of 77 numbers to maintain and re-tune on every retrain. Not worth it
until per-class error analysis says a specific intent is causing real harm.

**Easy mistake.** Tuning the threshold on the test set. It is chosen on
validation, and `evaluate.py` is the only code that touches test.

---

## D7 — LogisticRegression over LinearSVC (the one comparison run)

**Decision.** TF-IDF (word 1–2 + char_wb 3–5) → multinomial logistic regression.

**Reason.** LinearSVC + sigmoid calibration measured 0.910 macro F1 vs 0.912,
but its confidence separated correct from incorrect predictions less cleanly
(93.7% vs 94.8% accuracy on kept traffic at a 0.4 threshold). Confidence is a
*routing input* here, so probability quality outranks a tie on F1.

**Alternative.** Fine-tuned MiniLM/DistilBERT, ~0.93–0.94 published.

**Trade-off.** +2 F1 points for GPU-or-slow training, a torch dependency in the
serving image, ~400 MB, ~50 ms CPU inference. The router already handles
low-confidence traffic gracefully, so those two points buy less than they cost.
The Pipeline is a seam: anything with `predict_proba` and the same bundle
contract drops in.

**Deliberate non-goal.** No hyperparameter grid, no architecture search. The
classifier is a component of this system, not the subject of it.

---

## D8 — Two LLM providers, one interface, one needing no key

**Decision.** `LLMClient` with `AnthropicLLM` (httpx, timeouts, exponential
backoff + jitter, retry only on retryable status codes) and `ExtractiveLLM`
(deterministic sentence selection, offline).

**Reason.** CI must exercise the real request path with no key and no network
flake, and production needs a degraded mode. Making the offline path a real
implementation rather than a test mock means CI tests the code that ships.

**Alternative.** LangChain wrappers; or mocking httpx in tests.

**Trade-off.** ~80 hand-written lines and no LangChain ecosystem (callbacks,
tracing integrations, easy provider swaps). For one provider and one call shape,
fewer moving parts wins. Revisit at provider number two plus tool calling.

**Honesty requirement.** `ExtractiveLLM` is not a language model. `llm_model` in
the response says `extractive-v1` so no one mistakes stitched policy text for
generation.

**Easy mistake.** Retrying a 400 or a 401. Those never succeed on retry; you
just burn latency before failing.

---

## D9 — Deterministic answers for safety-critical intents

**Decision.** Lost/stolen card, compromised card, lost phone, swallowed card
return fixed text, no LLM, validated at startup against the chunk index.

**Reason.** These are the highest-cost intents to get subtly wrong, the wording
is compliance-reviewed, and skipping generation removes latency exactly where
speed matters most.

**Alternative.** RAG for everything, with a strong prompt.

**Trade-off.** Fixed text does not adapt to the specific phrasing, and the
policy set must be maintained by hand. Correct trade for four intents; it would
not scale to forty.

**Startup validation.** If a policy cites a chunk id that no longer exists (a KB
edit renamed a heading), the app refuses to start. A citation that 404s is worse
than no citation, and this catches it at boot rather than in front of a
customer.

---

## D10 — Artifacts baked into the image

**Decision.** `COPY models/ ./models/`; CI rebuilds them from source data on
every run.

**Reason.** The image tag then identifies the model, not just the code. Start-up
is deterministic and needs no network. Rollback restores code and model together.

**Alternative.** Pull artifacts from Blob Storage at boot.

**Trade-off.** Bigger images and a rebuild for a model-only change. Correct at
23 MB; wrong at 2 GB, or when models rotate faster than code — then mount them
and put the model version in an env var.

---

## D11 — SQLite for the prediction log and feedback

**Decision.** stdlib `sqlite3` behind a `FeedbackStore` class, no ORM.

**Reason.** Zero infrastructure, and the class boundary is what actually
matters — nothing outside that file knows the backend.

**Alternative.** Postgres from day one; or SQLAlchemy for portability.

**Trade-off.** SQLite is one writer, one file, dies with the container unless
mounted, and cannot be queried by a BI tool. The schema deliberately avoids
SQLite-specific types so the port is a driver swap. An ORM would remove even
that, at the cost of a dependency and a layer for four queries.

**Non-obvious point.** The `predictions` table exists to make late labels
joinable. Without a row written at answer time, a 👎 that arrives an hour later
has nothing to attach to and drift monitoring has no reference.

---

## D12 — In-process metrics rather than `prometheus_client`

**Decision.** A small registry emitting Prometheus text format, exact
percentiles over a bounded reservoir of recent samples.

**Reason.** No dependency, standard exposition format, exact p95/p99 for the
window instead of bucket approximations.

**Alternative.** `prometheus_client` with histograms and multiprocess mode.

**Trade-off.** Per-process and resets on restart, so it is not suited to exact
long-window quantiles across replicas. Fine while the scraper aggregates by
instance; swap when there is a real SLO to defend.

**Easy mistake.** Adding `request_id` or the question text as a metric label.
That is unbounded cardinality and it will take down the metrics backend.

---

## D13 — Container Apps, not Kubernetes

**Decision.** Azure Container Apps for the MVP; Kubernetes manifests provided as
reference only.

**Reason.** One stateless HTTP service. Container Apps gives revisions, traffic
splitting, autoscaling and managed TLS with no cluster to operate.

**Alternative.** AKS.

**Trade-off.** Less control (no custom schedulers, no mesh, no GPU pools) and
platform lock-in. Adopt Kubernetes when there is a second and third service, or
a model that needs GPUs — and note that `/ready` and `/health` are already
probe-shaped for that day.

---

## D14 — CI and CD are separate workflows

**Decision.** `ci.yml` proves correctness on every push and PR and never
deploys. `cd.yml` runs on main, pushes a SHA-tagged image, deploys to staging,
smoke tests it, then canaries production behind a required approval.

**Reason.** Different triggers, different permissions, different blast radius.
A PR from a fork must be able to run tests and must never be able to deploy.

**Alternative.** One workflow with `if:` conditions.

**Trade-off.** Some duplicated setup steps. Worth it for the permission
boundary — `cd.yml` is the only workflow with `id-token: write`.

**Two things CI does that are easy to skip.** It rebuilds the model artifacts
from source data, so the *training pipeline itself* is tested on every commit
rather than only the code that loads a stale pickle. And it boots the built
image and runs the smoke test against it, so "the image starts and answers" is
verified before anything is pushed.

**Rollback.** Immutable SHA tags and a warm previous revision make rollback a
traffic-weight change, seconds not a CI cycle. `latest` is never deployed:
it makes "what is running?" unanswerable.
