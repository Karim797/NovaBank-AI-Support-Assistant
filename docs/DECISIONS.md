# Engineering decisions

Every entry: **Decision → Reason → Alternative → Trade-off**, plus the mistake
that is easy to make at that spot. Read this next to your own implementation —
the disagreements are where the learning is.

---

## D1 — The router is deterministic code, the LLM is a text generator

**Decision.** Path selection, retrieval conditioning, answer admission and
escalation are `if` statements over a probability, a retrieval score and a JSON
lookup table (`src/app/router.py`). The LLM's only job is prose from supplied
passages.

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
by Reciprocal Rank Fusion. A confident intent contributes a **small topic ranking
prior**, never a hard candidate filter. Ranking uses RRF + prior; the relevance
floor still uses raw cosine.

**Reason.** The corpus is only 107 chunks of policy prose whose discriminating
tokens are often literal. Lexical retrieval has zero model download, low
latency, deterministic behaviour, and a strong measured ranking ceiling. On the
harder 50-query in-scope set, unfiltered Recall@4 is **0.92** and the soft topic
prior raises it to **0.94**. The actual production path is lower at **0.70**
because the 0.15 relevance floor intentionally rejects weak evidence.

**Alternative.** sentence-transformers or a hosted embedding API + FAISS/pgvector.

**Trade-off.** Dense retrieval wins on vocabulary mismatch — "money didn't land"
vs "balance not updated" — which is this retriever's measured weak spot. On 20
realistic paraphrases, production Recall@4 is **0.60**. `EmbeddingBackend` in
`retrieval.py` remains the seam: a dense arm can become a third rank list into
the same RRF, but it should be added only if it improves this measured gap
without weakening refusal behaviour.

**Why a prior instead of a filter.** The original implementation restricted
candidates to classifier-derived topics. That looked excellent on a small
corpus-derived set, but a confidently wrong classifier prediction could remove
the correct document before ranking even began. A soft prior still rewards an
intent-consistent document in a near tie while preserving lexical evidence from
the rest of the corpus.

**Why threshold on cosine, not RRF.** An RRF score is only meaningful relative to
the other candidates for that query, so a fixed floor on it means nothing.
Cosine is on a stable 0–1 scale across queries, so 0.15 means the same thing
every time.

**Easy mistake.** Min-max normalising the two score arrays and averaging. The
top hit then always scores 1.0, and any relevance floor becomes unreachable.
A second easy mistake is turning an upstream classifier prediction into a hard
retrieval constraint without measuring error propagation end to end.

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
The artifact answers it, and `/ready` echoes its operational version while CI
also records the source revision used to rebuild it.

**Alternative.** `joblib.dump(pipeline)` + a README.

**Trade-off.** Slightly more code and a schema to migrate. Nothing else.

**Easy mistake.** Saving the estimator but not the vectorizer, or saving both
separately and letting them drift out of sync. One Pipeline, one file.

---

## D6 — Threshold chosen from a curve, not by feel

**Decision.** Sweep 0.20–0.60 on validation, take the highest threshold that
still leaves ≥ 90% coverage. Selected: **0.45**.

**Reason.** Below the threshold nothing breaks — the router searches the whole
KB without the intent prior — so abstaining from intent conditioning is cheap
and a mild bias toward it is safe. Measured effect: 90.6% coverage, 95.5%
accuracy on confident traffic vs 91.2% overall.

The test-set reliability analysis also reports **ECE = 0.0895** over 10 bins.
The classifier is generally under-confident: the 0.4–0.5 bin averages ~0.454
confidence but ~0.621 accuracy. Confidence therefore separates easy from hard
traffic well enough to route on, but it is not presented as a perfectly
calibrated probability.

**Alternative.** A fixed 0.5, per-class thresholds, or explicit probability
calibration (temperature scaling / isotonic / Platt-style calibration).

**Trade-off.** Per-class thresholds would squeeze more out of the rare intents,
at the cost of 77 numbers to maintain and re-tune on every retrain. Explicit
calibration could make the probability semantics better, but the current router
needs ranking/separation more than literal probability correctness. Revisit if
confidence itself becomes customer-visible or cost decisions depend on it.

**Easy mistake.** Tuning the threshold on the test set. It is chosen on
validation, and `evaluate.py` is the only code that touches test. A second
mistake is reporting one F1 number without uncertainty: CI now reports a
stratified-bootstrap 95% interval of **0.9024–0.9218** around macro F1 0.9125.

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

**Decision.** stdlib `sqlite3` behind a `FeedbackStore` class, no ORM, and one
Uvicorn worker while SQLite is the live persistence layer.

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

**Decision.** Azure Container Apps for the reference cloud architecture;
Kubernetes manifests are provided as reference only. The current public demo is
Railway + Streamlit Community Cloud.

**Reason.** One stateless HTTP service. A managed container platform gives
revisions, autoscaling and managed TLS with no cluster to operate.

**Alternative.** AKS/Kubernetes.

**Trade-off.** Less control (no custom schedulers, no mesh, no GPU pools) and
platform lock-in. Adopt Kubernetes when there is a second and third service, or
a model that needs GPUs — and note that `/ready` and `/health` are already
probe-shaped for that day.

---

## D14 — CI proves correctness; deployment remains a separate concern

**Decision.** `ci.yml` proves correctness on every push and PR and never needs
production credentials. The current Railway integration deploys the merged
`main` revision separately; the repository's deployment references illustrate a
more formal staged/canary path for a larger environment.

**Reason.** Testing and deploying have different triggers, permissions and blast
radii. A PR must be able to run every correctness check without being able to
ship itself to production.

**Security scans.** `pip-audit --strict` and gitleaks are blocking gates. The
container Trivy scan keeps HIGH/CRITICAL findings visible as a SARIF artifact
for triage, but does not hide pip's vendored SBOM to manufacture a green scan.
Base-image findings are handled by updating/pinning the base rather than deleting
scanner evidence.

**Two things CI does that are easy to skip.** It rebuilds model/index artifacts
from source, so the training pipeline itself is tested on every commit rather
than only code that loads a stale pickle. And it boots the built image and runs
the smoke test against it, so "the image starts and answers" is verified before
a change is merged.

**Easy mistake.** Running ML quality-gate tests against stale checked-in JSON
reports. CI deliberately generates `evaluate.py` and `eval_retrieval.py` reports
first, then runs the `quality_gate` pytest marker against those fresh outputs.

---

## D15 — Retrieval evaluation must measure error propagation, not just retrieval in isolation

**Decision.** Keep three views of retrieval quality: (1) lexical ranking ceiling
with no floor or classifier conditioning, (2) classifier-derived soft topic
prior with no floor, and (3) the actual production path with the relevance
floor. Evaluate all three on seed questions, realistic paraphrases and a larger
out-of-scope set. Sweep the floor rather than reporting a single cherry-picked
number.

**Reason.** The old 34-query evaluation reported Recall@4 = 1.00 with a hard
intent-topic filter. That result was locally true and architecturally
misleading. On the harder 70-query benchmark, the useful picture is:

- unfiltered lexical Recall@4: **0.92**
- soft-prior Recall@4 before the floor: **0.94**
- production Recall@4 at floor 0.15: **0.70**
- production paraphrase Recall@4: **0.60**
- out-of-scope refusal: **0.90 (18/20)**

The prior itself no longer hurts recall; the dominant loss now comes from the
relevance floor and lexical vocabulary mismatch. That is a much more actionable
finding than a perfect score on an easy set.

**Alternative.** Report only retriever Recall@k, or only end-to-end answer
accuracy.

**Trade-off.** Isolated retrieval metrics are easier to compare across models;
end-to-end metrics are closer to product quality. Measuring both the ceiling and
the production path explains *where* the loss enters. The benchmark is still
hand-curated, so the next step is to replace or supplement it with sampled real
queries and human relevance labels.

**Easy mistake.** Treating a classifier as an oracle inside a RAG pipeline. A
77-class classifier at 91% accuracy is good, but any hard downstream filter can
turn its remaining errors into guaranteed retrieval misses. Soft priors degrade
gracefully; hard filters compound upstream mistakes.
