#!/usr/bin/env bash
# Post-deploy smoke test. Runs against any base URL; CD runs it against staging
# before promoting, and against production immediately after.
# Exit non-zero on the first failure so the pipeline stops.
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
AUTH=()
[ -n "$API_KEY" ] && AUTH=(-H "x-api-key: ${API_KEY}")

echo "smoke: ${BASE_URL}"

echo -n "  /health ... "
curl -fsS "${BASE_URL}/health" | grep -q '"status":"ok"' && echo OK

echo -n "  /ready ... "
curl -fsS "${BASE_URL}/ready" | grep -q '"status":"ready"' && echo OK

echo -n "  /chat happy path ... "
BODY=$(curl -fsS -X POST "${BASE_URL}/chat" "${AUTH[@]}" \
  -H 'content-type: application/json' \
  -d '{"question":"My card was charged twice for the same purchase"}')
echo "$BODY" | grep -q '"intent"' || { echo "FAIL: no intent"; exit 1; }
echo "$BODY" | grep -q '"sources"' || { echo "FAIL: no sources"; exit 1; }
echo OK

echo -n "  /chat safety path (lost card -> deterministic) ... "
curl -fsS -X POST "${BASE_URL}/chat" "${AUTH[@]}" -H 'content-type: application/json' \
  -d '{"question":"I lost my card, what do I do"}' | grep -q 'deterministic_policy' && echo OK

echo -n "  /chat rejects invalid input (422) ... "
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/chat" "${AUTH[@]}" \
  -H 'content-type: application/json' -d '{"question":"x"}')
[ "$CODE" = "422" ] && echo OK || { echo "FAIL: got $CODE"; exit 1; }

echo -n "  /metrics ... "
curl -fsS "${BASE_URL}/metrics" | grep -q 'nb_requests_total' && echo OK

echo "smoke passed"
