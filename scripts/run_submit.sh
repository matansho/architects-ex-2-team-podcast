#!/usr/bin/env bash
# Blind submit via the *stock* submit_runner.py.
# Timeout/retry lives inside contract.py /ask (RAG_ASK_ATTEMPTS + RAG_LLM_TIMEOUT).
#
# Usage:
#   bash scripts/run_gui.sh
#   bash scripts/run_submit.sh blind_questions.json submission_<team>.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh 2>/dev/null || true

QUESTIONS="${1:?usage: $0 <questions.json> [out.jsonl] [endpoint]}"
OUT="${2:-submission_out.jsonl}"
ENDPOINT="${3:-http://127.0.0.1:8000}"
# Client wait must cover in-pipeline retries (default 3 × ~LLM budget).
TIMEOUT="${SUBMIT_TIMEOUT:-600}"

echo "health:"
curl -sf "${ENDPOINT}/health" || {
  echo "ERROR: endpoint not ready at ${ENDPOINT} — start scripts/run_gui.sh first" >&2
  exit 1
}
echo
echo "=== submit_runner.py (vanilla) ${QUESTIONS} → ${OUT} (timeout=${TIMEOUT}s) ==="
echo "    retries are inside /ask (RAG_ASK_ATTEMPTS / RAG_LLM_TIMEOUT) — not in this client"
exec .venv/bin/python submit_runner.py \
  --questions "$QUESTIONS" \
  --endpoint "$ENDPOINT" \
  --out "$OUT" \
  --timeout "$TIMEOUT"
