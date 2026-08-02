#!/usr/bin/env bash
# Blind / dry-run submission helper.
# Usage:
#   # terminal 1 — champion server (warm)
#   bash scripts/run_gui.sh
#   # terminal 2 — batch ask
#   bash scripts/run_submit.sh reference_questions.json submission_dryrun.jsonl
#   bash scripts/run_submit.sh blind_questions.json submission_<team>.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh 2>/dev/null || true

QUESTIONS="${1:?usage: $0 <questions.json> [out.jsonl] [endpoint]}"
OUT="${2:-submission_out.jsonl}"
ENDPOINT="${3:-http://127.0.0.1:8000}"
# Eval had rare >120s outliers; default submit_runner 90s is too tight.
TIMEOUT="${SUBMIT_TIMEOUT:-300}"

echo "health:"
curl -sf "${ENDPOINT}/health" | .venv/bin/python -m json.tool || {
  echo "ERROR: endpoint not ready at ${ENDPOINT} — start scripts/run_gui.sh first" >&2
  exit 1
}

echo "=== submit_runner ${QUESTIONS} → ${OUT} (timeout=${TIMEOUT}s) ==="
exec .venv/bin/python submit_runner.py \
  --questions "$QUESTIONS" \
  --endpoint "$ENDPOINT" \
  --out "$OUT" \
  --timeout "$TIMEOUT"
