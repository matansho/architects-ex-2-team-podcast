#!/usr/bin/env bash
# Sandbox: hybrid agent WITH lexical grep tool on hard carve-out questions.
# Compare tool traces to reports/rag_answers_agent_probe_hard.jsonl (no grep).
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OPENAI_TIMEOUT=300
source scripts/activate.sh 2>/dev/null || true

ANSWERS=reports/rag_answers_agent_grep_probe_hard.jsonl
MODEL=moonshotai/Kimi-K3
IDS=dev-06-apartment-hard,dev-28-health-medium,dev-47-travel-hard,dev-17-car-hard,dev-02-apartment-easy

rm -f "$ANSWERS"
echo "=== AGENT+GREP PROBE: $IDS === $(date +%T)"
.venv/bin/python rag_runner.py \
  --agent --agent-max-rounds 2 --agent-grep \
  --questions reference_questions.json \
  --ids "$IDS" \
  --retrieve rerank --top-k 20 --window 2 --candidate-n 100 \
  --route --route-n 80 --route-global-n 20 --route-context --route-preview-n 20 \
  --catalog --catalog-min-score 8.0 \
  --cite passages \
  --model "$MODEL" --reasoning-effort low \
  --llm-timeout 300 \
  --out "$ANSWERS"
gen_rc=$?
echo "=== DONE rc=$gen_rc answers=$(wc -l < "$ANSWERS" 2>/dev/null || echo 0) === $(date +%T)"
[ "$gen_rc" -eq 0 ] || exit "$gen_rc"
.venv/bin/python -c "
import json
from pathlib import Path
for line in Path('$ANSWERS').read_text().splitlines():
    r = json.loads(line)
    tools = [t['name'] for t in r.get('retrieval',{}).get('agent',{}).get('tool_trace',[])]
    print(f\"{r['id']:28s} tools={tools}\")
"
