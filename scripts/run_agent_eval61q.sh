#!/usr/bin/env bash
# Hybrid ReAct agent on eval_questions2.json (61Q) + corpus judge.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OPENAI_TIMEOUT=300
source scripts/activate.sh 2>/dev/null || true

ANSWERS=reports/rag_answers_routectx_kimik3_low_agent_eval61q.jsonl
EVAL=reports/stage2/rag_routectx_kimik3_low_agent_eval61q_corpus_judge_eval.json
MODEL=moonshotai/Kimi-K3

RESUME_FLAG=()
if [ -s "$ANSWERS" ]; then
  RESUME_FLAG=(--resume)
  echo "=== RESUME from existing $(wc -l < "$ANSWERS") answers ==="
fi

echo "=== GEN: 61Q agent + Kimi-K3 low === $(date +%T)"
cmd=(
  .venv/bin/python rag_runner.py
  --agent --agent-max-rounds 2
  --questions eval_questions2.json
  --retrieve rerank --top-k 20 --window 2 --candidate-n 100
  --route --route-n 80 --route-global-n 20 --route-context --route-preview-n 20
  --catalog --catalog-min-score 8.0
  --cite passages
  --model "$MODEL" --reasoning-effort low
  --llm-timeout 300
  --out "$ANSWERS"
)
if [ ${#RESUME_FLAG[@]} -gt 0 ]; then
  cmd+=("${RESUME_FLAG[@]}")
fi
"${cmd[@]}"
gen_rc=$?
echo "=== GEN_DONE rc=$gen_rc answers=$(wc -l < "$ANSWERS" 2>/dev/null || echo 0) === $(date +%T)"
[ "$gen_rc" -eq 0 ] || { echo "=== ABORT: generation failed ==="; exit "$gen_rc"; }

# Drop empty/corrupt lines if a prior crash mid-wrote a record.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("reports/rag_answers_routectx_kimik3_low_agent_eval61q.jsonl")
qs = json.loads(Path("eval_questions2.json").read_text(encoding="utf-8"))
if isinstance(qs, dict):
    qs = qs["questions"]
order = [q["id"] for q in qs]
by = {}
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    by[r["id"]] = r
with p.open("w", encoding="utf-8") as f:
    for i in order:
        if i in by:
            f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
print(f"cleaned answers: {sum(1 for i in order if i in by)}/{len(order)}")
PY

echo "=== JUDGE: corpus judge via $MODEL === $(date +%T)"
.venv/bin/python run_eval.py \
  --answers "$ANSWERS" \
  --questions eval_questions2.json \
  --corpus-judge \
  --judge-model "$MODEL" \
  --judge-reasoning-effort low \
  --label routectx-kimik3-low-agent-eval61q \
  --out "$EVAL"
echo "=== ALL_DONE rc=$? === $(date +%T)"
