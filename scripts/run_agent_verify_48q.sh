#!/usr/bin/env bash
# Agent v2 hybrid topk25 + verify v2 (extras mode) on Train-48Q.
# Prefer scripts/run_verify_v2_det_postprocess_48q.sh first (cheap, no regen).
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OPENAI_TIMEOUT=300
source scripts/activate.sh 2>/dev/null || true

ANSWERS=reports/rag_answers_agent_v2_hybrid_topk25_verify_48q.jsonl
EVAL=reports/stage2/rag_agent_v2_hybrid_topk25_verify_48q_corpus_judge_eval.json
MODEL=moonshotai/Kimi-K3

RESUME_FLAG=()
if [ -s "$ANSWERS" ]; then
  RESUME_FLAG=(--resume)
  echo "=== RESUME from existing $(wc -l < "$ANSWERS") answers ==="
fi

echo "=== GEN: 48Q agent v2 + verify v2 extras + Kimi-K3 low === $(date +%T)"
cmd=(
  .venv/bin/python rag_runner.py
  --agent --agent-max-rounds 2
  --verify --verify-mode extras --verify-timeout 45 \
  --verify-model moonshotai/Kimi-K3
  --questions reference_questions.json
  --retrieve rerank --top-k 25 --window 2 --candidate-n 100
  --rerank-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  --rerank-refine-model BAAI/bge-reranker-v2-m3 --rerank-refine-n 40
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
[ "$gen_rc" -eq 0 ] || exit "$gen_rc"

.venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("$ANSWERS")
qs = json.loads(Path("reference_questions.json").read_text(encoding="utf-8"))
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

def vmeta(r):
    return ((r.get("retrieval") or {}).get("agent") or {}).get("verify") or {}

vals = list(by.values())
print(
    "verify: gated={} ran={} skipped={} changed={} det={}".format(
        sum(1 for r in vals if vmeta(r).get("gated")),
        sum(1 for r in vals if vmeta(r).get("ran")),
        sum(1 for r in vals if vmeta(r).get("skipped")),
        sum(1 for r in vals if vmeta(r).get("changed")),
        sum(1 for r in vals if vmeta(r).get("deterministic_stripped")),
    )
)
PY

echo "=== JUDGE: 48Q === $(date +%T)"
.venv/bin/python run_eval.py \
  --answers "$ANSWERS" \
  --questions reference_questions.json \
  --corpus-judge \
  --judge-model "$MODEL" \
  --judge-reasoning-effort low \
  --label agent-v2-hybrid-topk25-verify-48q \
  --out "$EVAL"
echo "=== ALL_DONE 48Q rc=$? === $(date +%T)"
