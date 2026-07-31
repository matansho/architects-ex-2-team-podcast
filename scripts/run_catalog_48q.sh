#!/usr/bin/env bash
# Full Train-48Q with catalog-filtered retrieval, judged by Kimi-K3.
# Answer prompt is the baseline. Only --catalog is new vs the
# routectx-kimik3-low baseline. Supports --resume after a stalled run.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
# Cap any litellm path that forgets to pass timeout= (module default is 6000s).
export OPENAI_TIMEOUT=300

ANSWERS=reports/rag_answers_routectx_kimik3_low_catalog_48q.jsonl
EVAL=reports/stage2/rag_routectx_kimik3_low_catalog_48q_corpus_judge_eval.json
MODEL=moonshotai/Kimi-K3

RESUME_FLAG=()
if [ -s "$ANSWERS" ]; then
  RESUME_FLAG=(--resume)
  echo "=== RESUME from existing $(wc -l < "$ANSWERS") answers ==="
fi

echo "=== GEN: 48Q routectx + Kimi-K3 low + catalog === $(date +%T)"
cmd=(
  .venv/bin/python rag_runner.py
  --questions reference_questions.json
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

echo "=== JUDGE: corpus judge via $MODEL === $(date +%T)"
.venv/bin/python run_eval.py \
  --answers "$ANSWERS" \
  --questions reference_questions.json \
  --corpus-judge \
  --judge-model "$MODEL" \
  --judge-reasoning-effort low \
  --label routectx-kimik3-low-catalog-48q \
  --out "$EVAL"
echo "=== ALL_DONE rc=$? === $(date +%T)"
