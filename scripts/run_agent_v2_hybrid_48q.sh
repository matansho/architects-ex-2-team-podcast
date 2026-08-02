#!/usr/bin/env bash
# Agent v2 (unread-map / early-final block / file-diverse seed / multi-page brief)
# + hybrid MiniLM→BGE refine40 + top-k 25 (best offline GT recall).
# Spends Nebius — do not run overnight budget-blind.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export OPENAI_TIMEOUT=300
source scripts/activate.sh 2>/dev/null || true
set -a && source .env && set +a
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.tokenfactory.nebius.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$NEBIUS_API_KEY}"

ANSWERS=reports/rag_answers_agent_v2_hybrid_topk25_48q.jsonl
EVAL=reports/stage2/rag_agent_v2_hybrid_topk25_48q_corpus_judge_eval.json
LOG=reports/stage3/agent_v2_hybrid_topk25_48q.log
MODEL=moonshotai/Kimi-K3

echo "=== GEN: agent v2 hybrid topk25 === $(date +%T)" | tee "$LOG"
.venv/bin/python rag_runner.py \
  --agent --agent-max-rounds 2 \
  --questions reference_questions.json \
  --retrieve rerank --top-k 25 --window 2 --candidate-n 100 \
  --rerank-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --rerank-refine-model BAAI/bge-reranker-v2-m3 \
  --rerank-refine-n 40 \
  --route --route-n 80 --route-global-n 20 --route-context --route-preview-n 20 \
  --catalog --catalog-min-score 8.0 \
  --cite passages \
  --model "$MODEL" --reasoning-effort low \
  --llm-timeout 300 \
  --out "$ANSWERS" 2>&1 | tee -a "$LOG"

echo "=== JUDGE === $(date +%T)" | tee -a "$LOG"
.venv/bin/python run_eval.py \
  --answers "$ANSWERS" \
  --questions reference_questions.json \
  --corpus-judge \
  --judge-model "$MODEL" \
  --judge-reasoning-effort low \
  --label agent-v2-hybrid-topk25-48q \
  --out "$EVAL" 2>&1 | tee -a "$LOG"
echo "=== ALL_DONE === $(date +%T)" | tee -a "$LOG"
