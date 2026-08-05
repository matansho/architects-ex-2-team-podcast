#!/usr/bin/env bash
# Serve champion /ask + chat GUI (voice via Web Speech API in the browser).
# Defaults match measured stack: MiniLM→BGE@40 · top-k 25 · LLM verify.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh 2>/dev/null || true
set -a && source .env 2>/dev/null && set +a

export PYTHONUNBUFFERED=1
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.tokenfactory.nebius.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${NEBIUS_API_KEY:-}}"

# Champion retrieve + agent knobs (override any with env before launch)
export RAG_TOP_K="${RAG_TOP_K:-25}"
export RAG_WINDOW="${RAG_WINDOW:-2}"
export RAG_CANDIDATE_N="${RAG_CANDIDATE_N:-100}"
export RAG_CATALOG_MIN_SCORE="${RAG_CATALOG_MIN_SCORE:-8.0}"
export RAG_ROUTE_N="${RAG_ROUTE_N:-80}"
export RAG_ROUTE_GLOBAL_N="${RAG_ROUTE_GLOBAL_N:-20}"
export RAG_ROUTE_PREVIEW_N="${RAG_ROUTE_PREVIEW_N:-20}"
export RAG_RERANK_MODEL="${RAG_RERANK_MODEL:-cross-encoder/mmarco-mMiniLMv2-L12-H384-v1}"
export RAG_RERANK_REFINE_MODEL="${RAG_RERANK_REFINE_MODEL:-BAAI/bge-reranker-v2-m3}"
export RAG_RERANK_REFINE_N="${RAG_RERANK_REFINE_N:-40}"
export RAG_MODEL="${RAG_MODEL:-moonshotai/Kimi-K3}"
export RAG_REASONING_EFFORT="${RAG_REASONING_EFFORT:-low}"
export RAG_AGENT_MAX_ROUNDS="${RAG_AGENT_MAX_ROUNDS:-2}"
# Keep LLM calls from wedging the single uvicorn worker (submit retries handle rare fails).
export RAG_LLM_TIMEOUT="${RAG_LLM_TIMEOUT:-120}"
export RAG_VERIFY="${RAG_VERIFY:-1}"
export RAG_VERIFY_MODE="${RAG_VERIFY_MODE:-extras}"
export RAG_VERIFY_TIMEOUT="${RAG_VERIFY_TIMEOUT:-45}"
# Warm index at startup so first blind /ask is not a cold load
export RAG_LAZY_LOAD="${RAG_LAZY_LOAD:-0}"

PORT="${PORT:-8000}"
echo "GUI → http://127.0.0.1:${PORT}/"
echo "API → POST http://127.0.0.1:${PORT}/ask"
echo "stack · top_k=${RAG_TOP_K} · ${RAG_RERANK_MODEL} → ${RAG_RERANK_REFINE_MODEL}@${RAG_RERANK_REFINE_N} · verify=${RAG_VERIFY}"
echo "Voice: use Chrome/Edge, allow microphone, speak Hebrew (he-IL)."
exec .venv/bin/uvicorn contract:app --host 0.0.0.0 --port "$PORT"
