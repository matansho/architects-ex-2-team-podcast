#!/usr/bin/env bash
#
# Regenerate the prompt-strategy comparison report (Stage 1 runs + RAG no-cite).
# Pure local render — reads the *_answers.jsonl / *_eval.json files, makes NO API calls.
#
# Requires the venv + .env:  source scripts/activate.sh
# Then:                      bash scripts/regen_dashboard.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/generate_compare_dashboard.py \
  --title "Prompt Strategy Comparison (Stage 1 + RAG)" \
  --run "Baseline|baseline_answers.jsonl" \
  --run "Few-shot + cite|reports/stage1/few_shot_answers.jsonl" \
  --run "Few-shot + concise|reports/stage1/concise_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 1)|reports/stage1/no_cite_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 2)|reports/stage1/no_cite_few_shot_2_answers.jsonl" \
  --run "Few-shot, contrast (run 1)|reports/stage1/contrast_answers.jsonl" \
  --run "Few-shot, contrast (run 2)|reports/stage1/contrast_2_answers.jsonl" \
  --run "Few-shot, contrast no far apart|reports/stage1/contrast_no_far_apart_answers.jsonl" \
  --run "RAG no-cite|rag_answers.jsonl" \
  --run "RAG no-cite k10±2|rag_answers_k10_w2.jsonl" \
  --run "RAG no-cite k20±2|rag_answers_k20_w2.jsonl" \
  --run "RAG no-cite k20±4|rag_answers_k20_w4.jsonl" \
  --run "RAG RRF k20±2|rag_answers_rrf_k20_w2.jsonl" \
  --run "RAG rerank 100→20±2|rag_answers_rerank_k20_w2.jsonl" \
  --run "RAG rerank + LLM tables|rag_answers_rerank_k20_w2_llm_tables.jsonl" \
  --run "RAG rerank + route + nofaq|reports/rag_answers_rerank_k20_w2_route_nofaq.jsonl" \
  --run "RAG rerank + nofaq|reports/rag_answers_rerank_k20_w2_nofaq.jsonl" \
  --run "RAG rerank + route80+20 + nofaq|reports/rag_answers_rerank_k20_w2_route80_20_nofaq.jsonl" \
  --eval "Baseline|reports/stage1/baseline_eval.json" \
  --eval "Few-shot + cite|reports/stage1/few_shot_eval.json" \
  --eval "Few-shot + concise|reports/stage1/concise_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 1)|reports/stage1/no_cite_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 2)|reports/stage1/no_cite_few_shot_2_eval.json" \
  --eval "Few-shot, contrast (run 1)|reports/stage1/contrast_eval.json" \
  --eval "Few-shot, contrast (run 2)|reports/stage1/contrast_2_eval.json" \
  --eval "Few-shot, contrast no far apart|reports/stage1/contrast_no_far_apart_eval.json" \
  --eval "RAG no-cite|reports/stage2/rag_no_cite_eval.json" \
  --eval "RAG no-cite k10±2|reports/stage2/rag_k10_w2_eval.json" \
  --eval "RAG no-cite k20±2|reports/stage2/rag_k20_w2_eval.json" \
  --eval "RAG no-cite k20±4|reports/stage2/rag_k20_w4_eval.json" \
  --eval "RAG RRF k20±2|reports/stage2/rag_rrf_k20_w2_eval.json" \
  --eval "RAG rerank 100→20±2|reports/stage2/rag_rerank_k20_w2_eval.json" \
  --eval "RAG rerank + LLM tables|reports/stage2/rag_rerank_k20_w2_llm_tables_eval.json" \
  --eval "RAG rerank + route + nofaq|reports/stage2/rag_rerank_k20_w2_route_nofaq_eval.json" \
  --eval "RAG rerank + nofaq|reports/stage2/rag_rerank_k20_w2_nofaq_eval.json" \
  --eval "RAG rerank + route80+20 + nofaq|reports/stage2/rag_rerank_k20_w2_route80_20_nofaq_eval.json" \
  --prompt "RAG no-cite|rag-no-cite" \
  --prompt "RAG no-cite k10±2|rag-no-cite" \
  --prompt "RAG no-cite k20±2|rag-no-cite" \
  --prompt "RAG no-cite k20±4|rag-no-cite" \
  --prompt "RAG RRF k20±2|rag-no-cite" \
  --prompt "RAG rerank 100→20±2|rag-no-cite" \
  --prompt "RAG rerank + LLM tables|rag-no-cite" \
  --prompt "RAG rerank + route + nofaq|rag-no-cite" \
  --prompt "RAG rerank + nofaq|rag-no-cite" \
  --prompt "RAG rerank + route80+20 + nofaq|rag-no-cite" \
  --out reports/stage1/stage1_prompt_strategy_comparison.html

cp reports/stage1/stage1_prompt_strategy_comparison.html deliverables/stage1_prompt_strategy_comparison.html
echo "Copied to deliverables/stage1_prompt_strategy_comparison.html"
