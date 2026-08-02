#!/usr/bin/env bash
#
# Regenerate the prompt-strategy comparison report as a multi-tab dashboard:
#   Train           → reference_questions.json (48 dev questions, ids dev-*)
#   Eval/Test       → eval_questions.json      (32 held-out questions, ids dev2-*)
#   Eval questions2 → eval_questions2.json     (61Q = 32 + 29 extend, ids eval-*)
#
# Each tab is a full comparison dashboard (its own runs/evals/corpus-evals)
# isolated in an iframe. Pure local render — reads the *_answers.jsonl /
# *_eval.json files, makes NO API calls.
#
# Requires the venv + .env:  source scripts/activate.sh
# Then:                      bash scripts/regen_dashboard.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/generate_compare_dashboard.py \
  --title "Prompt Strategy Comparison (Train vs Eval)" \
  --out reports/stage1/stage1_prompt_strategy_comparison.html \
  \
  --tab "Train (dev · 48Q)|reference_questions.json" \
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
  --run "RAG rerank + route80+20 + hits cite|reports/rag_answers_rerank_k20_w2_route80_20_nofaq_cite.jsonl" \
  --run "RAG rerank + route80+20 + passage cite|reports/rag_answers_rerank_k20_w2_route80_20_passage_cite.jsonl" \
  --run "RAG rerank + route40+10 + passage cite|reports/rag_answers_rerank_k20_w2_route40_10_passage_cite.jsonl" \
  --run "RAG route80+20 passage (Kimi judge)|reports/rag_answers_rerank_k20_w2_route80_20_passage_cite.jsonl" \
  --run "RAG route40+10 passage (Kimi judge)|reports/rag_answers_rerank_k20_w2_route40_10_passage_cite.jsonl" \
  --run "RAG routectx 80+20 + Kimi-K3 (low)|reports/rag_answers_rerank_k20_w2_route80_20_routectx_kimik3_low.jsonl" \
  --run "RAG routectx + catalog + Kimi-K3 (low)|reports/rag_answers_routectx_kimik3_low_catalog_48q.jsonl" \
  --run "RAG agent routectx+catalog + Kimi-K3 (low)|reports/rag_answers_routectx_kimik3_low_agent_48q.jsonl" \
  --run "RAG agent+verify + Kimi-K3 (low)|reports/rag_answers_routectx_kimik3_low_agent_verify_48q.jsonl" \
  --run "RAG agent paged brief + BGE|reports/rag_answers_agent_paged_48q.jsonl" \
  --run "RAG agent focused + MiniLM-L12|reports/rag_answers_agent_minilm_l12_focused_48q.jsonl" \
  --run "RAG agent integrated + MiniLM-L12|reports/rag_answers_agent_minilm_l12_integrated_48q.jsonl" \
  --run "RAG agent hybrid MiniLM→BGE|reports/rag_answers_agent_hybrid_ce_48q.jsonl" \
  --run "RAG no-agent hybrid MiniLM→BGE|reports/rag_answers_noagent_hybrid_ce_48q.jsonl" \
  --run "RAG agent v2 hybrid topk25|reports/rag_answers_agent_v2_hybrid_topk25_48q.jsonl" \
  --run "RAG agent v2 hybrid topk20|reports/rag_answers_agent_v2_hybrid_topk20_48q.jsonl" \
  --run "RAG no-agent hybrid topk25|reports/rag_answers_noagent_hybrid_topk25_48q.jsonl" \
  --run "RAG agent v2 fewshot+grep 48Q|reports/rag_answers_agent_v2_fewshot_grep_48q.jsonl" \
  --corpus-eval "RAG rerank + route80+20 + passage cite|reports/stage2/rag_rerank_k20_w2_route80_20_corpus_judge_eval.json" \
  --corpus-eval "RAG rerank + route40+10 + passage cite|reports/stage2/rag_rerank_k20_w2_route40_10_corpus_judge_eval.json" \
  --corpus-eval "RAG route80+20 passage (Kimi judge)|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_kimik3_corpus_judge_eval.json" \
  --corpus-eval "RAG route40+10 passage (Kimi judge)|reports/stage2/rag_rerank_k20_w2_route40_10_passage_cite_kimik3_corpus_judge_eval.json" \
  --corpus-eval "RAG routectx 80+20 + Kimi-K3 (low)|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_kimik3_low_corpus_judge_eval.json" \
  --corpus-eval "RAG routectx + catalog + Kimi-K3 (low)|reports/stage2/rag_routectx_kimik3_low_catalog_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent routectx+catalog + Kimi-K3 (low)|reports/stage2/rag_routectx_kimik3_low_agent_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent+verify + Kimi-K3 (low)|reports/stage2/rag_routectx_kimik3_low_agent_verify_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent paged brief + BGE|reports/stage2/rag_agent_paged_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent focused + MiniLM-L12|reports/stage2/rag_agent_minilm_l12_focused_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent integrated + MiniLM-L12|reports/stage2/rag_agent_minilm_l12_integrated_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent hybrid MiniLM→BGE|reports/stage2/rag_agent_hybrid_ce_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG no-agent hybrid MiniLM→BGE|reports/stage2/rag_noagent_hybrid_ce_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent v2 hybrid topk25|reports/stage2/rag_agent_v2_hybrid_topk25_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent v2 hybrid topk20|reports/stage2/rag_agent_v2_hybrid_topk20_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG no-agent hybrid topk25|reports/stage2/rag_noagent_hybrid_topk25_48q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent v2 fewshot+grep 48Q|reports/stage2/rag_agent_v2_fewshot_grep_48q_corpus_judge_eval.json" \
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
  --eval "RAG rerank + route80+20 + hits cite|reports/stage2/rag_rerank_k20_w2_route80_20_nofaq_cite_eval.json" \
  --eval "RAG rerank + route80+20 + passage cite|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval.json" \
  --eval "RAG rerank + route40+10 + passage cite|reports/stage2/rag_rerank_k20_w2_route40_10_passage_cite_eval.json" \
  --eval "RAG route80+20 passage (Kimi judge)|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_kimik3_judge_eval.json" \
  --eval "RAG route40+10 passage (Kimi judge)|reports/stage2/rag_rerank_k20_w2_route40_10_passage_cite_kimik3_judge_eval.json" \
  --eval "RAG routectx 80+20 + Kimi-K3 (low)|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_kimik3_low_eval.json" \
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
  --prompt "RAG rerank + route80+20 + hits cite|rag-no-cite" \
  --prompt "RAG rerank + route80+20 + passage cite|rag-no-cite" \
  --prompt "RAG rerank + route40+10 + passage cite|rag-no-cite" \
  --prompt "RAG route80+20 passage (Kimi judge)|rag-no-cite" \
  --prompt "RAG route40+10 passage (Kimi judge)|rag-no-cite" \
  --prompt "RAG routectx 80+20 + Kimi-K3 (low)|rag-no-cite" \
  --prompt "RAG routectx + catalog + Kimi-K3 (low)|rag-no-cite" \
  --prompt "RAG agent routectx+catalog + Kimi-K3 (low)|rag-no-cite" \
  --prompt "RAG agent+verify + Kimi-K3 (low)|rag-no-cite" \
  --prompt "RAG agent paged brief + BGE|rag-no-cite" \
  --prompt "RAG agent focused + MiniLM-L12|rag-no-cite" \
  --prompt "RAG agent integrated + MiniLM-L12|rag-no-cite" \
  --prompt "RAG agent hybrid MiniLM→BGE|rag-no-cite" \
  \
  --tab "Eval / Test (dev2 · 32Q)|eval_questions.json" \
  --run "RAG rerank + route80+20 + passage cite|reports/rag_answers_rerank_k20_w2_route80_20_passage_cite_eval_questions.jsonl" \
  --run "RAG rerank + route40+10 + passage cite|reports/rag_answers_rerank_k10_w2_route40_10_passage_cite_eval_questions.jsonl" \
  --eval "RAG rerank + route80+20 + passage cite|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval_questions_eval.json" \
  --eval "RAG rerank + route40+10 + passage cite|reports/stage2/rag_rerank_k10_w2_route40_10_passage_cite_eval_questions_eval.json" \
  \
  --tab "Eval questions2 (eval · 61Q)|eval_questions2.json" \
  --run "RAG rerank + route80+20 + passage cite|reports/rag_answers_rerank_k20_w2_route80_20_passage_cite_eval_questions2.jsonl" \
  --run "RAG routectx 80+20 + Qwen3-235B|reports/rag_answers_rerank_k20_w2_route80_20_routectx_qwen235_eval_questions2.jsonl" \
  --run "RAG routectx 80+20 + Kimi-K3 (low)|reports/rag_answers_rerank_k20_w2_route80_20_routectx_kimik3_low_eval_questions2.jsonl" \
  --run "RAG agent routectx+catalog + Kimi-K3 (low)|reports/rag_answers_routectx_kimik3_low_agent_eval61q.jsonl" \
  --run "RAG agent v2 hybrid topk25 61Q|reports/rag_answers_agent_v2_hybrid_topk25_eval61q.jsonl" \
  --eval "RAG rerank + route80+20 + passage cite|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval_questions2_eval.json" \
  --eval "RAG routectx 80+20 + Qwen3-235B|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_qwen235_eval_questions2_eval.json" \
  --eval "RAG routectx 80+20 + Kimi-K3 (low)|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_kimik3_low_eval_questions2_eval.json" \
  --corpus-eval "RAG rerank + route80+20 + passage cite|reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval_questions2_corpus_judge_eval.json" \
  --corpus-eval "RAG routectx 80+20 + Qwen3-235B|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_qwen235_eval_questions2_corpus_judge_eval.json" \
  --corpus-eval "RAG routectx 80+20 + Kimi-K3 (low)|reports/stage2/rag_rerank_k20_w2_route80_20_routectx_kimik3_low_eval_questions2_corpus_judge_eval.json" \
  --corpus-eval "RAG agent routectx+catalog + Kimi-K3 (low)|reports/stage2/rag_routectx_kimik3_low_agent_eval61q_corpus_judge_eval.json" \
  --corpus-eval "RAG agent v2 hybrid topk25 61Q|reports/stage2/rag_agent_v2_hybrid_topk25_eval61q_corpus_judge_eval.json" \
  --prompt "RAG rerank + route80+20 + passage cite|rag-no-cite" \
  --prompt "RAG routectx 80+20 + Qwen3-235B|rag-no-cite" \
  --prompt "RAG routectx 80+20 + Kimi-K3 (low)|rag-no-cite" \
  --prompt "RAG agent routectx+catalog + Kimi-K3 (low)|rag-no-cite"

cp reports/stage1/stage1_prompt_strategy_comparison.html deliverables/stage1_prompt_strategy_comparison.html
echo "Copied to deliverables/stage1_prompt_strategy_comparison.html"
