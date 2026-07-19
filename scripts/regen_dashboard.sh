#!/usr/bin/env bash
#
# Regenerate the Stage 1 prompt-strategy comparison report (8 runs, incl. contrast).
# Pure local render — reads the *_answers.jsonl / *_eval.json files, makes NO API calls.
#
# Requires the venv + .env:  source scripts/activate.sh
# Then:                      bash scripts/regen_dashboard.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/generate_compare_dashboard.py \
  --title "Stage 1 — Prompt Strategy Comparison" \
  --run "Baseline|baseline_answers.jsonl" \
  --run "Few-shot + cite|reports/few_shot_answers.jsonl" \
  --run "Few-shot + concise|reports/concise_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 1)|reports/no_cite_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 2)|reports/no_cite_few_shot_2_answers.jsonl" \
  --run "Few-shot, contrast (run 1)|reports/contrast_answers.jsonl" \
  --run "Few-shot, contrast (run 2)|reports/contrast_2_answers.jsonl" \
  --run "Few-shot, contrast no far apart|reports/contrast_no_far_apart_answers.jsonl" \
  --eval "Baseline|reports/baseline_eval.json" \
  --eval "Few-shot + cite|reports/few_shot_eval.json" \
  --eval "Few-shot + concise|reports/concise_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 1)|reports/no_cite_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 2)|reports/no_cite_few_shot_2_eval.json" \
  --eval "Few-shot, contrast (run 1)|reports/contrast_eval.json" \
  --eval "Few-shot, contrast (run 2)|reports/contrast_2_eval.json" \
  --eval "Few-shot, contrast no far apart|reports/contrast_no_far_apart_eval.json" \
  --out reports/stage1_prompt_strategy_comparison.html

cp reports/stage1_prompt_strategy_comparison.html deliverables/stage1_prompt_strategy_comparison.html
echo "Copied to deliverables/stage1_prompt_strategy_comparison.html"
