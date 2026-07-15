# Stage 1 — Baseline, Eval Harness & Prompt Experiments

Branch: `matan/stage1-setup`  
Model: `deepseek-ai/DeepSeek-V4-Pro` (Nebius Token Factory)

## Reports

| Report | File |
|---|---|
| 5-way prompt comparison | [`reports/stage1_prompt_strategy_comparison.html`](reports/stage1_prompt_strategy_comparison.html) |
| No-cite consistency (2 replicates) | [`reports/no_cite_consistency.html`](reports/no_cite_consistency.html) |
| Corpus & dev-set exploration | [`reports/data_exploration.html`](reports/data_exploration.html) |

## What we built

- **Eval harness** (`eval/`, `run_eval.py`)
  - Answer judge: relevance, hallucination, refusal (`eval/judge.py`)
  - Citation judge: resolve `{file, page}` → corpus text, LLM judges support for GT answer (`eval/citation.py`, `eval/corpus.py`)
  - Batch runner + aggregates (`eval/harness.py`)
  - Stage 1 runs used `--no-citation-judge` (bare model, no citations)
- **Few-shot baseline** (`few_shot_runner.py`)
  - Presets: `default` (with `מקורות:`), `concise`, `no-cite`
  - 10 examples in `few_shot_examples.json`; no-cite variant in `few_shot_examples_no_cite.json`
- **Comparison dashboards** (`scripts/generate_compare_dashboard.py`, `scripts/generate_data_report.py`)
- **Baseline run** (`baseline_runner.py` → `baseline_answers.jsonl`)

## Prompt experiments (48 dev questions)

Answer-judge eval, no citation judge:

| Run | Relevance | Hallucination | Refusal | Avg latency |
|---|---:|---:|---:|---:|
| Baseline | 27.1% | 60.4% | 6.2% | 6,488 ms |
| Few-shot + cite | 35.4% | 60.4% | 2.1% | 2,185 ms |
| Few-shot + concise | 25.0% | 70.8% | 4.2% | 1,544 ms |
| Few-shot, no cite (run 1) | 43.8% | 47.9% | 8.3% | 2,265 ms |
| Few-shot, no cite (run 2) | 43.8% | 52.1% | 4.2% | 1,505 ms |

**Finding:** few-shot without `מקורות:` blocks + no-cite system prompt performed best. Citation examples taught the model to invent file paths in `answer` text (counted as hallucination by the judge). Two no-cite replicates: 75% per-question relevance agreement, same aggregate relevance (43.8%).

## Files added

```
eval/                   # judge, citation, corpus, harness
run_eval.py
few_shot_runner.py
few_shot_examples.json
few_shot_examples_no_cite.json
baseline_answers.jsonl
reports/                # answers JSONL, eval JSON, HTML dashboards
scripts/activate.sh
scripts/generate_compare_dashboard.py
scripts/generate_data_report.py
.env.example
```

Not in git: `.env`, `.venv/`, `corpus/`

## Reproduce

```bash
source scripts/activate.sh

python baseline_runner.py --model deepseek-ai/DeepSeek-V4-Pro

python few_shot_runner.py --preset no-cite \
  --out reports/no_cite_few_shot_answers.jsonl

python run_eval.py \
  --answers reports/no_cite_few_shot_answers.jsonl \
  --out reports/no_cite_few_shot_eval.json \
  --no-citation-judge

python scripts/generate_compare_dashboard.py \
  --title "Stage 1 — Prompt Strategy Comparison" \
  --run "Baseline|baseline_answers.jsonl" \
  --run "Few-shot + cite|reports/few_shot_answers.jsonl" \
  --run "Few-shot + concise|reports/concise_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 1)|reports/no_cite_few_shot_answers.jsonl" \
  --run "Few-shot, no cite (run 2)|reports/no_cite_few_shot_2_answers.jsonl" \
  --eval "Baseline|reports/baseline_eval.json" \
  --eval "Few-shot + cite|reports/few_shot_eval.json" \
  --eval "Few-shot + concise|reports/concise_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 1)|reports/no_cite_few_shot_eval.json" \
  --eval "Few-shot, no cite (run 2)|reports/no_cite_few_shot_2_eval.json" \
  --out reports/stage1_prompt_strategy_comparison.html
```

Estimated API spend: ~$0.22 (192 generation + 192 judge calls).
