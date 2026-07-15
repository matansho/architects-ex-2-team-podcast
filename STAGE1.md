# Stage 1 — Baseline, Eval Harness & Prompt Experiments

Branch: `matan/stage1-setup`  
Model: `deepseek-ai/DeepSeek-V4-Pro` (Nebius Token Factory)

Stage 1 deliverables: bare LLM baseline on 48 dev questions, custom eval harness, at least two prompt strategies compared, and a baseline report with metrics.

## Reports

| Report | File | Contents |
|---|---|---|
| 5-way prompt comparison | [`reports/stage1_prompt_strategy_comparison.html`](reports/stage1_prompt_strategy_comparison.html) | Metrics, charts, per-question explorer, judge reasoning, sortable table across all runs |
| No-cite consistency | [`reports/no_cite_consistency.html`](reports/no_cite_consistency.html) | Two replicate runs of the best prompt (same config, different sample) |
| Corpus & dev-set exploration | [`reports/data_exploration.html`](reports/data_exploration.html) | Corpus size, domains, PDF vs web pages, dev question coverage |

## Process

1. **Explored the data** — ran `scripts/generate_data_report.py` to understand corpus structure (12 domains, ~571 docs) and dev-set difficulty mix before tuning prompts.
2. **Built the eval harness** — the course repo ships questions and runners but no scorer; we implemented answer + citation judges and a batch runner.
3. **Ran bare baseline** — `baseline_runner.py` on all 48 dev questions, no retrieval.
4. **Iterated on prompts** — three few-shot presets (`default`, `concise`, `no-cite`), scored each with the answer judge.
5. **Validated the winner** — ran no-cite few-shot twice to check consistency.
6. **Built comparison dashboards** — HTML reports so we can inspect failures question-by-question instead of reading raw JSONL.

## What we built

### Eval harness (`eval/`, `run_eval.py`)

- **Answer judge** (`eval/judge.py`) — LLM-as-judge for relevance, hallucination, refusal. Pinned model, temp=0, JSON output. If hallucination is detected, relevance is forced to false (a contradictory answer can't be relevant).
- **Citation judge** (`eval/citation.py`, `eval/corpus.py`) — aligned with the updated course spec: resolve cited `{file, page}` to corpus text, then LLM judges whether those pages establish the ground-truth answer. Invalid paths score 0 automatically. `score_citations_legacy()` kept for debugging retrieval against `ground_truth_sources`.
- **Batch runner** (`eval/harness.py`) — scores a full answers JSONL, aggregates by difficulty and domain, writes eval JSON.

Stage 1 bare-model runs used `--no-citation-judge` because the model produces no real citations without retrieval. The citation judge is implemented and ready for Stage 2.

### Few-shot baseline (`few_shot_runner.py`)

Ten exemplar Q&As in `few_shot_examples.json`, authored from uncited corpus pages (not from the dev set). Structure mirrors dev set: 4 easy / 4 medium / 2 hard. Examples teach answer style: כן/לא openings, exact numbers where known, safe refusal when uncertain.

| Preset | What it does | Why we tried it |
|---|---|---|
| `default` | Few-shot assistant turns include `מקורות:` citation blocks | Match course emphasis on citations |
| `concise` | Shorter system prompt asking for brief answers | Hypothesis: shorter answers → fewer invented facts |
| `no-cite` | Same Q&As without `מקורות:` blocks; system says not to cite | Hypothesis: citation examples teach the model to invent paths even when `citations: []` |

`few_shot_examples_no_cite.json` is the same 10 Q&As with citation blocks stripped.

### Dashboards (`scripts/`)

- `generate_compare_dashboard.py` — loads answer JSONL + eval JSON per run, renders Plotly charts and a question explorer.
- `generate_data_report.py` — corpus and dev-set statistics.
- `activate.sh` — sources venv + `.env` (zsh-safe).

## Results (48 dev questions, answer judge only)

| Run | Relevance | Hallucination | Refusal | Avg latency |
|---|---:|---:|---:|---:|
| Baseline | 27.1% | 60.4% | 6.2% | 6,488 ms |
| Few-shot + cite | 35.4% | 60.4% | 2.1% | 2,185 ms |
| Few-shot + concise | 25.0% | 70.8% | 4.2% | 1,544 ms |
| Few-shot, no cite (run 1) | 43.8% | 47.9% | 8.3% | 2,265 ms |
| Few-shot, no cite (run 2) | 43.8% | 52.1% | 4.2% | 1,505 ms |

Citation accuracy is 0% across all runs — expected without retrieval.

### Main finding: citation-shaped hallucinations

Few-shot examples with `מקורות:` blocks taught the model to append fake file paths inside `answer` text, even when structured `citations: []` in the JSONL output. The answer judge reads `answer` only, so those invented paths count as hallucinations.

Removing `מקורות:` from examples and instructing the model not to cite:
- Raised relevance from 27% (baseline) to ~44%
- Cut hallucination from ~60% to ~48–52%
- Increased refusals slightly (8% vs 6%) — the model says "I don't know" more often instead of inventing specifics

Few-shot + cite improved relevance (35%) without reducing hallucination — the examples helped answer style but the citation format caused fake paths.

Few-shot + concise was worse on both relevance and hallucination — brevity didn't reduce fabrication.

### Consistency check

Two identical no-cite runs agreed on relevance for 36/48 questions (75%) and produced the same aggregate relevance (43.8%). Hallucination labels agreed 67%. Some per-question variance from judge/answer stochasticity, but the headline metrics are stable.

### Failure patterns worth noting

**Plausible answers without the corpus.** On procedural questions (e.g. how to cancel dental insurance), the model often gives reasonable channel lists (call center, agent, website) from pretraining, not from Harel docs. Compare `dev-21-dental-medium` across runs in the dashboard: baseline invents specific emails/phones; no-cite runs are more generic. Plausible ≠ grounded.

**Easy vs hard split.** No-cite few-shot: easy questions ~69% relevance / ~25% hallucination; hard questions ~6–31% relevance / ~56–88% hallucination. Retrieval in Stage 2 should help hard questions most.

## Files added

```
eval/
  judge.py              # Answer + citation LLM judges
  citation.py           # Citation scoring (LLM judge + legacy matcher)
  corpus.py             # Resolve {file, page} → text
  harness.py            # Batch eval + aggregates
  fixtures/             # Smoke-test sample

run_eval.py
few_shot_runner.py
few_shot_examples.json
few_shot_examples_no_cite.json
baseline_answers.jsonl

reports/
  *_answers.jsonl       # Model outputs per experiment
  *_eval.json           # Full judge reports
  stage1_prompt_strategy_comparison.html
  no_cite_consistency.html
  data_exploration.html

scripts/
  activate.sh
  generate_compare_dashboard.py
  generate_data_report.py

.env.example
```

Not in git: `.env`, `.venv/`, `corpus/` (download with `python get_corpus.py`).

## Reproduce

```bash
cd architects-ex-2
python3 -m venv .venv
source scripts/activate.sh
pip install -r requirements.txt
python get_corpus.py                # once, ~105 MB

# Baseline
python baseline_runner.py --model deepseek-ai/DeepSeek-V4-Pro

# Best prompt so far
python few_shot_runner.py --preset no-cite \
  --out reports/no_cite_few_shot_answers.jsonl

# Eval (answer judge only for bare runs)
python run_eval.py \
  --answers reports/no_cite_few_shot_answers.jsonl \
  --out reports/no_cite_few_shot_eval.json \
  --no-citation-judge

# Regenerate 5-way dashboard
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

## Implications for Stage 2

- Use the **no-cite few-shot** style as the generation prompt inside RAG (or a variant that cites only from retrieved chunks).
- Populate structured `citations` from actual retrieved chunks, not from few-shot examples.
- Turn on the full citation judge in eval.
- Target: beat 43.8% relevance and ~50% hallucination while achieving real citation accuracy.

## Baseline report reflection prompts

1. **Where does the baseline succeed without Harel docs?** Procedural/how-to questions, general insurance law patterns, Hebrew fluency.
2. **Confidently wrong vs refusal?** Baseline hallucinates ~60% with only ~6% refusals. For an insurer, wrong specifics are worse than "I don't know" — but our best prompt still hallucinates ~50%.
3. **Judge disagreement?** Pick one question from the dashboard where you disagree with the judge — e.g. a vague-but-directionally-correct answer marked not relevant. The explorer shows judge reasoning per question.
