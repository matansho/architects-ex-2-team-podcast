# Stage 1 — Baseline, Eval Harness & Prompt Experiments

Branch: `matan/stage1-setup`  
Model: `deepseek-ai/DeepSeek-V4-Pro` (Nebius Token Factory)

Stage 1 deliverables: bare LLM baseline on 48 dev questions, custom eval harness, at least two prompt strategies compared, and a baseline report with metrics.

## Reports

| Report | File | Contents |
|---|---|---|
| 5-way prompt comparison | [`reports/stage1/stage1_prompt_strategy_comparison.html`](reports/stage1/stage1_prompt_strategy_comparison.html) | Metrics, charts, per-question explorer, judge reasoning, sortable table across all runs |
| No-cite consistency | [`reports/stage1/no_cite_consistency.html`](reports/stage1/no_cite_consistency.html) | Two replicate runs of the best prompt (same config, different sample) |
| Corpus & dev-set exploration | [`reports/exploration/data_exploration.html`](reports/exploration/data_exploration.html) | Corpus size, domains, PDF vs web pages, dev question coverage |

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
| Contrast (far-apart, run 1) | 45.8% | 22.9% | 14.6% | 3,897 ms |
| Contrast (far-apart, run 2) | 41.7% | 27.1% | 12.5% | 3,288 ms |
| Contrast (no far-apart) | 42% | 31% | 17% | 2,780 ms |

Citation accuracy is 0% across all runs — expected without retrieval.

The three contrast rows share one config's answer style but differ in the render-time
gate: **far-apart** hedges every specific whose stated alternative is materially
different from its best guess (in practice, all of them); **no far-apart** commits any
value the model self-labels `confident`. Files: `reports/stage1/contrast_*` (far-apart, run 1),
`reports/stage1/contrast_2_*` (far-apart, run 2), and the no-far-apart run.

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

### Contrast: single-call abstention gating

`contrast_runner.py` makes **one** LLM call per question and returns structured JSON: a
`qualitative` answer (yes/no + how-it-works, which earns relevance) plus a list of
`specifics`, each with a `best` value, a plausible `alt`ernative, and a `confident` flag.
A deterministic Python gate then decides per specific whether to commit the value or
replace it with a hedge — no second model call, so cost stays ≈ one generation.

**Robust finding:** hedging unverifiable specifics roughly **halves hallucination** vs
no-cite (~48% → ~23–31%) while holding relevance in the same ~42–46% band. Relevance is
carried by the qualitative core; hedging the numbers avoids the hallucination tax (a wrong
checkable fact forces `relevant=False`, so it costs both axes at once).

**Run-to-run variance is the dominant effect at n=48.** Two runs of the *identical*
far-apart config gave 45.8%/22.9% and 41.7%/27.1% — a ~4–5pp swing from generation + judge
stochasticity (even at `temperature=0`; the served MoE model is not fully deterministic).
The standard error on a ~45% rate over 48 questions is ≈7pp, so differences under ~10pp
between hedging variants are **not resolvable** on a single run. The one effect larger than
the noise is the ~20pp hallucination cut vs no-cite.

**`--far-apart` vs commit-on-confident (option 1).** The `confident` flag is only ~50%
precise (measured on the run's specifics: dev-12 correct; dev-24, dev-42 confidently wrong).
Committing those values (no far-apart) therefore injects double-penalty hallucinations and
scored worse (42%/31%). Keeping `--far-apart` — which in practice hedges every specific —
is the safer default for a bare model. Committing real numbers is what **retrieval (Stage 2)**
unlocks, not a smarter render gate.

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
  stage1/               # Prompt experiments: answers, evals, dashboards
  stage2/               # Docling / chunking reports + samples
  exploration/          # Corpus & dev-set exploration

scripts/
  activate.sh
  generate_compare_dashboard.py
  regen_dashboard.sh      # one-shot wrapper: regenerate the 8-way comparison report
  prompt_catalog.py       # resolves each run's system prompt + few-shot examples for the report
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
  --out reports/stage1/no_cite_few_shot_answers.jsonl

# Eval (answer judge only for bare runs)
python run_eval.py \
  --answers reports/stage1/no_cite_few_shot_answers.jsonl \
  --out reports/stage1/no_cite_few_shot_eval.json \
  --no-citation-judge

# Contrast — single-call abstention gating.
# Recommended mode: --far-apart hedges every unverifiable specific (bare-model default).
python contrast_runner.py --far-apart \
  --out reports/stage1/contrast_answers.jsonl
python run_eval.py \
  --answers reports/stage1/contrast_answers.jsonl \
  --out reports/stage1/contrast_eval.json \
  --no-citation-judge
#   -> rel 45.8% / hall 22.9% / ref 14.6%  (run 2: 41.7% / 27.1% / 12.5% -- run-to-run noise)

# Variant: commit values the model self-labels `confident` (no far-apart gate).
python contrast_runner.py \
  --out reports/stage1/contrast_no_far_apart_answers.jsonl
python run_eval.py \
  --answers reports/stage1/contrast_no_far_apart_answers.jsonl \
  --out reports/stage1/contrast_no_far_apart_eval.json \
  --no-citation-judge
#   -> rel 42% / hall 31% / ref 17%  (worse: ~50%-precise `confident` flag injects hallucinations)

# Regenerate the 8-way comparison report (baseline + few-shot presets + contrast).
# Pure local render, no API calls. Wraps generate_compare_dashboard.py with all runs.
bash scripts/regen_dashboard.sh
```

Estimated API spend: ~$0.30 (baseline + few-shot presets + two contrast runs; ~288 generation + ~288 judge calls). Note the ~4–5pp run-to-run swing on 48 questions — see the contrast rows above and the variance note under "Contrast: single-call abstention gating."

## Implications for Stage 2

- Use the **no-cite few-shot** style as the generation prompt inside RAG (or a variant that cites only from retrieved chunks).
- Populate structured `citations` from actual retrieved chunks, not from few-shot examples.
- Turn on the full citation judge in eval.
- Target: beat 43.8% relevance and ~50% hallucination while achieving real citation accuracy.

## Baseline report reflection prompts

1. **Where does the baseline succeed without Harel docs?** Procedural/how-to questions, general insurance law patterns, Hebrew fluency.
2. **Confidently wrong vs refusal?** Baseline hallucinates ~60% with only ~6% refusals. For an insurer, wrong specifics are worse than "I don't know" — but our best prompt still hallucinates ~50%.
3. **Judge disagreement?** Pick one question from the dashboard where you disagree with the judge — e.g. a vague-but-directionally-correct answer marked not relevant. The explorer shows judge reasoning per question.
