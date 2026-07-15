# Stage 1 — Baseline, Eval Harness & Prompt Experiments

Welcome! This document describes everything we built on branch `matan/stage1-setup` for **Exercise 2, Stage 1** (baseline + evaluation harness + prompt strategy comparison).

If you're joining the project: start with the **interactive reports** below, then skim **Key findings**. The rest is here when you need to reproduce or extend something.

---

## Start here — interactive reports

Open these in a browser (double-click or `open reports/...`):

| Report | File | What you'll see |
|---|---|---|
| **5-way prompt comparison** (main) | [`reports/stage1_prompt_strategy_comparison.html`](reports/stage1_prompt_strategy_comparison.html) | Side-by-side metrics, charts, per-question explorer, and full table for all 5 runs |
| **No-cite consistency check** | [`reports/no_cite_consistency.html`](reports/no_cite_consistency.html) | Two replicate runs of our best prompt — same config, different sample |
| **Corpus & dev-set exploration** | [`reports/data_exploration.html`](reports/data_exploration.html) | Corpus size, domains, PDF vs web pages, dev question coverage, text lengths |

```bash
open reports/stage1_prompt_strategy_comparison.html
open reports/data_exploration.html
```

The **5-way dashboard** is the one to use for team discussions and the Stage 1 write-up. It includes summary cards, Plotly charts (relevance, hallucination, refusal, latency, difficulty breakdowns), a question-by-question explorer with judge reasoning, and a sortable table across all runs.

---

## What we were trying to do

Stage 1 asks us to:

1. Run a **bare LLM baseline** on the 48 dev questions (no retrieval).
2. **Build our own eval harness** (relevance, hallucination, citation, latency).
3. Try **at least two prompt strategies** and compare failure profiles.
4. Write a short baseline report with metrics + reflection.

We went further: five prompt variants, two consistency replicates of the winner, corpus exploration, and HTML dashboards so we can actually *see* what's failing.

**Model used throughout:** `deepseek-ai/DeepSeek-V4-Pro` via Nebius Token Factory.

---

## Results at a glance

Answer-judge eval on 48 dev questions (no citations in any run — citation accuracy is 0% everywhere):

| Run | Relevance | Hallucination | Refusal | Avg latency |
|---|---:|---:|---:|---:|
| Baseline | 27.1% | 60.4% | 6.2% | 6,488 ms |
| Few-shot + cite | 35.4% | 60.4% | 2.1% | 2,185 ms |
| Few-shot + concise | 25.0% | 70.8% | 4.2% | 1,544 ms |
| **Few-shot, no cite (run 1)** | **43.8%** | **47.9%** | 8.3% | 2,265 ms |
| **Few-shot, no cite (run 2)** | **43.8%** | **52.1%** | 4.2% | 1,505 ms |

**Best strategy so far:** few-shot examples **without** citation blocks in the prompt, plus a system instruction not to cite. Relevance ~44%; hallucination ~48–52% (vs ~60% baseline).

**Consistency:** two identical no-cite runs agreed on relevance **75%** of questions (36/48) and on aggregate relevance (both 43.8%). Hallucination labels agreed 67% — some judge/answer variance, but the headline metric is stable.

---

## Why we made these decisions

### Eval harness (`eval/`, `run_eval.py`)

The course repo ships questions and runners but **no scorer** — building the harness is the deliverable.

We implemented:

- **Answer judge** (`eval/judge.py`) — LLM-as-judge for relevance, hallucination, refusal. Pinned model, temp=0, JSON output. If hallucination → force not relevant (contradiction can't be "relevant").
- **Citation judge** (`eval/citation.py` + `eval/corpus.py`) — aligned with the **updated course spec**: resolve cited `{file, page}` to corpus text, then LLM judges whether those pages establish the ground-truth answer. Invalid paths → automatic 0. `score_citations_legacy()` kept for debugging retrieval against `ground_truth_sources`.
- **Batch runner** (`eval/harness.py`, `run_eval.py`) — aggregates by difficulty and domain, writes JSON reports.

For Stage 1 bare-model runs we used `--no-citation-judge` (no citations to score). The citation judge is ready for Stage 2.

### Few-shot examples (`few_shot_examples.json`)

Ten exemplar Q&As authored from **uncited** corpus pages (not from the dev set), mirroring dev structure: 4 easy / 4 medium / 2 hard. They teach answer *style*: כן/לא openings, exact numbers, safe refusal.

### Prompt experiments (`few_shot_runner.py`)

| Preset | Motivation |
|---|---|
| **default** (+ cite) | Match course emphasis on citations; few-shot assistant turns include `מקורות:` blocks |
| **concise** | Shorter answers → fewer invented facts? (didn't help — more hallucination) |
| **no-cite** | Hypothesis: citation examples teach the model to invent paths in `answer` text even when `citations: []` |

**The big discovery:** removing `מקורות:` from few-shot examples *and* telling the model not to cite cut hallucination ~12 points and raised relevance ~8 points. The examples were teaching citation *shape*; without retrieval, that became fake file paths counted as hallucinations.

### Reports (`scripts/generate_compare_dashboard.py`)

We needed to compare runs without reading 48 × N JSONL files. The dashboard loads answer JSONL + eval JSON, renders Plotly charts, and embeds a question explorer. Regenerate after new runs:

```bash
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

### Data exploration (`scripts/generate_data_report.py`)

Before tuning prompts we wanted to understand the corpus and dev set: 12 domains, ~571 docs, PDF vs scraped web pages, question difficulty mix. Output: `reports/data_exploration.html`.

---

## Repository map (what we added)

```
eval/
  judge.py              # Answer + citation LLM judges
  citation.py           # Citation scoring (LLM judge + legacy matcher)
  corpus.py             # Resolve {file, page} → text for citation judge
  harness.py            # Batch eval + aggregates
  fixtures/             # Tiny sample for smoke tests

run_eval.py             # CLI: score any answers JSONL

few_shot_runner.py      # Few-shot baseline (--preset default|concise|no-cite)
few_shot_examples.json          # 10 examples with מקורות blocks
few_shot_examples_no_cite.json    # Same Q&As, no citations

baseline_answers.jsonl  # Bare model, 48 questions

reports/
  *_answers.jsonl       # Model outputs per experiment
  *_eval.json           # Full judge reports
  stage1_prompt_strategy_comparison.html   # ← main 5-way dashboard
  no_cite_consistency.html
  data_exploration.html
  baseline_eval.json

scripts/
  activate.sh           # source venv + .env (zsh-safe)
  generate_compare_dashboard.py
  generate_data_report.py

.env.example            # Template for API key (never commit .env)
```

**Not in git** (gitignored): `.env`, `.venv/`, `corpus/` (download with `python get_corpus.py`).

---

## How to reproduce

```bash
cd architects-ex-2
python3 -m venv .venv
source scripts/activate.sh          # loads .env with NEBIUS_API_KEY

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

# Regenerate dashboard
# (use the long generate_compare_dashboard.py command above)
```

**Estimated API spend so far:** ~$0.22 on the shared course key (192 generation calls + 192 judge calls). See `tf_client.py` for per-call cost estimates.

---

## Interesting cases to discuss

### "Perfect" answers without the corpus

On procedural questions (e.g. *how do I cancel dental insurance?*), the model often gives plausible channel lists (call center, agent, website) from **pretraining memory** — not browsing, not our corpus. Compare runs on `dev-21-dental-medium` in the dashboard: baseline invents specific emails/phones; no-cite run 1 is generic-but-right; run 2 gives different specifics. **Plausible ≠ grounded.**

### Citation-shaped hallucinations

With cite few-shot, models append `מקורות:` + fake paths inside `answer` text even though `citations: []` in JSONL. The judge reads `answer` only — those invented paths count as hallucinations.

### Easy vs hard

No-cite few-shot: **easy** ~69% relevance / ~25% hallucination; **hard** ~6–31% relevance / ~56–88% hallucination. Stage 2 retrieval should help hard questions most.

---

## What's next (Stage 2)

- [ ] Use **few-shot, no cite** (or similar) as the generation prompt inside RAG
- [ ] Actually populate structured `citations` from retrieved chunks
- [ ] Turn on full citation judge in eval
- [ ] Beat the no-cite few-shot baseline on relevance + hallucination + citation accuracy

---

## Open questions for the baseline report

1. **Where does the baseline succeed without Harel docs?** Procedural/how-to questions, general insurance law patterns, Hebrew fluency.
2. **Confidently wrong vs refusal?** Baseline hallucinates ~60%; refusals ~6%. For an insurer, wrong specifics are worse than "I don't know" — but our best prompt still hallucinates ~50%.
3. **Judge disagreement?** Pick one question from the dashboard where you disagree with the judge — e.g. a vague-but-directionally-correct answer marked not relevant.

---

## Questions?

Ping in the team channel or open `reports/stage1_prompt_strategy_comparison.html` and walk through a few questions together — the explorer + judge reasoning columns are the fastest way to build shared intuition before Stage 2.
