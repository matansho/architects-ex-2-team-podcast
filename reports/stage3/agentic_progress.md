# Stage 3 — Agentic path: progress report

**Date:** 2026-07-30 · **Branch focus:** corpus scheme + structured retrieval tools ahead of a full ReAct loop  
**Answer / judge model:** `moonshotai/Kimi-K3` (`reasoning_effort=low`)  
**Router / decomposer:** `google/gemma-3-27b-it`  
**Baseline to beat:** routectx hybrid 80+20 + passage cite + no-FAQ — **40/48 ok (83.3%)**, gt_covered **85.4%**, fully_supported **97.9%**

---

## Goal

Stage 3 asks for a production-style agentic system. We are **not** jumping straight to a free-form multi-agent loop. Plan:

1. Give the system a navigable **scheme of the corpus** (catalog → TOC/outline later).
2. Add **structured retrieval tools** (catalog-filtered search, then section expand).
3. Only then wrap a **single ReAct-style agent** that calls those tools.

This report covers what shipped toward that plan, and measured Train-48Q impact.

---

## Architecture so far

```
question
  ├─ (optional) query decompose — rival products only
  ├─ dense preview → domain router (routectx)
  ├─ catalog lexical match → maybe narrow route pool to product files
  ├─ hybrid dense (route_n in pool + route_global_n from full corpus)
  ├─ cross-encoder rerank → ±window expand
  ├─ hybrid ReAct (--agent): peek_titles / fetch_chunks / search / catalog_lookup
  └─ answer LLM + USED_PASSAGES citations
```

Deterministic helpers (catalog, decompose, auto title-peek) still available; `--agent` wraps seed retrieve in a short tool loop.

---

## What we built

### 1. Corpus catalog (scheme layer 1)

| Piece | Location |
| --- | --- |
| Builder | `scripts/build_catalog.py` |
| Module | `rag/catalog.py` |
| Artifact | `artifacts/catalog.json` (557 files, 77 products, 12 domains) |
| Runner flag | `--catalog` / `--catalog-min-score` / `--catalog-path` |

Each entry: domain, `doc_type`, title, aliases, file path, chunk count, blurb, `related_files` (token-overlap heuristic to same-domain PDFs).

**Agent-facing APIs already in the module** (usable by a future tool loop):

- `list_products(domain=…)`
- `search_catalog(query, domain=…)`
- `catalog_files_for_query(…)` → files for filtered retrieval
- `catalog_summary_for_prompt(…)` — compact text for a system prompt

**Filtered retrieval wiring** (`rag_runner.py`):

1. Lexical catalog search, scoped to routed domain(s).
2. Activate only if best score ≥ `min_score` (48Q used **8.0**).
3. Prefer product hits; attach branded related PDFs/forms that share distinctive tokens.
4. Map files → chunk idxs; require ≥10 chunks or fall back to domain pool.
5. Intersect with domain `route_idxs` → hybrid dense uses that as the **route** leg; global leg still searches the full corpus.

Catalog matching details: Hebrew stopwords dropped, proclitic strip (`ה/ו/ל/ב`), longer tokens weigh more, product-first cluster + margin.

### 2. Query decomposition (tight)

| Piece | Location |
| --- | --- |
| Module | `rag/decompose.py` |
| Runner flag | `--decompose` |
| Run script | `scripts/run_decompose_48q.sh` |

Splits **rival product** comparisons into sub-questions; budgets (`top_k`, `candidate_n`, `route_n`) are **split**, not multiplied. Aspect / join questions collapse back to the original (gate: `split_kind == "rival_products"`).

### 3. Supporting infra (agent-ready)

- **Route-context** — dense preview reused as router evidence + hybrid global seed (`--route-context`).
- **`--llm-timeout`** + `num_retries=0` + `--resume` — Nebius stalls no longer hang for 6000s.
- **Query-form bench** — declarative rewrite explored (`scripts/bench_query_form.py`); best-of-both ≥0.97 suggests fusion later, **not** wired as replace.

### 4. Explicitly *not* done yet

- Document / TOC outline tools (`expand`, `get_section`)
- ReAct / tool-calling loop over catalog + retrieve + expand
- FastAPI `/ask` packaging of the agent
- Multi-agent orchestration (deferred; prefer one agent + tools)

---

## Measured results (corpus judge v4)

Same stack as baseline unless noted: rerank k20 ±2, candidate 100, route 80+20, routectx, no-FAQ, passage cite, Kimi-K3 low.

### Train-48Q (`reference_questions.json`)

| Run | ok | gt_covered | fully_supported | Notes |
| --- | ---: | ---: | ---: | --- |
| **Baseline** routectx | **40/48 (83.3%)** | 85.4% | 97.9% | Reference |
| **+ decompose** (loose then tight) | **40/48 (83.3%)** | 85.4% | 97.9% | Flat overall; 7 splits on loose 48Q |
| **+ catalog** (`min_score=8`) | **41/48 (85.4%)** | **87.5%** | 97.9% | Activated on **25/48** questions |
| **+ hybrid ReAct agent** | **42/48 (87.5%)** | **89.6%** | 97.9% | Best 48Q so far |
| **+ agent + gated verify/strip** | **33/48 (68.8%)** | 72.9% | 93.8% | **Regressed** — strip removed GT facts |

### Eval-61Q (`eval_questions2.json`)

| Run | ok | gt_covered | fully_supported | Notes |
| --- | ---: | ---: | ---: | --- |
| Prior routectx Kimi-K3 low | **44/61 (72.1%)** | 85.2% | 82.0% | Pre-agent reference |
| **+ hybrid ReAct agent** | **48/61 (78.7%)** | **91.8%** | 82.0% | +4 ok vs prior; overclaim still dominates fails |
| Agent + verify | — | — | — | Aborted at 18/61 after 48Q regression; not judged |

### Catalog flips vs baseline

| ID | Baseline → Catalog | Catalog fired? | Comment |
| --- | --- | --- | --- |
| `dev-17-car-hard` | ✗ → ✓ | yes | Switch pool pinned; main intended win |
| `dev-47-travel-hard` | ✗ → ✓ | yes | First Class product focus |
| `dev-13-car-easy` | ✗ → ✓ | no | Likely LLM variance (no filter) |
| `dev-06-apartment-hard` | ✓ → ✗ | yes | gt covered but cite support failed |
| `dev-28-health-medium` | ✓ → ✗ | yes | omitted “chronic-only” GT fact |

### Decompose probes

- Loose 48Q: flat 40/48; real win `dev-17`, real regression `dev-41` (wrong split dropped `*2735` / `safe-mortgage`).
- Tight gate (rival-only) on 7 hard cases: **6/7 ok**; `dev-41` fixed; aspect splits (`dev-23`, `dev-35`) no longer split.
- Full 48Q with **tight** decompose not re-run end-to-end (extrapolated ~41/48); catalog alone already delivered +1 on full 48Q.

### Prompt completeness v6

Tried forcing qualifiers / full lists → noisy +1 with grounding regressions and +51% longer answers. **Reverted**; patch kept at `reports/prompt_completeness_v6.patch`.

### 3. Title peek + section fetch (scheme layer 2, local)

| Piece | Location |
| --- | --- |
| Module | `rag/peek.py` |
| Offline probe | `scripts/probe_peek.py` |
| 48Q runner | `scripts/run_peek_48q.sh` |
| Runner flags | `--peek-titles` / `--peek-radius` / `--peek-max-fetches` / `--peek-always` |

Index has no semantic heading field; titles are inferred from each chunk’s **first line** (chunker already promotes headers). After CE expand, for top anchors:

1. Scan ±`peek-radius` sequential neighbors for first-line titles.
2. If the question has coverage/exception cues (or `--peek-always`), fetch chunks whose titles look like carve-outs / limits / waiting periods / liability tables.
3. Attach them as extra prev/next neighbors (budgeted, de-duped by title).

**Offline probe** (no answer LLM) on hard cases showed real fetches:

- `dev-06`: `חריגים` / `השתתפות עצמית` / `סייגים לכיסוי` near hits (also pulls wrong business PDFs without catalog — use with `--catalog`).
- `dev-28`: `תקופת המתנה` near service-letter hits.
- `dev-47`: `גבולות אחריות` tables + `חריגים` sections.

Full Train-48Q with catalog+peek **not judged yet** — run `bash scripts/run_peek_48q.sh`.

### 4. Hybrid ReAct agent (single agent + tools)

| Piece | Location |
| --- | --- |
| Tools + state | `rag/agent_tools.py` |
| Loop + answer | `rag/agent.py` |
| Runner | `rag_runner.py --agent` |
| Probe | `scripts/run_agent_probe.sh` |
| 48Q | `scripts/run_agent_48q.sh` |
| API | `contract.py` `/ask` (warm index on startup) |

Flow: **seed retrieve** (route + catalog + CE + window, no auto-peek) → agent LLM with native tools (`peek_titles`, `fetch_chunks`, `search`, `catalog_lookup`, `final_answer`) for up to **2 rounds** → grounded answer via existing `SYSTEM_PASSAGE_CITE`.

**Train-48Q:** **42/48 ok (87.5%)**, gt_covered **89.6%**, fully_supported **97.9%** (beats catalog 41/48). Mean latency ~46s. Tool mix: 32/48 final-only; peek on 6Q, fetch on 12Q, search on 10Q, catalog_lookup on 1Q.

**Eval-61Q:** **48/61 ok (78.7%)**, gt_covered **91.8%**, fully_supported **82.0%** (beats prior routectx 44/61). Fail mode mostly overclaim (cov✓ / sup✗).

Train-48Q agent failures (6): incomplete GT (`dev-29`, `dev-32`, `dev-35`, `dev-45`, `dev-48`) plus one overclaim (`dev-46`).

### 5. Gated verify + strip (anti-overclaim) — measured regression

| Piece | Location |
| --- | --- |
| Module | `rag/verify.py` |
| Flags | `--verify` / `--verify-model` / `--verify-timeout` / `--verify-force` |
| 48Q script | `scripts/run_agent_verify_48q.sh` |
| 61Q script | `scripts/run_agent_verify_eval61q.sh` |

Design: **(1)** risk gate, **(2)** tiny Gemma on cited passages only, **(3)** strip unsupported sentences (no regen), **(4)** hard timeout → answer as-is.

**Train-48Q result:** **33/48 ok (68.8%)**, gt_covered **72.9%**, fully_supported **93.8%** — **−9 ok vs agent-only**. Strip deleted checkable GT facts (amounts, timing, conditions), not only inventions. Eval-61Q verify run aborted mid-gen (18/61) after seeing the 48Q collapse; not judged.

**Decision:** keep verify **off by default** (`RAG_VERIFY=0` in `/ask`; pass `--verify` only for experiments). Do not ship strip-verify until it can drop clear inventions without touching answer-critical facts.

---

## How to run

```bash
source scripts/activate.sh

# Rebuild catalog (writes data/catalog.json; shipped copy in artifacts/)
.venv/bin/python scripts/build_catalog.py

# Catalog-filtered 48Q (resume-safe)
bash scripts/run_catalog_48q.sh

# Catalog + title-peek 48Q
bash scripts/run_peek_48q.sh

# Hybrid ReAct agent 48Q / 61Q
bash scripts/run_agent_48q.sh
bash scripts/run_agent_eval61q.sh

# Agent + verify (experimental; regressed on 48Q — off by default)
bash scripts/run_agent_verify_48q.sh

# Agent hard-case probe
bash scripts/run_agent_probe.sh

# API (warm index)
uvicorn contract:app --port 8000

# Offline peek probe (no answer LLM)
.venv/bin/python scripts/probe_peek.py --catalog --ids dev-06-apartment-hard,dev-28-health-medium

# Or one-shot
.venv/bin/python rag_runner.py \
  --questions reference_questions.json \
  --retrieve rerank --top-k 20 --window 2 --candidate-n 100 \
  --route --route-n 80 --route-global-n 20 --route-context --route-preview-n 20 \
  --catalog --catalog-min-score 8.0 \
  --cite passages \
  --model moonshotai/Kimi-K3 --reasoning-effort low \
  --llm-timeout 300 \
  --out reports/rag_answers_routectx_kimik3_low_catalog_48q.jsonl
```

Judge:

```bash
.venv/bin/python run_eval.py \
  --answers reports/rag_answers_routectx_kimik3_low_catalog_48q.jsonl \
  --questions reference_questions.json \
  --corpus-judge \
  --judge-model moonshotai/Kimi-K3 --judge-reasoning-effort low \
  --label routectx-kimik3-low-catalog-48q \
  --out reports/stage2/rag_routectx_kimik3_low_catalog_48q_corpus_judge_eval.json
```

---

## Key artifacts

| Artifact | Path |
| --- | --- |
| This report | `reports/stage3/agentic_progress.md` |
| Catalog JSON | `artifacts/catalog.json` |
| Catalog 48Q answers | `reports/rag_answers_routectx_kimik3_low_catalog_48q.jsonl` |
| Catalog 48Q corpus judge | `reports/stage2/rag_routectx_kimik3_low_catalog_48q_corpus_judge_eval.json` |
| Decompose 48Q judge | `reports/stage2/rag_routectx_kimik3_low_decompose_48q_corpus_judge_eval.json` |
| Tight decompose probe7 | `reports/stage2/rag_probe7_decompose_tight_corpus_judge_eval.json` |
| Baseline corpus judge | `reports/stage2/rag_rerank_k20_w2_route80_20_routectx_kimik3_low_corpus_judge_eval.json` |
| Agent 48Q answers | `reports/rag_answers_routectx_kimik3_low_agent_48q.jsonl` |
| Agent 48Q corpus judge | `reports/stage2/rag_routectx_kimik3_low_agent_48q_corpus_judge_eval.json` |
| Agent 61Q answers | `reports/rag_answers_routectx_kimik3_low_agent_eval61q.jsonl` |
| Agent 61Q corpus judge | `reports/stage2/rag_routectx_kimik3_low_agent_eval61q_corpus_judge_eval.json` |
| Agent+verify 48Q answers | `reports/rag_answers_routectx_kimik3_low_agent_verify_48q.jsonl` |
| Agent+verify 48Q corpus judge | `reports/stage2/rag_routectx_kimik3_low_agent_verify_48q_corpus_judge_eval.json` |
| Agent+verify 61Q answers (partial) | `reports/rag_answers_routectx_kimik3_low_agent_verify_eval61q.jsonl` |
| Agent hard probe | `reports/rag_answers_agent_probe_hard.jsonl` |

---

## Next steps (ordered)

1. **Ship agent without verify** as the Stage 3 default (best measured stack).
2. If revisiting verify: only strip high-confidence inventions (phones/ages not in cites); never drop numeric/timing GT-bearing clauses.
3. **Richer outline** — per-file TOC + stronger `fetch_chunks` when titles sit outside ±window.
4. Tighten answer completeness on multi-part GT omissions without reintroducing overclaim.
5. Multi-turn `session_id` in `/ask` if needed for blind eval conversations.

---

## Bottom line

Best measured Stage 3 stack is **hybrid ReAct agent (no verify)** on routectx+catalog: Train-48Q **42/48 (87.5%)**, Eval-61Q **48/61 (78.7%)**. Gated verify/strip looked attractive for overclaims but **regressed Train-48Q to 33/48** by deleting grounded facts — left off by default.
