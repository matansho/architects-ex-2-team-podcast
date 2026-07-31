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
  └─ answer LLM + USED_PASSAGES citations
```

Nothing here is a free agent loop yet. Catalog + decompose are **deterministic / gated LLM helpers** that reshape retrieval before generation.

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

## Measured results (Train-48Q, corpus judge v4)

Same stack as baseline unless noted: rerank k20 ±2, candidate 100, route 80+20, routectx, no-FAQ, passage cite, Kimi-K3 low.

| Run | ok | gt_covered | fully_supported | Notes |
| --- | ---: | ---: | ---: | --- |
| **Baseline** routectx | **40/48 (83.3%)** | 85.4% | 97.9% | Reference |
| **+ decompose** (loose then tight) | **40/48 (83.3%)** | 85.4% | 97.9% | Flat overall; 7 splits on loose 48Q |
| **+ catalog** (`min_score=8`) | **41/48 (85.4%)** | **87.5%** | 97.9% | Activated on **25/48** questions |

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

---

## How to run

```bash
source scripts/activate.sh

# Rebuild catalog (writes data/catalog.json; shipped copy in artifacts/)
.venv/bin/python scripts/build_catalog.py

# Catalog-filtered 48Q (resume-safe)
bash scripts/run_catalog_48q.sh

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

---

## Next steps (ordered)

1. **Outline / section tools** — TOC per file + `get_section` / `expand` so carve-outs and long policies are addressable without stuffing the whole PDF into context.
2. **Combine** tight `--decompose` + `--catalog` on full 48Q (expect complementary wins on rival pairs).
3. **Single ReAct agent** with tools: `search_catalog`, `filtered_retrieve`, `expand_section`, `answer` — keep budgets explicit.
4. Expose behind `contract.py` `/ask` with latency/cost logging.

---

## Bottom line

We have a **corpus scheme (catalog) + filtered retrieval tool path** and a **tight rival-product decomposer**, sitting on the Stage 2 routectx RAG core. Catalog alone moves Train-48Q from **40 → 41 ok** and lifts gt_covered to **87.5%**, with clear product-name wins (Switch) and known failure modes when the filter fires on the wrong product cluster. Full agent loop and outline tools are the remaining Stage 3 work.
