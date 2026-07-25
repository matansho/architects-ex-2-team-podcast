# Stage 2 — Retrieval Pipeline (RAG Core)

Generation: `deepseek-ai/DeepSeek-V4-Pro` (Nebius Token Factory)  
Embed: `intfloat/multilingual-e5-base` · Rerank: `BAAI/bge-reranker-v2-m3`  
Table describe / domain route: `google/gemma-3-27b-it`

Stage 2: parse and chunk the Harel corpus, build a searchable index, retrieve grounded context, and answer the 48 dev questions without inventing policy facts. Index is on-disk numpy (no Qdrant yet).

Best end-to-end result: **85.4% relevance** / **91.7% citation accuracy** with passage-index citations (vs **43.8%** Stage 1; vs **83.3%** / **81.2%** retrieval-hit cite MVP).

## Reports

| Report | File | Contents |
| --- | --- | --- |
| Strategy comparison (Stage 1 + all RAG) | [stage1_prompt_strategy_comparison.html](reports/stage1/stage1_prompt_strategy_comparison.html) | Metrics, charts, per-question explorer |
| Docling explorer | [docling_explorer.html](reports/stage2/docling_explorer.html) | Parsed PDF/TXT samples — layout, sections, tables |
| PDF chunking | [chunking_report.html](reports/stage2/chunking_report.html) | Hierarchical / structural packing (~500 tok target) |
| TXT chunking | [txt_chunking_report.html](reports/stage2/txt_chunking_report.html) | Web-page cleaning and packs |
| Dense retrieval vs GT | [retrieval_eval.html](reports/stage2/retrieval_eval.html) | File/page hit rates @1/3/5/10 |
| Embedding space | [embedding_tsne.html](reports/stage2/embedding_tsne.html) | PCA / t-SNE of the index |

## Process

1. **Explored Docling** — sample PDFs/TXTs; structure and tables (`docling_explorer.html`).
2. **Designed chunking** — hierarchical PDF packs (~500 / 700 max) + cleaned TXT packs.
3. **Built the first index** — `scripts/embed_corpus.py` → `data/index/` (~20.5k vectors); per-PDF cache under `data/cache/pdf_chunks/`.
4. **Diagnosed retrieval** — dense vs GT sources (~58% file hit @10); embedding viz.
5. **First RAG loop** — dense top-k + neighbor window, Stage 1 **no-cite** prompt → answer judge (`--no-citation-judge`).
6. **Swept context size** — k ∈ {5,10,20}, window ±2 vs ±4. Best dense: **k20 ±2 (68.8%)**.
7. **Tried hybrids** — RRF underperformed dense; cascade implemented but not batch-eval’d.
8. **Added cross-encoder rerank** — dense 100 → `bge-reranker-v2-m3` → 20 ±2 → **70.8%** (hard **50%**).
9. **Fixed tables + dual representation** — full CSV; embed short description, return full table to the generator; LLM `embed_passage` on ~3.6k tables.
10. **Re-ran best RAG** — same rerank on LLM-table index → **79.2%**; refreshed comparison dashboard.
11. **Shipped table descriptions** — `artifacts/table_descriptions/` in git; `embed_corpus.py` applies them so teammates rebuild without Gemma.
12. **Table describe prompt v2** — aboutness + sketch inventory; fold notables into `embed_text`; re-described + re-embedded all table rows.
13. **Exclude FAQs at retrieve** — scraped `*/pages/faq.txt` are question titles with almost no answers; dropped by default (`--include-faq` to opt in).
14. **Domain router** — Gemma picks corpus domains before dense; pure domain filter → **81.2%** but some misroutes refused.
15. **Hybrid route pool** — dense **80 from routed domains + 20 unique from full corpus**, then CE → **83.3%**, **0% refusal**.
16. **Citation MVP (hits)** — structured `{file, page}` from top retrieval hits → **81.2%** cite acc.
17. **Passage-index citations (B)** — model ends with `USED_PASSAGES: 1, 3`; map labels → `{file, page}`; path-free answer body → **85.4%** rel / **91.7%** cite (48/48 parse ok).

## What we built

### Index pipeline

```
corpus/ (PDFs + pages/*.txt)
  → Docling parse (rag/parse.py) + hierarchical chunk (rag/chunk.py)
  → TXT clean/pack (rag/txt.py)
  → cache: data/cache/pdf_chunks/          (local, gitignored)
  → heuristic table dual-rep (rag/table_describe.py)
       embed_text = short NL description (search)
       text       = caption + full CSV (generator)
  → apply shipped LLM descriptions (rag/table_cache.py)
       artifacts/table_descriptions/       (3,612 files, git-tracked; prompt v2)
  → E5 embed → data/index/                 (local, gitignored)
```

Live index: **20,520** vectors (528 TXT + 19,992 PDF), dim 768, L2-normalized; **3,612** with table `embed_text` (3,607 LLM + 5 heuristic).

### Inference pipeline (`rag_runner.py`)

```
question
  → (optional) LLM domain route          [--route]
  → FAQ filter (default ON)              [--include-faq to disable]
  → dense candidates
       no route: top candidate_n (100)
       --route + rerank: hybrid route_n (80) + route_global_n (20 unique)
  → cross-encoder top_k (20)             [--retrieve rerank]
  → ±window neighbor expand              [--window 2]
  → grounded generation (path-free answer)
       default --cite passages: final line USED_PASSAGES: 1, 3
       fallback --cite hits: top unique hit (file, page)
  → map indices / hits → structured {file, page}
  → answers JSONL
```

### Retrieval (`rag/retrieve.py`, `rag/route.py`)

| Mode | Mechanism | Eval’d? |
| --- | --- | --- |
| `dense` | Cosine over E5; expand ±window neighbors by chunk id | Yes — k/window sweep |
| `cascade` | File BM25 → dense inside those files | Code only |
| `rrf` | Fuse dense + chunk-BM25 (RRF), then expand | Yes — no win vs dense |
| `rerank` | Dense candidates → CE on **`embed_text`** → top_k → expand | Yes — best path |
| `--route` | Gemma → domains; with rerank uses **80+20 hybrid** pool | Yes — best overall |

### Generation

`rag/generate.py` — Stage 1 **no-cite** system prompt; context from retrieved chunks (payload `text`). Runner still writes `citations: []` (see todos).

### Table dual representation

Insurance tariffs live in Docling tables. Embedding raw CSV matched poorly; truncating tables lost numbers like **59.12**.

| Field | Role |
| --- | --- |
| `embed_text` | Fluent passage for E5 + reranker (v2: aboutness + inventory/notables) |
| `text` | Full table body for the LLM answer |

Describe with a **sketch** (headers + sample rows + neighbors). Gemma returns JSON with `embed_passage` (+ axes / `notable_values` / `product_hints` for debugging).

**Reproducibility:** descriptions live in `artifacts/table_descriptions/` (one JSON per table chunk id). `scripts/embed_corpus.py` applies them by default (`--table-desc-cache`). `scripts/llm_describe_tables.py` reads/writes the same store (API only on cache misses; ~$0.30–1.50 for a full cold run). Optional local backup of pre-v2 files: `artifacts/table_descriptions_v1/` (not required in git).

## Results (48 questions, answer judge)

| Run | Relevance | Hallucination | Refusal | Avg latency |
| --- | --- | --- | --- | --- |
| Stage 1 few-shot no-cite (ref) | 43.8% | ~48–52% | ~4–8% | ~1.5–2.3 s |
| Dense k5 ±1 | 52.1% | 10.4% | 31.2% | 7.3 s |
| Dense k10 ±2 | 64.6% | 6.2% | 18.8% | 4.0 s |
| Dense k20 ±2 | 68.8% | 8.3% | 10.4% | 5.4 s |
| Dense k20 ±4 | 62.5% | 14.6% | 12.5% | 7.4 s |
| RRF k20 ±2 | 64.6% | 14.6% | 16.7% | 7.0 s |
| Rerank 100→20 ±2 | 70.8% | 12.5% | 10.4% | 18.6 s |
| Rerank + LLM tables | 79.2% | 10.4% | 4.2% | 14.4 s |
| Rerank + nofaq | 75.0% | 14.6% | 4.2% | 17.5 s |
| Rerank + route (domain-only) + nofaq | 81.2% | 12.5% | 6.2% | 22.1 s |
| Rerank + route80+20 + nofaq | 83.3% | 14.6% | 0.0% | 28.6 s |
| **Rerank + route80+20 + passage cite** | **85.4%** | **12.5%** | 2.1% | 16.0 s |

### Citations

| Run | Citation acc | fully / partial / none | Notes |
| --- | --- | --- | --- |
| route80+20 + hits MVP | 81.2% | 34 / 10 / 4 | Backfilled top-5 hit locations onto 83.3% answers |
| **route80+20 + passage cite (B)** | **91.7%** | **41 / 6 / 1** | Fresh gen; `USED_PASSAGES` → `{file,page}`; 48/48 parsed |

`--cite passages` (default): `SYSTEM_PASSAGE_CITE` with worked examples; parse footer; strip from answer; map 1-based `[N]` to hit locations. `--cite hits` keeps the MVP. Skip paths missing under `corpus/`. Never put file paths in answer prose (Stage 1 lesson).

Eval artifacts:
- hits MVP: `reports/rag_answers_rerank_k20_w2_route80_20_nofaq_cite.jsonl` + `_cite_eval.json`
- passage B: `reports/rag_answers_rerank_k20_w2_route80_20_passage_cite.jsonl` + `reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval.json`

### By difficulty (relevance)

| Run | Easy | Medium | Hard |
| --- | --- | --- | --- |
| Dense k20 ±2 | 93.8% | 81.2% | 31.2% |
| Rerank 100→20 ±2 | 87.5% | 75.0% | 50.0% |
| Rerank + LLM tables | 87.5% | **87.5%** | **62.5%** |
| **Rerank + route80+20 + nofaq** | 81% | **94%** | **75%** |

Hard questions are where retrieval quality matters most; tables + rerank moved hard from ~31% (dense) → 50% → 62.5%; hybrid routing pushed hard to **75%**.

### Offline dense retrieval (GT sources, window ±2)

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| File hit | 17% | 42% | 48% | 58% |
| Page group | 14% | 34% | 36% | 44% |

Answer quality can exceed file@10 because neighbor expand + generation tolerate near-misses — but missing the right tariff page still fails hard numbers.

## Main findings

1. **Retrieval beats prompt-only** — grounded RAG cut hallucination from ~50% (Stage 1) to ~10–15% and lifted relevance into the 50–83% range.
2. **More context isn’t always better** — k20 ±2 beat k20 ±4; extra neighbors diluted the signal.
3. **RRF didn’t help** — chunk BM25 fusion underperformed plain dense k20 on this corpus.
4. **Rerank lifts hard questions** — cross-encoder on the right string is worth the latency (~15–29 s/q).
5. **Tables need dual representation** — large Stage 2 jump (**70.8% → 79.2%**). Dense + CE must rank the **description**; the generator still needs the **full CSV**.
6. **FAQ pages are noise** — titles without answers; exclude at retrieve (default).
7. **Domain route alone is brittle** — misroutes (e.g. apartment→health/PA) caused refusals; **80 routed + 20 global unique** keeps domain focus while recovering cross-domain gold (0% refusal, **83.3%**).

### Flagship: `dev-30` (UPGRADE annual premium)

> אני בן 35 … UPGRADE משלים שב"ן. כמה יעלה לי הביטוח לשנה…  
> **GT:** 709.44 ₪ (= **59.12** × 12)

Earlier runs refused or retrieved the wrong sheet. With LLM table passages, the tariff chunk is rerank **#1** and the model answers correctly. Used as the debugging probe for the table bug — not special-cased in code.

**Heuristic `embed_text` (weak):**

> גילוי נאות לניתוחים upgrade … כותרת/שורה ראשונה: 0,1,2,3 … לדוגמה: 2735* טלפון…

**LLM `embed_passage` (searchable):**

> תוכנית UPGRADE מציעה כיסוי … עלות … משתנה בהתאם לגיל … עד גיל 20: **17.94 ₪**, גילאי 31–40: **59.12 ₪**, … מעל 66: **364.99 ₪**.

Other description examples:

- **מכלול** — פרמיה **₪ 350** ↔ גבול אחריות **₪ 40,000** (יחידות אירוח).
- **דרכון First Class** — גבולות אחריות / השתתפות עצמית in **$** by coverage type.

### Router misroutes rescued by 80+20

Pure domain filter refused `dev-04` (gun legal fees) and `dev-10` (clinical trial) after wrong routes. Hybrid CE pool pulled apartment/business gold into top hits; both scored relevant under route80+20.

## Todos / gaps

### Citations (follow-ups)

- [x] Hits MVP: `{file, page}` from retrieval (`citations_from_hits`)
- [x] Passage-index B: `USED_PASSAGES` → map to `{file, page}` (**91.7%** cite / **85.4%** rel)
- [x] Keep path-free answer body (no Hebrew `מקורות`)
- [x] Skip stale index paths missing on disk when emitting citations
- [ ] Note pypdf-vs-Docling risk on table pages (cite judge reads raw PDF text, not our CSV)

### Other open items

- [x] **Reproducibility (table descriptions)** — shipped in `artifacts/table_descriptions/`; `embed_corpus.py` / `llm_describe_tables.py` load them (no API needed to rebuild embeddings from cache)
- [x] **Lose FAQs (retrieve)** — excluded by default in `rag_runner.py`; optional: drop from index at build time
- [x] **Domain routing + hybrid pool** — `rag/route.py` + `hybrid_dense_candidates` (80+20)
- [x] **Citation MVP + passage indices** — `--cite hits|passages`
- [ ] **Cascade eval** — `--retrieve cascade` implemented; no scored JSONL yet
- [ ] **Qdrant** — client pinned; still on-disk numpy index (`rag/index_store.py`)
- [ ] **Tighten `looks_like_table`** — detector is loose (comma-heavy prose/forms count as tables); require stronger signals (numeric cells, Docling `TableItem`, etc.)
- [ ] **Re-describe the 5 heuristic leftovers** — those chunks embed a filename+first-line template (empty LLM `embed_passage`); optional LLM pass + re-embed (low impact; not real tariff grids)
- [ ] **Hierarchy into embeddings** — prepend / aggregate ancestor section titles (and maybe short parent context) into each chunk’s `embed_text` so dense search sees “§3 → 3.1 → clause” path, not only the leaf text; keep payload `text` as the local chunk for generation
- [ ] **Utilize TOC** — use PDF tables of contents for cleaner section hierarchy than OCR/number heuristics alone
- [ ] **FastAPI `/ask`** — Stage 3 (`contract.py` stub)

## Files

| Path | Role |
| --- | --- |
| `rag/parse.py` | Docling → blocks; full table CSV |
| `rag/chunk.py` / `rag/txt.py` | PDF hierarchical + TXT packs |
| `rag/corpus_chunks.py` | Corpus collect, PDF cache, table enrich |
| `rag/table_describe.py` | Sketch, LLM JSON (v2), embed_passage, repair |
| `rag/table_cache.py` | Load/save/apply shipped table descriptions |
| `artifacts/table_descriptions/` | Git-tracked LLM describe cache (~3.6k, prompt v2) |
| `rag/embed.py` | multilingual-e5-base |
| `rag/index_store.py` | On-disk vectors + embeddings.npy |
| `rag/retrieve.py` / `rag/bm25.py` / `rag/rerank.py` | dense / cascade / rrf / rerank + FAQ filter + hybrid route pool + `citations_from_hits` |
| `rag/route.py` | Domain router (Gemma JSON) |
| `rag/generate.py` | Grounded prompts; `USED_PASSAGES` parse + index→cite map |
| `rag_runner.py` | End-to-end retrieve → generate → structured citations (`--cite`) |
| `scripts/embed_corpus.py` | Full index build (+ apply shipped descriptions) |
| `scripts/llm_describe_tables.py` | Second-pass LLM describe + re-embed |
| `scripts/check_table_desc_quality.py` | Heuristic quality gate on describe cache |
| `scripts/eval_retrieval.py` | GT hit-rate HTML |
| `scripts/generate_*_report.py` / `regen_dashboard.sh` | Diagnostics + comparison HTML |
| `run_eval.py` | Answer (+ optional citation) judge |

## Reproduce

What is / isn’t in git:

| Path | In git? | Notes |
| --- | --- | --- |
| Code + `STAGE2.md` | yes | |
| `artifacts/table_descriptions/` | yes | LLM `embed_text` + payloads; no API needed to apply |
| `data/index/`, `data/cache/pdf_chunks/` | no | Rebuild locally (Docling + E5) |
| `.env` / API keys | no | Needed only to regenerate descriptions or run RAG/eval |

```bash
source scripts/activate.sh   # venv + .env (for RAG / eval; not for applying shipped descriptions)

# Full re-index (Docling; slow). Applies artifacts/table_descriptions/ by default.
python scripts/embed_corpus.py

# Optional: regenerate descriptions via API, then re-embed table rows
python scripts/llm_describe_tables.py --workers 8

# Best RAG config (hybrid route + FAQ exclusion + passage-index cites)
python rag_runner.py --retrieve rerank --candidate-n 100 --top-k 20 --window 2 \
  --route --route-n 80 --route-global-n 20 --cite passages \
  --out reports/rag_answers_rerank_k20_w2_route80_20_passage_cite.jsonl

# Eval with answer + citation judges
python run_eval.py --answers reports/rag_answers_rerank_k20_w2_route80_20_passage_cite.jsonl \
  --out reports/stage2/rag_rerank_k20_w2_route80_20_passage_cite_eval.json

# Dashboard (Stage 1 + RAG)
bash scripts/regen_dashboard.sh
```

## Implications for Stage 3

- Citations are wired for `/ask` (`--cite passages` preferred); Stage 3 still needs FastAPI wiring.
- Optionally move the index to Qdrant (or similar) behind the same retrieve API.
- Expose `/ask` via the course contract; keep path-free answer text; put `{file,page}` in the structured field.
- Best inference knobs: `--retrieve rerank --route --route-n 80 --route-global-n 20 --cite passages` (FAQ exclusion on).
