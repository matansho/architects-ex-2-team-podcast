# Stage 2 — Retrieval Pipeline (RAG Core)

Generation: `deepseek-ai/DeepSeek-V4-Pro` (Nebius Token Factory)  
Embed: `intfloat/multilingual-e5-base` · Rerank: `BAAI/bge-reranker-v2-m3`  
Table describe: `google/gemma-3-27b-it`

Stage 2 deliverables: parse and chunk the Harel corpus, build a searchable index, retrieve grounded context, and answer the 48 dev questions without inventing policy facts. We stayed on an on-disk numpy index (no Qdrant yet) and deferred structured citations to Stage 3.

Best end-to-end result: **79.2% relevance** (vs **43.8%** Stage 1 no-cite, no retrieval).

## Reports


| Report                                  | File                                                                                                             | Contents                                                                   |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Strategy comparison (Stage 1 + all RAG) | `[reports/stage1/stage1_prompt_strategy_comparison.html](reports/stage1/stage1_prompt_strategy_comparison.html)` | Metrics, charts, per-question explorer across prompt runs and RAG variants |
| Docling explorer                        | `[reports/stage2/docling_explorer.html](reports/stage2/docling_explorer.html)`                                   | Parsed PDF/TXT samples — layout, sections, tables                          |
| PDF chunking                            | `[reports/stage2/chunking_report.html](reports/stage2/chunking_report.html)`                                     | Hierarchical / structural packing (~500 tok target)                        |
| TXT chunking                            | `[reports/stage2/txt_chunking_report.html](reports/stage2/txt_chunking_report.html)`                             | Web-page cleaning and packs                                                |
| Dense retrieval vs GT                   | `[reports/stage2/retrieval_eval.html](reports/stage2/retrieval_eval.html)`                                       | File/page hit rates @1/3/5/10                                              |
| Embedding space                         | `[reports/stage2/embedding_tsne.html](reports/stage2/embedding_tsne.html)`                                       | PCA / t-SNE of the index                                                   |


## Process

1. **Explored Docling** — converted sample PDFs/TXTs, inspected structure and tables (`docling_explorer.html`).
2. **Designed chunking** — hierarchical PDF packs (~500 target / 700 max tokens) plus cleaned TXT packs; reported both.
3. **Built the first index** — `scripts/embed_corpus.py`: Docling parse → chunk → E5 embed → `data/index/` (~20.5k vectors). Per-PDF cache under `data/cache/pdf_chunks/`.
4. **Diagnosed retrieval** — dense hit rates vs ground-truth sources (~58% file hit @10); embedding visualization.
5. **First RAG loop** — dense top-k + neighbor window, Stage 1 **no-cite** grounded prompt → generate → answer judge (`--no-citation-judge`).
6. **Swept context size** — k ∈ {5,10,20}, window ±2 vs ±4. Best dense: **k20 ±2 (68.8%)**; wider window added noise.
7. **Tried hybrids** — RRF (dense + chunk BM25) underperformed dense; cascade (file BM25 → dense) implemented but not batch-eval’d.
8. **Added cross-encoder rerank** — dense 100 → `bge-reranker-v2-m3` → 20 ±2 → **70.8%**, hard questions **50%**.
9. **Fixed tables + dual representation** — full CSV extraction; embed a short description, return full table to the generator; LLM `embed_passage` from sketches on ~3.6k tables.
10. **Re-ran best RAG** — same rerank config on the LLM-table index → **79.2%**; regenerated the comparison dashboard.

## What we built

### Index pipeline

```
corpus/ (PDFs + pages/*.txt)
  → Docling parse (rag/parse.py) + hierarchical chunk (rag/chunk.py)
  → TXT clean/pack (rag/txt.py)
  → cache: data/cache/pdf_chunks/
  → table dual-rep (rag/table_describe.py)
       embed_text = short NL description (search)
       text       = caption + full CSV (generator)
  → apply shipped LLM descriptions: artifacts/table_descriptions/ (3,612, git-tracked)
  → E5 embed → data/index/{config.json, vectors.jsonl, embeddings.npy}
```

Live index: **20,520** vectors (528 TXT + 19,992 PDF), dim 768, L2-normalized; **3,612** with table `embed_text` (3,607 LLM + 5 heuristic).

### Retrieval (`rag/retrieve.py`, `rag_runner.py --retrieve`)


| Mode      | Mechanism                                                                    | Eval’d?               |
| --------- | ---------------------------------------------------------------------------- | --------------------- |
| `dense`   | Cosine over E5; expand ±window neighbors by chunk id                         | Yes — k/window sweep  |
| `cascade` | File BM25 → dense inside those files                                         | Code only             |
| `rrf`     | Fuse dense + chunk-BM25 (RRF), then expand                                   | Yes — no win vs dense |
| `rerank`  | Dense top-`candidate_n` → cross-encoder on `**embed_text**` → top_k → expand | Yes — best path       |


### Generation

`rag/generate.py` — Stage 1 **no-cite** system prompt; context from retrieved chunks (payload `text`). Runner still writes `citations: []` (citation wiring left for Stage 3).

### Table dual representation (late Stage 2)

Insurance tariffs live in Docling tables. Embedding raw CSV matched poorly; truncating tables lost numbers like **59.12**.


| Field        | Role                                   |
| ------------ | -------------------------------------- |
| `embed_text` | 2–3 fluent sentences for E5 + reranker |
| `text`       | Full table body for the LLM answer     |


Describe with a **sketch** (headers + sample rows + neighbors), not the whole grid. Gemma returns JSON with `embed_passage` (+ axes / `notable_values` / `product_hints` for debugging). Descriptions are shipped under `artifacts/table_descriptions/` and applied at embed time (`--table-desc-cache`). `scripts/llm_describe_tables.py` can regenerate/patch an existing index (~$0.30–1.50 one-time if cache is empty).

## Results (48 questions, answer judge)


| Run                            | Relevance | Hallucination | Refusal  | Avg latency |
| ------------------------------ | --------- | ------------- | -------- | ----------- |
| Stage 1 few-shot no-cite (ref) | 43.8%     | ~48–52%       | ~4–8%    | ~1.5–2.3 s  |
| Dense k5 ±1                    | 52.1%     | 10.4%         | 31.2%    | 7.3 s       |
| Dense k10 ±2                   | 64.6%     | 6.2%          | 18.8%    | 4.0 s       |
| Dense k20 ±2                   | 68.8%     | 8.3%          | 10.4%    | 5.4 s       |
| Dense k20 ±4                   | 62.5%     | 14.6%         | 12.5%    | 7.4 s       |
| RRF k20 ±2                     | 64.6%     | 14.6%         | 16.7%    | 7.0 s       |
| Rerank 100→20 ±2               | 70.8%     | 12.5%         | 10.4%    | 18.6 s      |
| **Rerank + LLM tables**        | **79.2%** | **10.4%**     | **4.2%** | 14.4 s      |


Citation accuracy is 0% — expected (`citations: []`, `--no-citation-judge`).

### By difficulty (relevance)


| Run                     | Easy  | Medium    | Hard      |
| ----------------------- | ----- | --------- | --------- |
| Dense k20 ±2            | 93.8% | 81.2%     | 31.2%     |
| Rerank 100→20 ±2        | 87.5% | 75.0%     | 50.0%     |
| **Rerank + LLM tables** | 87.5% | **87.5%** | **62.5%** |


Hard questions are where retrieval quality matters most; tables + rerank moved hard from ~31% (dense) → 50% → **62.5%**.

### Offline dense retrieval (GT sources, window ±2)


| Metric     | @1  | @3  | @5  | @10 |
| ---------- | --- | --- | --- | --- |
| File hit   | 17% | 42% | 48% | 58% |
| Page group | 14% | 34% | 36% | 44% |


Answer quality can exceed file@10 because neighbor expand + generation tolerate near-misses — but missing the right tariff page still fails hard numbers.

## Main findings

1. **Retrieval beats prompt-only** — grounded RAG cut hallucination from ~50% (Stage 1) to ~10% and lifted relevance into the 50–80% range.
2. **More context isn’t always better** — k20 ±2 beat k20 ±4; extra neighbors diluted the signal.
3. **RRF didn’t help** — chunk BM25 fusion underperformed plain dense k20 on this corpus.
4. **Rerank lifts hard questions** — cross-encoder on the right string is worth the latency (~15–19 s/q).
5. **Tables need dual representation** — largest single jump in Stage 2 (**70.8% → 79.2%**). Dense + CE must rank the **description**; the generator still needs the **full CSV**.

### Flagship: `dev-30` (UPGRADE annual premium)

> אני בן 35 … UPGRADE משלים שב"ן. כמה יעלה לי הביטוח לשנה…  
> **GT:** 709.44 ₪ (= **59.12** × 12)

Earlier runs refused or cited the wrong sheet. With LLM table passages, the tariff chunk is rerank **#1** and the model answers correctly.

**Heuristic `embed_text` (weak):**

> גילוי נאות לניתוחים upgrade … כותרת/שורה ראשונה: 0,1,2,3 … לדוגמה: 2735* טלפון…

**LLM `embed_passage` (searchable):**

> תוכנית UPGRADE מציעה כיסוי … עלות … משתנה בהתאם לגיל … עד גיל 20: **17.94 ₪**, גילאי 31–40: **59.12 ₪**, … מעל 66: **364.99 ₪**.

Other description examples:

- **מכלול** — פרמיה **₪ 350** ↔ גבול אחריות **₪ 40,000** (יחידות אירוח).
- **דרכון First Class** — גבולות אחריות / השתתפות עצמית in **$** by coverage type.

## Todos / gaps

### Next: citations + citation judge

- [ ] Emit real `{file, page}` in `rag_runner.py` (today: `citations: []`)
- [ ] Keep **no-cite** answer prompt (don’t reintroduce Hebrew `מקורות` — Stage 1 showed that hurts relevance)
- [ ] MVP: fill citations from retrieved hit locations (dedupe top‑N unique `(file, page)`; TXT → `page: null`) — not free-form model paths
- [ ] Re-run best RAG config → `run_eval.py` **without** `--no-citation-judge`
- [ ] Check relevance/hallucination stay near 79.2% / 10.4%; note pypdf-vs-Docling risk on table pages (judge reads raw PDF text, not our CSV)
- [ ] If cite score is weak while answers stay strong: hybrid (model returns passage indices `[1],[3]` → map to `{file, page}`)

### Other open items

- [x] **Reproducibility (table descriptions)** — shipped in `artifacts/table_descriptions/`; `embed_corpus.py` / `llm_describe_tables.py` load them (no API needed to rebuild embeddings from cache)
- [ ] **Cascade eval** — `--retrieve cascade` implemented; no scored JSONL yet
- [ ] **Qdrant** — client pinned; still on-disk numpy index (`rag/index_store.py`)
- [ ] **Tighten `looks_like_table`** — detector is loose (comma-heavy prose/forms count as tables); require stronger signals (numeric cells, Docling `TableItem`, etc.)
- [ ] **Re-describe the 5 heuristic leftovers** — those chunks embed a filename+first-line template (empty LLM `embed_passage`); optional LLM pass + re-embed (low impact; not real tariff grids)
- [ ] **Hierarchy into embeddings** — prepend / aggregate ancestor section titles (and maybe short parent context) into each chunk’s `embed_text` so dense search sees “§3 → 3.1 → clause” path, not only the leaf text; keep payload `text` as the local chunk for generation
- [ ] **Lose FAQs** — don’t index TXT pages that are only question lists with no answers
- [ ] **Utilize TOC** — use PDF tables of contents for cleaner section hierarchy than OCR/number heuristics alone
- [ ] **FastAPI `/ask`** — Stage 3 (`contract.py` stub)

## Files


| Path                                                  | Role                                    |
| ----------------------------------------------------- | --------------------------------------- |
| `rag/parse.py`                                        | Docling → blocks; full table CSV        |
| `rag/chunk.py` / `rag/txt.py`                         | PDF hierarchical + TXT packs            |
| `rag/corpus_chunks.py`                                | Corpus collect, PDF cache, table enrich |
| `rag/table_describe.py`                               | Sketch, LLM JSON, embed_passage, repair |
| `rag/table_cache.py`                                  | Load/save/apply shipped table descriptions |
| `artifacts/table_descriptions/`                       | Git-tracked LLM describe cache (~3.6k) |
| `rag/embed.py`                                        | multilingual-e5-base                    |
| `rag/index_store.py`                                  | On-disk vectors + embeddings.npy        |
| `rag/retrieve.py` / `rag/bm25.py` / `rag/rerank.py`   | dense / cascade / rrf / rerank          |
| `rag/generate.py`                                     | Grounded no-cite prompts                |
| `rag_runner.py`                                       | End-to-end retrieve → generate          |
| `scripts/embed_corpus.py`                             | Full index build                        |
| `scripts/llm_describe_tables.py`                      | Second-pass LLM describe + re-embed     |
| `scripts/eval_retrieval.py`                           | GT hit-rate HTML                        |
| `scripts/generate_*_report.py` / `regen_dashboard.sh` | Diagnostics + comparison HTML           |
| `run_eval.py`                                         | Answer (+ optional citation) judge      |


## Reproduce

```bash
source scripts/activate.sh   # venv + .env

# Full re-index (Docling; slow). Applies shipped table descriptions by default
# from artifacts/table_descriptions/ — no Gemma call needed.
python scripts/embed_corpus.py

# Optional: regenerate / patch descriptions via API, then re-embed table rows
python scripts/llm_describe_tables.py --workers 8

# Best RAG config
python rag_runner.py --retrieve rerank --candidate-n 100 --top-k 20 --window 2 \
  --out rag_answers_rerank_k20_w2_llm_tables.jsonl

# Eval
python run_eval.py --answers rag_answers_rerank_k20_w2_llm_tables.jsonl \
  --no-citation-judge --out reports/stage2/rag_rerank_k20_w2_llm_tables_eval.json

# Dashboard (Stage 1 + RAG)
bash scripts/regen_dashboard.sh
```

## Implications for Stage 3

- Citations + judge are now a near-term Stage 2 todo (see above); Stage 3 still needs `/ask` wiring.
- Optionally move the index to Qdrant (or similar) behind the same retrieve API.
- Expose `/ask` via the course contract; keep the no-cite answer style unless the API requires citations in text.

