# BM25 Optimization Summary And Rerank Integration Plan

Date: 2026-07-25

## Scope

This summary captures the BM25 optimization work completed so far, the metric used for tuning, the final parameter choices, and how BM25 was integrated into the full rerank flow.

No additional code execution was performed for this summary.

## Optimization Goal

Improve lexical retrieval quality before generation, specifically for exact document and page matching from reference questions.

## Evaluator And Target Metric

Standalone evaluator script:

- scripts/eval_bm25_retrieval.py

Question source:

- reference_questions.json

Primary optimization metric:

- Full doc+page hit@20

Supporting metrics tracked:

- Full doc-only hit@20
- Page-group recall@20
- Doc-group recall@20
- Any page-group hit@20

## BM25 Strategy That Was Tuned

The tuned scorer used:

1. Chunk-level BM25 ranking for lexical matches.
2. Light file-level BM25 prior to stabilize document selection.
3. Diversity constraints:
   - unique (file, page) references
   - per-file cap

This was tuned as a standalone retrieval module, independent from generator behavior.

## Baseline vs Tuned Result

Initial standalone BM25 setup (top-20 refs):

- Full doc+page hit@20: 45.8%

Tuned setup (top-20 refs):

- Full doc+page hit@20: 47.9%
- Full doc-only hit@20: 60.4%
- Page-group recall@20: 52.1%
- Doc-group recall@20: 63.5%
- Any page-group hit@20: 56.2%

Net gain on primary metric:

- +2.1 points on full doc+page hit@20

## Final Tuned BM25 Parameters

- alpha_file_prior = 0.15
- chunk_pool = 150
- max_per_file = 4
- file_top_n = 40
- evaluation depth = top_k 20

Interpretation:

- Low file prior worked better than higher prior values.
- Larger chunk pools did not show meaningful gains in this sweep.
- Per-file cap around 4 improved diversity without hurting relevance too much.

## Integration Into Full Rerank Flow

### Code Integration

Integrated files:

- rag/retrieve.py
- rag_runner.py

What was added:

1. Optional BM25 keyword candidate injection before cross-encoder reranking.
2. Injection is disabled by default to preserve existing behavior.
3. BM25 indexes are built only when injection is enabled.

New retrieval helper:

- bm25_keyword_candidates(...) in rag/retrieve.py

### New Runner Controls

Added CLI flags in rag_runner.py:

- --bm25-inject-n
- --bm25-chunk-pool
- --bm25-file-top-n
- --bm25-alpha-file-prior
- --bm25-max-per-file

### Recommended Conservative Mix

For routed rerank mode:

- route dense: 80
- global dense: 8
- BM25 injected: 12

Rationale:

- Keep dense routed backbone intact.
- Replace part of exploration budget with lexical rescue candidates.
- Maintain approximately stable reranker workload.

## Operational Notes

Current strong baseline (full eval run):

- relevance: 85.4%
- hallucination: 12.5%
- citation accuracy: 92.7%

Because full evaluation is expensive (about 2 hours on local machine), the intended rollout is:

1. run a 12-question mixed pilot,
2. compare against the current rerank baseline,
3. expand only if pilot shows non-regressive relevance and improved citation behavior.
