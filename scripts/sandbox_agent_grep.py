#!/usr/bin/env python3
"""Offline sandbox: lexical grep (bm25/literal) vs seed-retrieve coverage.

No answer LLM. Loads the index, builds BM25 once, runs a few Hebrew queries
(and optionally question ids) and prints whether hits are already in the
dense seed context.

Usage:
  source scripts/activate.sh
  .venv/bin/python scripts/sandbox_agent_grep.py
  .venv/bin/python scripts/sandbox_agent_grep.py --ids dev-02-apartment-easy,dev-28-health-medium
  .venv/bin/python scripts/sandbox_agent_grep.py --query "התיישנות תביעה" --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.agent_tools import (  # noqa: E402
    AgentState,
    RetrievalBundle,
    ensure_chunk_bm25,
    seed_retrieve,
    tool_grep,
)
from rag.catalog import DEFAULT_CATALOG_PATH, load_catalog  # noqa: E402
from rag.embed import Embedder  # noqa: E402
from rag.index_store import load_embeddings, load_vectors_meta  # noqa: E402
from rag.rerank import Reranker  # noqa: E402
from rag.retrieve import build_id_index, idxs_excluding_faqs  # noqa: E402


DEFAULT_QUERIES = [
    ("התיישנות תביעה ביטוח", "limitation / statute"),
    ("תקופת המתנה", "waiting period"),
    ("חריגים לכיסוי", "exclusions"),
    ("גבולות אחריות", "liability caps"),
    ("03-9294000", "claims phone"),
]


def load_bundle(
    index_dir: Path,
    catalog_path: Path,
    *,
    need_seed: bool,
) -> RetrievalBundle:
    print(f"Loading index {index_dir} …", flush=True)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir) if need_seed else None
    id_to_idx = build_id_index(vectors)
    catalog = load_catalog(catalog_path) if need_seed else None
    embedder = Embedder() if need_seed else None
    reranker = None
    if need_seed:
        print("Loading reranker (needed for seed_retrieve) …", flush=True)
        reranker = Reranker()
        reranker._load()
    bundle = RetrievalBundle(
        vectors=vectors,
        emb=emb if emb is not None else __import__("numpy").zeros((0, 1)),
        id_to_idx=id_to_idx,
        embedder=embedder,
        reranker=reranker,
        catalog=catalog,
        non_faq_idxs=idxs_excluding_faqs(vectors),
        enable_grep=True,
        use_route=True,
        use_catalog=True,
    )
    t0 = time.perf_counter()
    ensure_chunk_bm25(bundle)
    print(f"BM25 ready in {(time.perf_counter() - t0):.1f}s · {len(vectors)} chunks", flush=True)
    return bundle


def show_grep(
    bundle: RetrievalBundle,
    state: AgentState,
    query: str,
    *,
    mode: str,
    top_n: int,
) -> None:
    obs = tool_grep(state, bundle, query=query, mode=mode, top_n=top_n)
    data = json.loads(obs)
    print(f"\n--- grep mode={mode} q={query!r} ms={data.get('grep_ms')} hits={data.get('n_hits')} ---")
    for h in data.get("hits") or []:
        flag = "IN" if h.get("in_context") else "new"
        print(
            f"  [{flag}] {h['score']:>7}  {h['chunk_id']}"
            f"  p={h.get('page')}  {h['file'][-60:]}"
        )
        snip = (h.get("snippet") or "")[:140]
        print(f"         {snip}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--ids", default="", help="Comma-separated question ids (seed+grep)")
    ap.add_argument("--query", action="append", default=[], help="Extra query (repeatable)")
    ap.add_argument("--mode", choices=("bm25", "literal", "both"), default="both")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--skip-seed", action="store_true", help="Skip dense seed (grep only)")
    args = ap.parse_args()

    need_seed = bool(args.ids) and not args.skip_seed
    bundle = load_bundle(Path(args.index), Path(args.catalog), need_seed=need_seed)
    modes = ["bm25", "literal"] if args.mode == "both" else [args.mode]

    queries: list[tuple[str, str]] = list(DEFAULT_QUERIES)
    for q in args.query:
        queries.append((q, "cli"))

    # Standalone queries (empty route state — full non-FAQ pool).
    if not args.ids:
        state = AgentState(question="(sandbox)")
        for q, note in queries:
            print(f"\n===== {note}: {q} =====")
            for m in modes:
                show_grep(bundle, state, q, mode=m, top_n=args.top_n)
        return

    qs = {
        r["id"]: r
        for r in json.loads(Path(args.questions).read_text(encoding="utf-8"))
    }
    for qid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        q = qs.get(qid)
        if not q:
            print(f"missing id {qid}", flush=True)
            continue
        print(f"\n########## {qid} ##########")
        print(q["question"][:200])
        if args.skip_seed:
            state = AgentState(question=q["question"])
        else:
            t0 = time.perf_counter()
            state = seed_retrieve(bundle, q["question"])
            print(
                f"seed: {len(state.passages)} passages · route={state.route_domains} · "
                f"{(time.perf_counter() - t0)*1000:.0f}ms",
                flush=True,
            )
        # Grep with the full question and a couple of keyword extracts.
        for probe_q in (q["question"], *([qq for qq, _ in DEFAULT_QUERIES[:2]])):
            for m in modes:
                show_grep(bundle, state, probe_q, mode=m, top_n=args.top_n)


if __name__ == "__main__":
    main()
