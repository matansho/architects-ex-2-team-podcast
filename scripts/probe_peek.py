#!/usr/bin/env python3
"""Offline probe: dense+CE retrieve → title peek → print interesting/fetched.

No answer LLM. Optional domain route (LLM) and catalog filter match the
catalog-48Q stack when --route / --catalog are set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.catalog import catalog_files_for_query, idxs_for_files, load_catalog
from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.peek import apply_title_peek, peek_summary, question_wants_peek
from rag.rerank import DEFAULT_RERANK_MODEL, Reranker
from rag.retrieve import (
    build_id_index,
    idxs_excluding_faqs,
    intersect_idxs,
    search,
    search_reranked,
)
from rag.route import idxs_for_domains, route_question


DEFAULT_IDS = [
    "dev-06-apartment-hard",
    "dev-17-car-hard",
    "dev-28-health-medium",
    "dev-47-travel-hard",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--catalog-path", default="artifacts/catalog.json")
    ap.add_argument("--catalog-min-score", type=float, default=8.0)
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--route-preview-n", type=int, default=20)
    ap.add_argument("--peek-radius", type=int, default=6)
    ap.add_argument("--peek-max-fetches", type=int, default=6)
    ap.add_argument("--peek-always", action="store_true")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--candidate-n", type=int, default=100)
    ap.add_argument("--route-n", type=int, default=80)
    ap.add_argument("--route-global-n", type=int, default=20)
    args = ap.parse_args()

    id_filter = {x.strip() for x in args.ids.split(",") if x.strip()}
    questions = [
        q
        for q in json.loads(Path(args.questions).read_text(encoding="utf-8"))
        if q["id"] in id_filter
    ]
    if not questions:
        raise SystemExit(f"No questions matched ids={sorted(id_filter)}")

    index_dir = Path(args.index)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    id_to_idx = build_id_index(vectors)
    non_faq = idxs_excluding_faqs(vectors)
    catalog = load_catalog(Path(args.catalog_path)) if args.catalog else None

    print(f"Loading reranker {DEFAULT_RERANK_MODEL}…", flush=True)
    reranker = Reranker(model_name=DEFAULT_RERANK_MODEL)
    reranker._load()
    embedder = Embedder()
    q_embs = embedder.embed_queries([q["question"] for q in questions])

    for qi, q in enumerate(questions):
        print(f"\n=== {q['id']} ===", flush=True)
        print(f"Q: {q['question'][:140]}", flush=True)
        print(f"wants_peek={question_wants_peek(q['question'])}", flush=True)

        route_idxs = None
        global_seed = None
        route_domains = None
        if args.route:
            preview = search(
                q_embs[qi],
                emb,
                vectors,
                top_k=args.route_preview_n,
                candidate_idxs=non_faq,
            )
            global_seed = preview
            route = route_question(q["question"], preview_hits=preview)
            if route.domains and not route.fallback_all:
                route_domains = route.domains
                route_idxs = intersect_idxs(
                    idxs_for_domains(vectors, route.domains), non_faq
                )
                print(f"route → {route.domains}", flush=True)
            else:
                print(f"route fallback · {route.reason}", flush=True)

        if catalog is not None:
            files, matched, _dbg = catalog_files_for_query(
                catalog,
                q["question"],
                domains=route_domains,
                min_score=args.catalog_min_score,
            )
            if files:
                cat_idxs = intersect_idxs(idxs_for_files(vectors, files), non_faq)
                if cat_idxs and len(cat_idxs) >= 10:
                    base = route_idxs if route_idxs is not None else non_faq
                    narrowed = intersect_idxs(cat_idxs, base)
                    if narrowed and len(narrowed) >= 10:
                        route_idxs = narrowed
                        print(
                            f"catalog: {[e.title[:40] for e in matched[:3]]} "
                            f"→ {len(route_idxs)} chunks",
                            flush=True,
                        )

        result = search_reranked(
            q["question"],
            q_embs[qi],
            emb,
            vectors,
            reranker,
            candidate_n=args.candidate_n,
            top_k=args.top_k,
            window=args.window,
            id_to_idx=id_to_idx,
            candidate_idxs=non_faq,
            route_idxs=route_idxs,
            route_n=args.route_n,
            route_global_n=args.route_global_n,
            global_seed=global_seed,
        )
        expanded = result.expanded
        expanded, peek_results = apply_title_peek(
            expanded,
            vectors,
            id_to_idx,
            q["question"],
            radius=args.peek_radius,
            max_fetches=args.peek_max_fetches,
            always_interesting=args.peek_always,
        )
        summary = peek_summary(peek_results)
        print(f"fetched={summary['n_fetched']}", flush=True)
        for a in summary["anchors"]:
            if not a["interesting"] and not a["fetched"]:
                continue
            print(f"  anchor {a['anchor_id']}", flush=True)
            for t in a["interesting"]:
                mark = (
                    "FETCH"
                    if t["id"] in a["fetched"]
                    else ("in-ctx" if t["in_context"] else "skip")
                )
                print(
                    f"    [{mark}] p{t['page']} {t['id']}: {t['title'][:70]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
