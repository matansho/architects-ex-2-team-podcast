#!/usr/bin/env python3
"""Standalone BM25 retrieval evaluator against reference_questions.json.

Evaluates whether top-k proposed references contain the correct ground-truth
document and page pointers (any_of groups) per question.

Example:
    python scripts/eval_bm25_retrieval.py --top-k 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.corpus import normalize_path  # noqa: E402
from rag.bm25 import build_chunk_bm25, build_file_bm25  # noqa: E402
from rag.index_store import load_vectors_meta  # noqa: E402
from rag.retrieve import Hit, groups_hit, hit_matches_source  # noqa: E402


def _load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or []
    return list(data)


def _group_hit_file_only(group: dict[str, Any], hits: list[Hit]) -> bool:
    for opt in group.get("any_of") or []:
        exp = normalize_path(opt.get("file") or "")
        if not exp:
            continue
        for h in hits:
            got = normalize_path(h.vector.location.file)
            if got == exp:
                return True
    return False


def _groups_hit_file_only(
    ground_truth_sources: list[dict[str, Any]],
    hits: list[Hit],
) -> tuple[int, int]:
    total = len(ground_truth_sources or [])
    sat = sum(1 for g in (ground_truth_sources or []) if _group_hit_file_only(g, hits))
    return sat, total


def _collect_match_flags(hit: Hit, groups: list[dict[str, Any]]) -> tuple[bool, bool]:
    loc = hit.vector.location
    page_match = any(
        hit_matches_source(loc, opt)
        for g in groups
        for opt in (g.get("any_of") or [])
    )
    doc_match = any(
        normalize_path(loc.file) == normalize_path((opt.get("file") or ""))
        for g in groups
        for opt in (g.get("any_of") or [])
    )
    return page_match, doc_match


def rank_bm25_refs(
    query: str,
    *,
    vectors,
    chunk_bm25,
    file_bm25,
    top_k: int,
    chunk_pool: int,
    file_top_n: int,
    alpha_file_prior: float,
    max_per_file: int,
) -> list[Hit]:
    """Optimized sparse ranking: chunk BM25 + file-level BM25 prior + diversity.

    Steps:
    1) score chunks by keyword overlap,
    2) boost chunks whose file is highly ranked by file-level BM25,
    3) keep unique (file, page) references and limit per-file domination.
    """
    chunk_hits = chunk_bm25.search(query, top_n=max(chunk_pool, top_k))
    if not chunk_hits:
        return []

    file_hits = file_bm25.search_files(query, top_n=file_top_n)
    file_score_map = {f: s for f, s in file_hits}

    top_chunk_score = max(s for _, s in chunk_hits) or 1.0
    top_file_score = max((s for _, s in file_hits), default=0.0)

    scored: list[tuple[int, float, float]] = []
    for idx, c_score in chunk_hits:
        vec = vectors[idx]
        f_score = file_score_map.get(vec.location.file, 0.0)

        c_norm = c_score / top_chunk_score
        f_norm = (f_score / top_file_score) if top_file_score > 0 else 0.0
        combined = (1.0 - alpha_file_prior) * c_norm + alpha_file_prior * f_norm
        scored.append((idx, combined, c_score))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    out: list[Hit] = []
    seen_ref: set[tuple[str, int | None]] = set()
    per_file: defaultdict[str, int] = defaultdict(int)

    for idx, combined, _raw in scored:
        vec = vectors[idx]
        f = vec.location.file
        p = vec.location.page
        key = (normalize_path(f), p)

        if key in seen_ref:
            continue
        if per_file[f] >= max_per_file:
            continue

        seen_ref.add(key)
        per_file[f] += 1
        out.append(Hit(rank=len(out) + 1, score=float(combined), vector=vec, role="match"))
        if len(out) >= top_k:
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone BM25 retrieval evaluator")
    ap.add_argument("--index", default="data/index", help="Index directory")
    ap.add_argument("--questions", default="reference_questions.json", help="Questions JSON")
    ap.add_argument("--top-k", type=int, default=20, help="Top references to evaluate")
    ap.add_argument("--chunk-pool", type=int, default=150, help="Initial chunk BM25 pool")
    ap.add_argument("--file-top-n", type=int, default=40, help="File BM25 prior depth")
    ap.add_argument(
        "--alpha-file-prior",
        type=float,
        default=0.15,
        help="Weight of file-level prior in [0,1]",
    )
    ap.add_argument(
        "--max-per-file",
        type=int,
        default=4,
        help="Max selected refs from a single file",
    )
    ap.add_argument(
        "--out",
        default="reports/stage2/bm25_retrieval_eval_top20.json",
        help="Write detailed JSON report",
    )
    args = ap.parse_args()

    if not (0.0 <= args.alpha_file_prior <= 1.0):
        raise SystemExit("--alpha-file-prior must be in [0,1]")

    index_dir = ROOT / args.index
    questions_path = ROOT / args.questions

    print("Loading vectors and building BM25 indexes...", flush=True)
    vectors = load_vectors_meta(index_dir)
    file_bm25 = build_file_bm25(vectors)
    chunk_bm25 = build_chunk_bm25(vectors)

    questions = _load_questions(questions_path)
    print(f"Loaded {len(questions)} questions", flush=True)

    n = len(questions)
    page_full = 0
    doc_full = 0
    page_group_sum = 0.0
    doc_group_sum = 0.0
    any_page_hit = 0
    any_doc_hit = 0
    by_domain: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0,
            "page_full": 0,
            "doc_full": 0,
            "page_group_sum": 0.0,
            "doc_group_sum": 0.0,
        }
    )

    rows: list[dict[str, Any]] = []

    for q in questions:
        qid = q.get("id", "")
        query = q.get("question", "")
        groups = q.get("ground_truth_sources") or []

        hits = rank_bm25_refs(
            query,
            vectors=vectors,
            chunk_bm25=chunk_bm25,
            file_bm25=file_bm25,
            top_k=args.top_k,
            chunk_pool=args.chunk_pool,
            file_top_n=args.file_top_n,
            alpha_file_prior=args.alpha_file_prior,
            max_per_file=args.max_per_file,
        )

        page_sat, page_total, page_details = groups_hit(groups, hits)
        doc_sat, doc_total = _groups_hit_file_only(groups, hits)

        page_frac = (page_sat / page_total) if page_total else 1.0
        doc_frac = (doc_sat / doc_total) if doc_total else 1.0
        page_ok = page_total > 0 and page_sat == page_total
        doc_ok = doc_total > 0 and doc_sat == doc_total

        has_page = page_sat > 0
        has_doc = doc_sat > 0

        page_full += int(page_ok)
        doc_full += int(doc_ok)
        page_group_sum += page_frac
        doc_group_sum += doc_frac
        any_page_hit += int(has_page)
        any_doc_hit += int(has_doc)

        domain = q.get("domain", "")
        d = by_domain[domain]
        d["n"] += 1
        d["page_full"] += int(page_ok)
        d["doc_full"] += int(doc_ok)
        d["page_group_sum"] += page_frac
        d["doc_group_sum"] += doc_frac

        top_refs = []
        for h in hits:
            page_match, doc_match = _collect_match_flags(h, groups)
            top_refs.append(
                {
                    "rank": h.rank,
                    "score": h.score,
                    "file": h.vector.location.file,
                    "page": h.vector.location.page,
                    "doc_match": doc_match,
                    "page_match": page_match,
                }
            )

        rows.append(
            {
                "id": qid,
                "domain": domain,
                "difficulty": q.get("difficulty", ""),
                "question": query,
                "page_groups_hit": page_sat,
                "page_groups_total": page_total,
                "doc_groups_hit": doc_sat,
                "doc_groups_total": doc_total,
                "page_group_recall": page_frac,
                "doc_group_recall": doc_frac,
                "full_page_hit": page_ok,
                "full_doc_hit": doc_ok,
                "group_details": page_details,
                "top_refs": top_refs,
            }
        )

        status = "PAGE" if page_ok else ("DOC" if doc_ok else "MISS")
        print(
            f"{qid:28s} {status:4s} page={page_sat}/{page_total} doc={doc_sat}/{doc_total}",
            flush=True,
        )

    summary_by_domain: dict[str, dict[str, float]] = {}
    for domain, d in sorted(by_domain.items()):
        dn = d["n"] or 1
        summary_by_domain[domain] = {
            "n": d["n"],
            "full_page_hit_rate": d["page_full"] / dn,
            "full_doc_hit_rate": d["doc_full"] / dn,
            "page_group_recall": d["page_group_sum"] / dn,
            "doc_group_recall": d["doc_group_sum"] / dn,
        }

    summary = {
        "n_questions": n,
        "top_k": args.top_k,
        "metrics": {
            "full_page_hit_rate": (page_full / n) if n else 0.0,
            "full_doc_hit_rate": (doc_full / n) if n else 0.0,
            "page_group_recall": (page_group_sum / n) if n else 0.0,
            "doc_group_recall": (doc_group_sum / n) if n else 0.0,
            "any_page_group_hit_rate": (any_page_hit / n) if n else 0.0,
            "any_doc_group_hit_rate": (any_doc_hit / n) if n else 0.0,
        },
        "config": {
            "index": args.index,
            "questions": args.questions,
            "chunk_pool": args.chunk_pool,
            "file_top_n": args.file_top_n,
            "alpha_file_prior": args.alpha_file_prior,
            "max_per_file": args.max_per_file,
        },
        "by_domain": summary_by_domain,
    }

    print("\n=== BM25 Retrieval Summary ===")
    print(f"Questions: {n}")
    print(f"Top-K references: {args.top_k}")
    print(f"Full doc+page hit@{args.top_k}: {summary['metrics']['full_page_hit_rate']:.1%}")
    print(f"Full doc-only hit@{args.top_k}: {summary['metrics']['full_doc_hit_rate']:.1%}")
    print(f"Page-group recall@{args.top_k}: {summary['metrics']['page_group_recall']:.1%}")
    print(f"Doc-group recall@{args.top_k}: {summary['metrics']['doc_group_recall']:.1%}")
    print(f"Any page-group hit@{args.top_k}: {summary['metrics']['any_page_group_hit_rate']:.1%}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
