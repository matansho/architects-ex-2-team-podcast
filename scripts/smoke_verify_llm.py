#!/usr/bin/env python3
"""Smoke the LLM verifier on frozen champion answers (no agent regen).

Builds evidence from index chunks matching each answer's citations when possible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag.index_store import load_vectors
from rag.retrieve import ExpandedHit, normalize_cite_file
from rag.verify import DEFAULT_VERIFY_MODEL, verify_and_strip, verify_meta


def _load_questions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["questions"] if isinstance(data, dict) and "questions" in data else data
    return {r["id"]: r["question"] for r in rows if r.get("id") and r.get("question")}


def _passages_from_citations(vectors, citations: list[dict], *, limit: int = 6) -> list[ExpandedHit]:
    if not citations or vectors is None:
        return []
    want: list[tuple[str, int | None]] = []
    for c in citations[:limit]:
        f = normalize_cite_file(c.get("file") or "")
        p = c.get("page")
        if f:
            want.append((f, p if isinstance(p, int) else None))
    out: list[ExpandedHit] = []
    used: set[str] = set()
    for file, page in want:
        for v in vectors:
            vf = normalize_cite_file(v.location.file)
            if vf != file:
                continue
            pages = set(v.location.pages or [])
            if v.location.page is not None:
                pages.add(v.location.page)
            if page is not None and page not in pages and v.location.page != page:
                continue
            if v.id in used:
                continue
            used.add(v.id)
            out.append(
                ExpandedHit(rank=len(out) + 1, score=1.0, match=v, neighbors=[])
            )
            break
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--answers",
        type=Path,
        default=Path("reports/rag_answers_agent_v2_hybrid_topk25_48q.jsonl"),
    )
    ap.add_argument("--questions", type=Path, default=Path("reference_questions.json"))
    ap.add_argument(
        "--ids",
        default=(
            "dev-41-mortgage-hard,dev-43-travel-easy,dev-46-travel-medium,"
            "dev-06-apartment-hard,dev-02-apartment-easy,dev-45-travel-medium"
        ),
    )
    ap.add_argument("--model", default=os.environ.get("RAG_VERIFY_MODEL", DEFAULT_VERIFY_MODEL))
    ap.add_argument("--mode", choices=("extras", "strict"), default="extras")
    ap.add_argument("--index-dir", type=Path, default=Path("data/index"))
    ap.add_argument("--no-index", action="store_true", help="Skip evidence load")
    args = ap.parse_args()

    focus = {x.strip() for x in args.ids.split(",") if x.strip()}
    qmap = _load_questions(args.questions)
    vectors = None
    if not args.no_index and args.index_dir.exists():
        print(f"loading index {args.index_dir} …", flush=True)
        vectors = load_vectors(args.index_dir)
        print(f"  {len(vectors)} vectors", flush=True)

    by = {}
    with args.answers.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in focus:
                by[r["id"]] = r

    for qid in sorted(focus):
        rec = by.get(qid)
        if not rec:
            print(f"\n== {qid} MISSING")
            continue
        q = qmap.get(qid, "")
        a = rec.get("answer") or ""
        passages = _passages_from_citations(
            vectors, rec.get("citations") or []
        )
        print(f"\n== {qid}  evidence_passages={len(passages)}", flush=True)
        v = verify_and_strip(
            q,
            a,
            passages,
            passage_indices=list(range(1, len(passages) + 1)) or None,
            force=True,
            model=args.model,
            mode=args.mode,
            apply_deterministic=False,
            deterministic_only=False,
        )
        meta = verify_meta(v)
        print(
            f"  changed={meta['changed']} drop={meta.get('drop_ids')} "
            f"reverted={meta['reverted']} ({meta['latency_ms']:.0f}ms) "
            f"ask={meta.get('question_ask')!r}"
        )
        if meta["changed"]:
            print("  BEFORE:", a[:220].replace("\n", " | "))
            print("  AFTER: ", v.answer_after[:220].replace("\n", " | "))
            print("  STRIP: ", [s[:120].replace("\n", " ") for s in v.stripped])
        elif v.skipped:
            print("  skipped:", v.skip_reason)
        else:
            print("  (no change) keep_summary=", meta.get("keep_summary"))


if __name__ == "__main__":
    main()
