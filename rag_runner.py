"""
RAG runner (no citations): retrieve → generate from context → answers JSONL.

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
    python rag_runner.py --model deepseek-ai/DeepSeek-V4-Pro
    python rag_runner.py --retrieve rerank --top-k 20 --window 2 --candidate-n 100
    python run_eval.py --answers rag_answers.jsonl --no-citation-judge
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import litellm

from rag.bm25 import build_chunk_bm25, build_file_bm25
from rag.embed import DEFAULT_MODEL, Embedder
from rag.generate import SYSTEM_NO_CITE, build_context, build_messages
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import DEFAULT_RERANK_MODEL, Reranker
from rag.retrieve import (
    build_id_index,
    idxs_excluding_faqs,
    intersect_idxs,
    search_cascade,
    search_expanded,
    search_reranked,
    search_rrf,
)
from rag.route import DEFAULT_ROUTE_MODEL, idxs_for_domains, route_question


def resolve_model(model: str) -> tuple[str, dict]:
    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"
    return model, kwargs


def load_questions(
    path: Path,
    limit: int | None,
    ids: set[str] | None = None,
) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["questions"]
    if ids:
        want = ids
        data = [q for q in data if q.get("id") in want]
        missing = want - {q["id"] for q in data}
        if missing:
            raise SystemExit(f"Unknown question ids: {sorted(missing)}")
    if limit:
        data = data[:limit]
    return data


def render_prompt_preview(messages: list[dict]) -> str:
    parts = []
    for i, m in enumerate(messages):
        parts.append(f"{'=' * 60}\n[{i}] {m['role'].upper()}\n{'=' * 60}\n{m['content']}\n")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG runner (grounded, no citations)")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--system-prompt", default=SYSTEM_NO_CITE)
    ap.add_argument("--out", default="rag_answers.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument(
        "--retrieve",
        choices=("dense", "cascade", "rrf", "rerank"),
        default="dense",
        help="dense | cascade | rrf | rerank (dense candidates → cross-encoder)",
    )
    ap.add_argument("--file-top-n", type=int, default=20, help="Cascade: BM25 files")
    ap.add_argument(
        "--candidate-n",
        type=int,
        default=100,
        help="RRF/rerank: first-stage candidate count (rerank default 100)",
    )
    ap.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    ap.add_argument(
        "--route",
        action="store_true",
        help=(
            "Pre-route question to corpus domains (LLM). For rerank: dense "
            "route-n from those domains + route-global-n unique from the full "
            "pool (default 80+20). Other retrieve modes stay domain-only."
        ),
    )
    ap.add_argument(
        "--route-model",
        default=DEFAULT_ROUTE_MODEL,
        help="Model for domain router (smaller/cheaper OK)",
    )
    ap.add_argument(
        "--route-n",
        type=int,
        default=80,
        help="With --route + rerank: dense candidates from routed domains",
    )
    ap.add_argument(
        "--route-global-n",
        type=int,
        default=20,
        help="With --route + rerank: extra unique dense candidates from full pool",
    )
    ap.add_argument(
        "--include-faq",
        action="store_true",
        help="Allow retrieving from scraped */pages/faq.txt (excluded by default)",
    )
    ap.add_argument(
        "--ids",
        default=None,
        help="Comma-separated question ids to run (e.g. dev-13-car-easy,dev-24-dental-hard)",
    )
    ap.add_argument("--limit", type=int, help="Only run first N questions")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index)
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    embed_model = config.get("model", DEFAULT_MODEL)

    print(f"Loading index from {index_dir}…", flush=True)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    if len(vectors) != len(emb):
        raise SystemExit(f"Mismatch: {len(vectors)} vectors vs {len(emb)} embeddings")
    id_to_idx = build_id_index(vectors)
    print(f"  {len(vectors)} vectors · embed={embed_model}", flush=True)

    file_bm25 = None
    chunk_bm25 = None
    reranker = None
    if args.retrieve == "cascade":
        print("Building file-level BM25…", flush=True)
        file_bm25 = build_file_bm25(vectors)
        print(f"  {len(file_bm25.files)} files indexed", flush=True)
    elif args.retrieve == "rrf":
        print("Building chunk-level BM25…", flush=True)
        chunk_bm25 = build_chunk_bm25(vectors)
        print(f"  {chunk_bm25.n} chunks indexed", flush=True)
    elif args.retrieve == "rerank":
        print(f"Loading reranker {args.rerank_model}…", flush=True)
        reranker = Reranker(model_name=args.rerank_model)
        reranker._load()
        print("  ready", flush=True)

    id_filter = (
        {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    )
    questions = load_questions(Path(args.questions), args.limit, id_filter)
    if args.route:
        print(f"Domain router ON · model={args.route_model}", flush=True)
    exclude_faq = not args.include_faq
    non_faq_idxs = idxs_excluding_faqs(vectors) if exclude_faq else None
    if exclude_faq:
        print(
            f"FAQ exclusion ON · {len(non_faq_idxs)}/{len(vectors)} chunks eligible",
            flush=True,
        )
    print(f"Embedding {len(questions)} queries…", flush=True)
    embedder = Embedder(model_name=embed_model)
    q_emb = embedder.embed_queries([q["question"] for q in questions])

    llm_model, kwargs = resolve_model(args.model)

    def retrieve_one(qi: int, question: str):
        route_meta = None
        route_idxs = None  # domain filter for hybrid / domain-only modes
        # Global pool always applies FAQ exclusion when enabled.
        global_idxs = non_faq_idxs
        if args.route:
            route_meta = route_question(question, model=args.route_model)
            if route_meta.fallback_all or not route_meta.domains:
                print(
                    f"    route fallback (all domains): {route_meta.reason}",
                    flush=True,
                )
                route_idxs = None
            else:
                route_idxs = intersect_idxs(
                    idxs_for_domains(vectors, route_meta.domains),
                    non_faq_idxs,
                )
                if args.retrieve == "rerank":
                    print(
                        f"    route → {route_meta.domains} "
                        f"(hybrid {args.route_n}+{args.route_global_n} "
                        f"from {len(route_idxs)} domain chunks) · "
                        f"{route_meta.reason}",
                        flush=True,
                    )
                else:
                    print(
                        f"    route → {route_meta.domains} "
                        f"({len(route_idxs)} chunks) · {route_meta.reason}",
                        flush=True,
                    )

        # Non-rerank modes: restrict entirely to routed domains (old behavior).
        domain_only_idxs = (
            route_idxs if route_idxs is not None else global_idxs
        )

        if args.retrieve == "cascade":
            assert file_bm25 is not None
            result = search_cascade(
                question,
                q_emb[qi],
                emb,
                vectors,
                file_bm25,
                top_k=args.top_k,
                window=args.window,
                file_top_n=args.file_top_n,
                id_to_idx=id_to_idx,
                candidate_idxs=domain_only_idxs,
            )
            return result.expanded, result.mode, result.file_hits, route_meta
        if args.retrieve == "rrf":
            assert chunk_bm25 is not None
            result = search_rrf(
                question,
                q_emb[qi],
                emb,
                vectors,
                chunk_bm25,
                top_k=args.top_k,
                window=args.window,
                candidate_n=args.candidate_n,
                id_to_idx=id_to_idx,
                candidate_idxs=domain_only_idxs,
            )
            return result.expanded, result.mode, result.file_hits, route_meta
        if args.retrieve == "rerank":
            assert reranker is not None
            result = search_reranked(
                question,
                q_emb[qi],
                emb,
                vectors,
                reranker,
                candidate_n=args.candidate_n,
                top_k=args.top_k,
                window=args.window,
                id_to_idx=id_to_idx,
                candidate_idxs=global_idxs,
                route_idxs=route_idxs,
                route_n=args.route_n,
                route_global_n=args.route_global_n,
            )
            return result.expanded, result.mode, result.file_hits, route_meta
        expanded = search_expanded(
            q_emb[qi],
            emb,
            vectors,
            top_k=args.top_k,
            window=args.window,
            id_to_idx=id_to_idx,
            candidate_idxs=domain_only_idxs,
        )
        return expanded, "dense", [], route_meta

    approach = {
        "dense": "rag-no-cite",
        "cascade": "rag-cascade-no-cite",
        "rrf": "rag-rrf-no-cite",
        "rerank": "rag-rerank-no-cite",
    }[args.retrieve]

    if args.show_prompt:
        expanded, mode, _, route_meta = retrieve_one(0, questions[0]["question"])
        context = build_context(expanded)
        messages = build_messages(
            questions[0]["question"], context, system_prompt=args.system_prompt
        )
        print(render_prompt_preview(messages))
        print(
            f"\n(retrieve={mode}, candidates={args.candidate_n}, top_k={args.top_k}, "
            f"window=±{args.window}, {len(expanded)} hits, "
            f"route={None if route_meta is None else route_meta.domains}, "
            f"prompt chars≈{sum(len(m['content']) for m in messages)})"
        )
        return

    with open(args.out, "w", encoding="utf-8") as out:
        for qi, q in enumerate(questions):
            t0 = time.time()
            expanded, mode, file_hits, route_meta = retrieve_one(qi, q["question"])
            context = build_context(expanded)
            messages = build_messages(
                q["question"], context, system_prompt=args.system_prompt
            )
            t_retr = time.time()
            resp = litellm.completion(
                model=llm_model,
                messages=messages,
                timeout=120,
                **kwargs,
            )
            latency_ms = (time.time() - t0) * 1000
            answer = resp.choices[0].message.content or ""
            rec = {
                "id": q["id"],
                "answer": answer,
                "citations": [],
                "latency_ms": latency_ms,
                "tokens": {
                    "prompt": resp.usage.prompt_tokens,
                    "completion": resp.usage.completion_tokens,
                },
                "retrieval": {
                    "mode": mode,
                    "top_k": args.top_k,
                    "window": args.window,
                    "candidate_n": args.candidate_n
                    if args.retrieve in ("rrf", "rerank")
                    else None,
                    "rerank_model": args.rerank_model if args.retrieve == "rerank" else None,
                    "route_domains": None
                    if route_meta is None
                    else route_meta.domains,
                    "route_reason": None if route_meta is None else route_meta.reason,
                    "route_fallback_all": None
                    if route_meta is None
                    else route_meta.fallback_all,
                    "route_n": args.route_n
                    if args.route and args.retrieve == "rerank"
                    else None,
                    "route_global_n": args.route_global_n
                    if args.route and args.retrieve == "rerank"
                    else None,
                    "exclude_faq": exclude_faq,
                    "bm25_files": [
                        {"file": f, "score": round(s, 4)} for f, s in file_hits
                    ],
                    "hits": [
                        {
                            "rank": ex.rank,
                            "score": round(ex.score, 4),
                            "file": ex.match.location.file,
                            "page": ex.match.location.page,
                            "id": ex.match.id,
                            "domain": ex.match.location.domain,
                        }
                        for ex in expanded
                    ],
                },
                "approach": approach
                + (
                    f"+route{args.route_n}+{args.route_global_n}"
                    if args.route and args.retrieve == "rerank"
                    else ("+route" if args.route else "")
                )
                + ("" if args.include_faq else "+nofaq"),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            preview = answer.replace("\n", " ")[:70]
            top = expanded[0].score if expanded else 0.0
            print(
                f"  {q['id']:28s} {mode:8s} "
                f"retr={(t_retr - t0)*1000:.0f}ms  "
                f"total={latency_ms:.0f}ms  "
                f"top={top:.3f}  "
                f"{preview!r}…",
                flush=True,
            )

    print(
        f"\nwrote {args.out} — score with: "
        f"python run_eval.py --answers {args.out} --no-citation-judge"
    )


if __name__ == "__main__":
    main()
