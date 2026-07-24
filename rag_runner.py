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
    search_cascade,
    search_expanded,
    search_reranked,
    search_rrf,
)


def resolve_model(model: str) -> tuple[str, dict]:
    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"
    return model, kwargs


def load_questions(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["questions"]
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

    questions = load_questions(Path(args.questions), args.limit)
    print(f"Embedding {len(questions)} queries…", flush=True)
    embedder = Embedder(model_name=embed_model)
    q_emb = embedder.embed_queries([q["question"] for q in questions])

    llm_model, kwargs = resolve_model(args.model)

    def retrieve_one(qi: int, question: str):
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
            )
            return result.expanded, result.mode, result.file_hits
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
            )
            return result.expanded, result.mode, result.file_hits
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
            )
            return result.expanded, result.mode, result.file_hits
        expanded = search_expanded(
            q_emb[qi],
            emb,
            vectors,
            top_k=args.top_k,
            window=args.window,
            id_to_idx=id_to_idx,
        )
        return expanded, "dense", []

    approach = {
        "dense": "rag-no-cite",
        "cascade": "rag-cascade-no-cite",
        "rrf": "rag-rrf-no-cite",
        "rerank": "rag-rerank-no-cite",
    }[args.retrieve]

    if args.show_prompt:
        expanded, mode, _ = retrieve_one(0, questions[0]["question"])
        context = build_context(expanded)
        messages = build_messages(
            questions[0]["question"], context, system_prompt=args.system_prompt
        )
        print(render_prompt_preview(messages))
        print(
            f"\n(retrieve={mode}, candidates={args.candidate_n}, top_k={args.top_k}, "
            f"window=±{args.window}, {len(expanded)} hits, "
            f"prompt chars≈{sum(len(m['content']) for m in messages)})"
        )
        return

    with open(args.out, "w", encoding="utf-8") as out:
        for qi, q in enumerate(questions):
            t0 = time.time()
            expanded, mode, file_hits = retrieve_one(qi, q["question"])
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
                        }
                        for ex in expanded
                    ],
                },
                "approach": approach,
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
