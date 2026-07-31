"""
RAG runner: retrieve → generate from context → answers JSONL.

Citations (default: passage indices):
  Model ends with USED_PASSAGES: 1, 3 — mapped to {file, page}.
  Fallback --cite hits: top retrieval locations (no model indices).

Answer body stays path-free (no מקורות section).

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
    python rag_runner.py --retrieve rerank --route --top-k 20 --window 2
    python run_eval.py --answers rag_answers.jsonl
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
from rag.generate import (
    SYSTEM_NO_CITE,
    SYSTEM_PASSAGE_CITE,
    build_context,
    build_messages,
    citations_from_passage_indices,
    parse_answer_with_passages,
)
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import DEFAULT_RERANK_MODEL, Reranker
from rag.retrieve import (
    build_id_index,
    citations_from_hits,
    idxs_excluding_faqs,
    intersect_idxs,
    search,
    search_cascade,
    search_expanded,
    search_reranked,
    search_rrf,
)
from rag.catalog import (
    DEFAULT_CATALOG_PATH,
    catalog_files_for_query,
    idxs_for_files,
    load_catalog,
)
from rag.decompose import (
    DEFAULT_DECOMPOSE_MODEL,
    MAX_SUBQUESTIONS,
    decompose_question,
    merge_hits,
    split_budget,
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
    ap = argparse.ArgumentParser(
        description="RAG runner (grounded answers; structured citations)"
    )
    ap.add_argument(
        "--cite",
        choices=("passages", "hits"),
        default="passages",
        help=(
            "passages: model emits USED_PASSAGES indices → {file,page} "
            "(default). hits: top retrieval locations (MVP)."
        ),
    )
    ap.add_argument(
        "--max-citations",
        type=int,
        default=5,
        help="Max unique (file, page) citations",
    )
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default=None,
        help=(
            "Pass reasoning_effort to the answer LLM (Kimi-K3: low/high/max; "
            "default max on provider). Omit to leave provider default."
        ),
    )
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Pass thinking={type:disabled} to the answer LLM when supported "
            "(Nebius Kimi may accept this; official K3 prefers --reasoning-effort)."
        ),
    )
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
        "--route-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --route: dense-preview top-N snippets for the router, then "
            "reuse that preview as the hybrid global leg (default: on)"
        ),
    )
    ap.add_argument(
        "--route-preview-n",
        type=int,
        default=20,
        help="Dense preview size for --route-context (also reused as global seed)",
    )
    ap.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Split multi-subject questions into single-subject sub-questions "
            "(LLM) and retrieve for each. top-k and candidate-n are split "
            "across sub-questions, so context size and cross-encoder work stay "
            "constant. Single-subject questions are unaffected."
        ),
    )
    ap.add_argument(
        "--decompose-model",
        default=DEFAULT_DECOMPOSE_MODEL,
        help="Model for query decomposition (smaller/cheaper OK)",
    )
    ap.add_argument(
        "--decompose-max",
        type=int,
        default=MAX_SUBQUESTIONS,
        help="Maximum sub-questions per question",
    )
    ap.add_argument(
        "--catalog",
        action="store_true",
        help=(
            "Use data/catalog.json to restrict the hybrid dense pool to matched "
            "product pages (+ related PDFs) when the catalog score is strong "
            "enough; otherwise keep domain routing."
        ),
    )
    ap.add_argument(
        "--catalog-path",
        default=str(DEFAULT_CATALOG_PATH),
        help="Catalog JSON path (default data/catalog.json)",
    )
    ap.add_argument(
        "--catalog-min-score",
        type=float,
        default=6.0,
        help="Minimum catalog search score to activate file filtering",
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip question ids already present in --out and append new answers. "
            "Use after a killed/stalled run so completed work is not lost."
        ),
    )
    ap.add_argument(
        "--llm-timeout",
        type=float,
        default=90.0,
        help="Hard per-attempt timeout (seconds) for the answer LLM (default 90)",
    )
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
    out_mode = "w"
    if args.resume and Path(args.out).exists():
        done: set[str] = set()
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        before = len(questions)
        questions = [q for q in questions if q["id"] not in done]
        out_mode = "a"
        print(
            f"Resume: skipping {before - len(questions)} done, "
            f"{len(questions)} remaining → append {args.out}",
            flush=True,
        )
        if not questions:
            print("Nothing left to run.", flush=True)
            return
    # Cap litellm's module-level default (6000s) so a hung socket cannot sit
    # for hours even if a call site forgets to pass timeout=.
    litellm.request_timeout = float(args.llm_timeout)
    if args.route:
        ctx = (
            f" · context preview={args.route_preview_n} (reused as global seed)"
            if args.route_context
            else " · no context preview"
        )
        print(f"Domain router ON · model={args.route_model}{ctx}", flush=True)
    catalog = None
    if args.catalog:
        catalog = load_catalog(Path(args.catalog_path))
        print(
            f"Catalog ON · {catalog.n_entries} files · "
            f"min_score={args.catalog_min_score} · {args.catalog_path}",
            flush=True,
        )
    exclude_faq = not args.include_faq
    non_faq_idxs = idxs_excluding_faqs(vectors) if exclude_faq else None
    if exclude_faq:
        print(
            f"FAQ exclusion ON · {len(non_faq_idxs)}/{len(vectors)} chunks eligible",
            flush=True,
        )
    if args.system_prompt == SYSTEM_NO_CITE and args.cite == "passages":
        args.system_prompt = SYSTEM_PASSAGE_CITE
    print(f"Cite mode: {args.cite}", flush=True)
    print(f"Embedding {len(questions)} queries…", flush=True)
    embedder = Embedder(model_name=embed_model)
    q_emb = embedder.embed_queries([q["question"] for q in questions])

    llm_model, kwargs = resolve_model(args.model)
    llm_extra: dict = {}
    extra_body: dict = {}
    if args.reasoning_effort:
        # Nebius OpenAI-compat: litellm rejects top-level reasoning_effort.
        extra_body["reasoning_effort"] = args.reasoning_effort
    if args.no_thinking:
        extra_body["thinking"] = {"type": "disabled"}
    if extra_body:
        llm_extra["extra_body"] = extra_body
        print(f"Answer LLM extras: {extra_body}", flush=True)

    def retrieve_one(qi: int, question: str):
        route_meta = None
        route_idxs = None  # domain filter for hybrid / domain-only modes
        global_seed = None  # dense preview reused as hybrid global leg
        timings: dict[str, float] = {}
        decomp = None
        catalog_info: dict | None = None
        if args.decompose:
            t_dec = time.perf_counter()
            decomp = decompose_question(
                question,
                model=args.decompose_model,
                max_n=args.decompose_max,
            )
            timings["decompose_ms"] = round(
                (time.perf_counter() - t_dec) * 1000, 1
            )
            if decomp.failed:
                print(f"    decompose failed: {decomp.reason}", flush=True)
            elif decomp.split:
                print(
                    f"    decompose → {len(decomp.subquestions)} subjects · "
                    f"{decomp.reason}",
                    flush=True,
                )
                for s in decomp.subquestions:
                    print(f"        · {s}", flush=True)

        def catalog_route_idxs(sub_q: str, domain_hint: list[str] | None):
            """Optional tighter pool from the catalog; None = keep domain pool."""
            nonlocal catalog_info
            if catalog is None:
                return None
            t_cat = time.perf_counter()
            files, matched, debug = catalog_files_for_query(
                catalog,
                sub_q,
                domain=(
                    domain_hint[0]
                    if domain_hint is not None and len(domain_hint) == 1
                    else None
                ),
                domains=domain_hint if domain_hint and len(domain_hint) != 1 else None,
                min_score=args.catalog_min_score,
            )
            timings["catalog_ms"] = round(
                timings.get("catalog_ms", 0)
                + (time.perf_counter() - t_cat) * 1000,
                1,
            )
            if not files:
                return None
            idxs = intersect_idxs(idxs_for_files(vectors, files), non_faq_idxs)
            if len(idxs) < 10:
                # Too thin — don't replace a healthy domain pool.
                return None
            info = {
                "files": files,
                "matched": [
                    {
                        "id": e.id,
                        "title": e.title,
                        "doc_type": e.doc_type,
                        "file": e.file,
                    }
                    for e in matched
                ],
                "n_chunks": len(idxs),
                "top_debug": debug[:5],
            }
            if catalog_info is None:
                catalog_info = {"queries": [info]}
            else:
                catalog_info.setdefault("queries", []).append(info)
            titles = ", ".join(e.title[:40] for e in matched[:3])
            print(
                f"    catalog → {len(matched)} hit(s), {len(files)} files, "
                f"{len(idxs)} chunks · {titles}",
                flush=True,
            )
            return idxs

        def apply_catalog(sub_q: str, base_idxs: list[int] | None) -> list[int] | None:
            """Narrow a domain/global pool with catalog files when match is strong."""
            domains = None if route_meta is None else route_meta.domains
            cat_idxs = catalog_route_idxs(sub_q, domains)
            if cat_idxs is None:
                return base_idxs
            narrowed = intersect_idxs(cat_idxs, base_idxs)
            if narrowed is None or len(narrowed) < 10:
                return base_idxs
            return narrowed

        # Global pool always applies FAQ exclusion when enabled.
        global_idxs = non_faq_idxs
        if args.route:
            preview_hits = None
            if args.route_context:
                t_prev = time.perf_counter()
                preview_hits = search(
                    q_emb[qi],
                    emb,
                    vectors,
                    top_k=args.route_preview_n,
                    candidate_idxs=global_idxs,
                )
                timings["dense_preview_ms"] = round(
                    (time.perf_counter() - t_prev) * 1000, 1
                )
                global_seed = preview_hits
            t_route = time.perf_counter()
            route_meta = route_question(
                question,
                model=args.route_model,
                preview_hits=preview_hits,
            )
            timings["route_llm_ms"] = round(
                (time.perf_counter() - t_route) * 1000, 1
            )
            if route_meta.fallback_all or not route_meta.domains:
                print(
                    f"    route fallback (all domains): {route_meta.reason}",
                    flush=True,
                )
                route_idxs = None
                global_seed = None  # fallback: full-corpus dense, no hybrid seed
            else:
                route_idxs = intersect_idxs(
                    idxs_for_domains(vectors, route_meta.domains),
                    non_faq_idxs,
                )
                preview_note = (
                    f", preview={len(preview_hits)} reused"
                    if preview_hits is not None
                    else ""
                )
                if args.retrieve == "rerank":
                    print(
                        f"    route → {route_meta.domains} "
                        f"(hybrid {args.route_n}+{args.route_global_n} "
                        f"from {len(route_idxs)} domain chunks{preview_note}) · "
                        f"{route_meta.reason}",
                        flush=True,
                    )
                else:
                    print(
                        f"    route → {route_meta.domains} "
                        f"({len(route_idxs)} chunks{preview_note}) · "
                        f"{route_meta.reason}",
                        flush=True,
                    )

        # Catalog can narrow the routed (or full) pool to matched product files.
        # For decompose+rerank, narrowing happens per sub-question below.
        multi_sub = bool(decomp and decomp.split and args.retrieve == "rerank")
        if catalog is not None and not multi_sub:
            route_idxs = apply_catalog(question, route_idxs)

        # Non-rerank modes: restrict entirely to routed domains (old behavior).
        # When catalog alone set route_idxs (no --route), use that pool.
        domain_only_idxs = (
            route_idxs if route_idxs is not None else global_idxs
        )

        if args.retrieve == "cascade":
            assert file_bm25 is not None
            t_ret = time.perf_counter()
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
            timings["retrieve_ms"] = round(
                (time.perf_counter() - t_ret) * 1000, 1
            )
            return (
                result.expanded,
                result.mode,
                result.file_hits,
                route_meta,
                global_seed,
                timings,
                decomp,
                catalog_info,
            )
        if args.retrieve == "rrf":
            assert chunk_bm25 is not None
            t_ret = time.perf_counter()
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
            timings["retrieve_ms"] = round(
                (time.perf_counter() - t_ret) * 1000, 1
            )
            return (
                result.expanded,
                result.mode,
                result.file_hits,
                route_meta,
                global_seed,
                timings,
                decomp,
                catalog_info,
            )
        if args.retrieve == "rerank":
            assert reranker is not None
            subs = decomp.subquestions if (decomp and decomp.split) else [question]
            if len(subs) == 1:
                sub_embs = [q_emb[qi]]
            else:
                # Sub-questions are rewritten text, so they need their own vectors.
                sub_embs = list(embedder.embed_queries(subs))
            # Budgets are split, not multiplied: total CE input and final
            # context size stay the same as the single-query path.
            k_budget = split_budget(args.top_k, len(subs))
            cand_budget = split_budget(args.candidate_n, len(subs))
            route_budget = split_budget(args.route_n, len(subs))
            gseed_budget = split_budget(args.route_global_n, len(subs))
            per_sub = []
            mode = "rerank"
            file_hits: list = []
            for si, sub in enumerate(subs):
                sub_route = route_idxs
                if catalog is not None and multi_sub:
                    sub_route = apply_catalog(sub, route_idxs)
                result = search_reranked(
                    sub,
                    sub_embs[si],
                    emb,
                    vectors,
                    reranker,
                    candidate_n=cand_budget[si],
                    top_k=k_budget[si],
                    window=args.window,
                    id_to_idx=id_to_idx,
                    candidate_idxs=global_idxs,
                    route_idxs=sub_route,
                    route_n=route_budget[si],
                    route_global_n=gseed_budget[si],
                    global_seed=global_seed if sub_route is not None else None,
                )
                per_sub.append(result.expanded)
                mode = result.mode
                file_hits = result.file_hits
                for key, val in result.timings_ms.items():
                    timings[key] = round(timings.get(key, 0) + val, 1)
            expanded = (
                per_sub[0]
                if len(per_sub) == 1
                else merge_hits(per_sub, top_k=args.top_k)
            )
            return (
                expanded,
                mode,
                file_hits,
                route_meta,
                global_seed,
                timings,
                decomp,
                catalog_info,
            )
        t_ret = time.perf_counter()
        expanded = search_expanded(
            q_emb[qi],
            emb,
            vectors,
            top_k=args.top_k,
            window=args.window,
            id_to_idx=id_to_idx,
            candidate_idxs=domain_only_idxs,
        )
        timings["retrieve_ms"] = round((time.perf_counter() - t_ret) * 1000, 1)
        return (
            expanded,
            "dense",
            [],
            route_meta,
            global_seed,
            timings,
            decomp,
            catalog_info,
        )

    approach = {
        "dense": "rag-no-cite",
        "cascade": "rag-cascade-no-cite",
        "rrf": "rag-rrf-no-cite",
        "rerank": "rag-rerank-no-cite",
    }[args.retrieve]

    if args.show_prompt:
        expanded, mode, _, route_meta, _seed, _timings, _decomp, _cat = retrieve_one(
            0, questions[0]["question"]
        )
        context = build_context(expanded)
        messages = build_messages(
            questions[0]["question"], context, system_prompt=args.system_prompt
        )
        print(render_prompt_preview(messages))
        preview_note = ""
        if route_meta is not None:
            preview_note = (
                f", preview={'on' if route_meta.used_preview else 'off'}"
            )
        print(
            f"\n(retrieve={mode}, candidates={args.candidate_n}, top_k={args.top_k}, "
            f"window=±{args.window}, {len(expanded)} hits, "
            f"route={None if route_meta is None else route_meta.domains}"
            f"{preview_note}, "
            f"prompt chars≈{sum(len(m['content']) for m in messages)})"
        )
        return

    latency_rows: list[dict[str, float]] = []
    with open(args.out, out_mode, encoding="utf-8") as out:
        for qi, q in enumerate(questions):
            t0 = time.perf_counter()
            (
                expanded,
                mode,
                file_hits,
                route_meta,
                _seed,
                stage_ms,
                decomp,
                catalog_info,
            ) = retrieve_one(qi, q["question"])
            t_prompt0 = time.perf_counter()
            context = build_context(expanded)
            messages = build_messages(
                q["question"], context, system_prompt=args.system_prompt
            )
            stage_ms["prompt_build_ms"] = round(
                (time.perf_counter() - t_prompt0) * 1000, 1
            )
            t_retr = time.perf_counter()
            stage_ms["retrieve_total_ms"] = round((t_retr - t0) * 1000, 1)
            last_err: Exception | None = None
            resp = None
            t_llm0 = time.perf_counter()
            # num_retries=0: litellm's own retries stack with ours and can leave
            # multiple hung SSL sockets open against Nebius (seen: 4 ESTABLISHED
            # while processing one question). One hard timeout, one outer retry.
            for attempt in range(2):
                try:
                    resp = litellm.completion(
                        model=llm_model,
                        messages=messages,
                        timeout=args.llm_timeout,
                        num_retries=0,
                        **kwargs,
                        **llm_extra,
                    )
                    break
                except Exception as e:
                    last_err = e
                    name = type(e).__name__
                    if "Timeout" not in name and "timeout" not in str(e).lower():
                        raise
                    print(
                        f"    LLM timeout on {q['id']} "
                        f"(attempt {attempt + 1}/2, {args.llm_timeout:.0f}s)",
                        flush=True,
                    )
                    time.sleep(2 * (attempt + 1))
            if resp is None:
                raise last_err  # type: ignore[misc]
            stage_ms["llm_ms"] = round((time.perf_counter() - t_llm0) * 1000, 1)
            latency_ms = (time.perf_counter() - t0) * 1000
            stage_ms["total_ms"] = round(latency_ms, 1)
            latency_rows.append(dict(stage_ms))
            msg = resp.choices[0].message
            raw = (msg.content or "").strip()
            # Some thinking models spend the token budget on reasoning_content.
            if not raw:
                raw = (
                    getattr(msg, "reasoning_content", None)
                    or (msg.get("reasoning_content") if isinstance(msg, dict) else None)
                    or ""
                )
                if isinstance(raw, str):
                    raw = raw.strip()
                else:
                    raw = ""
                if raw:
                    print(
                        f"    note: empty content; fell back to reasoning_content "
                        f"({len(raw)} chars)",
                        flush=True,
                    )
            passage_meta: dict | None = None
            if args.cite == "passages":
                parsed = parse_answer_with_passages(raw)
                answer = parsed.answer
                if parsed.passage_indices:
                    citations = citations_from_passage_indices(
                        expanded,
                        parsed.passage_indices,
                        max_citations=args.max_citations,
                    )
                    cite_source = "passages"
                else:
                    # Fallback so we never ship empty cites on parse miss / none
                    citations = citations_from_hits(
                        expanded, max_citations=args.max_citations
                    )
                    cite_source = "hits_fallback"
                passage_meta = {
                    "parse_ok": parsed.parse_ok,
                    "used_passages_raw": parsed.used_passages_raw,
                    "passage_indices": parsed.passage_indices,
                    "cite_source": cite_source,
                }
            else:
                answer = raw
                citations = citations_from_hits(
                    expanded, max_citations=args.max_citations
                )
                cite_source = "hits"
            rec = {
                "id": q["id"],
                "answer": answer,
                "citations": citations,
                "latency_ms": latency_ms,
                "latency_breakdown_ms": stage_ms,
                "reasoning_effort": args.reasoning_effort,
                "no_thinking": bool(args.no_thinking),
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
                    "route_context": bool(args.route and args.route_context),
                    "route_preview_n": (
                        args.route_preview_n
                        if args.route and args.route_context
                        else None
                    ),
                    "route_used_preview": None
                    if route_meta is None
                    else route_meta.used_preview,
                    "decompose": None
                    if decomp is None
                    else {
                        "subquestions": decomp.subquestions,
                        "split": decomp.split,
                        "reason": decomp.reason,
                        "failed": decomp.failed,
                        "notes": decomp.field_notes,
                    },
                    "catalog": catalog_info,
                    "exclude_faq": exclude_faq,
                    "cite_mode": args.cite,
                    "passage_cite": passage_meta,
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
                + (
                    "+routectx"
                    if args.route and args.route_context
                    else ""
                )
                + ("" if args.include_faq else "+nofaq")
                + ("+catalog" if args.catalog else "")
                + f"+cite_{args.cite}",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            preview = answer.replace("\n", " ")[:70]
            top = expanded[0].score if expanded else 0.0
            cite_bit = ""
            if passage_meta is not None:
                cite_bit = (
                    f"  cites={passage_meta['cite_source']}:"
                    f"{passage_meta['passage_indices'] or '∅'}"
                )
            bits = []
            for key in (
                "dense_preview_ms",
                "route_llm_ms",
                "dense_candidates_ms",
                "cross_encoder_ms",
                "llm_ms",
            ):
                if key in stage_ms:
                    short = key.removesuffix("_ms")
                    bits.append(f"{short}={stage_ms[key]:.0f}")
            profile_bit = ("  [" + " ".join(bits) + "]") if bits else ""
            print(
                f"  {q['id']:28s} {mode:8s} "
                f"retr={stage_ms.get('retrieve_total_ms', (t_retr - t0) * 1000):.0f}ms  "
                f"total={latency_ms:.0f}ms  "
                f"top={top:.3f}  "
                f"{preview!r}…{cite_bit}{profile_bit}",
                flush=True,
            )

    if latency_rows:
        keys = sorted({k for row in latency_rows for k in row})
        print("\nLatency profile (ms):", flush=True)
        print(
            f"  {'stage':22s} {'n':>4} {'p50':>8} {'p95':>8} {'max':>8}",
            flush=True,
        )
        for key in keys:
            vals = sorted(row[key] for row in latency_rows if key in row)
            if not vals:
                continue
            p50 = vals[len(vals) // 2]
            p95 = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
            print(
                f"  {key:22s} {len(vals):4d} {p50:8.0f} {p95:8.0f} {vals[-1]:8.0f}",
                flush=True,
            )

    print(
        f"\nwrote {args.out} — score with: "
        f"python run_eval.py --answers {args.out}"
    )


if __name__ == "__main__":
    main()
