"""Tool implementations + OpenAI-style schemas for the hybrid ReAct agent.

Tools mutate `AgentState` (working memory). Observations stay short so the
agent loop does not dump whole PDFs into the chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag.bm25 import ChunkBM25Index, build_chunk_bm25, tokenize
from rag.catalog import Catalog, catalog_files_for_query, idxs_for_files, search_catalog
from rag.index_store import IndexedVector
from rag.peek import peek_titles, title_from_text
from rag.retrieve import (
    ExpandedHit,
    Hit,
    _chunk_seq,
    idxs_excluding_faqs,
    intersect_idxs,
    search,
    search_reranked,
)
from rag.route import idxs_for_domains, route_question

MAX_OBS_CHARS = 3500
MAX_SNIPPET = 280  # primary text in agent brief (neighbors listed separately)
MAX_NEIGHBOR_SNIPPET = 140
# Primaries shown per seed / more_passages page. Answer LLM gets the revealed
# prefix plus any post-seed additions (search / standalone fetch), not unread pages.
BRIEF_BATCH = 5
MAX_FETCH_PER_CALL = 4
MAX_EXTRA_FETCHED = 6
DEFAULT_PEEK_RADIUS = 6
MAX_GREP_HITS = 10
GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Sandbox lexical search over chunk text (no embeddings). "
            "Use for exact Hebrew phrases, product codes, form titles, phone "
            "numbers, or statute wording that dense search may miss. "
            "Does NOT attach passages — call fetch_chunks on promising ids. "
            "mode=bm25 ranks by keyword score; mode=literal requires all "
            "space-separated terms as substrings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords / exact phrase (Hebrew OK)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["bm25", "literal"],
                    "description": "bm25 (default) or literal substring AND",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max hits to return (default 8, max 10)",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ToolTraceEntry:
    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class AgentState:
    question: str
    passages: list[ExpandedHit] = field(default_factory=list)
    title_map: list[dict[str, Any]] = field(default_factory=list)
    route_domains: list[str] | None = None
    catalog_files: list[str] = field(default_factory=list)
    catalog_matched: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    fetched_ids: list[str] = field(default_factory=list)
    # How many primaries have been revealed in the controller brief (pagination).
    brief_shown: int = 0
    # Passage count right after seed retrieve (before search/fetch appends).
    seed_passage_n: int = 0
    done: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalBundle:
    """Warm index + models shared across seed retrieve and tools."""

    vectors: list[IndexedVector]
    emb: np.ndarray
    id_to_idx: dict[str, int]
    embedder: Any
    reranker: Any
    catalog: Catalog | None = None
    non_faq_idxs: list[int] | None = None
    catalog_min_score: float = 8.0
    top_k: int = 20
    window: int = 2
    candidate_n: int = 100
    route_n: int = 80
    route_global_n: int = 20
    route_preview_n: int = 20
    route_model: str | None = None
    use_route: bool = True
    use_catalog: bool = True
    # Sandbox: expose lexical grep tool (off by default).
    enable_grep: bool = False
    chunk_bm25: ChunkBM25Index | None = None


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "peek_titles",
            "description": (
                "Scan inferred section titles (first lines) of chunks near a "
                "retrieved chunk_id. Use when you need exclusions, limits, "
                "waiting periods, or liability tables near a hit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "Anchor chunk id from seed passages (e.g. pdf:…#33)",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Sequential neighbor radius (default 6)",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunks",
            "description": (
                "Load full text for chunk ids (from peek_titles or search) into "
                "the working passage set. Prefer interesting exclusion/limit titles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 4 chunk ids to fetch",
                    },
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "catalog_lookup",
            "description": (
                "Search the corpus catalog for products/files matching a query. "
                "Use when the seed passages look like the wrong product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Product or document name / keywords",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Run another hybrid retrieve+rerank for a tighter query. "
                "By default searches the routed domain pool only. Set "
                "use_catalog=true to narrow to files already matched by seed "
                "or a prior catalog_lookup (does not re-run catalog on this query)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (Hebrew OK)",
                    },
                    "use_catalog": {
                        "type": "boolean",
                        "description": (
                            "If true, restrict to catalog files already in state "
                            "(from seed or catalog_lookup). Default false."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "more_passages",
            "description": (
                "Reveal the next batch of seed/working primaries (with neighbors) "
                "that were not shown yet. Passage [N] labels match the answer "
                "context. Prefer this over search when you only need to scan more "
                "of the already-retrieved list. Free vs tool-round budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Stop gathering evidence and answer from the current passages. "
                "Call when evidence is enough or clearly absent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief why you are ready to answer",
                    },
                },
                "required": [],
            },
        },
    },
]


def tool_schemas(*, enable_grep: bool = False) -> list[dict[str, Any]]:
    """OpenAI tool list; grep is opt-in sandbox only."""
    if not enable_grep:
        return TOOL_SCHEMAS
    # Insert before final_answer so the model sees it with the other retrieve tools.
    out = list(TOOL_SCHEMAS)
    out.insert(-1, GREP_SCHEMA)
    return out


def ensure_chunk_bm25(bundle: RetrievalBundle) -> ChunkBM25Index:
    if bundle.chunk_bm25 is None:
        bundle.chunk_bm25 = build_chunk_bm25(bundle.vectors)
    return bundle.chunk_bm25


def _truncate(s: str, n: int = MAX_OBS_CHARS) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 20] + "\n…[truncated]"


def _context_ids(passages: list[ExpandedHit]) -> set[str]:
    out: set[str] = set()
    for ex in passages:
        out.add(ex.match.id)
        for h in ex.neighbors:
            out.add(h.vector.id)
    return out


def _brief_snip(text: str, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) > n:
        return t[: n - 1] + "…"
    return t


def format_passages_brief(
    passages: list[ExpandedHit],
    *,
    offset: int = 0,
    max_n: int = BRIEF_BATCH,
    snippet: int = MAX_SNIPPET,
    neighbor_snippet: int = MAX_NEIGHBOR_SNIPPET,
) -> str:
    """Thin view for the agent controller — primaries + attached neighbors.

    Neighbors are already in the answer-LLM context (window expand / fetch).
    All window neighbors are listed (★ interesting titles first). Listing them
    here prevents the controller from grepping/searching for text it cannot see
    in primary snippets alone.

    `[N]` labels are 1-based into `passages` (same indices the answer LLM uses).
    Use `offset` / `max_n` to page; call more_passages for the next page.
    """
    from rag.peek import is_interesting_title, title_from_text

    offset = max(0, int(offset))
    max_n = max(0, int(max_n))
    page = passages[offset : offset + max_n]
    blocks: list[str] = []
    for i, ex in enumerate(page, start=offset + 1):
        loc = ex.match.location
        text = _brief_snip(ex.match.text or "", snippet)
        parts = [
            f"[{i}] id={ex.match.id} file={loc.file} page={loc.page} "
            f"score={ex.score:.3f}",
            text,
        ]
        if ex.neighbors:
            # Interesting titles first, then document order.
            ranked = sorted(
                ex.neighbors,
                key=lambda h: (
                    0
                    if is_interesting_title(title_from_text(h.vector.text or ""))
                    else 1,
                    (_chunk_seq(h.vector.id) or ("", 0))[1],
                ),
            )
            parts.append(
                "  neighbors (already in answer context — prefer final_answer; "
                "fetch only if you need fuller text):"
            )
            for h in ranked:
                title = title_from_text(h.vector.text or "")
                flag = " ★" if is_interesting_title(title) else ""
                snip = _brief_snip(h.vector.text or "", neighbor_snippet)
                parts.append(
                    f"  - id={h.vector.id} page={h.vector.location.page} "
                    f"title={title[:80]}{flag}"
                )
                parts.append(f"    {snip}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks) if blocks else "(no passages)"


def brief_page_footer(
    *,
    shown: int,
    total: int,
    batch: int = BRIEF_BATCH,
) -> str:
    """Short paging note for seed / more_passages observations."""
    if total <= 0:
        return "No passages in working set."
    if shown >= total:
        return f"Showing all {total} passages. No further batches."
    remaining = total - shown
    nxt = min(batch, remaining)
    return (
        f"Showing primaries 1–{shown} of {total}. "
        f"Call more_passages for the next {nxt} "
        f"(does not consume tool-round budget)."
    )


def seed_retrieve(bundle: RetrievalBundle, question: str) -> AgentState:
    """Deterministic first retrieve: route → catalog → hybrid CE → expand."""
    import time

    state = AgentState(question=question)
    t0 = time.perf_counter()
    q_emb = bundle.embedder.embed_queries([question])[0]
    state.timings_ms["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    route_idxs = None
    global_seed = None
    non_faq = bundle.non_faq_idxs
    if non_faq is None:
        non_faq = idxs_excluding_faqs(bundle.vectors)

    if bundle.use_route:
        t_r = time.perf_counter()
        preview = search(
            q_emb,
            bundle.emb,
            bundle.vectors,
            top_k=bundle.route_preview_n,
            candidate_idxs=non_faq,
        )
        global_seed = preview
        route = route_question(
            question,
            model=bundle.route_model or "google/gemma-3-27b-it",
            preview_hits=preview,
        )
        state.timings_ms["route_ms"] = round((time.perf_counter() - t_r) * 1000, 1)
        if route.domains and not route.fallback_all:
            state.route_domains = list(route.domains)
            route_idxs = intersect_idxs(
                idxs_for_domains(bundle.vectors, route.domains), non_faq
            )
        else:
            state.route_domains = None

    if bundle.use_catalog and bundle.catalog is not None:
        t_c = time.perf_counter()
        files, matched, _dbg = catalog_files_for_query(
            bundle.catalog,
            question,
            domains=state.route_domains,
            min_score=bundle.catalog_min_score,
        )
        state.timings_ms["catalog_ms"] = round(
            (time.perf_counter() - t_c) * 1000, 1
        )
        if files:
            cat_idxs = intersect_idxs(idxs_for_files(bundle.vectors, files), non_faq)
            if cat_idxs and len(cat_idxs) >= 10:
                base = route_idxs if route_idxs is not None else non_faq
                narrowed = intersect_idxs(cat_idxs, base)
                if narrowed and len(narrowed) >= 10:
                    route_idxs = narrowed
                    state.catalog_files = list(files)
                    state.catalog_matched = [
                        {
                            "id": e.id,
                            "title": e.title,
                            "doc_type": e.doc_type,
                            "file": e.file,
                        }
                        for e in matched
                    ]

    t_ret = time.perf_counter()
    result = search_reranked(
        question,
        q_emb,
        bundle.emb,
        bundle.vectors,
        bundle.reranker,
        candidate_n=bundle.candidate_n,
        top_k=bundle.top_k,
        window=bundle.window,
        id_to_idx=bundle.id_to_idx,
        candidate_idxs=non_faq,
        route_idxs=route_idxs,
        route_n=bundle.route_n,
        route_global_n=bundle.route_global_n,
        global_seed=global_seed,
    )
    state.timings_ms["retrieve_ms"] = round(
        (time.perf_counter() - t_ret) * 1000, 1
    )
    state.timings_ms.update(result.timings_ms or {})
    state.passages = list(result.expanded)
    state.seed_passage_n = len(state.passages)
    return state


def passages_for_answer(state: AgentState) -> list[ExpandedHit]:
    """Passages the answer LLM should see.

    Includes the controller-revealed seed prefix (`brief_shown`) plus any
    passages appended after seed (search / standalone fetch). Unread seed
    pages stay out of the answer prompt until revealed via more_passages.
    """
    passages = state.passages
    if not passages:
        return []
    n = len(passages)
    shown = max(0, min(int(state.brief_shown or 0), n))
    seed_n = max(0, min(int(state.seed_passage_n or 0), n))
    if shown <= 0 and seed_n <= 0:
        # Fallback if state was built outside the agent loop.
        return list(passages)

    out: list[ExpandedHit] = []
    seen: set[str] = set()
    for ex in passages[:shown]:
        out.append(ex)
        seen.add(ex.match.id)
    # Post-seed additions (even if middle seed pages were never revealed).
    for ex in passages[seed_n:]:
        if ex.match.id in seen:
            continue
        out.append(ex)
        seen.add(ex.match.id)
    return out or list(passages[: min(BRIEF_BATCH, n)])


def tool_peek_titles(
    state: AgentState,
    bundle: RetrievalBundle,
    *,
    chunk_id: str,
    radius: int = DEFAULT_PEEK_RADIUS,
) -> str:
    if chunk_id not in bundle.id_to_idx:
        return f"Unknown chunk_id: {chunk_id}"
    already = _context_ids(state.passages)
    titles = peek_titles(
        bundle.vectors,
        bundle.id_to_idx,
        chunk_id,
        radius=max(1, min(int(radius), 12)),
        already=already,
    )
    rows = []
    for th in titles:
        row = {
            "id": th.chunk_id,
            "title": th.title,
            "page": th.page,
            "interesting": th.interesting,
            "in_context": th.already_in_context,
        }
        rows.append(row)
        state.title_map.append(row)
    interesting = [r for r in rows if r["interesting"]]
    body = {
        "anchor": chunk_id,
        "n": len(rows),
        "interesting": interesting[:20],
        "other_sample": [r for r in rows if not r["interesting"]][:8],
    }
    return _truncate(json.dumps(body, ensure_ascii=False, indent=2))


def _attach_chunk(
    state: AgentState,
    bundle: RetrievalBundle,
    chunk_id: str,
) -> bool:
    """Attach chunk as neighbor of best same-file passage, or as new hit."""
    j = bundle.id_to_idx.get(chunk_id)
    if j is None:
        return False
    if chunk_id in _context_ids(state.passages):
        return False
    if len(state.fetched_ids) >= MAX_EXTRA_FETCHED:
        return False

    vec = bundle.vectors[j]
    parsed = _chunk_seq(chunk_id)
    # Prefer attaching to an existing passage from the same file stem.
    best: ExpandedHit | None = None
    if parsed is not None:
        stem, seq = parsed
        for ex in state.passages:
            p = _chunk_seq(ex.match.id)
            if p and p[0] == stem:
                best = ex
                break
    if best is not None and parsed is not None:
        role = "prev" if parsed[1] < (_chunk_seq(best.match.id) or ("", 0))[1] else "next"
        best.neighbors.append(
            Hit(
                rank=best.rank,
                score=best.score,
                vector=vec,
                role=role,
                anchor_id=best.match.id,
            )
        )
        best.neighbors.sort(
            key=lambda h: (_chunk_seq(h.vector.id) or ("", 0))[1]
        )
    else:
        # Standalone passage at the end.
        state.passages.append(
            ExpandedHit(
                rank=len(state.passages) + 1,
                score=0.0,
                match=vec,
                neighbors=[],
            )
        )
    state.fetched_ids.append(chunk_id)
    return True


def tool_fetch_chunks(
    state: AgentState,
    bundle: RetrievalBundle,
    *,
    chunk_ids: list[str],
) -> str:
    ids = [c for c in chunk_ids if isinstance(c, str) and c.strip()]
    ids = ids[:MAX_FETCH_PER_CALL]
    added: list[str] = []
    skipped: list[str] = []
    for cid in ids:
        if len(state.fetched_ids) >= MAX_EXTRA_FETCHED:
            skipped.append(f"{cid}: budget_exhausted")
            continue
        if cid not in bundle.id_to_idx:
            skipped.append(f"{cid}: unknown")
            continue
        if cid in _context_ids(state.passages):
            skipped.append(f"{cid}: already_in_context")
            continue
        ok = _attach_chunk(state, bundle, cid)
        if ok:
            v = bundle.vectors[bundle.id_to_idx[cid]]
            title = title_from_text(v.text or "")
            added.append(f"{cid} · p{v.location.page} · {title[:60]}")
        else:
            skipped.append(f"{cid}: failed")
    body = {
        "added": added,
        "skipped": skipped,
        "n_fetched_total": len(state.fetched_ids),
        "budget_left": max(0, MAX_EXTRA_FETCHED - len(state.fetched_ids)),
    }
    return _truncate(json.dumps(body, ensure_ascii=False, indent=2))


def tool_catalog_lookup(
    state: AgentState,
    bundle: RetrievalBundle,
    *,
    query: str,
) -> str:
    if bundle.catalog is None:
        return "Catalog not loaded."
    scored = search_catalog(bundle.catalog, query, limit=8)
    rows = [
        {
            "title": e.title,
            "domain": e.domain,
            "doc_type": e.doc_type,
            "file": e.file,
            "score": round(s, 2),
            "aliases": e.aliases[:4],
        }
        for e, s in scored
    ]
    if scored:
        state.catalog_files = list(
            dict.fromkeys([*state.catalog_files, *[e.file for e, _ in scored if e.file]])
        )
    return _truncate(json.dumps({"query": query, "hits": rows}, ensure_ascii=False, indent=2))


def _grep_allow_idxs(state: AgentState, bundle: RetrievalBundle) -> list[int] | None:
    """Prefer non-FAQ + routed domains when available."""
    non_faq = bundle.non_faq_idxs
    if non_faq is None:
        non_faq = idxs_excluding_faqs(bundle.vectors)
    if state.route_domains:
        route_idxs = intersect_idxs(
            idxs_for_domains(bundle.vectors, state.route_domains), non_faq
        )
        if route_idxs:
            return route_idxs
    return non_faq


def _literal_grep(
    vectors: list[IndexedVector],
    query: str,
    *,
    allow: list[int] | None,
    top_n: int,
) -> list[tuple[int, float]]:
    """Substring AND over space-separated terms; score = total occurrences."""
    terms = [t for t in query.split() if t.strip()]
    if not terms:
        return []
    terms_l = [t.casefold() for t in terms]
    pool = allow if allow is not None else range(len(vectors))
    scored: list[tuple[int, float]] = []
    for i in pool:
        text = (vectors[i].text or "").casefold()
        if not text:
            continue
        if any(t not in text for t in terms_l):
            continue
        score = float(sum(text.count(t) for t in terms_l))
        scored.append((i, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_n]


def tool_grep(
    state: AgentState,
    bundle: RetrievalBundle,
    *,
    query: str,
    mode: str = "bm25",
    top_n: int = 8,
) -> str:
    """Lexical sandbox search. Returns hits only — does not mutate passages."""
    import time

    q = (query or "").strip()
    if not q:
        return "grep: empty query"
    mode_l = (mode or "bm25").strip().lower()
    if mode_l not in ("bm25", "literal"):
        return f"grep: unknown mode {mode!r} (use bm25|literal)"
    n = max(1, min(int(top_n or 8), MAX_GREP_HITS))
    allow = _grep_allow_idxs(state, bundle)
    allow_set = set(allow) if allow is not None else None
    have = _context_ids(state.passages)

    t0 = time.perf_counter()
    if mode_l == "bm25":
        bm25 = ensure_chunk_bm25(bundle)
        # Over-fetch then filter to allow-set (BM25 is global).
        raw = bm25.search(q, top_n=max(n * 8, 40))
        hits = [(i, s) for i, s in raw if allow_set is None or i in allow_set][:n]
    else:
        hits = _literal_grep(bundle.vectors, q, allow=allow, top_n=n)
    ms = round((time.perf_counter() - t0) * 1000, 1)

    rows = []
    for i, score in hits:
        v = bundle.vectors[i]
        text = (v.text or "").replace("\n", " ").strip()
        if len(text) > MAX_SNIPPET:
            text = text[: MAX_SNIPPET - 1] + "…"
        rows.append(
            {
                "chunk_id": v.id,
                "score": round(float(score), 3),
                "file": v.location.file,
                "page": v.location.page,
                "in_context": v.id in have,
                "snippet": text,
            }
        )
    body = {
        "query": q,
        "mode": mode_l,
        "grep_ms": ms,
        "tokens": tokenize(q) if mode_l == "bm25" else q.split(),
        "n_hits": len(rows),
        "note": "Call fetch_chunks on useful chunk_ids; grep does not attach text.",
        "hits": rows,
    }
    return _truncate(json.dumps(body, ensure_ascii=False, indent=2))


def tool_search(
    state: AgentState,
    bundle: RetrievalBundle,
    *,
    query: str,
    use_catalog: bool = False,
) -> str:
    import time

    t0 = time.perf_counter()
    q_emb = bundle.embedder.embed_queries([query])[0]
    non_faq = bundle.non_faq_idxs
    if non_faq is None:
        non_faq = idxs_excluding_faqs(bundle.vectors)

    route_idxs = None
    if state.route_domains:
        route_idxs = intersect_idxs(
            idxs_for_domains(bundle.vectors, state.route_domains), non_faq
        )

    # Opt-in: reuse files already matched by seed / catalog_lookup.
    # Do not re-run catalog lexical match on the rewritten search query —
    # that fights agent-chosen keywords and can silently re-narrow the pool.
    if use_catalog and state.catalog_files:
        cat_idxs = intersect_idxs(
            idxs_for_files(bundle.vectors, state.catalog_files), non_faq
        )
        if cat_idxs and len(cat_idxs) >= 10:
            base = route_idxs if route_idxs is not None else non_faq
            narrowed = intersect_idxs(cat_idxs, base)
            if narrowed and len(narrowed) >= 10:
                route_idxs = narrowed
            else:
                # Catalog files outside the route pool — still prefer them.
                route_idxs = cat_idxs
        elif cat_idxs:
            route_idxs = cat_idxs

    # Smaller follow-up search to limit cost.
    result = search_reranked(
        query,
        q_emb,
        bundle.emb,
        bundle.vectors,
        bundle.reranker,
        candidate_n=min(60, bundle.candidate_n),
        top_k=min(8, bundle.top_k),
        window=bundle.window,
        id_to_idx=bundle.id_to_idx,
        candidate_idxs=non_faq,
        route_idxs=route_idxs,
        route_n=min(40, bundle.route_n),
        route_global_n=min(10, bundle.route_global_n),
    )
    # Merge new primary matches not already present.
    have = {ex.match.id for ex in state.passages}
    added = 0
    for ex in result.expanded:
        if ex.match.id in have:
            continue
        state.passages.append(ex)
        have.add(ex.match.id)
        added += 1
    ms = round((time.perf_counter() - t0) * 1000, 1)
    brief = format_passages_brief(result.expanded, max_n=8)
    return _truncate(
        f"search_ms={ms} added_new={added}\n\n{brief}"
    )


def tool_final_answer(state: AgentState, *, reason: str = "") -> str:
    state.done = True
    return f"ready_to_answer: {reason or 'ok'}"


def tool_more_passages(
    state: AgentState,
    *,
    batch: int = BRIEF_BATCH,
) -> str:
    """Reveal the next brief page of working-set primaries (no retrieve)."""
    total = len(state.passages)
    start = state.brief_shown
    if start >= total:
        return (
            f"No more passages to reveal (already shown {start} of {total}). "
            "Call final_answer, or search/peek if you need new evidence."
        )
    batch = max(1, int(batch) or BRIEF_BATCH)
    end = min(start + batch, total)
    body = format_passages_brief(state.passages, offset=start, max_n=end - start)
    state.brief_shown = end
    footer = brief_page_footer(shown=end, total=total, batch=batch)
    return _truncate(f"Passages {start + 1}–{end} of {total}:\n\n{body}\n\n{footer}")


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    state: AgentState,
    bundle: RetrievalBundle,
) -> str:
    """Run one tool; append trace; return observation string."""
    try:
        if name == "peek_titles":
            obs = tool_peek_titles(
                state,
                bundle,
                chunk_id=str(arguments.get("chunk_id", "")),
                radius=int(arguments.get("radius") or DEFAULT_PEEK_RADIUS),
            )
        elif name == "fetch_chunks":
            ids = arguments.get("chunk_ids") or []
            if isinstance(ids, str):
                ids = [ids]
            obs = tool_fetch_chunks(state, bundle, chunk_ids=list(ids))
        elif name == "catalog_lookup":
            obs = tool_catalog_lookup(
                state, bundle, query=str(arguments.get("query", ""))
            )
        elif name == "search":
            # Default false: agent must opt into catalog-file narrowing.
            raw_uc = arguments.get("use_catalog", False)
            if isinstance(raw_uc, str):
                use_cat = raw_uc.strip().lower() in ("1", "true", "yes")
            else:
                use_cat = bool(raw_uc)
            obs = tool_search(
                state,
                bundle,
                query=str(arguments.get("query", state.question)),
                use_catalog=use_cat,
            )
        elif name == "more_passages":
            obs = tool_more_passages(state, batch=BRIEF_BATCH)
        elif name == "grep":
            if not bundle.enable_grep:
                obs = "grep tool disabled (pass --agent-grep / RAG_AGENT_GREP=1)"
            else:
                obs = tool_grep(
                    state,
                    bundle,
                    query=str(arguments.get("query", state.question)),
                    mode=str(arguments.get("mode") or "bm25"),
                    top_n=int(arguments.get("top_n") or 8),
                )
        elif name == "final_answer":
            obs = tool_final_answer(
                state, reason=str(arguments.get("reason", ""))
            )
        else:
            obs = f"Unknown tool: {name}"
            state.tool_trace.append(
                ToolTraceEntry(name=name, arguments=arguments, ok=False, summary=obs)
            )
            return obs
        state.tool_trace.append(
            ToolTraceEntry(
                name=name,
                arguments=arguments,
                ok=True,
                summary=obs[:240].replace("\n", " "),
            )
        )
        return obs
    except Exception as e:
        msg = f"tool error ({name}): {type(e).__name__}: {e}"
        state.tool_trace.append(
            ToolTraceEntry(name=name, arguments=arguments, ok=False, summary=msg)
        )
        return msg


def trace_as_json(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "arguments": t.arguments,
            "ok": t.ok,
            "summary": t.summary,
        }
        for t in state.tool_trace
    ]
