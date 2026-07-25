"""In-memory dense retrieval over a saved corpus index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag.bm25 import ChunkBM25Index, FileBM25Index
from rag.index_store import IndexedVector, Location


def is_faq_file(path: str) -> bool:
    """True for scraped Harel FAQ pages (question titles, little/no answer body)."""
    p = (path or "").replace("\\", "/").lower()
    name = p.rsplit("/", 1)[-1]
    return name == "faq.txt" or name.startswith("faq.")


def normalize_cite_file(path: str) -> str:
    """Corpus-relative path for structured citations (matches eval normalize)."""
    p = (path or "").strip().replace("\\", "/")
    if p.startswith("corpus/"):
        p = p[len("corpus/") :]
    return p.replace(".aspx.txt", ".txt")


def citations_from_locations(
    locations: list[tuple[str, int | None]],
    *,
    max_citations: int = 5,
    corpus_root: str | None = "corpus",
) -> list[dict[str, str | int | None]]:
    """Dedupe `(file, page)` in order; TXT pages stay `null`.

    Skips paths that do not exist under `corpus_root` (stale index ghosts),
    because the citation judge scores 0 if any citation fails to resolve.
    """
    root = Path(corpus_root) if corpus_root else None
    out: list[dict[str, str | int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for raw_file, page in locations:
        file = normalize_cite_file(raw_file)
        if not file:
            continue
        if root is not None and not (root / file).exists():
            continue
        key = (file, page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": file, "page": page})
        if len(out) >= max_citations:
            break
    return out


def citations_from_hits(
    expanded: list["ExpandedHit"],
    *,
    max_citations: int = 5,
    corpus_root: str | None = "corpus",
) -> list[dict[str, str | int | None]]:
    """Structured citations from primary match locations (not neighbors alone)."""
    locs = [
        (ex.match.location.file, ex.match.location.page) for ex in expanded
    ]
    return citations_from_locations(
        locs, max_citations=max_citations, corpus_root=corpus_root
    )


def citations_from_hit_records(
    hits: list[dict],
    *,
    max_citations: int = 5,
    corpus_root: str | None = "corpus",
) -> list[dict[str, str | int | None]]:
    """Same as citations_from_hits for JSONL `retrieval.hits` rows."""
    locs = [(h.get("file") or "", h.get("page")) for h in hits]
    return citations_from_locations(
        locs, max_citations=max_citations, corpus_root=corpus_root
    )


def idxs_excluding_faqs(vectors: list[IndexedVector]) -> list[int]:
    return [i for i, v in enumerate(vectors) if not is_faq_file(v.location.file)]


def intersect_idxs(
    base: list[int] | None,
    restrict: list[int] | None,
) -> list[int] | None:
    """Intersect optional index lists. None means 'no restriction'."""
    if restrict is None:
        return base
    if base is None:
        return list(restrict)
    allow = set(restrict)
    return [i for i in base if i in allow]


@dataclass
class Hit:
    rank: int
    score: float
    vector: IndexedVector
    role: str = "match"  # match | prev | next
    anchor_id: str | None = None  # primary hit id when role is neighbor


@dataclass
class ExpandedHit:
    """A primary match plus ordered neighboring chunks (same file)."""

    rank: int
    score: float
    match: IndexedVector
    neighbors: list[Hit] = field(default_factory=list)

    @property
    def chunks(self) -> list[IndexedVector]:
        """Preceding → match → trailing, document order."""
        prev = [h.vector for h in self.neighbors if h.role == "prev"]
        nxt = [h.vector for h in self.neighbors if h.role == "next"]
        return [*prev, self.match, *nxt]

    def merged_text(self, *, separator: str = "\n\n") -> str:
        return separator.join(c.text for c in self.chunks if (c.text or "").strip())


@dataclass
class CascadeResult:
    expanded: list[ExpandedHit]
    mode: str  # cascade | dense-fallback | rrf
    file_hits: list[tuple[str, float]] = field(default_factory=list)


def rrf_fuse(rank_lists: list[list[int]], *, k: int = 60, top_k: int = 20) -> list[int]:
    """Reciprocal Rank Fusion over lists of corpus indices (rank 1 = best)."""
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]



def _chunk_seq(vector_id: str) -> tuple[str, int] | None:
    if "#" not in vector_id:
        return None
    stem, n = vector_id.rsplit("#", 1)
    if not n.isdigit():
        return None
    return stem, int(n)


def build_id_index(vectors: list[IndexedVector]) -> dict[str, int]:
    return {v.id: i for i, v in enumerate(vectors)}


def search(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    *,
    top_k: int = 10,
    candidate_idxs: list[int] | None = None,
) -> list[Hit]:
    """Cosine top-k (assumes L2-normalized rows when normalize=True at index time)."""
    if len(vectors) == 0:
        return []
    q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
    qn = float(np.linalg.norm(q))
    if qn > 0:
        q = q / qn

    if candidate_idxs is not None:
        if not candidate_idxs:
            return []
        sub_emb = corpus_emb[candidate_idxs]
        norms = np.linalg.norm(sub_emb, axis=1, keepdims=True)
        mat = sub_emb / np.clip(norms, 1e-9, None)
        scores = mat @ q
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        local = np.argpartition(-scores, k - 1)[:k]
        local = local[np.argsort(-scores[local])]
        return [
            Hit(
                rank=r + 1,
                score=float(scores[j]),
                vector=vectors[int(candidate_idxs[int(j)])],
                role="match",
            )
            for r, j in enumerate(local)
        ]

    norms = np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    mat = corpus_emb / np.clip(norms, 1e-9, None)
    scores = mat @ q
    k = min(top_k, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        Hit(rank=r + 1, score=float(scores[i]), vector=vectors[int(i)], role="match")
        for r, i in enumerate(idx)
    ]


def expand_hit(
    hit: Hit,
    vectors: list[IndexedVector],
    id_to_idx: dict[str, int],
    *,
    window: int = 1,
) -> ExpandedHit:
    """Attach ±window neighboring chunks from the same file (by sequential #id)."""
    if window <= 0:
        return ExpandedHit(rank=hit.rank, score=hit.score, match=hit.vector, neighbors=[])

    parsed = _chunk_seq(hit.vector.id)
    neighbors: list[Hit] = []
    if parsed is None:
        return ExpandedHit(rank=hit.rank, score=hit.score, match=hit.vector, neighbors=[])

    stem, n = parsed
    for delta in range(-window, 0):
        nid = f"{stem}#{n + delta}"
        j = id_to_idx.get(nid)
        if j is not None:
            neighbors.append(
                Hit(
                    rank=hit.rank,
                    score=hit.score,
                    vector=vectors[j],
                    role="prev",
                    anchor_id=hit.vector.id,
                )
            )
    for delta in range(1, window + 1):
        nid = f"{stem}#{n + delta}"
        j = id_to_idx.get(nid)
        if j is not None:
            neighbors.append(
                Hit(
                    rank=hit.rank,
                    score=hit.score,
                    vector=vectors[j],
                    role="next",
                    anchor_id=hit.vector.id,
                )
            )
    return ExpandedHit(
        rank=hit.rank, score=hit.score, match=hit.vector, neighbors=neighbors
    )


def search_expanded(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    *,
    top_k: int = 10,
    window: int = 1,
    id_to_idx: dict[str, int] | None = None,
    candidate_idxs: list[int] | None = None,
) -> list[ExpandedHit]:
    """Top-k dense hits, each expanded with preceding/trailing chunks."""
    hits = search(
        query_emb,
        corpus_emb,
        vectors,
        top_k=top_k,
        candidate_idxs=candidate_idxs,
    )
    index = id_to_idx if id_to_idx is not None else build_id_index(vectors)
    return [expand_hit(h, vectors, index, window=window) for h in hits]


def hybrid_dense_candidates(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    *,
    route_idxs: list[int],
    global_idxs: list[int] | None = None,
    route_n: int = 80,
    global_n: int = 20,
) -> list[Hit]:
    """Dense top-`route_n` in routed domains, plus `global_n` unique from full pool.

    The global leg searches everything in `global_idxs` (or the full corpus),
    including routed domains, and skips ids already taken by the route leg.
    """
    routed = search(
        query_emb,
        corpus_emb,
        vectors,
        top_k=route_n,
        candidate_idxs=route_idxs,
    )
    seen = {h.vector.id for h in routed}

    # Worst case the global top-route_n mirror the route hits; fetch that many
    # extras so we can still collect global_n uniques.
    fetch_n = route_n + global_n
    pool_size = len(global_idxs) if global_idxs is not None else len(vectors)
    fetch_n = min(fetch_n, pool_size)

    extras: list[Hit] = []
    if global_n > 0 and fetch_n > 0:
        global_hits = search(
            query_emb,
            corpus_emb,
            vectors,
            top_k=fetch_n,
            candidate_idxs=global_idxs,
        )
        for h in global_hits:
            if h.vector.id in seen:
                continue
            extras.append(h)
            seen.add(h.vector.id)
            if len(extras) >= global_n:
                break
        # Rare: heavy overlap — widen the global fetch once.
        if len(extras) < global_n and fetch_n < pool_size:
            global_hits = search(
                query_emb,
                corpus_emb,
                vectors,
                top_k=min(pool_size, fetch_n * 3),
                candidate_idxs=global_idxs,
            )
            for h in global_hits:
                if h.vector.id in seen:
                    continue
                extras.append(h)
                seen.add(h.vector.id)
                if len(extras) >= global_n:
                    break

    out: list[Hit] = []
    for rank, h in enumerate([*routed, *extras], start=1):
        out.append(
            Hit(rank=rank, score=h.score, vector=h.vector, role="match")
        )
    return out


def search_cascade(
    query: str,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    file_bm25: FileBM25Index,
    *,
    top_k: int = 10,
    window: int = 1,
    file_top_n: int = 20,
    id_to_idx: dict[str, int] | None = None,
    candidate_idxs: list[int] | None = None,
) -> CascadeResult:
    """BM25 file filter → dense within those files; full-corpus dense if filter empty.

    If `candidate_idxs` is set (e.g. domain route), BM25/dense are intersected with it.
    """
    allow = set(candidate_idxs) if candidate_idxs is not None else None
    file_hits = file_bm25.search_files(query, top_n=file_top_n)
    idxs: list[int] = []
    for path, _score in file_hits:
        for i in file_bm25.file_to_idxs.get(path) or []:
            if allow is None or i in allow:
                idxs.append(i)

    if not idxs:
        expanded = search_expanded(
            query_emb,
            corpus_emb,
            vectors,
            top_k=top_k,
            window=window,
            id_to_idx=id_to_idx,
            candidate_idxs=candidate_idxs,
        )
        mode = "dense-fallback" if candidate_idxs is None else "dense-domain-fallback"
        return CascadeResult(expanded=expanded, mode=mode, file_hits=[])

    expanded = search_expanded(
        query_emb,
        corpus_emb,
        vectors,
        top_k=top_k,
        window=window,
        id_to_idx=id_to_idx,
        candidate_idxs=idxs,
    )
    return CascadeResult(expanded=expanded, mode="cascade", file_hits=file_hits)


def search_rrf(
    query: str,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    chunk_bm25: ChunkBM25Index,
    *,
    top_k: int = 10,
    window: int = 1,
    candidate_n: int = 50,
    rrf_k: int = 60,
    id_to_idx: dict[str, int] | None = None,
    candidate_idxs: list[int] | None = None,
) -> CascadeResult:
    """Fuse dense + chunk-BM25 rankings with RRF, then neighbor-expand."""
    index = id_to_idx if id_to_idx is not None else build_id_index(vectors)
    allow = set(candidate_idxs) if candidate_idxs is not None else None
    dense_hits = search(
        query_emb,
        corpus_emb,
        vectors,
        top_k=candidate_n,
        candidate_idxs=candidate_idxs,
    )
    dense_idxs = [index[h.vector.id] for h in dense_hits]
    dense_score = {index[h.vector.id]: h.score for h in dense_hits}

    bm25_hits = chunk_bm25.search(query, top_n=candidate_n)
    bm25_idxs = [i for i, _ in bm25_hits if allow is None or i in allow]

    fused = rrf_fuse([dense_idxs, bm25_idxs], k=rrf_k, top_k=top_k)
    if not fused:
        expanded = search_expanded(
            query_emb,
            corpus_emb,
            vectors,
            top_k=top_k,
            window=window,
            id_to_idx=index,
            candidate_idxs=candidate_idxs,
        )
        return CascadeResult(expanded=expanded, mode="dense-fallback", file_hits=[])

    hits: list[Hit] = []
    for rank, idx in enumerate(fused, start=1):
        hits.append(
            Hit(
                rank=rank,
                score=float(dense_score.get(idx, 0.0)),
                vector=vectors[idx],
                role="match",
            )
        )
    expanded = [expand_hit(h, vectors, index, window=window) for h in hits]
    return CascadeResult(expanded=expanded, mode="rrf", file_hits=[])


def search_reranked(
    query: str,
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    vectors: list[IndexedVector],
    reranker,
    *,
    candidate_n: int = 100,
    top_k: int = 20,
    window: int = 2,
    id_to_idx: dict[str, int] | None = None,
    candidate_idxs: list[int] | None = None,
    route_idxs: list[int] | None = None,
    route_n: int = 80,
    route_global_n: int = 20,
) -> CascadeResult:
    """Dense top-candidate_n → cross-encoder top_k → neighbor expand.

    Cross-encoder scores embed_text when set (same string used for dense
    embedding); generator still receives vector.text via context building.
    Optional `candidate_idxs` restricts the dense pool (e.g. FAQ filter).

    When `route_idxs` is set: dense `route_n` from routed domains + `route_global_n`
    unique from the general pool (`candidate_idxs` / full corpus), then CE.
    """
    index = id_to_idx if id_to_idx is not None else build_id_index(vectors)
    if route_idxs is not None:
        candidates = hybrid_dense_candidates(
            query_emb,
            corpus_emb,
            vectors,
            route_idxs=route_idxs,
            global_idxs=candidate_idxs,
            route_n=route_n,
            global_n=route_global_n,
        )
    else:
        candidates = search(
            query_emb,
            corpus_emb,
            vectors,
            top_k=candidate_n,
            candidate_idxs=candidate_idxs,
        )
    if not candidates:
        return CascadeResult(expanded=[], mode="rerank", file_hits=[])

    # Rank the same string we embed (description for tables; else payload).
    passages = [
        (h.vector.embed_text or h.vector.text or "") for h in candidates
    ]
    scores = reranker.score(query, passages)
    order = np.argsort(-scores)[: min(top_k, len(candidates))]

    hits: list[Hit] = []
    for rank, j in enumerate(order, start=1):
        base = candidates[int(j)]
        hits.append(
            Hit(
                rank=rank,
                score=float(scores[int(j)]),
                vector=base.vector,
                role="match",
            )
        )
    expanded = [expand_hit(h, vectors, index, window=window) for h in hits]
    return CascadeResult(expanded=expanded, mode="rerank", file_hits=[])



def flatten_expanded(expanded: list[ExpandedHit]) -> list[Hit]:
    """Unique chunks from expanded hits (match first, then neighbors), stable order."""
    seen: set[str] = set()
    out: list[Hit] = []
    for ex in expanded:
        primary = Hit(
            rank=ex.rank, score=ex.score, vector=ex.match, role="match"
        )
        for h in [primary, *ex.neighbors]:
            if h.vector.id in seen:
                continue
            seen.add(h.vector.id)
            out.append(h)
    return out


def hit_matches_source(loc: Location, source: dict) -> bool:
    """True if chunk location covers a GT {file, page} pointer."""
    from eval.corpus import normalize_path

    if normalize_path(loc.file) != normalize_path(source.get("file") or ""):
        return False
    exp_page = source.get("page")
    if exp_page is None:
        return True
    pages = set(loc.pages or [])
    if loc.page is not None:
        pages.add(loc.page)
    return int(exp_page) in pages


def groups_hit(
    ground_truth_sources: list[dict],
    hits: list[Hit],
) -> tuple[int, int, list[str]]:
    """How many any_of groups are satisfied by retrieved hits."""
    details: list[str] = []
    satisfied = 0
    for i, group in enumerate(ground_truth_sources or [], start=1):
        matched = None
        for opt in group.get("any_of") or []:
            for h in hits:
                if hit_matches_source(h.vector.location, opt):
                    matched = opt.get("file")
                    break
            if matched:
                break
        if matched:
            satisfied += 1
            details.append(f"group {i}: hit ({matched})")
        else:
            opts = [o.get("file") for o in (group.get("any_of") or [])]
            details.append(f"group {i}: miss (expected one of {opts})")
    return satisfied, len(ground_truth_sources or []), details
