"""Local title peek around retrieved chunks + optional section fetch.

After dense/CE retrieval lands on a hit, scan neighboring chunk *first lines*
(inferred section titles — the index has no semantic heading field) and pull
in chunks whose titles look like carve-outs / limits when they are not already
in the ±window expand.

This is the deterministic v1 of:
  retrieve → peek titles around hits → fetch missing sections → answer
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rag.index_store import IndexedVector
from rag.retrieve import ExpandedHit, Hit, _chunk_seq

# First-line titles that usually repay a fetch (Hebrew insurance PDFs).
# Prefer phrases that look like section headings, not body mentions.
_TITLE_HINTS = (
    "חריגים",
    "חריג",
    "סייגים",
    "סייג",
    "הגבלות",
    "הגבלה",
    "תקופת המתנה",
    "תקופת אכשרה",
    "הוצאות מן הכלל",
    "מקרים שאינם",
    "מה אינו מכוסה",
    "מה לא מכוסה",
    "אינו כלול",
    "אינם כלולים",
    "תנאים מוקדמים",
    "הגבלת אחריות",
    "גבולות אחריות",
    "גבול אחריות",
    "השתתפות עצמית",
    "ביטול הפוליסה",
    "ביטול הביטוח",
    "התיישנות",
    # Policy extensions / coverage add-ons (often hold caps the agent needs).
    "הרחבה",
    "הרחבות",
)

# Question cues that make a title-peek fetch worthwhile.
_QUESTION_CUES = (
    "חריג",
    "חריגים",
    "סייג",
    "מכוסה",
    "כיסוי",
    "לא כולל",
    "מוחרג",
    "הגבלה",
    "תקופת המתנה",
    "אכשרה",
    "האם",
    "מתי",
    "תנאי",
    "ביטול",
    "החזר",
    "התיישנות",
    "צד ג",
    "נזק",
)


@dataclass
class TitleHit:
    chunk_id: str
    title: str
    seq: int
    page: int | None
    interesting: bool
    already_in_context: bool


@dataclass
class PeekResult:
    anchor_id: str
    titles: list[TitleHit] = field(default_factory=list)
    fetched_ids: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()


def title_from_text(text: str, *, max_len: int = 80) -> str:
    """Infer a section title from the leading line(s) of a chunk."""
    if not text:
        return ""
    for line in text.splitlines():
        t = _norm(line).lstrip(".·•–—)(-")
        # Skip pure page noise / very short fragments.
        if len(t) < 3:
            continue
        if re.fullmatch(r"[\d\W]+", t):
            continue
        return t[:max_len]
    return _norm(text)[:max_len]


def is_interesting_title(title: str) -> bool:
    """True when the inferred first line looks like a carve-out heading."""
    t = _norm(title)
    if not t or len(t) > 80:
        # Long first lines are almost always body text, not section titles.
        return False
    tl = t.lower()
    for h in _TITLE_HINTS:
        pos = tl.find(h)
        if pos == -1:
            continue
        # Heading-like: cue near the start, short line, or a table caption.
        if pos <= 12 or len(t) <= 48 or tl.startswith("טבלה"):
            return True
    return False


def question_wants_peek(question: str) -> bool:
    q = _norm(question).lower()
    return any(c in q for c in _QUESTION_CUES)


def _context_ids(expanded: list[ExpandedHit]) -> set[str]:
    out: set[str] = set()
    for ex in expanded:
        out.add(ex.match.id)
        for h in ex.neighbors:
            out.add(h.vector.id)
    return out


def peek_titles(
    vectors: list[IndexedVector],
    id_to_idx: dict[str, int],
    chunk_id: str,
    *,
    radius: int = 6,
    already: set[str] | None = None,
) -> list[TitleHit]:
    """Scan ±radius sequential chunks around `chunk_id`; return inferred titles."""
    parsed = _chunk_seq(chunk_id)
    if parsed is None:
        return []
    stem, n = parsed
    already = already or set()
    out: list[TitleHit] = []
    for delta in range(-radius, radius + 1):
        if delta == 0:
            continue
        nid = f"{stem}#{n + delta}"
        j = id_to_idx.get(nid)
        if j is None:
            continue
        v = vectors[j]
        title = title_from_text(v.text or "")
        if not title:
            continue
        out.append(
            TitleHit(
                chunk_id=nid,
                title=title,
                seq=n + delta,
                page=getattr(v.location, "page", None),
                interesting=is_interesting_title(title),
                already_in_context=nid in already,
            )
        )
    return out


def apply_title_peek(
    expanded: list[ExpandedHit],
    vectors: list[IndexedVector],
    id_to_idx: dict[str, int],
    question: str,
    *,
    radius: int = 6,
    max_anchors: int = 5,
    max_fetches: int = 6,
    always_interesting: bool = False,
) -> tuple[list[ExpandedHit], list[PeekResult]]:
    """Peek around top hits; fetch interesting titles not already in context.

    Mutates neighbor lists on the expanded hits (adds prev/next by seq).
    Returns (expanded, per-anchor peek diagnostics).
    """
    if not expanded:
        return expanded, []

    want = always_interesting or question_wants_peek(question)
    already = _context_ids(expanded)
    results: list[PeekResult] = []
    fetches_left = max_fetches
    # Prefer fetching titles that are interesting; de-dupe by chunk id.
    planned: list[tuple[ExpandedHit, TitleHit]] = []

    for ex in expanded[:max_anchors]:
        titles = peek_titles(
            vectors,
            id_to_idx,
            ex.match.id,
            radius=radius,
            already=already,
        )
        pr = PeekResult(anchor_id=ex.match.id, titles=titles)
        results.append(pr)
        if not want:
            continue
        for th in titles:
            if not th.interesting or th.already_in_context:
                continue
            planned.append((ex, th))

    # Stable: closer to anchor first, then by title.
    def _dist(th: TitleHit, anchor_id: str) -> int:
        p = _chunk_seq(anchor_id)
        return abs(th.seq - p[1]) if p else 999

    planned.sort(key=lambda pair: (_dist(pair[1], pair[0].match.id), pair[1].title))

    seen_fetch: set[str] = set()
    # Cap near-duplicate table/section titles so one PDF cannot fill the budget.
    title_fetch_counts: dict[str, int] = {}
    for ex, th in planned:
        if fetches_left <= 0:
            break
        if th.chunk_id in seen_fetch or th.chunk_id in already:
            continue
        title_key = re.sub(r"\s+", " ", th.title.lower())[:48]
        if title_fetch_counts.get(title_key, 0) >= 2:
            continue
        j = id_to_idx.get(th.chunk_id)
        if j is None:
            continue
        parsed_anchor = _chunk_seq(ex.match.id)
        parsed_t = _chunk_seq(th.chunk_id)
        if parsed_anchor is None or parsed_t is None:
            continue
        role = "prev" if parsed_t[1] < parsed_anchor[1] else "next"
        ex.neighbors.append(
            Hit(
                rank=ex.rank,
                score=ex.score,
                vector=vectors[j],
                role=role,
                anchor_id=ex.match.id,
            )
        )
        # Keep neighbors in document order for merged_text / chunks.
        ex.neighbors.sort(
            key=lambda h: (_chunk_seq(h.vector.id) or ("", 0))[1]
        )
        already.add(th.chunk_id)
        seen_fetch.add(th.chunk_id)
        title_fetch_counts[title_key] = title_fetch_counts.get(title_key, 0) + 1
        fetches_left -= 1
        for pr in results:
            if pr.anchor_id == ex.match.id:
                pr.fetched_ids.append(th.chunk_id)
                break

    return expanded, results


def peek_summary(results: list[PeekResult]) -> dict:
    """Compact JSON-serializable summary for answer records / logs."""
    return {
        "anchors": [
            {
                "anchor_id": r.anchor_id,
                "interesting": [
                    {
                        "id": t.chunk_id,
                        "title": t.title,
                        "page": t.page,
                        "in_context": t.already_in_context,
                    }
                    for t in r.titles
                    if t.interesting
                ],
                "fetched": r.fetched_ids,
            }
            for r in results
        ],
        "n_fetched": sum(len(r.fetched_ids) for r in results),
    }
