"""Drop / scrub passages that look like OCR/extraction garbage before answer gen.

Keeps the answer LLM from treating DummyText placeholders and similar junk as
citeable policy facts (seen when forms hallucinate caps the judge rejects).
"""

from __future__ import annotations

import re

from rag.retrieve import ExpandedHit, Hit

# Explicit extractor/placeholder markers (avoid bare "name" — matches English prose).
_PLACEHOLDER_RE = re.compile(
    r"(?i)("
    r"\bDummyText\b|\bDummy\s*Text\b|\bLorem\s+ipsum\b|\bplaceholder\b|"
    r"\bloremipsum\b|TODO:\s*text|\[insert[^\]]*\]|<<name>>|<name>"
    r")"
)

# Replacement char / private-use spam from broken PDF extractors.
_MOJIBAKE_RE = re.compile(r"[\ufffd\uFFFE\uFFFF]")


def passage_text_is_garbage(text: str | None) -> bool:
    """True if chunk text should not be shown to the answer model as evidence."""
    t = (text or "").strip()
    if not t:
        return True
    if _PLACEHOLDER_RE.search(t):
        return True
    # Dense replacement characters → unreadable extract.
    if len(_MOJIBAKE_RE.findall(t)) >= 3:
        return True
    # Symbol soup with almost no letters/digits (keep numeric ₪ tables).
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 24:
        alnum = sum(ch.isalnum() for ch in compact)
        if alnum / len(compact) < 0.15:
            return True
    return False


def sanitize_passages_for_answer(
    expanded: list[ExpandedHit],
) -> tuple[list[ExpandedHit], dict]:
    """Filter garbage primaries/neighbors; return cleaned list + drop stats."""
    kept: list[ExpandedHit] = []
    dropped_primary = 0
    dropped_neighbors = 0
    for ex in expanded:
        if passage_text_is_garbage(ex.match.text):
            dropped_primary += 1
            continue
        clean_neighbors: list[Hit] = []
        for h in ex.neighbors:
            if passage_text_is_garbage(h.vector.text):
                dropped_neighbors += 1
                continue
            clean_neighbors.append(h)
        if clean_neighbors is ex.neighbors:
            kept.append(ex)
        else:
            kept.append(
                ExpandedHit(
                    rank=ex.rank,
                    score=ex.score,
                    match=ex.match,
                    neighbors=clean_neighbors,
                )
            )
    stats = {
        "dropped_primary": dropped_primary,
        "dropped_neighbors": dropped_neighbors,
        "kept": len(kept),
        "input": len(expanded),
    }
    return kept, stats
