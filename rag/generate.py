"""Grounded answer generation from retrieved chunks (no citations yet)."""

from __future__ import annotations

from rag.retrieve import ExpandedHit

SYSTEM_NO_CITE = (
    "You are a customer-support assistant for Harel Insurance (Israel). "
    "Answer the customer's question using ONLY the provided context passages. "
    "Answer in the language of the question. "
    "Be direct: start yes/no questions with כן or לא when appropriate. "
    "For numeric questions, state the exact number, limit, or period. "
    "Do not cite sources, document paths, page numbers, or a מקורות section. "
    "If the context does not contain enough information for a specific fact, "
    "say clearly that you do not have enough information — do not guess or "
    "use outside knowledge."
)


def format_passage(ex: ExpandedHit, *, index: int) -> str:
    """One expanded hit as a labeled context block."""
    loc = ex.match.location
    page = loc.page
    pages = sorted(set(loc.pages or []) | ({page} if page is not None else set()))
    page_bit = f", page {pages[0]}" if len(pages) == 1 else (
        f", pages {pages[0]}-{pages[-1]}" if pages else ""
    )
    header = f"[{index}] {loc.file}{page_bit} (score={ex.score:.3f})"
    body = ex.merged_text().strip()
    return f"{header}\n{body}" if body else header


def build_context(
    expanded: list[ExpandedHit],
    *,
    top_k: int | None = None,
) -> str:
    hits = expanded[:top_k] if top_k is not None else expanded
    blocks = [format_passage(ex, index=i + 1) for i, ex in enumerate(hits)]
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str,
    context: str,
    *,
    system_prompt: str = SYSTEM_NO_CITE,
) -> list[dict]:
    user = (
        "Context passages from Harel insurance documents:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Customer question:\n{question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
