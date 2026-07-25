"""Grounded answer generation from retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.retrieve import ExpandedHit, citations_from_locations

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

# Passage-index citing: answer stays path-free; model names which [N] blocks it used.
SYSTEM_PASSAGE_CITE = (
    "You are a customer-support assistant for Harel Insurance (Israel). "
    "Answer the customer's question using ONLY the provided context passages. "
    "Each passage is labeled [1], [2], [3], … at the start of its header.\n\n"
    "Answer in the language of the question. "
    "Be direct: start yes/no questions with כן or לא when appropriate. "
    "For numeric questions, state the exact number, limit, or period. "
    "Do not write document paths, file names, page numbers, or a מקורות section "
    "in the answer body.\n\n"
    "After the answer, on its own final line, list the passage numbers you actually "
    "relied on, using exactly this format (comma-separated integers, no brackets):\n"
    "USED_PASSAGES: 1, 3\n\n"
    "Rules for USED_PASSAGES:\n"
    "- Include every passage that supports a fact in your answer.\n"
    "- Use only numbers that appear as [N] labels in the context.\n"
    "- Prefer the fewest passages that establish the answer (usually 1–3).\n"
    "- If you refuse for lack of information, use: USED_PASSAGES: none\n"
    "- Put USED_PASSAGES on the last line by itself — nothing after it.\n\n"
    "Examples of the FULL reply format (follow this shape exactly):\n\n"
    "Example 1 — yes/no with one supporting passage:\n"
    "לא, הגשת תביעה לגוף מוסדי אינה עוצרת את מרוץ ההתיישנות. "
    "רק הגשת תביעה לבית משפט עוצרת אותו.\n"
    "USED_PASSAGES: 2\n\n"
    "Example 2 — numeric answer using two passages:\n"
    "העלות החודשית לגילאי 31–40 היא 59.12 ש״ח, כלומר 709.44 ש״ח לשנה.\n"
    "USED_PASSAGES: 1, 5\n\n"
    "Example 3 — refusal when context is insufficient:\n"
    "אין בידי מספיק מידע במסמכים שסופקו כדי לענות על השאלה.\n"
    "USED_PASSAGES: none\n\n"
    "Example 4 — English question:\n"
    "Yes. Third-party liability is included up to $150,000 with no extra premium.\n"
    "USED_PASSAGES: 1, 2\n\n"
    "Wrong (do NOT do this):\n"
    "- Putting paths in the answer: מקורות: car/pages/foo.txt\n"
    "- Using brackets in the footer: USED_PASSAGES: [1], [2]\n"
    "- Omitting the USED_PASSAGES line\n"
    "- Writing USED_PASSAGES before the answer"
)

_USED_PASSAGES_RE = re.compile(
    r"(?im)^\s*USED_PASSAGES\s*:\s*(.+?)\s*$"
)


@dataclass
class ParsedGeneration:
    answer: str
    passage_indices: list[int]  # 1-based; empty if none/unparseable
    used_passages_raw: str | None
    parse_ok: bool


def parse_answer_with_passages(raw: str) -> ParsedGeneration:
    """Split model output into answer body + USED_PASSAGES indices."""
    text = (raw or "").strip()
    if not text:
        return ParsedGeneration("", [], None, False)

    matches = list(_USED_PASSAGES_RE.finditer(text))
    if not matches:
        return ParsedGeneration(text, [], None, False)

    last = matches[-1]
    raw_list = last.group(1).strip()
    answer = text[: last.start()].rstrip()

    if re.fullmatch(r"(?i)none|n/a|no|אין|ללא", raw_list):
        return ParsedGeneration(answer, [], raw_list, True)

    indices: list[int] = []
    seen: set[int] = set()
    for part in re.split(r"[,;\s]+", raw_list):
        part = part.strip().strip("[]().")
        if not part:
            continue
        if not part.isdigit():
            continue
        n = int(part)
        if n < 1 or n in seen:
            continue
        seen.add(n)
        indices.append(n)

    ok = bool(indices) or bool(
        re.fullmatch(r"(?i)none|n/a|no|אין|ללא", raw_list)
    )
    # Digits-only list that parsed empty → not ok
    if not indices and not re.fullmatch(r"(?i)none|n/a|no|אין|ללא", raw_list):
        ok = False
    return ParsedGeneration(answer, indices, raw_list, ok)


def citations_from_passage_indices(
    expanded: list[ExpandedHit],
    indices: list[int],
    *,
    max_citations: int = 5,
    corpus_root: str | None = "corpus",
) -> list[dict[str, str | int | None]]:
    """Map 1-based passage labels to `{file, page}` (context order = expanded order)."""
    locs: list[tuple[str, int | None]] = []
    n = len(expanded)
    for i in indices:
        if 1 <= i <= n:
            loc = expanded[i - 1].match.location
            locs.append((loc.file, loc.page))
    return citations_from_locations(
        locs, max_citations=max_citations, corpus_root=corpus_root
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
        f"Customer question:\n{question}\n\n"
    )
    if system_prompt == SYSTEM_PASSAGE_CITE or "USED_PASSAGES" in system_prompt:
        user += (
            "Reply with your answer, then a final line exactly like:\n"
            "USED_PASSAGES: 1, 3\n"
            "(or USED_PASSAGES: none if you cannot answer from the context)."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
