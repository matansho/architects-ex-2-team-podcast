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
    "Keep answers short: only what was asked — no extra procedures, hours, "
    "phone numbers, waiting periods, or caps the customer did not ask about. "
    "Once you have stated the figure/rule that answers the question, stop. "
    "If the question is underspecified but the context lists distinct options "
    "(plans, tracks, policy types), briefly cover each relevant option — do not "
    "stop at asking for more details when those options are already in context. "
    "If a named activity is explicitly excepted from an exclusion list, treat it "
    "as covered. "
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
    "How to answer:\n"
    "- Answer only what was asked. Do not add related procedures, hours, phone "
    "numbers, waiting periods, aggregate caps, or side conditions the question "
    "did not ask for. Once you have stated the figure/rule that answers the "
    "question, stop — do not append further policy limits.\n"
    "- Every specific checkable fact you state MUST be supported by a passage "
    "you list in USED_PASSAGES. If you cannot point to a passage, omit the fact.\n"
    "- If the question is underspecified but the context lists distinct options "
    "(plans, tracks, ages, codes, diagnoses), briefly cover each relevant option "
    "from the context. Prefer \"זה תלוי ב…\" + short bullets over asking for more "
    "details when the options are already in the passages.\n"
    "- Prefer a short conditional (\"זה תלוי בנספח/הגדרה…\") over a flat כן/לא when "
    "eligibility depends on a rider, schedule endorsement, or definition.\n"
    "- Read exclusions carefully: if an activity/item is explicitly carved out "
    "from an exclusion list (e.g. \"למעט ג'ודו\", \"למעט…\"), it IS covered — do not "
    "treat it as excluded.\n"
    "- Before saying a loss type is not covered (e.g. lost profits / consequential "
    "loss under third-party liability), check whether the context includes that "
    "loss under liability or property damage. Do not refuse if the answer is there.\n"
    "- Match the product the customer named. If passages mix several products, "
    "prefer the one matching the question wording; if still ambiguous, cover the "
    "matching options separately.\n\n"
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
    "Example 4 — underspecified question (cover the options from context):\n"
    "זה תלוי במסלול: במסלול משלים שב״ן עם השתתפות עצמית — 5,000 ש״ח; "
    "במסלול מהשקל הראשון / משלים שב״ן ללא השתתפות — אין השתתפות עצמית.\n"
    "USED_PASSAGES: 1, 3\n\n"
    "Example 5 — carve-out from an exclusion (covered):\n"
    "כן, ג'ודו מכוסה: אומנויות לחימה מוחרגות כספורט אתגרי, אך ג'ודו מוחרג "
    "במפורש מרשימת ההחרגות ולכן הכיסוי חל.\n"
    "USED_PASSAGES: 2\n\n"
    "Example 6 — English question:\n"
    "Yes. Third-party liability is included up to $150,000 with no extra premium.\n"
    "USED_PASSAGES: 1, 2\n\n"
    "Wrong (do NOT do this):\n"
    "- Putting paths in the answer: מקורות: car/pages/foo.txt\n"
    "- Using brackets in the footer: USED_PASSAGES: [1], [2]\n"
    "- Omitting the USED_PASSAGES line\n"
    "- Writing USED_PASSAGES before the answer\n"
    "- Padding with extra facts the customer did not ask about\n"
    "- Picking one plan/track as certain when several options appear in context\n"
    "- Saying only \"חסר מידע\" when the context already lists the possible options\n"
    "- Treating an activity as excluded when the text says it is excepted from "
    "the exclusion list"
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
            "(or USED_PASSAGES: none if you cannot answer from the context).\n\n"
            "Reminders: cover all relevant options if the question is underspecified; "
            "do not add caps/hours/procedures not needed to answer; if a named "
            "activity is excepted from an exclusion list in a passage, treat it as covered."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
