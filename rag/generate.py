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
    "Keep answers short and complete: answer only what was asked. "
    "Use at most 2 short paragraphs, or up to 3 short bullets only when the "
    "question clearly asks to compare options. "
    "Do not restate the question or add extra procedures, hours, phone "
    "numbers, waiting periods, aggregate caps, or side conditions the "
    "customer did not ask about. Once you have stated the figure/rule that "
    "answers the question, stop. "
    "Cover every part of a multi-part question (amounts, caps, timing, who is paid, "
    "conditions that decide yes/no) when those facts appear in the context — then "
    "stop. "
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
    "- Keep the answer short and complete: use at most 2 short paragraphs, or "
    "up to 3 short bullets only when the question clearly asks to compare "
    "options. Do not restate the question.\n"
    "- Answer only what was asked. Do not add related procedures, hours, phone "
    "numbers, waiting periods, aggregate caps, or side conditions the question "
    "did not ask for. Once you have stated the figure/rule that answers the "
    "question, stop.\n"
    "- Scope: answer every clause of the customer's question (amounts, caps, "
    "timing/effective date, who is paid, purchase prerequisites, and conditions "
    "that decide כן/לא) when those facts are in the passages. Then stop.\n"
    "- Completeness check: before finishing, re-read the question and confirm each "
    "asked sub-point is answered or explicitly marked as missing from the context. "
    "Do not stop after the first figure if the question also asks for a condition, "
    "cap, beneficiary, or effective-date rule present in the passages.\n"
    "- Every specific checkable fact you state MUST be supported by a passage "
    "you list in USED_PASSAGES. If you cannot point to a passage, omit the fact. "
    "Never invent exceptions, age cutoffs, refund destinations, purchase "
    "prerequisites, or medical carve-outs. Do not reuse a condition from a "
    "different section/coverage as if it applied here.\n"
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
    "- Include every passage [N] that supports a fact in your answer.\n"
    "- If you name a product, form, page, limit, or joining path from the "
    "context, its supporting [N] MUST appear in USED_PASSAGES — do not state "
    "it from memory of the brief while citing only related passages.\n"
    "- Use only numbers that appear as [N] labels in the context.\n"
    "- Each [N] block may list primary + neighbor chunk ids/pages — pick the [N] "
    "whose text (including neighbors) actually states the fact. Do not cite a "
    "different [N] that is merely related (e.g. boilers vs garden extension).\n"
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
    "- Answering only the first number when the question also asks who is paid, "
    "a time limit, a cap, or a purchase prerequisite that appears in context\n"
    "- Inventing exceptions or rules not stated in the cited passages\n"
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
    """Map 1-based passage labels to `{file, page}`.

    Includes the primary hit and its attached neighbors (same expanded block),
    so citing [N] can surface the page that actually holds a windowed clause.
    """
    locs: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    n = len(expanded)
    for i in indices:
        if not (1 <= i <= n):
            continue
        for chunk in expanded[i - 1].chunks:
            loc = chunk.location
            key = (loc.file, loc.page)
            if key in seen:
                continue
            seen.add(key)
            locs.append(key)
    return citations_from_locations(
        locs, max_citations=max_citations, corpus_root=corpus_root
    )


def format_passage(ex: ExpandedHit, *, index: int) -> str:
    """One expanded hit as a labeled context block (primary + neighbors)."""
    from rag.peek import title_from_text

    loc = ex.match.location
    pages = sorted(
        {
            c.location.page
            for c in ex.chunks
            if c.location.page is not None
        }
        | ({loc.page} if loc.page is not None else set())
    )
    if len(pages) == 1:
        page_bit = f", page {pages[0]}"
    elif pages:
        page_bit = f", pages {pages[0]}-{pages[-1]}"
    else:
        page_bit = ""
    header = f"[{index}] {loc.file}{page_bit} (score={ex.score:.3f})"
    # Explicit sub-chunk map so the model cites the block that holds the fact.
    sub: list[str] = []
    for c in ex.chunks:
        role = "primary" if c.id == ex.match.id else "neighbor"
        title = title_from_text(c.text or "")
        sub.append(
            f"  ({role}) id={c.id} page={c.location.page}"
            + (f" — {title}" if title else "")
        )
    map_block = "\n".join(sub)
    body = ex.merged_text().strip()
    parts = [header]
    if map_block:
        parts.append(map_block)
    if body:
        parts.append(body)
    return "\n".join(parts)


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
            "Reminders: keep the answer short and complete; cover all relevant "
            "options if the question is underspecified; do not add caps/hours/"
            "procedures not needed to answer; if a named activity is excepted "
            "from an exclusion list in a passage, treat it as covered."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
