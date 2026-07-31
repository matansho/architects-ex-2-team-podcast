"""Query decomposition: split multi-subject questions before retrieval.

A single embedding for "what's the difference between Switch and regular
comprehensive?" averages both subjects into one vector, and the distinctive
named product dominates it. On dev-17 that produced 5 passages about Switch
against 2 about comprehensive, and the answer model filled the gap by inventing
the contrast instead of citing it.

Splitting gives each subject its own retrieval budget. Most questions are single
subject, so the prompt is biased hard toward returning one — a split that isn't
needed costs recall, because each sub-question gets a smaller slice of top-k.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rag.route import _strip_json_fences

DEFAULT_DECOMPOSE_MODEL = "google/gemma-3-27b-it"

# Hard ceiling regardless of what the model returns: top-k is split across
# sub-questions, so more than 3 leaves too few passages per subject.
MAX_SUBQUESTIONS = 3

DECOMPOSE_SYSTEM = """\
You decide whether a Hebrew insurance question needs to be split for search.

Corpus = ONE insurer (הראל). Prefer NOT splitting. Wrong splits hurt recall.

You may split ONLY when the customer names two or more RIVAL products/policies
and asks about both (comparison, or the same question about each). Examples of
rival pairs: הראל סוויץ' vs ביטוח מקיף רגיל; פוליסת שיר לעסק vs הראל ביט;
ביטוח בית משותף vs ביטוח אוסף אמנות.

Everything else is ONE subject — return the original question unchanged:
- catalog of a family + how to join one item in that family
  (e.g. "אילו ביטוחי משכנתא יש, ואיך מצטרפים לביטוח החיים למשכנתא?")
- several aspects of ONE product (cover, cost, join, phone, email, forms)
- unnamed "others" / competitors
- scenarios and yes/no questions

When you split: one sub-question per rival product; same language as the
original; keep ONLY what the customer asked — do not invent extra aspects.

Return ONLY valid JSON with this shape:
  {"split_kind": "rival_products" | "none",
   "subquestions": ["..."],
   "reason": "short reason"}

If split_kind is "none", subquestions must be a one-element list containing
the original question (or a trivial cleanup of it).
If split_kind is "rival_products", subquestions has 2 or 3 items, one per
named rival product.
"""

DECOMPOSE_USER = """\
Question:
{question}

Return the JSON object now.
"""


@dataclass
class DecomposeResult:
    subquestions: list[str]
    reason: str = ""
    raw: str = ""
    failed: bool = False  # parse/call failed → fell back to the original
    field_notes: list[str] = field(default_factory=list)

    @property
    def split(self) -> bool:
        return len(self.subquestions) > 1


def parse_decompose_json(
    raw: str,
    original: str,
    *,
    max_n: int = MAX_SUBQUESTIONS,
) -> tuple[list[str], str, list[str]]:
    """Parse decomposer JSON → (subquestions, reason, notes).

    Only `split_kind == "rival_products"` may yield multiple sub-questions.
    Any other kind (or a missing/unknown kind with multiple items) collapses
    to the original question — Gemma has repeatedly split catalog+join and
    same-product aspects despite prompt text forbidding it.
    """
    data = json.loads(_strip_json_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("decompose JSON must be an object")
    subs = data.get("subquestions")
    if isinstance(subs, str):
        subs = [subs]
    if not isinstance(subs, list):
        raise ValueError("subquestions must be a list")

    out: list[str] = []
    seen: set[str] = set()
    for s in subs:
        text = " ".join(str(s).split())
        # Too short to retrieve on; usually a fragment or a stray label.
        if len(text) < 8:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)

    reason = str(data.get("reason") or "")
    notes: list[str] = []
    kind = str(data.get("split_kind") or "").strip().lower().replace("-", "_")
    if not out:
        return [original.strip()], reason, notes

    # Only an explicit rival-products label may produce a multi-query split.
    # Anything else collapses to the original — Gemma has repeatedly emitted
    # multiple subquestions for catalog+join / same-product aspects.
    if kind != "rival_products":
        if len(out) > 1:
            notes.append(f"collapsed non-rival split (split_kind={kind!r})")
        return [original.strip()], reason, notes

    return out[:max_n], reason, notes


def decompose_question(
    question: str,
    *,
    model: str = DEFAULT_DECOMPOSE_MODEL,
    max_n: int = MAX_SUBQUESTIONS,
    complete=None,
) -> DecomposeResult:
    """Split `question` into single-subject sub-questions.

    Never raises: any failure returns the original question unchanged with
    `failed=True`, so the caller keeps the current single-query behaviour.
    """
    original = question.strip()
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": DECOMPOSE_USER.format(question=original)},
    ]

    def _call() -> str:
        import os

        import litellm

        kwargs: dict = {}
        base = os.environ.get("OPENAI_BASE_URL")
        m = model
        if base:
            kwargs["api_base"] = base
            m = f"openai/{model.removeprefix('openai/')}"
        elif "/" not in model:
            m = f"openai/{model}"
        resp = litellm.completion(
            model=m, messages=messages, temperature=0, timeout=60, num_retries=0, **kwargs
        )
        return resp.choices[0].message.content or ""

    try:
        raw = complete(messages) if complete is not None else _call()
        subs, reason, notes = parse_decompose_json(raw, original, max_n=max_n)
    except Exception as e:  # noqa: BLE001 - decomposition is best-effort
        return DecomposeResult(
            subquestions=[original],
            reason=f"decompose failed: {e!r}",
            raw="",
            failed=True,
        )

    # A "split" that just echoes the question back twice buys nothing but costs
    # each copy half the budget.
    if len(subs) > 1 and all(s.strip() == original for s in subs):
        notes.append("degenerate split (all copies of original)")
        subs = [original]

    return DecomposeResult(
        subquestions=subs, reason=reason, raw=raw, failed=False, field_notes=notes
    )


def split_budget(total: int, n: int, *, minimum: int = 1) -> list[int]:
    """Split a retrieval budget across `n` sub-questions, remainder to the first.

    Keeps the summed budget constant so decomposition does not silently inflate
    cross-encoder work or the size of the answer context.
    """
    if n <= 1:
        return [total]
    base, rem = divmod(max(total, n * minimum), n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _hit_key(h) -> object:
    """Identity for dedup: the matched chunk id, falling back to object id."""
    match = getattr(h, "match", None)
    return getattr(match, "id", None) or getattr(h, "id", None) or id(h)


def merge_hits(per_sub, *, top_k: int):
    """Round-robin interleave per-sub-question hits, dropping duplicates.

    Round-robin rather than concatenation so that when the subjects overlap the
    survivors stay balanced across subjects, instead of the first sub-question
    consuming the whole budget. Ranks are renumbered over the merged list.
    """
    out = []
    seen: set = set()
    for tier in range(max((len(h) for h in per_sub), default=0)):
        for hits in per_sub:
            if tier >= len(hits):
                continue
            h = hits[tier]
            key = _hit_key(h)
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= top_k:
                break
        if len(out) >= top_k:
            break
    for rank, h in enumerate(out, start=1):
        if hasattr(h, "rank"):
            h.rank = rank
    return out
