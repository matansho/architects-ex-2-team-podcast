"""Post-answer verifier: question-aware LLM edits, then safe strip (no regen).

Design (v3 — LLM-first):
1. Soft gate — skip empty / refusals / tiny one-liners
2. Sentence-level LLM pass — drop only *ancillary* content the customer did not ask for
   (and that is unsupported or clearly a side product/limit). Never drop the core answer.
3. Apply drops by sentence index (verbatim units we numbered for the model)
4. covers_question safety net — revert if the edit breaks the ask
5. Optional deterministic aside trim (`apply_deterministic`) — legacy hack, off by default
6. Hard timeout — on error keep answer as-is
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import litellm

from rag.retrieve import ExpandedHit

# Prefer the answer-class model for judgment quality; override with --verify-model.
DEFAULT_VERIFY_MODEL = "moonshotai/Kimi-K3"
DEFAULT_VERIFY_TIMEOUT_S = 45.0
DEFAULT_MAX_TOKENS = 700
DEFAULT_VERIFY_MODE: Literal["extras", "strict"] = "extras"
DEFAULT_VERIFY_REASONING_EFFORT = "low"

# --- Legacy deterministic helpers (opt-in; not the main path) -----------------

_ASIDE_PREFIX = re.compile(
    r"^(?:"
    r"שים לב(?:\s*:)?|"
    r"לידיעתך(?:\s*:)?|"
    r"לתשומת לבך(?:\s*:)?|"
    r"ראוי לציין(?:\s*:)?|"
    r"הערה(?:\s*:)?|"
    r"בנוסף(?:\s*לכך)?(?:\s*:)?|"
    r"חשוב לציין(?:\s*:)?|"
    r"יצוין(?:\s*:)?|"
    r"note(?:\s*:)?"
    r")\s*",
    re.IGNORECASE,
)
_ASIDE_INLINE = re.compile(
    r"(?:^|[\n.])\s*(?:שים לב|לידיעתך|לתשומת לבך|ראוי לציין)\s*:",
    re.IGNORECASE,
)
_NUM_RE = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
)
_INLINE_ASIDE_SPLIT = re.compile(
    r"(?<=[.!?。…\n])\s*(?=(?:שים לב|לידיעתך|לתשומת לבך|ראוי לציין|חשוב לציין)\s*:)",
    re.IGNORECASE,
)

VERIFY_SYSTEM = """\
You are a surgical editor for a Hebrew insurance support chatbot.

You receive:
1) the customer QUESTION
2) a DRAFT answer, already split into numbered units S1..Sn
3) EVIDENCE passages (may be incomplete)

Your job: decide which units to DROP. A unit may be dropped ONLY if it is
ANCILLARY to the question — a different product, an unasked cap/age/phone,
or a "by the way" note about something the customer did not ask — AND it is
either unsupported by the evidence OR clearly about a different product/limit.

KEEP units that qualify or complete the asked answer, even if phrased as a
caveat (e.g. Q about limitation period → KEEP "only a court filing stops the
clock"; Q about a payout → KEEP the amount/conditions). Drop side notes about
other policies or unasked purchase limits.

NEVER drop a unit that is needed to answer the question, including:
- yes/no to the ask
- amounts, dates, conditions, procedures the question seeks
- the only sentence that states the main conclusion
Even if that unit looks poorly grounded, KEEP it (do not invent a replacement).

When evidence is thin/missing: still drop only clearly unasked ancillary units;
do not drop core answer units just because evidence is incomplete.

Return ONLY JSON:
{
  "question_ask": "short English paraphrase of what the customer wants",
  "keep_summary": "short English: what the draft's core answer is",
  "drop": [
    {"id": "S3", "why": "unasked side product / unsupported cap / ..."}
  ]
}

If nothing should be removed: {"question_ask":"...","keep_summary":"...","drop":[]}
Do not rewrite units. Do not invent new text. ids must be from the list given.
"""

VERIFY_SYSTEM_STRICT = """\
You are a grounding checker for a Hebrew insurance support chatbot.

You receive a QUESTION, a DRAFT answer split into S1..Sn, and EVIDENCE passages.

Drop a unit ONLY when BOTH are true:
1) It makes a checkable factual claim NOT supported by the evidence, AND
2) Removing it does not remove the direct answer to the question.

Prefer dropping ancillary asides over core answer units. If unsure, KEEP.

Return ONLY JSON:
{
  "question_ask": "short paraphrase",
  "keep_summary": "core answer kept",
  "drop": [{"id": "S2", "why": "unsupported: ..."}]
}
If nothing to drop: {"question_ask":"...","keep_summary":"...","drop":[]}
"""


@dataclass
class VerifyResult:
    ran: bool
    gated: bool
    gate_reason: str
    skipped: bool = False
    skip_reason: str | None = None
    unsupported: list[str] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)
    deterministic_stripped: list[str] = field(default_factory=list)
    drop_ids: list[str] = field(default_factory=list)
    question_ask: str | None = None
    keep_summary: str | None = None
    answer_before: str = ""
    answer_after: str = ""
    model: str | None = None
    mode: str = DEFAULT_VERIFY_MODE
    latency_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    reverted: bool = False
    revert_reason: str | None = None


def resolve_verify_model(
    model: str,
    *,
    reasoning_effort: str | None = DEFAULT_VERIFY_REASONING_EFFORT,
) -> tuple[str, dict]:
    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"
    # Nebius OpenAI-compat: Kimi rejects top-level reasoning_effort; use extra_body.
    if reasoning_effort and "kimi" in model.lower():
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    return model, kwargs


def extract_numbers(text: str) -> set[str]:
    out: set[str] = set()
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        if not raw:
            continue
        out.add(raw)
        if "." in raw:
            out.add(raw.rstrip("0").rstrip("."))
    return out


def has_aside_marker(answer: str) -> bool:
    ans = answer or ""
    if _ASIDE_INLINE.search(ans):
        return True
    for line in ans.splitlines():
        if _ASIDE_PREFIX.match(line.strip()):
            return True
    return False


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•")) or re.match(r"^\d+[\).]", line):
            parts.append(line)
            continue
        chunks = re.split(r"(?<=[.!?。…])\s+|(?<=[;؛])\s+", line)
        for c in chunks:
            c = c.strip()
            if c:
                parts.append(c)
    return parts


def split_units(answer: str) -> list[str]:
    """Paragraph/sentence units, with inline caveat clauses split out.

    Splitting caveat tails into their own unit lets the LLM drop them without
    also dropping a needed procedure that shares the same paragraph.
    """
    ans = (answer or "").strip()
    if not ans:
        return []
    if "\n\n" in ans:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", ans) if b.strip()]
        raw = blocks if len(blocks) >= 2 else split_sentences(ans)
    else:
        raw = split_sentences(ans)

    out: list[str] = []
    for u in raw:
        core, aside = _split_inline_aside(u)
        if aside and core:
            out.append(core)
            out.append(aside)
        elif aside and not core:
            out.append(aside)
        elif core:
            out.append(core)
        else:
            out.append(u)
    return out


def is_aside_unit(unit: str) -> bool:
    u = (unit or "").strip()
    if not u:
        return False
    if _ASIDE_PREFIX.match(u):
        return True
    first = u.splitlines()[0].strip()
    return bool(_ASIDE_PREFIX.match(first))


def should_verify(
    question: str,
    answer: str,
    *,
    passage_indices: list[int] | None = None,
    n_citations: int = 0,
    used_tools: bool = False,
) -> tuple[bool, str]:
    """Soft gate: skip empties / refusals / tiny single-clause answers."""
    del passage_indices, n_citations, used_tools
    ans = (answer or "").strip()
    if not ans:
        return False, "empty_answer"
    if re.search(r"אין בידי|לא מספיק מידע|אין מספיק", ans):
        return False, "refusal"
    units = split_units(ans)
    if len(units) >= 2:
        return True, f"multi_unit:{len(units)}"
    if len(ans) >= 180:
        return True, "long_single_unit"
    # Still worth a pass when the one unit mixes core + caveat.
    if has_aside_marker(ans):
        return True, "aside_in_single_unit"
    return False, "low_risk"


def _split_inline_aside(unit: str) -> tuple[str, str | None]:
    u = (unit or "").strip()
    if not u:
        return unit, None
    if is_aside_unit(u):
        return "", u
    parts = _INLINE_ASIDE_SPLIT.split(u, maxsplit=1)
    if len(parts) < 2:
        m = re.search(
            r"\s*((?:שים לב|לידיעתך|לתשומת לבך|ראוי לציין|חשוב לציין)\s*:.*)$",
            u,
            flags=re.IGNORECASE | re.S,
        )
        if not m:
            return u, None
        return u[: m.start()].rstrip(), m.group(1).strip()
    return parts[0].strip(), parts[1].strip()


def strip_unasked_asides(question: str, answer: str) -> tuple[str, list[str]]:
    """Legacy opt-in: remove aside tails with numbers not present in the question."""
    q_nums = extract_numbers(question)
    units = split_units(answer)
    if not units:
        return answer, []

    kept: list[str] = []
    removed: list[str] = []
    for u in units:
        core, aside = _split_inline_aside(u)
        if aside:
            a_nums = extract_numbers(aside)
            if a_nums and any(n not in q_nums for n in a_nums):
                removed.append(aside)
                if core:
                    kept.append(core)
                continue
        if core:
            kept.append(core)
        elif u and not aside:
            kept.append(u)

    if not removed or not kept:
        return answer, []
    if "\n\n" in (answer or ""):
        new = "\n\n".join(kept)
    elif any(x.startswith(("-", "*", "•")) or re.match(r"^\d+[\).]", x) for x in kept):
        new = "\n".join(kept)
    else:
        new = "\n\n".join(kept) if len(kept) > 1 else kept[0]
    return new.strip(), removed


def cited_passage_text(
    passages: list[ExpandedHit],
    indices: list[int] | None,
    *,
    max_chars: int = 10000,
    include_neighbors: bool = True,
) -> str:
    if not passages:
        return "(no evidence passages provided)"
    cited = [i for i in (indices or []) if 1 <= i <= len(passages)]
    if not cited:
        cited = list(range(1, min(len(passages), 4) + 1))

    ordered: list[int] = []
    seen: set[int] = set()
    for i in cited:
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    if include_neighbors:
        for i in range(1, len(passages) + 1):
            if i not in seen:
                ordered.append(i)
                seen.add(i)

    blocks: list[str] = []
    total = 0
    for i in ordered:
        ex = passages[i - 1]
        body = ex.merged_text().strip()
        if len(body) > 2000:
            body = body[:2000] + "…"
        block = f"[{i}] {ex.match.location.file} p{ex.match.location.page}\n{body}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks) if blocks else "(no evidence passages provided)"


def format_units_for_prompt(units: list[str]) -> str:
    lines = []
    for i, u in enumerate(units, 1):
        lines.append(f"S{i}: {u}")
    return "\n\n".join(lines)


def _parse_drop_response(raw: str, n_units: int) -> tuple[list[int], str | None, str | None]:
    """Return 0-based unit indices to drop, plus optional summaries."""
    text = (raw or "").strip()
    if not text:
        return [], None, None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return [], None, None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [], None, None
    if not isinstance(data, dict):
        return [], None, None

    ask = data.get("question_ask")
    keep = data.get("keep_summary")
    ask_s = ask.strip() if isinstance(ask, str) else None
    keep_s = keep.strip() if isinstance(keep, str) else None

    drop_idxs: list[int] = []
    drops = data.get("drop") or data.get("unsupported") or []
    if not isinstance(drops, list):
        return [], ask_s, keep_s

    for item in drops:
        if isinstance(item, dict):
            sid = item.get("id") or item.get("span") or item.get("unit")
        else:
            sid = item
        if not isinstance(sid, str):
            continue
        sid = sid.strip()
        m = re.match(r"[Ss](\d+)$", sid)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n_units:
                drop_idxs.append(i)
            continue
        # Fallback: treat as verbatim span — match containing unit
        # (handled by caller via unsupported list path)
    # unique preserve order
    seen: set[int] = set()
    out: list[int] = []
    for i in drop_idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out, ask_s, keep_s


def _parse_unsupported_spans(raw: str) -> list[str]:
    """Fallback parser for legacy {"unsupported": [...]} span lists."""
    text = (raw or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, dict):
        return []
    uns = data.get("unsupported") or data.get("unsupported_spans") or []
    if not isinstance(uns, list):
        return []
    # Also collect drop entries that used span instead of id
    drops = data.get("drop") or []
    out: list[str] = []
    for u in uns:
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    if isinstance(drops, list):
        for item in drops:
            if isinstance(item, dict):
                span = item.get("span")
                if isinstance(span, str) and span.strip() and not re.match(
                    r"[Ss]\d+$", span.strip()
                ):
                    out.append(span.strip())
    return out


def filter_protected_spans(
    question: str,
    answer: str,
    unsupported: list[str],
) -> list[str]:
    if not unsupported:
        return []
    q_nums = extract_numbers(question)
    units = split_units(answer)
    core = units[0] if units else answer
    out: list[str] = []
    for span in unsupported:
        span = span.strip()
        if not span:
            continue
        if extract_numbers(span) & q_nums:
            continue
        if core and (span in core or core in span) and not is_aside_unit(core):
            # Allow drop only if there are other units and this isn't the sole core
            if len(units) <= 1:
                continue
        if re.match(r"^(כן|לא)\b", span) and not is_aside_unit(span):
            continue
        out.append(span)
    return out


def strip_unsupported(answer: str, unsupported: list[str]) -> tuple[str, list[str]]:
    if not unsupported or not answer:
        return answer, []
    sents = split_sentences(answer)
    if not sents:
        return answer, []
    removed: list[str] = []
    kept: list[str] = []
    for s in sents:
        drop = False
        for span in unsupported:
            span = span.strip()
            if not span:
                continue
            if span in s or s in span:
                drop = True
                break
            if len(span) >= 12 and span[:20] in s:
                drop = True
                break
        if drop:
            removed.append(s)
        else:
            kept.append(s)
    if not removed or not kept:
        return answer, []
    if any(x.startswith(("-", "*", "•")) or re.match(r"^\d+[\).]", x) for x in kept):
        new = "\n".join(kept)
    elif "\n\n" in answer:
        new = "\n\n".join(kept)
    else:
        new = " ".join(kept)
    return new.strip(), removed


def drop_units_by_index(answer: str, units: list[str], drop_idxs: list[int]) -> tuple[str, list[str]]:
    if not drop_idxs:
        return answer, []
    drop_set = set(drop_idxs)
    # Never drop everything
    if len(drop_set) >= len(units):
        return answer, []
    kept = [u for i, u in enumerate(units) if i not in drop_set]
    removed = [u for i, u in enumerate(units) if i in drop_set]
    if not kept:
        return answer, []
    if "\n\n" in (answer or ""):
        new = "\n\n".join(kept)
    elif any(x.startswith(("-", "*", "•")) or re.match(r"^\d+[\).]", x) for x in kept):
        new = "\n".join(kept)
    else:
        new = "\n\n".join(kept) if len(kept) > 1 else kept[0]
    return new.strip(), removed


def _polarity(text: str) -> str | None:
    t = (text or "").lstrip()
    if t.startswith("כן"):
        return "yes"
    if t.startswith("לא"):
        return "no"
    return None


def covers_question(question: str, before: str, after: str) -> tuple[bool, str]:
    after = (after or "").strip()
    before = (before or "").strip()
    if not after:
        return False, "empty_after"
    if len(after) < max(40, int(0.35 * len(before))):
        return False, "too_short"
    q_nums = extract_numbers(question)
    if q_nums:
        needed = q_nums & extract_numbers(before)
        missing = needed - extract_numbers(after)
        if missing:
            return False, f"missing_q_numbers:{','.join(sorted(missing))}"
    pol_b, pol_a = _polarity(before), _polarity(after)
    if pol_b and pol_a and pol_b != pol_a:
        return False, "polarity_flip"
    # Must keep at least one unit
    if not split_units(after):
        return False, "no_units"
    return True, "ok"


def verify_and_strip(
    question: str,
    answer: str,
    passages: list[ExpandedHit],
    *,
    passage_indices: list[int] | None = None,
    n_citations: int = 0,
    used_tools: bool = False,
    model: str = DEFAULT_VERIFY_MODEL,
    timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force: bool = False,
    mode: Literal["extras", "strict"] = DEFAULT_VERIFY_MODE,
    deterministic_only: bool = False,
    apply_deterministic: bool = False,
    reasoning_effort: str | None = DEFAULT_VERIFY_REASONING_EFFORT,
) -> VerifyResult:
    """LLM-first verify; strip ancillary units; never regenerate."""
    result = VerifyResult(
        ran=False,
        gated=False,
        gate_reason="",
        answer_before=answer,
        answer_after=answer,
        mode=mode,
    )
    ok, reason = should_verify(
        question,
        answer,
        passage_indices=passage_indices,
        n_citations=n_citations,
        used_tools=used_tools,
    )
    result.gate_reason = reason
    if not ok and not force:
        result.gated = True
        return result

    working = answer

    # Optional legacy hack (off by default).
    if apply_deterministic or deterministic_only:
        det_ans, det_removed = strip_unasked_asides(question, working)
        if det_removed:
            ok_cov, cov_reason = covers_question(question, answer, det_ans)
            if ok_cov:
                working = det_ans
                result.deterministic_stripped = det_removed
            else:
                result.reverted = True
                result.revert_reason = f"deterministic:{cov_reason}"
        if deterministic_only:
            result.ran = bool(result.deterministic_stripped) or force
            result.answer_after = working
            result.stripped = list(result.deterministic_stripped)
            return result

    units = split_units(working)
    if len(units) < 1:
        result.skipped = True
        result.skip_reason = "no_units"
        result.answer_after = working
        return result

    evidence = cited_passage_text(passages, passage_indices)
    llm_model, kwargs = resolve_verify_model(
        model, reasoning_effort=reasoning_effort
    )
    result.model = llm_model
    system = VERIFY_SYSTEM if mode == "extras" else VERIFY_SYSTEM_STRICT
    user = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT UNITS (drop by id only):\n{format_units_for_prompt(units)}\n\n"
        f"EVIDENCE:\n{evidence}\n"
    )

    t0 = time.perf_counter()
    raw = ""
    try:
        resp = litellm.completion(
            model=llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=float(timeout_s),
            num_retries=0,
            max_tokens=int(max_tokens),
            temperature=0.0,
            **kwargs,
        )
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        result.ran = True
        usage = getattr(resp, "usage", None)
        if usage:
            result.tokens = {
                "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion": int(getattr(usage, "completion_tokens", 0) or 0),
            }
            try:
                from tf_client import EST_PRICE

                result.cost_usd = (
                    result.tokens["prompt"] * EST_PRICE[0]
                    + result.tokens["completion"] * EST_PRICE[1]
                ) / 1e6
            except Exception:
                pass
        raw = resp.choices[0].message.content or ""
        # Truncated reasoning-model replies sometimes leave content empty.
        if not raw.strip():
            result.skipped = True
            result.skip_reason = "empty_llm_content"
            result.answer_after = working
            result.stripped = list(result.deterministic_stripped)
            return result
    except Exception as e:
        name = type(e).__name__
        result.skipped = True
        result.skip_reason = (
            f"timeout:{e}"
            if "Timeout" in name or "timeout" in str(e).lower()
            else f"error:{name}"
        )
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        result.answer_after = working
        result.stripped = list(result.deterministic_stripped)
        return result
    drop_idxs, ask_s, keep_s = _parse_drop_response(raw, len(units))
    result.question_ask = ask_s
    result.keep_summary = keep_s
    result.drop_ids = [f"S{i+1}" for i in drop_idxs]

    final = working
    llm_removed: list[str] = []

    if drop_idxs:
        # Never drop unit 0 if it's the only core-looking opener — still allow if
        # model asks, but covers_question will revert polarity/length issues.
        new_ans, removed = drop_units_by_index(working, units, drop_idxs)
        ok_cov, cov_reason = covers_question(question, answer, new_ans)
        if ok_cov and removed:
            final = new_ans
            llm_removed = removed
            result.unsupported = removed
        elif removed:
            result.reverted = True
            result.revert_reason = (
                f"{result.revert_reason};llm:{cov_reason}"
                if result.revert_reason
                else f"llm:{cov_reason}"
            )
    else:
        # Fallback: legacy span list
        spans = filter_protected_spans(
            question, working, _parse_unsupported_spans(raw)
        )
        result.unsupported = spans
        if spans:
            new_ans, removed = strip_unsupported(working, spans)
            ok_cov, cov_reason = covers_question(question, answer, new_ans)
            if ok_cov and removed:
                final = new_ans
                llm_removed = removed
            elif removed:
                result.reverted = True
                result.revert_reason = (
                    f"{result.revert_reason};llm:{cov_reason}"
                    if result.revert_reason
                    else f"llm:{cov_reason}"
                )

    result.answer_after = final
    result.stripped = list(result.deterministic_stripped) + llm_removed
    return result


def verify_meta(v: VerifyResult) -> dict[str, Any]:
    return {
        "ran": v.ran,
        "gated": v.gated,
        "gate_reason": v.gate_reason,
        "skipped": v.skipped,
        "skip_reason": v.skip_reason,
        "unsupported": v.unsupported,
        "stripped": v.stripped,
        "deterministic_stripped": v.deterministic_stripped,
        "drop_ids": v.drop_ids,
        "question_ask": v.question_ask,
        "keep_summary": v.keep_summary,
        "changed": v.answer_before != v.answer_after,
        "model": v.model,
        "mode": v.mode,
        "latency_ms": v.latency_ms,
        "tokens": v.tokens,
        "cost_usd": v.cost_usd,
        "reverted": v.reverted,
        "revert_reason": v.revert_reason,
    }
