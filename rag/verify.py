"""Cheap post-answer grounding check: gate → tiny LLM → strip unsupported spans.

Designed not to double end-to-end latency:
1. Gate — skip easy / short / single-fact answers
2. Tiny call — small model, cited passages only, short JSON, low max_tokens
3. Strip — delete unsupported sentences; do not regenerate the answer
4. Budget — hard verify timeout; on timeout/error keep answer as-is (once)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from rag.retrieve import ExpandedHit
from rag.route import DEFAULT_ROUTE_MODEL

DEFAULT_VERIFY_MODEL = DEFAULT_ROUTE_MODEL  # gemma — cheap vs answer model
DEFAULT_VERIFY_TIMEOUT_S = 15.0
DEFAULT_MAX_TOKENS = 350
MIN_ANSWER_CHARS_FOR_GATE = 280

# Strong multi-part markers (need ≥1 of these, or ≥2 weaker signals).
_MULTIPART_STRONG = (
    "ומה",
    "וכמה",
    "ומי",
    "ואיך",
    "ועד",
    "עד איזה",
    "בכמה",
    "מה הסכום",
    "מה התנאי",
    "ואיזה",
    "וגם כמה",
)
_MULTIPART_WEAK = ("מתי", "למי", "כמה", "איזה סכום")

VERIFY_SYSTEM = """\
You check whether an insurance chatbot answer is grounded in the CITED passages only.
Find checkable claims in the answer that are NOT supported by those passages
(invented numbers, phones, ages, caps, exceptions, refund rules, dates, conditions
from a different coverage section).

Return ONLY JSON:
{"unsupported": ["exact sentence or short span copied from the answer", ...]}

Rules:
- unsupported entries MUST be substrings of the answer (copy verbatim).
- If everything is supported, return {"unsupported": []}.
- Do not rewrite the answer. Do not add new facts.
- Ignore stylistic wording; only flag factual claims missing from the citations.
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
    answer_before: str = ""
    answer_after: str = ""
    model: str | None = None
    latency_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None


def resolve_verify_model(model: str) -> tuple[str, dict]:
    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"
    return model, kwargs


def should_verify(
    question: str,
    answer: str,
    *,
    passage_indices: list[int] | None = None,
    n_citations: int = 0,
    used_tools: bool = False,
) -> tuple[bool, str]:
    """Gate: only run verify when overclaim risk is high."""
    ans = (answer or "").strip()
    q = (question or "").strip()
    if not ans:
        return False, "empty_answer"
    if re.search(r"אין בידי|לא מספיק מידע|אין מספיק", ans):
        return False, "refusal"

    reasons: list[str] = []
    if len(ans) >= MIN_ANSWER_CHARS_FOR_GATE:
        reasons.append(f"long_answer>={MIN_ANSWER_CHARS_FOR_GATE}")
    # Multiple numbers / currency-like tokens.
    if len(re.findall(r"\d[\d,.]{0,12}", ans)) >= 2:
        reasons.append("multi_number")
    if used_tools:
        reasons.append("used_tools")
    if n_citations >= 2 or (passage_indices and len(passage_indices) >= 2):
        reasons.append("multi_cite")
    q_lower = q.lower()
    if any(c in q_lower for c in _MULTIPART_STRONG):
        reasons.append("multipart_question")
    elif sum(1 for c in _MULTIPART_WEAK if c in q_lower) >= 2:
        reasons.append("multipart_question")
    elif q.count("?") + q.count("؟") >= 2:
        reasons.append("multipart_question")
    # "X ו-Y" style asks (amount and condition).
    elif " ו" in q and ("כמה" in q_lower or "סכום" in q_lower) and (
        "תנאי" in q_lower or "מתי" in q_lower or "האם" in q_lower or "מי" in q_lower
    ):
        reasons.append("multipart_question")

    if not reasons:
        return False, "low_risk"
    return True, "+".join(reasons)


def cited_passage_text(
    passages: list[ExpandedHit],
    indices: list[int] | None,
    *,
    max_chars: int = 6000,
) -> str:
    """Build verifier context from USED_PASSAGES indices (1-based)."""
    if not passages:
        return ""
    idxs = list(indices or [])
    if not idxs:
        # Fallback: top 2 passages only (keep call small).
        idxs = [1] if passages else []
        if len(passages) > 1:
            idxs.append(2)
    blocks: list[str] = []
    total = 0
    for i in idxs:
        if not (1 <= i <= len(passages)):
            continue
        ex = passages[i - 1]
        body = ex.merged_text().strip()
        if len(body) > 1800:
            body = body[:1800] + "…"
        block = f"[{i}] {ex.match.location.file} p{ex.match.location.page}\n{body}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def _parse_unsupported(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    # Strip markdown fences if present.
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
    out: list[str] = []
    for u in uns:
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    return out


def split_sentences(text: str) -> list[str]:
    """Light sentence split for Hebrew/English answers."""
    text = (text or "").strip()
    if not text:
        return []
    # Keep bullet lines as units.
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


def strip_unsupported(answer: str, unsupported: list[str]) -> tuple[str, list[str]]:
    """Remove sentences that contain an unsupported span. Deterministic; no regen."""
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
            # Fuzzy: significant overlap of digits+words
            if len(span) >= 12 and span[:20] in s:
                drop = True
                break
        if drop:
            removed.append(s)
        else:
            kept.append(s)
    if not removed:
        return answer, []
    if not kept:
        # Don't nuke the whole answer — keep original.
        return answer, []
    # Rebuild: preserve bullet newlines when present.
    if any(x.startswith(("-", "*", "•")) or re.match(r"^\d+[\).]", x) for x in kept):
        new = "\n".join(kept)
    else:
        new = " ".join(kept)
    return new.strip(), removed


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
) -> VerifyResult:
    """Run gated verify once; strip unsupported spans; never regenerate."""
    result = VerifyResult(
        ran=False,
        gated=False,
        gate_reason="",
        answer_before=answer,
        answer_after=answer,
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

    cited = cited_passage_text(passages, passage_indices)
    if not cited.strip():
        result.skipped = True
        result.skip_reason = "no_cited_text"
        return result

    llm_model, kwargs = resolve_verify_model(model)
    result.model = llm_model
    user = (
        f"Customer question:\n{question}\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Cited passages (only evidence allowed):\n{cited}\n"
    )
    t0 = time.perf_counter()
    try:
        resp = litellm.completion(
            model=llm_model,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": user},
            ],
            timeout=float(timeout_s),
            num_retries=0,
            max_tokens=int(max_tokens),
            temperature=0.0,
            **kwargs,
        )
    except Exception as e:
        name = type(e).__name__
        result.skipped = True
        result.skip_reason = (
            f"timeout:{e}"
            if "Timeout" in name or "timeout" in str(e).lower()
            else f"error:{name}"
        )
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Budget hard: answer as-is.
        return result

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
    unsupported = _parse_unsupported(raw)
    result.unsupported = unsupported
    if unsupported:
        new_ans, removed = strip_unsupported(answer, unsupported)
        result.answer_after = new_ans
        result.stripped = removed
    else:
        result.answer_after = answer
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
        "changed": v.answer_before != v.answer_after,
        "model": v.model,
        "latency_ms": v.latency_ms,
        "tokens": v.tokens,
        "cost_usd": v.cost_usd,
    }
