"""LLM-as-judge for answer relevance and hallucination detection."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import litellm

from tf_client import BASE_URL, EST_PRICE

DEFAULT_JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

JUDGE_SYSTEM = """You are an evaluation judge for a Hebrew insurance Q&A system.
Score ONE model answer against a ground-truth reference answer.

Return ONLY valid JSON with these fields:
- relevant (bool): true if the model answer agrees with the ground truth on the core fact asked.
  Minor wording differences are fine. Partial overlap on a multi-part answer counts as false
  unless every required fact is present and correct.
- refusal (bool): true if the model declines to answer, says it lacks information, or gives
  only a generic disclaimer without committing to a fact.
- hallucination (bool): true if the model states any specific checkable fact — a number,
  date, duration, monetary limit, percentage, yes/no, or named exclusion/inclusion — that
  either (a) contradicts the ground truth, or (b) is absent from the ground truth (invented).
  General background or procedural advice that does not assert a checkable fact is NOT
  hallucination. A refusal is NOT a hallucination. An empty answer is NOT a hallucination.
  Being wrong but vague (no specific fact committed) is NOT hallucination.
- reasoning (string): one or two sentences explaining your verdict."""

JUDGE_USER = """Question: {question}

Ground-truth answer: {ground_truth}

Model answer: {answer}

Respond with JSON only."""


CITATION_JUDGE_SYSTEM = """You are an evaluation judge for cited evidence in a Hebrew insurance Q&A system.
Given a customer question, a ground-truth answer, and the text of cited corpus pages, decide whether
the cited pages establish the ground-truth answer.

Return ONLY valid JSON with these fields:
- establishes (string): one of "fully", "partially", "not_at_all"
  - fully: the cited pages contain the core facts needed to support the ground-truth answer
  - partially: the cited pages support some but not all required facts
  - not_at_all: the cited pages do not substantively support the ground-truth answer
- reasoning (string): one or two sentences explaining your verdict"""

CITATION_JUDGE_USER = """Question: {question}

Ground-truth answer: {ground_truth}

Cited source pages:
{sources}

Respond with JSON only."""


@dataclass
class CitationJudgeScore:
    establishes: str
    reasoning: str
    raw: str
    cost_usd: float


def _default_reasoning_effort(model: str) -> str | None:
    """Kimi always thinks; without a low budget, content often comes back empty."""
    name = model.lower()
    if "kimi" in name or "moonshot" in name:
        return "low"
    return None


def _judge_call_kwargs(model: str, *, reasoning_effort: str | None = None) -> dict[str, Any]:
    effort = reasoning_effort if reasoning_effort is not None else _default_reasoning_effort(model)
    kwargs: dict[str, Any] = {}
    if effort:
        # Nebius OpenAI-compat: litellm rejects top-level reasoning_effort.
        kwargs["extra_body"] = {"reasoning_effort": effort}
    return kwargs


def _assistant_text(message: Any) -> str:
    """Prefer content; fall back to reasoning_content (Kimi sometimes empties content)."""
    content = getattr(message, "content", None) or ""
    if isinstance(content, str) and content.strip():
        return content
    extra = getattr(message, "reasoning_content", None)
    if extra is None and hasattr(message, "model_extra") and isinstance(message.model_extra, dict):
        extra = message.model_extra.get("reasoning_content")
    if isinstance(extra, str) and extra.strip():
        return extra
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        for key in ("content", "reasoning_content", "reasoning"):
            val = d.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return content if isinstance(content, str) else ""


def _format_sources(resolved: list) -> str:
    blocks: list[str] = []
    for i, r in enumerate(resolved, start=1):
        page = f" (page {r.page})" if r.page else ""
        body = r.text_for_judge() if hasattr(r, "text_for_judge") else (r.text or "")
        blocks.append(f"--- Citation {i}: {r.file}{page} ---\n{body}")
    return "\n\n".join(blocks)


CORPUS_JUDGE_SYSTEM = """You are an evaluation judge for a Hebrew insurance Q&A system.
Score ONE model answer with two INDEPENDENT checks:
  (A) gt_covered  — does the answer contain the ground-truth facts?
  (B) fully_supported — is everything the answer asserts backed by its own cited
      corpus pages (extracted page text and table payloads from our index)?
Never let one check decide the other: (A) looks only at the ground truth, (B)
only at the cited sources. An answer can be supported but not covered, or
covered but not supported.

Return ONLY valid JSON, exactly these keys in this order:
{"reasoning": "...", "gt_covered": true, "fully_supported": true, "refusal": false}

- reasoning (string): first, in ONE or TWO sentences of English, name the GT
  facts you checked and the decisive evidence. Decide the booleans from it.
- gt_covered (bool): true if EVERY core *policy fact* in the ground-truth answer
  is present and correct in the model answer. Minor wording differences are fine.
  The model MAY add extra details; that alone does NOT make this false.
  But the model may NOT omit a core ground-truth fact — especially any fact that
  answers a part of the customer's question. On a multi-part question/GT, all
  asked parts must be answered with the GT facts.

  Hedging / "not in the documents" is NOT coverage: if the ground truth states a
  specific fact (e.g. where a refund is paid) and the model says it does not know
  or that the documents omit it, gt_covered is false for that fact.

  Framing exception (כן/לא polarity only): judge policy substance, not the
  opening yes/no. Example — GT: "לא. הפרק מחריג חבות כלפי קבלן משנה ועובדיו."
  Model: "כן, אך בתנאי… ככלל החבות מוחרגת… עם זאת אם נרכשה הרחבה X אז יש כיסוי."
  → gt_covered true (default exclusion is stated; optional rider is allowed
  extra). Do NOT fail solely for opening with כן while stating the GT exclusion.
  Still fail if the model never states the GT exclusion/rule, or claims
  unconditional coverage when GT says excluded.
- fully_supported (bool): true if EVERY specific checkable fact in the model
  answer — a number, date, duration, monetary amount, percentage, yes/no on
  coverage, or named exclusion/inclusion — is supported by the cited source
  texts below. Judge this against the sources ONLY: a fact that the sources
  support stays supported even if it goes beyond or conflicts with the ground
  truth. Generic procedural advice ("check your schedule", "contact your agent")
  asserts no checkable fact and never makes this false. If the model answer has
  no checkable facts, true only when it is a refusal or empty. If there are no
  usable cited sources, false (unless the answer is a pure refusal with no
  checkable facts).
- refusal (bool): true only if the answer AS A WHOLE declines — it commits to no
  policy fact, or is only a generic disclaimer. An answer that resolves the main
  question but hedges one detail is NOT a refusal (it just fails gt_covered).
  A refusal always implies gt_covered false.

Treat as the SAME fact: equivalent number/currency formats (24,000 ₪ / 24 אלף
ש"ח / ILS 24000), equivalent date or duration phrasings, and Hebrew vs English
product names. Extraction noise in the source text (garbled Hebrew, broken
tables, RTL artifacts) is not evidence of a contradiction — judge substance."""

CORPUS_JUDGE_USER = """Question: {question}

Ground-truth answer: {ground_truth}

Model answer: {answer}

Cited source pages (model citations):
{sources}

Respond with JSON only."""


@dataclass
class CorpusJudgeScore:
    gt_covered: bool
    fully_supported: bool
    refusal: bool
    reasoning: str
    raw: str
    cost_usd: float

    @property
    def ok(self) -> bool:
        return self.gt_covered and self.fully_supported and not self.refusal


def judge_citations(
    question: str,
    ground_truth_answer: str,
    resolved: list,
    model: str = DEFAULT_JUDGE_MODEL,
    quiet: bool = False,
    reasoning_effort: str | None = None,
) -> CitationJudgeScore:
    key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY (or OPENAI_API_KEY) not set")

    messages = [
        {"role": "system", "content": CITATION_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": CITATION_JUDGE_USER.format(
                question=question,
                ground_truth=ground_truth_answer,
                sources=_format_sources(resolved),
            ),
        },
    ]

    resp = litellm.completion(
        model=f"openai/{model}",
        api_base=BASE_URL,
        api_key=key,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        timeout=180,
        response_format={"type": "json_object"},
        **_judge_call_kwargs(model, reasoning_effort=reasoning_effort),
    )
    raw = _assistant_text(resp.choices[0].message)
    parsed = _extract_json(raw)
    establishes = str(parsed.get("establishes", "not_at_all")).strip().lower()
    if establishes not in {"fully", "partially", "not_at_all"}:
        establishes = "not_at_all"

    u = resp.usage
    cost = (u.prompt_tokens * EST_PRICE[0] + u.completion_tokens * EST_PRICE[1]) / 1e6
    if not quiet:
        print(
            f"  [cite-judge] {u.prompt_tokens}+{u.completion_tokens} tokens ~${cost:.4f} -> {establishes}",
            file=sys.stderr,
        )

    return CitationJudgeScore(
        establishes=establishes,
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
        cost_usd=cost,
    )


@dataclass
class JudgeScore:
    relevant: bool
    hallucination: bool
    refusal: bool
    reasoning: str
    raw: str
    cost_usd: float


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Judge returned non-JSON: {text[:200]!r}")


def judge_answer_corpus(
    question: str,
    ground_truth: str,
    answer: str,
    resolved: list,
    model: str = DEFAULT_JUDGE_MODEL,
    quiet: bool = False,
    reasoning_effort: str | None = None,
) -> CorpusJudgeScore:
    """Second judge: GT ⊆ answer, and answer ⊆ cited sources (+ tables)."""
    key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY (or OPENAI_API_KEY) not set")

    usable = [r for r in resolved if getattr(r, "ok", False)]
    sources = _format_sources(usable) if usable else "(no usable citations resolved)"

    messages = [
        {"role": "system", "content": CORPUS_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": CORPUS_JUDGE_USER.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer or "(empty answer)",
                sources=sources,
            ),
        },
    ]

    resp = litellm.completion(
        model=f"openai/{model}",
        api_base=BASE_URL,
        api_key=key,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        timeout=180,
        response_format={"type": "json_object"},
        **_judge_call_kwargs(model, reasoning_effort=reasoning_effort),
    )
    raw = _assistant_text(resp.choices[0].message)
    parsed = _extract_json(raw)
    gt_covered = bool(parsed.get("gt_covered", False))
    fully_supported = bool(parsed.get("fully_supported", False))
    refusal = bool(parsed.get("refusal", False))
    if refusal:
        gt_covered = False
    u = resp.usage
    cost = (u.prompt_tokens * EST_PRICE[0] + u.completion_tokens * EST_PRICE[1]) / 1e6
    if not quiet:
        ok = gt_covered and fully_supported and not refusal
        print(
            f"  [corpus-judge] {u.prompt_tokens}+{u.completion_tokens} tokens "
            f"~${cost:.4f} -> covered={gt_covered} supported={fully_supported} "
            f"refusal={refusal} ok={ok}",
            file=sys.stderr,
        )

    return CorpusJudgeScore(
        gt_covered=gt_covered,
        fully_supported=fully_supported,
        refusal=refusal,
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
        cost_usd=cost,
    )


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    model: str = DEFAULT_JUDGE_MODEL,
    quiet: bool = False,
    reasoning_effort: str | None = None,
) -> JudgeScore:
    key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY (or OPENAI_API_KEY) not set")

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer or "(empty answer)",
            ),
        },
    ]

    resp = litellm.completion(
        model=f"openai/{model}",
        api_base=BASE_URL,
        api_key=key,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        timeout=180,
        response_format={"type": "json_object"},
        **_judge_call_kwargs(model, reasoning_effort=reasoning_effort),
    )
    raw = _assistant_text(resp.choices[0].message)
    parsed = _extract_json(raw)
    relevant = bool(parsed.get("relevant", False))
    hallucination = bool(parsed.get("hallucination", False))
    refusal = bool(parsed.get("refusal", False))
    if hallucination:
        relevant = False  # contradiction/invented fact cannot be relevant
    u = resp.usage
    cost = (u.prompt_tokens * EST_PRICE[0] + u.completion_tokens * EST_PRICE[1]) / 1e6
    if not quiet:
        print(
            f"  [judge] {u.prompt_tokens}+{u.completion_tokens} tokens ~${cost:.4f}",
            file=sys.stderr,
        )

    return JudgeScore(
        relevant=relevant,
        hallucination=hallucination,
        refusal=refusal,
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
        cost_usd=cost,
    )
