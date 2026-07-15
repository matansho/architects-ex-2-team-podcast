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


def _format_sources(resolved: list) -> str:
    blocks: list[str] = []
    for i, r in enumerate(resolved, start=1):
        page = f" (page {r.page})" if r.page else ""
        blocks.append(f"--- Citation {i}: {r.file}{page} ---\n{r.text}")
    return "\n\n".join(blocks)


def judge_citations(
    question: str,
    ground_truth_answer: str,
    resolved: list,
    model: str = DEFAULT_JUDGE_MODEL,
    quiet: bool = False,
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
        max_tokens=512,
        temperature=0.0,
        timeout=120,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
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


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    model: str = DEFAULT_JUDGE_MODEL,
    quiet: bool = False,
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
        max_tokens=512,
        temperature=0.0,
        timeout=120,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
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
