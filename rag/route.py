"""Domain router: small LLM picks corpus sections before retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Top-level corpus folders (must match IndexedVector.location.domain).
CORPUS_DOMAINS: tuple[str, ...] = (
    "apartment",
    "business",
    "car",
    "dental",
    "diseases-disabilities",
    "health",
    "life",
    "long-term-care",
    "loss-of-working-ability",
    "mortgage",
    "personal-accident",
    "travel",
)

DEFAULT_ROUTE_MODEL = "google/gemma-3-27b-it"

ROUTE_SYSTEM = """\
You route Hebrew insurance-customer questions to corpus domains for retrieval.
Pick ONLY from the allowed domain list. Prefer the smallest set that can answer
the question (usually 1, at most 3). If the question clearly spans domains,
include all relevant ones. If unsure, include the most likely domain rather than
returning an empty list.

Return ONLY valid JSON:
  {"domains": ["domain-id", ...], "reason": "short English or Hebrew reason"}
"""

ROUTE_USER = """\
Allowed domains (use these exact ids):
{domain_list}

Question:
{question}

Return the JSON object now.
"""


@dataclass
class RouteResult:
    domains: list[str]
    reason: str = ""
    raw: str = ""
    fallback_all: bool = False  # True if we ignored a bad/empty route


def _strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if "{" in text and not text.lstrip().startswith("{"):
        text = text[text.index("{") :]
    if "}" in text:
        text = text[: text.rindex("}") + 1]
    return text.strip()


def parse_route_json(raw: str, *, allowed: tuple[str, ...] = CORPUS_DOMAINS) -> list[str]:
    """Parse router JSON → ordered unique allowed domain ids."""
    text = _strip_json_fences(raw)
    text = re.sub(r",\s*}", "}", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("route JSON must be an object")
    domains = data.get("domains")
    if domains is None and "domain" in data:
        domains = data["domain"]
    if isinstance(domains, str):
        domains = [domains]
    if not isinstance(domains, list):
        raise ValueError("domains must be a list")
    allow = set(allowed)
    out: list[str] = []
    seen: set[str] = set()
    for d in domains:
        key = str(d).strip().lower().replace("_", "-").replace(" ", "-")
        # common aliases
        aliases = {
            "longtermcare": "long-term-care",
            "long-term-care": "long-term-care",
            "lossofworkingability": "loss-of-working-ability",
            "loss-of-working-ability": "loss-of-working-ability",
            "diseases": "diseases-disabilities",
            "disability": "diseases-disabilities",
            "personalaccident": "personal-accident",
        }
        key = aliases.get(key, key)
        if key in allow and key not in seen:
            seen.add(key)
            out.append(key)
    return out[:3]


def route_question(
    question: str,
    *,
    model: str = DEFAULT_ROUTE_MODEL,
    allowed: tuple[str, ...] = CORPUS_DOMAINS,
    complete=None,
) -> RouteResult:
    """Call LLM router. On failure / empty → fallback_all with empty domains."""
    domain_list = "\n".join(f"- {d}" for d in allowed)
    messages = [
        {"role": "system", "content": ROUTE_SYSTEM},
        {
            "role": "user",
            "content": ROUTE_USER.format(
                domain_list=domain_list, question=question.strip()
            ),
        },
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
            model=m, messages=messages, temperature=0, timeout=60, **kwargs
        )
        return resp.choices[0].message.content or ""

    try:
        raw = complete(messages) if complete is not None else _call()
        domains = parse_route_json(raw, allowed=allowed)
        reason = ""
        try:
            reason = str(json.loads(_strip_json_fences(raw)).get("reason") or "")
        except Exception:
            pass
        if not domains:
            return RouteResult(
                domains=[], reason=reason or "empty route", raw=raw, fallback_all=True
            )
        return RouteResult(domains=domains, reason=reason, raw=raw, fallback_all=False)
    except Exception as e:
        return RouteResult(
            domains=[],
            reason=f"route failed: {e!r}",
            raw="",
            fallback_all=True,
        )


def idxs_for_domains(
    vectors,
    domains: list[str] | set[str] | tuple[str, ...],
) -> list[int]:
    """Corpus indices whose location.domain is in `domains`."""
    want = {str(d).strip() for d in domains if str(d).strip()}
    if not want:
        return []
    return [
        i
        for i, v in enumerate(vectors)
        if (getattr(v.location, "domain", None) or "") in want
    ]
