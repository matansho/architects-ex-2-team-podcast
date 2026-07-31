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

# When a dense preview is available (named products, jargon), prefer corpus evidence.
ROUTE_SYSTEM_CTX = """\
You route Hebrew insurance-customer questions to corpus domains for retrieval.
Pick ONLY from the allowed domain list. Prefer the smallest set that can answer
the question (usually 1, at most 3). If the question clearly spans domains,
include all relevant ones.

You are also given a PREVIEW of the top dense-retrieved corpus snippets for this
question (domain, file, page, short text). Use them as evidence of where the
answer likely lives — especially for named products/policies — but do not invent
domains that are not in the allowed list. If the preview domains disagree with
your prior, prefer the preview when it clearly matches the product or topic.

Return ONLY valid JSON:
  {"domains": ["domain-id", ...], "reason": "short English or Hebrew reason"}
"""

ROUTE_USER_CTX = """\
Allowed domains (use these exact ids):
{domain_list}

Question:
{question}

Top retrieved snippets (dense preview — may be imperfect):
{snippets}

Return the JSON object now.
"""


@dataclass
class RouteResult:
    domains: list[str]
    reason: str = ""
    raw: str = ""
    fallback_all: bool = False  # True if we ignored a bad/empty route
    used_preview: bool = False


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


def format_preview_snippets(
    hits,
    *,
    max_chars: int = 220,
) -> str:
    """Format dense Hit list for the context-aware router prompt."""
    lines: list[str] = []
    for rank, h in enumerate(hits, start=1):
        v = h.vector if hasattr(h, "vector") else h
        loc = getattr(v, "location", None)
        domain = (getattr(loc, "domain", None) or "?") if loc else "?"
        path = (getattr(loc, "file", None) or "") if loc else ""
        page = getattr(loc, "page", None) if loc else None
        base = path.rsplit("/", 1)[-1] if path else "?"
        body = getattr(v, "embed_text", None) or getattr(v, "text", None) or ""
        body = " ".join(str(body).split())[:max_chars]
        lines.append(
            f"{rank}. domain={domain} | file={base[:55]} | page={page} | {body}"
        )
    return "\n".join(lines)


def route_question(
    question: str,
    *,
    model: str = DEFAULT_ROUTE_MODEL,
    allowed: tuple[str, ...] = CORPUS_DOMAINS,
    complete=None,
    preview_hits=None,
) -> RouteResult:
    """Call LLM router. On failure / empty → fallback_all with empty domains.

    If `preview_hits` is a non-empty list of Hits (or objects with `.vector`),
    the router gets those snippets as corpus evidence.
    """
    domain_list = "\n".join(f"- {d}" for d in allowed)
    use_preview = bool(preview_hits)
    if use_preview:
        messages = [
            {"role": "system", "content": ROUTE_SYSTEM_CTX},
            {
                "role": "user",
                "content": ROUTE_USER_CTX.format(
                    domain_list=domain_list,
                    question=question.strip(),
                    snippets=format_preview_snippets(preview_hits),
                ),
            },
        ]
    else:
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
            model=m, messages=messages, temperature=0, timeout=60, num_retries=0, **kwargs
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
                domains=[],
                reason=reason or "empty route",
                raw=raw,
                fallback_all=True,
                used_preview=use_preview,
            )
        return RouteResult(
            domains=domains,
            reason=reason,
            raw=raw,
            fallback_all=False,
            used_preview=use_preview,
        )
    except Exception as e:
        return RouteResult(
            domains=[],
            reason=f"route failed: {e!r}",
            raw="",
            fallback_all=True,
            used_preview=use_preview,
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
