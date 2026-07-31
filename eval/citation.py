"""Citation scoring: corpus resolution + optional LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.corpus import ResolvedCitation, normalize_path, resolve_citations
from eval.judge import CitationJudgeScore, judge_citations

ESTABLISHES_SCORE = {
    "fully": 1.0,
    "partially": 0.5,
    "not_at_all": 0.0,
}


@dataclass
class CitationScore:
    score: float
    establishes: str | None
    resolved: int
    total: int
    invalid: int
    details: list[str]
    reasoning: str | None = None
    judge_cost_usd: float | None = None


def _single_match(expected: dict[str, Any], citation: dict[str, Any]) -> bool:
    exp_file = normalize_path(expected["file"])
    act_file = normalize_path(citation.get("file", ""))
    if exp_file != act_file:
        return False
    exp_page = expected.get("page")
    act_page = citation.get("page")
    if exp_page is None:
        return True
    return act_page == exp_page


def _group_satisfied(group: dict[str, Any], citations: list[dict[str, Any]]) -> str | None:
    for opt in group.get("any_of", []):
        for cite in citations:
            if _single_match(opt, cite):
                return opt["file"]
    return None


def score_citations_legacy(
    ground_truth_sources: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None,
) -> CitationScore:
    """Debug helper: match citations against reference source pointers."""
    citations = citations or []
    if not ground_truth_sources:
        return CitationScore(1.0, "fully", 0, 0, 0, ["no reference sources listed"])

    hits = 0
    details: list[str] = []
    for i, group in enumerate(ground_truth_sources, start=1):
        matched = _group_satisfied(group, citations)
        if matched:
            hits += 1
            details.append(f"group {i}: reference hit ({matched})")
        else:
            expected = [normalize_path(o["file"]) for o in group.get("any_of", [])]
            details.append(f"group {i}: reference miss (expected one of {expected})")

    total = len(ground_truth_sources)
    score = hits / total
    establishes = "fully" if score == 1.0 else "partially" if score > 0 else "not_at_all"
    return CitationScore(score, establishes, hits, total, total - hits, details)


def score_citations(
    question: str,
    ground_truth_answer: str,
    citations: list[dict[str, Any]] | None,
    *,
    corpus_root: Path | str = "corpus",
    use_judge: bool = True,
    judge_model: str = "google/gemma-3-27b-it",
    quiet: bool = False,
) -> CitationScore:
    citations = citations or []
    corpus_root = Path(corpus_root)

    if not citations:
        return CitationScore(
            0.0,
            "not_at_all",
            0,
            0,
            0,
            ["no citations provided"],
        )

    resolved = resolve_citations(corpus_root, citations)
    invalid = [r for r in resolved if not r.ok]
    valid = [r for r in resolved if r.ok]

    details = []
    for r in resolved:
        if r.ok:
            page = f" p.{r.page}" if r.page else ""
            details.append(f"resolved {r.file}{page}")
        else:
            details.append(f"invalid {r.file}: {r.error}")

    if invalid:
        return CitationScore(
            0.0,
            "not_at_all",
            len(valid),
            len(resolved),
            len(invalid),
            details + ["score=0 because at least one citation is invalid"],
        )

    if not use_judge:
        return CitationScore(
            0.0,
            None,
            len(valid),
            len(resolved),
            0,
            details + ["citation judge skipped (--no-citation-judge)"],
        )

    judged: CitationJudgeScore = judge_citations(
        question=question,
        ground_truth_answer=ground_truth_answer,
        resolved=valid,
        model=judge_model,
        quiet=quiet,
    )
    score = ESTABLISHES_SCORE.get(judged.establishes, 0.0)
    return CitationScore(
        score,
        judged.establishes,
        len(valid),
        len(resolved),
        0,
        details,
        reasoning=judged.reasoning,
        judge_cost_usd=judged.cost_usd,
    )
