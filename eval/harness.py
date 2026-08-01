"""Aggregate scoring for a batch of answers against the dev set."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.citation import score_citations
from eval.corpus import (
    enrich_resolved_with_tables,
    load_table_payloads_by_location,
    resolve_citations,
)
from eval.judge import CorpusJudgeScore, JudgeScore, judge_answer, judge_answer_corpus


@dataclass
class QuestionResult:
    id: str
    domain: str
    difficulty: str
    question: str
    answer: str
    ground_truth_answer: str
    latency_ms: float | None
    citation_score: float
    citation_establishes: str | None = None
    citation_details: list[str] = field(default_factory=list)
    citation_reasoning: str | None = None
    relevant: bool | None = None
    hallucination: bool | None = None
    refusal: bool | None = None
    judge_reasoning: str | None = None
    judge_cost_usd: float | None = None


@dataclass
class EvalReport:
    run_label: str
    total: int
    relevance_rate: float
    hallucination_rate: float
    refusal_rate: float
    citation_accuracy: float
    latency_ms_avg: float
    latency_ms_p50: float
    judge_cost_usd: float
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_domain: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)


def load_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["questions"]
    return {q["id"]: q for q in data}


def load_answers(path: str | Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        answers[rec["id"]] = rec
    return answers


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(results: list[QuestionResult], key_fn) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        buckets[key_fn(r)].append(r)

    out: dict[str, dict[str, float]] = {}
    for key, rows in sorted(buckets.items()):
        judged = [r for r in rows if r.relevant is not None]
        out[key] = {
            "count": len(rows),
            "relevance_rate": _rate([bool(r.relevant) for r in judged]),
            "hallucination_rate": _rate([bool(r.hallucination) for r in judged]),
            "refusal_rate": _rate([bool(r.refusal) for r in judged]),
            "citation_accuracy": sum(r.citation_score for r in rows) / len(rows),
            "latency_ms_avg": _avg([r.latency_ms for r in rows if r.latency_ms is not None]),
        }
    return out


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _p50(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[len(s) // 2]


def run_eval(
    answers_path: str | Path,
    questions_path: str | Path = "reference_questions.json",
    corpus_path: str | Path = "corpus",
    run_label: str = "eval",
    judge: bool = True,
    citation_judge: bool = True,
    judge_model: str = "deepseek-ai/DeepSeek-V4-Pro",
    judge_reasoning_effort: str | None = None,
    limit: int | None = None,
    quiet_judge: bool = False,
) -> EvalReport:
    questions = load_questions(questions_path)
    answers = load_answers(answers_path)

    ids = [qid for qid in questions if qid in answers]
    if limit:
        ids = ids[:limit]

    results: list[QuestionResult] = []
    judge_cost = 0.0

    for qid in ids:
        q = questions[qid]
        a = answers[qid]
        print(f"scoring {qid}...", flush=True)
        cite = score_citations(
            question=q["question"],
            ground_truth_answer=q["ground_truth_answer"],
            citations=a.get("citations"),
            corpus_root=corpus_path,
            use_judge=judge and citation_judge,
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
            quiet=quiet_judge,
        )

        result = QuestionResult(
            id=qid,
            domain=q["domain"],
            difficulty=q["difficulty"],
            question=q["question"],
            answer=a.get("answer", ""),
            ground_truth_answer=q["ground_truth_answer"],
            latency_ms=a.get("latency_ms"),
            citation_score=cite.score,
            citation_establishes=cite.establishes,
            citation_details=cite.details,
            citation_reasoning=cite.reasoning,
        )

        judge_cost += cite.judge_cost_usd or 0.0

        if judge:
            js: JudgeScore = judge_answer(
                question=q["question"],
                ground_truth=q["ground_truth_answer"],
                answer=a.get("answer", ""),
                model=judge_model,
                quiet=quiet_judge,
                reasoning_effort=judge_reasoning_effort,
            )
            result.relevant = js.relevant
            result.hallucination = js.hallucination
            result.refusal = js.refusal
            result.judge_reasoning = js.reasoning
            result.judge_cost_usd = (cite.judge_cost_usd or 0.0) + js.cost_usd
            judge_cost += js.cost_usd

        results.append(result)

    judged = [r for r in results if r.relevant is not None]
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]

    return EvalReport(
        run_label=run_label,
        total=len(results),
        relevance_rate=_rate([bool(r.relevant) for r in judged]),
        hallucination_rate=_rate([bool(r.hallucination) for r in judged]),
        refusal_rate=_rate([bool(r.refusal) for r in judged]),
        citation_accuracy=sum(r.citation_score for r in results) / len(results) if results else 0.0,
        latency_ms_avg=_avg(latencies),
        latency_ms_p50=_p50(latencies),
        judge_cost_usd=judge_cost,
        by_difficulty=_aggregate(results, lambda r: r.difficulty),
        by_domain=_aggregate(results, lambda r: r.domain),
        results=results,
    )


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    d = asdict(report)
    return d


def print_summary(report: EvalReport) -> None:
    print(f"\n=== {report.run_label} ({report.total} questions) ===")
    print(f"  relevance:      {report.relevance_rate:.1%}")
    print(f"  hallucination:  {report.hallucination_rate:.1%}")
    print(f"  refusal:        {report.refusal_rate:.1%}")
    print(f"  citation acc:   {report.citation_accuracy:.1%}")
    print(f"  latency avg:    {report.latency_ms_avg:.0f} ms")
    print(f"  latency p50:    {report.latency_ms_p50:.0f} ms")
    if report.judge_cost_usd:
        print(f"  judge cost:     ~${report.judge_cost_usd:.4f} (answer + citation judges)")

    print("\n  by difficulty:")
    for diff, stats in report.by_difficulty.items():
        print(
            f"    {diff:6s}  rel={stats['relevance_rate']:.0%}  "
            f"hall={stats['hallucination_rate']:.0%}  cite={stats['citation_accuracy']:.0%}"
        )


@dataclass
class CorpusQuestionResult:
    id: str
    domain: str
    difficulty: str
    question: str
    answer: str
    ground_truth_answer: str
    gt_covered: bool
    fully_supported: bool
    refusal: bool
    ok: bool
    citations_resolved: int
    citations_total: int
    tables_attached: int
    judge_reasoning: str
    judge_cost_usd: float


@dataclass
class CorpusEvalReport:
    run_label: str
    total: int
    gt_covered_rate: float
    fully_supported_rate: float
    refusal_rate: float
    ok_rate: float
    judge_cost_usd: float
    results: list[CorpusQuestionResult] = field(default_factory=list)


def run_corpus_judge_eval(
    answers_path: str | Path,
    questions_path: str | Path = "reference_questions.json",
    corpus_path: str | Path = "corpus",
    index_dir: str | Path = "data/index",
    run_label: str = "corpus-judge",
    judge_model: str = "deepseek-ai/DeepSeek-V4-Pro",
    judge_reasoning_effort: str | None = None,
    limit: int | None = None,
    quiet_judge: bool = False,
) -> CorpusEvalReport:
    """Second judge only: GT ⊆ answer and answer ⊆ cited sources (+ index tables)."""
    questions = load_questions(questions_path)
    answers = load_answers(answers_path)
    ids = [qid for qid in questions if qid in answers]
    if limit:
        ids = ids[:limit]

    print("Loading index table payloads…", flush=True)
    table_by_loc = load_table_payloads_by_location(index_dir)
    print(f"  {len(table_by_loc)} (file,page) locations with tables", flush=True)

    results: list[CorpusQuestionResult] = []
    judge_cost = 0.0
    corpus_root = Path(corpus_path)

    for qid in ids:
        q = questions[qid]
        a = answers[qid]
        print(f"scoring {qid}...", flush=True)
        citations = a.get("citations") or []
        resolved = resolve_citations(corpus_root, citations)
        enrich_resolved_with_tables(resolved, table_by_loc)
        tables_attached = sum(len(r.table_payloads or []) for r in resolved)

        js: CorpusJudgeScore = judge_answer_corpus(
            question=q["question"],
            ground_truth=q["ground_truth_answer"],
            answer=a.get("answer", ""),
            resolved=resolved,
            model=judge_model,
            quiet=quiet_judge,
            reasoning_effort=judge_reasoning_effort,
        )
        judge_cost += js.cost_usd
        results.append(
            CorpusQuestionResult(
                id=qid,
                domain=q["domain"],
                difficulty=q["difficulty"],
                question=q["question"],
                answer=a.get("answer", ""),
                ground_truth_answer=q["ground_truth_answer"],
                gt_covered=js.gt_covered,
                fully_supported=js.fully_supported,
                refusal=js.refusal,
                ok=js.ok,
                citations_resolved=sum(1 for r in resolved if r.ok),
                citations_total=len(resolved),
                tables_attached=tables_attached,
                judge_reasoning=js.reasoning,
                judge_cost_usd=js.cost_usd,
            )
        )

    return CorpusEvalReport(
        run_label=run_label,
        total=len(results),
        gt_covered_rate=_rate([r.gt_covered for r in results]),
        fully_supported_rate=_rate([r.fully_supported for r in results]),
        refusal_rate=_rate([r.refusal for r in results]),
        ok_rate=_rate([r.ok for r in results]),
        judge_cost_usd=judge_cost,
        results=results,
    )


def print_corpus_summary(report: CorpusEvalReport) -> None:
    print(f"\n=== {report.run_label} ({report.total} questions) ===")
    print(f"  gt_covered:       {report.gt_covered_rate:.1%}")
    print(f"  fully_supported:  {report.fully_supported_rate:.1%}")
    print(f"  refusal:          {report.refusal_rate:.1%}")
    print(f"  ok (both):        {report.ok_rate:.1%}")
    if report.judge_cost_usd:
        print(f"  judge cost:       ~${report.judge_cost_usd:.4f}")

    fails = [r for r in report.results if not r.ok]
    if fails:
        print(f"\n  failures ({len(fails)}):")
        for r in fails:
            print(
                f"    {r.id}: covered={r.gt_covered} supported={r.fully_supported} "
                f"refusal={r.refusal} tables={r.tables_attached}"
            )
            if r.judge_reasoning:
                print(f"      {r.judge_reasoning}")
