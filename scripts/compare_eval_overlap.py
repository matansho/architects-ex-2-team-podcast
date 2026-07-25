#!/usr/bin/env python3
"""Compare evaluation metrics on overlapping question IDs only.

Default mode compares two existing eval JSON reports.

Optional slow mode (explicit opt-in):
candidate is answers JSONL (--candidate-answers) and is evaluated on the fly.

Examples:
    python scripts/compare_eval_overlap.py \
      --baseline-eval eval_rag_rerank_k20_w2_route80_20_cite_mp.json \
      --candidate-eval reports/stage2/pilot12_bm25inj_eval.json

    python scripts/compare_eval_overlap.py \
      --baseline-eval eval_rag_rerank_k20_w2_route80_20_cite_mp.json \
      --candidate-answers reports/stage2/pilot12_rerank_route80_8_bm25inj12.jsonl \
      --candidate-label bm25inj_pilot12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.harness import report_to_dict, run_eval


def _load_eval_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in payload or not isinstance(payload["results"], list):
        raise SystemExit(f"{path} is not an eval JSON report (missing 'results')")
    label = str(payload.get("run_label") or path.stem)
    rows = payload["results"]
    return label, rows


def _rate(values: list[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    rel = [bool(r.get("relevant")) for r in rows if r.get("relevant") is not None]
    hall = [bool(r.get("hallucination")) for r in rows if r.get("hallucination") is not None]
    refu = [bool(r.get("refusal")) for r in rows if r.get("refusal") is not None]
    cite = [float(r.get("citation_score", 0.0)) for r in rows]
    lat = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]

    return {
        "n": float(len(rows)),
        "relevance_rate": _rate(rel),
        "hallucination_rate": _rate(hall),
        "refusal_rate": _rate(refu),
        "citation_accuracy": (sum(cite) / len(cite)) if cite else 0.0,
        "latency_ms_avg": (sum(lat) / len(lat)) if lat else 0.0,
        "latency_ms_p50": float(median(lat)) if lat else 0.0,
    }


def _print_compare(base_label: str, cand_label: str, base: dict[str, float], cand: dict[str, float]) -> None:
    print("=== Overlap Comparison (common question IDs only) ===")
    print(f"baseline: {base_label}")
    print(f"candidate: {cand_label}")
    print(f"common questions: {int(base['n'])}")
    print()
    print("metric                 baseline      candidate     delta")
    print("-----------------------------------------------------------")

    def row(name: str, key: str, pct: bool) -> None:
        b = base[key]
        c = cand[key]
        d = c - b
        if pct:
            print(f"{name:22s} {b:9.2%}   {c:9.2%}   {d:+8.2%}")
        else:
            print(f"{name:22s} {b:9.1f}   {c:9.1f}   {d:+8.1f}")

    row("relevance_rate", "relevance_rate", True)
    row("hallucination_rate", "hallucination_rate", True)
    row("refusal_rate", "refusal_rate", True)
    row("citation_accuracy", "citation_accuracy", True)
    row("latency_ms_avg", "latency_ms_avg", False)
    row("latency_ms_p50", "latency_ms_p50", False)


def _changed_rows(base_rows: dict[str, dict[str, Any]], cand_rows: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qid in ids:
        b = base_rows[qid]
        c = cand_rows[qid]
        changed = (
            b.get("relevant") != c.get("relevant")
            or b.get("hallucination") != c.get("hallucination")
            or b.get("refusal") != c.get("refusal")
            or float(b.get("citation_score", 0.0)) != float(c.get("citation_score", 0.0))
        )
        if not changed:
            continue
        out.append(
            {
                "id": qid,
                "domain": c.get("domain") or b.get("domain"),
                "difficulty": c.get("difficulty") or b.get("difficulty"),
                "baseline": {
                    "relevant": b.get("relevant"),
                    "hallucination": b.get("hallucination"),
                    "refusal": b.get("refusal"),
                    "citation_score": b.get("citation_score"),
                },
                "candidate": {
                    "relevant": c.get("relevant"),
                    "hallucination": c.get("hallucination"),
                    "refusal": c.get("refusal"),
                    "citation_score": c.get("citation_score"),
                },
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline and candidate eval on overlap IDs")
    ap.add_argument("--baseline-eval", required=True, help="Path to baseline eval JSON")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate-eval", help="Path to candidate eval JSON")
    group.add_argument(
        "--candidate-answers",
        help="Path to candidate answers JSONL (slow; requires --allow-eval-from-answers)",
    )
    ap.add_argument(
        "--allow-eval-from-answers",
        action="store_true",
        help="Allow running eval from candidate answers JSONL",
    )

    ap.add_argument("--candidate-label", default="candidate", help="Run label if candidate is answers JSONL")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-citation-judge", action="store_true")
    ap.add_argument("--quiet-judge", action="store_true")
    ap.add_argument("--out-json", help="Write comparison JSON to this path")
    args = ap.parse_args()

    base_label, base_results = _load_eval_results(Path(args.baseline_eval))

    if args.candidate_eval:
        cand_label, cand_results = _load_eval_results(Path(args.candidate_eval))
    else:
        if not args.allow_eval_from_answers:
            raise SystemExit(
                "Refusing to run eval from answers JSONL without --allow-eval-from-answers. "
                "Use --candidate-eval with an existing eval JSON for fast compare."
            )
        report = run_eval(
            answers_path=args.candidate_answers,
            questions_path=args.questions,
            corpus_path=args.corpus,
            run_label=args.candidate_label,
            judge=not args.no_judge,
            citation_judge=not args.no_citation_judge,
            judge_model=args.judge_model,
            limit=None,
            quiet_judge=args.quiet_judge,
        )
        cand_payload = report_to_dict(report)
        cand_label = str(cand_payload.get("run_label") or args.candidate_label)
        cand_results = cand_payload["results"]

    base_by_id = {r["id"]: r for r in base_results}
    cand_by_id = {r["id"]: r for r in cand_results}
    common_ids = sorted(set(base_by_id) & set(cand_by_id))
    if not common_ids:
        raise SystemExit("No common question IDs between baseline and candidate")

    base_subset = [base_by_id[i] for i in common_ids]
    cand_subset = [cand_by_id[i] for i in common_ids]
    base_summary = _subset_summary(base_subset)
    cand_summary = _subset_summary(cand_subset)

    _print_compare(base_label, cand_label, base_summary, cand_summary)

    changed = _changed_rows(base_by_id, cand_by_id, common_ids)
    if changed:
        print(f"\nChanged per-question outcomes: {len(changed)}")
        for r in changed:
            b = r["baseline"]
            c = r["candidate"]
            print(
                f"- {r['id']} ({r['domain']}/{r['difficulty']}): "
                f"rel {b['relevant']}->{c['relevant']}, "
                f"hall {b['hallucination']}->{c['hallucination']}, "
                f"cite {b['citation_score']}->{c['citation_score']}"
            )
    else:
        print("\nNo per-question label changes on overlap.")

    if args.out_json:
        out = {
            "baseline": base_label,
            "candidate": cand_label,
            "common_ids": common_ids,
            "baseline_summary": base_summary,
            "candidate_summary": cand_summary,
            "delta": {
                k: cand_summary[k] - base_summary[k]
                for k in (
                    "relevance_rate",
                    "hallucination_rate",
                    "refusal_rate",
                    "citation_accuracy",
                    "latency_ms_avg",
                    "latency_ms_p50",
                )
            },
            "changed_rows": changed,
        }
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
