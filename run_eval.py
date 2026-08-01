#!/usr/bin/env python3
"""
Score an answers JSONL file against reference_questions.json.

    # Fast smoke test (no LLM judges):
    python run_eval.py --answers baseline_answers.jsonl --no-judge

    # Answer judge only (skip citation LLM judge):
    python run_eval.py --answers answers.jsonl --no-citation-judge

    # Full eval with answer + citation LLM judges:
    python run_eval.py --answers baseline_answers.jsonl --out reports/stage1/baseline_eval.json

    # Second judge only (GT ⊆ answer, answer ⊆ cites + index tables):
    python run_eval.py --answers reports/rag_answers_….jsonl --corpus-judge \
      --out reports/stage2/…_corpus_judge.json

    # Quick smoke test on 3 questions:
    python run_eval.py --answers baseline_answers.jsonl --limit 3
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from eval.harness import (
    print_corpus_summary,
    print_summary,
    report_to_dict,
    run_corpus_judge_eval,
    run_eval,
)


def main():
    ap = argparse.ArgumentParser(description="APEX Exercise 2 evaluation harness")
    ap.add_argument("--answers", required=True, help="Answers JSONL from baseline_runner or submit_runner")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--corpus", default="corpus", help="Path to downloaded corpus directory")
    ap.add_argument("--index", default="data/index", help="Index dir (table payloads for --corpus-judge)")
    ap.add_argument("--out", help="Write full JSON report to this path")
    ap.add_argument("--label", default="eval", help="Label for this run")
    ap.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument(
        "--judge-reasoning-effort",
        choices=["low", "high", "max"],
        help="Pass reasoning_effort to the judge LLM (auto-low for Kimi if omitted)",
    )
    ap.add_argument("--no-judge", action="store_true", help="Skip all LLM judges (citation resolve only)")
    ap.add_argument(
        "--no-citation-judge",
        action="store_true",
        help="Skip citation LLM judge (still runs answer judge unless --no-judge)",
    )
    ap.add_argument(
        "--corpus-judge",
        action="store_true",
        help="Run second judge only: GT covered + answer fully supported by citations",
    )
    ap.add_argument("--limit", type=int, help="Only evaluate first N questions")
    ap.add_argument("--quiet-judge", action="store_true")
    args = ap.parse_args()

    if args.corpus_judge:
        report = run_corpus_judge_eval(
            answers_path=args.answers,
            questions_path=args.questions,
            corpus_path=args.corpus,
            index_dir=args.index,
            run_label=args.label if args.label != "eval" else "corpus-judge",
            judge_model=args.judge_model,
            judge_reasoning_effort=args.judge_reasoning_effort,
            limit=args.limit,
            quiet_judge=args.quiet_judge,
        )
        print_corpus_summary(report)
        payload = asdict(report)
    else:
        report = run_eval(
            answers_path=args.answers,
            questions_path=args.questions,
            corpus_path=args.corpus,
            run_label=args.label,
            judge=not args.no_judge,
            citation_judge=not args.no_citation_judge,
            judge_model=args.judge_model,
            judge_reasoning_effort=args.judge_reasoning_effort,
            limit=args.limit,
            quiet_judge=args.quiet_judge,
        )
        print_summary(report)
        payload = report_to_dict(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
