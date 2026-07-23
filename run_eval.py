#!/usr/bin/env python3
"""
Score an answers JSONL file against reference_questions.json.

    # Fast smoke test (no LLM judges):
    python run_eval.py --answers baseline_answers.jsonl --no-judge

    # Answer judge only (skip citation LLM judge):
    python run_eval.py --answers answers.jsonl --no-citation-judge

    # Full eval with answer + citation LLM judges:
    python run_eval.py --answers baseline_answers.jsonl --out reports/stage1/baseline_eval.json

    # Quick smoke test on 3 questions:
    python run_eval.py --answers baseline_answers.jsonl --limit 3
"""
import argparse
import json

from eval.harness import print_summary, report_to_dict, run_eval


def main():
    ap = argparse.ArgumentParser(description="APEX Exercise 2 evaluation harness")
    ap.add_argument("--answers", required=True, help="Answers JSONL from baseline_runner or submit_runner")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--corpus", default="corpus", help="Path to downloaded corpus directory")
    ap.add_argument("--out", help="Write full JSON report to this path")
    ap.add_argument("--label", default="eval", help="Label for this run")
    ap.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--no-judge", action="store_true", help="Skip all LLM judges (citation resolve only)")
    ap.add_argument(
        "--no-citation-judge",
        action="store_true",
        help="Skip citation LLM judge (still runs answer judge unless --no-judge)",
    )
    ap.add_argument("--limit", type=int, help="Only evaluate first N questions")
    ap.add_argument("--quiet-judge", action="store_true")
    args = ap.parse_args()

    report = run_eval(
        answers_path=args.answers,
        questions_path=args.questions,
        corpus_path=args.corpus,
        run_label=args.label,
        judge=not args.no_judge,
        citation_judge=not args.no_citation_judge,
        judge_model=args.judge_model,
        limit=args.limit,
        quiet_judge=args.quiet_judge,
    )
    print_summary(report)

    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
