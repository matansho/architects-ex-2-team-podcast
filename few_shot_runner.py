"""
Few-shot baseline runner: same as baseline_runner.py but prepends exemplar Q&A
pairs to the conversation before each evaluation question.

    python few_shot_runner.py --model deepseek-ai/DeepSeek-V4-Pro
    python few_shot_runner.py --show-prompt --limit 1   # preview prompt only

The few-shot examples live in few_shot_examples.json. They are NOT from the
dev evaluation set — they were authored from uncited corpus pages to teach
answer style, citation format, and safe refusal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a customer-support assistant for Harel Insurance (Israel). "
    "Answer the customer's question in the language it was asked. "
    "Be direct: start yes/no questions with כן or לא when appropriate. "
    "For numeric questions, state the exact number, limit, or period. "
    "When you know the source, cite the exact document path "
    "(and page number for PDFs). "
    "If you lack enough information for a specific fact, say so clearly — "
    "do not guess."
)

CONCISE_SYSTEM = (
    "You are a customer-support assistant for Harel Insurance (Israel). "
    "Answer the customer's question in the language it was asked. "
    "Be concise: include only the facts required to answer the question. "
    "No introductions, filler, background, or generic disclaimers. "
    "Start yes/no questions with כן or לא when appropriate. "
    "For numeric questions, state only the exact number, limit, or period. "
    "If you lack enough information for a specific fact, say so briefly — "
    "do not guess."
)

NO_CITE_SYSTEM = (
    "You are a customer-support assistant for Harel Insurance (Israel). "
    "Answer the customer's question in the language it was asked. "
    "Be direct: start yes/no questions with כן or לא when appropriate. "
    "For numeric questions, state the exact number, limit, or period. "
    "Do not cite sources, document paths, page numbers, or a מקורות section. "
    "If you lack enough information for a specific fact, say so clearly — "
    "do not guess."
)

PRESETS = {
    "default": DEFAULT_SYSTEM,
    "concise": CONCISE_SYSTEM,
    "no-cite": NO_CITE_SYSTEM,
}

DEFAULT_EXAMPLES = "few_shot_examples.json"
PRESET_EXAMPLES = {
    "default": DEFAULT_EXAMPLES,
    "concise": DEFAULT_EXAMPLES,
    "no-cite": "few_shot_examples_no_cite.json",
}


def load_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("examples", data.get("few_shot", []))
    return data


def format_assistant_answer(example: dict) -> str:
    """Render assistant turn: answer text + citation line if present."""
    answer = example["answer"]
    cites = example.get("citations") or []
    if not cites:
        return answer
    lines = [answer, "", "מקורות:"]
    for c in cites:
        loc = c["file"]
        if c.get("page") is not None:
            loc += f", עמוד {c['page']}"
        lines.append(f"- {loc}")
    return "\n".join(lines)


def build_messages(
    question: str,
    examples: list[dict],
    system_prompt: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": format_assistant_answer(ex)})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt_preview(messages: list[dict]) -> str:
    """Human-readable prompt dump for inspection."""
    parts = []
    for i, m in enumerate(messages):
        parts.append(f"{'='*60}\n[{i}] {m['role'].upper()}\n{'='*60}\n{m['content']}\n")
    return "\n".join(parts)


def resolve_model(model: str) -> tuple[str, dict]:
    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
        model = f"openai/{model.removeprefix('openai/')}"
    elif "/" not in model:
        model = f"openai/{model}"
    return model, kwargs


def main():
    ap = argparse.ArgumentParser(description="Few-shot baseline runner")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default",
                    help="System prompt preset (default | concise | no-cite)")
    ap.add_argument("--system-prompt", help="Override system prompt text")
    ap.add_argument("--out", default="few_shot_answers.jsonl")
    ap.add_argument("--limit", type=int, help="Only run first N questions")
    ap.add_argument("--show-prompt", action="store_true",
                    help="Print the full prompt for the first question and exit")
    args = ap.parse_args()
    system_prompt = args.system_prompt or PRESETS[args.preset]
    examples_path = Path(args.examples)
    if args.examples == DEFAULT_EXAMPLES and args.preset in PRESET_EXAMPLES:
        examples_path = Path(PRESET_EXAMPLES[args.preset])

    examples = load_examples(examples_path)
    questions = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]
    if args.limit:
        questions = questions[: args.limit]

    model, kwargs = resolve_model(args.model)

    if args.show_prompt:
        q0 = questions[0]["question"]
        messages = build_messages(q0, examples, system_prompt)
        print(render_prompt_preview(messages))
        print(f"\n({len(messages)} messages, {len(examples)} few-shot examples)")
        return

    import litellm  # lazy: keep module import litellm-free (slow) for prompt reuse

    with open(args.out, "w", encoding="utf-8") as out:
        for q in questions:
            messages = build_messages(q["question"], examples, system_prompt)
            t0 = time.time()
            resp = litellm.completion(
                model=model,
                messages=messages,
                timeout=120,
                **kwargs,
            )
            rec = {
                "id": q["id"],
                "answer": resp.choices[0].message.content,
                "citations": [],
                "latency_ms": (time.time() - t0) * 1000,
                "tokens": {
                    "prompt": resp.usage.prompt_tokens,
                    "completion": resp.usage.completion_tokens,
                },
                "few_shot_examples": len(examples),
                "examples_file": str(examples_path),
                "preset": args.preset if not args.system_prompt else "custom",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {rec['answer'][:70]!r}... ({rec['latency_ms']:.0f} ms)")
    print(f"\nwrote {args.out} — score with: python run_eval.py --answers {args.out}")


if __name__ == "__main__":
    main()
