"""
Approach C — In-generation self-contrast.

Single LLM call per question. Instead of K separate samples (self-consistency),
the model exposes its own uncertainty *inside one completion*: for every specific
value it must give its `best` guess AND the most plausible `alt`ernative, plus a
`confident` flag. A deterministic Python gate then commits `best` only when the
model is confident with a clear preference; otherwise it hedges. This approximates
sampling-variance abstention at the price of a few extra output tokens on one call.

    python contrast_runner.py --out reports/stage1/contrast_answers.jsonl
    python contrast_runner.py --show-prompt --limit 1
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from few_shot_runner import resolve_model

# Helpers kept inline so this runner is self-contained (no cross-runner dependency).
HEDGE = "את הפרט המדויק יש לוודא בדף פרטי הביטוח שלך או מול הראל."

_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"(\*\d{3,4}\b|\b0\d[\d\-\s]{6,}\d\b|\b1[\-\s]?800[\-\s\d]{5,}\b)")


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"non-JSON model output: {text[:200]!r}")


def _strip_contacts(text: str) -> str:
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()

CONTRAST_SYSTEM = (
    "You are a customer-support assistant for Harel Insurance (Israel). You have "
    "NO access to Harel's policy documents. Answer in the language the customer used.\n"
    "Return ONLY a JSON object with exactly these fields:\n"
    '  "qualitative": string — the directional answer (yes/no, covered/excluded, whether a '
    "limit or waiting period exists, how it works, statutory or logical facts). Put NO "
    "specific numbers, amounts, percentages, ages, dates, emails, phones, or product names here.\n"
    '  "specifics": array, one object per checkable value the answer needs, each '
    '{ "claim_type": string (what the value is, e.g. "תקרת כיסוי"), '
    '"best": string (your single best value, in the answer\'s language), '
    '"alt": string (the most plausible DIFFERENT value it could be — give a real alternative, '
    'not a copy of best), '
    '"confident": boolean (true only if you are certain enough to rule the alternative out) }.\n'
    "Be honest: if the value is a Harel-chosen constant you cannot verify (a specific cap, "
    "sub-limit, deductible %, waiting period, join age, product name, email, phone), the "
    "alternative is genuinely plausible, so confident MUST be false. Only set confident = true "
    "for values fixed by law, logic, or arithmetic on numbers stated in the question."
)

DEFAULT_EXAMPLES = "few_shot_examples_contrast.json"


def load_examples(path: str = DEFAULT_EXAMPLES) -> list[dict]:
    """Load contrast-schema exemplars: [{question, answer:{qualitative, specifics}}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("examples", data.get("few_shot", []))
    return data


def build_messages(question: str, examples: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": CONTRAST_SYSTEM}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append(
            {"role": "assistant", "content": json.dumps(ex["answer"], ensure_ascii=False)}
        )
    messages.append({"role": "user", "content": question})
    return messages


def _num_tokens(s: str) -> list[float]:
    """Pull comparable numeric magnitudes out of a value string (ignores separators)."""
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s or "")]


def _far_apart(best: str, alt: str) -> bool:
    """True if best and alt are materially different values."""
    if not alt or alt.strip() == best.strip():
        return False  # no real alternative offered
    nb, na = _num_tokens(best), _num_tokens(alt)
    if nb and na:
        b, a = nb[0], na[0]
        hi = max(abs(b), abs(a), 1.0)
        return abs(b - a) / hi > 0.05  # >5% apart = materially different
    return True  # non-numeric (e.g. two different emails/names) → treat as far apart


def render_answer(parsed: dict, use_far_apart: bool = False) -> str:
    qualitative = _strip_contacts(str(parsed.get("qualitative", "")).strip())
    specifics = parsed.get("specifics") or []
    kept: list[str] = []
    hedged = False
    for s in specifics:
        if not isinstance(s, dict):
            hedged = True
            continue
        best = str(s.get("best", "")).strip()
        alt = str(s.get("alt", "")).strip()
        confident = bool(s.get("confident", False))
        if not best:
            continue
        # Commit when the model is confident. With --far-apart, additionally require
        # its own alternative to sit close to its best (low self-spread); a far-apart
        # alternative then means the model sees a materially different plausible value
        # → hedge. Without the flag (default) trust the confident flag alone (option 1).
        if confident and (not use_far_apart or not _far_apart(best, alt)):
            kept.append(best)
        else:
            hedged = True

    parts = [qualitative] if qualitative else []
    parts.extend(kept)
    if hedged:
        parts.append(HEDGE)
    return " ".join(p for p in parts if p).strip() or HEDGE


def main():
    ap = argparse.ArgumentParser(description="Approach C — self-contrast runner")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--out", default="reports/stage1/contrast_answers.jsonl")
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--far-apart", action="store_true",
                    help="also hedge a confident specific whose alt is far from its best "
                         "(self-spread gate); OFF by default = commit iff confident")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    examples = load_examples(args.examples)
    questions = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]
    if args.limit:
        questions = questions[: args.limit]

    model, kwargs = resolve_model(args.model)

    if args.show_prompt:
        for m in build_messages(questions[0]["question"], examples):
            print(f"{'='*60}\n{m['role'].upper()}\n{'='*60}\n{m['content']}\n")
        return

    import litellm  # lazy: keep module import litellm-free (slow) for prompt reuse

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for q in questions:
            messages = build_messages(q["question"], examples)
            t0 = time.time()
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=args.temperature,
                timeout=120,
                response_format={"type": "json_object"},
                **kwargs,
            )
            raw = resp.choices[0].message.content or ""
            try:
                parsed = _extract_json(raw)
                answer = render_answer(parsed, use_far_apart=args.far_apart)
            except (ValueError, json.JSONDecodeError):
                answer = raw
            rec = {
                "id": q["id"],
                "answer": answer,
                "citations": [],
                "latency_ms": (time.time() - t0) * 1000,
                "tokens": {
                    "prompt": resp.usage.prompt_tokens,
                    "completion": resp.usage.completion_tokens,
                },
                "approach": "contrast",
                "raw": raw,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {answer[:70]!r}... ({rec['latency_ms']:.0f} ms)")
    eval_out = re.sub(r"_?answers\.jsonl$", "_eval.json", args.out)
    if eval_out == args.out:
        eval_out = "reports/stage1/contrast_eval.json"
    print(f"\nwrote {args.out} — score with: python run_eval.py --answers {args.out} "
          f"--out {eval_out} --no-citation-judge")


if __name__ == "__main__":
    main()
