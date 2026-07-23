"""Resolve and render prompt configurations for dashboard reports."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from baseline_runner import DEFAULT_SYSTEM as BASELINE_SYSTEM
from few_shot_runner import (
    PRESET_EXAMPLES,
    PRESETS,
    build_messages,
    format_assistant_answer,
    load_examples,
)


def infer_preset_from_label(label: str) -> str | None:
    lower = label.lower()
    if "cascade" in lower or "rag" in lower:
        return "rag-no-cite"
    if "baseline" in lower:
        return "baseline"
    if "contrast" in lower:
        return "contrast"
    if "no cite" in lower or "no-cite" in lower:
        return "no-cite"
    if "concise" in lower:
        return "concise"
    if "few-shot" in lower or "few shot" in lower:
        return "default"
    return None


def infer_preset_from_answers(answers_path: Path) -> str | None:
    if not answers_path.exists():
        return None
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        approach = rec.get("approach") or ""
        if approach.startswith("rag"):
            return "rag-no-cite"
        if approach == "contrast":
            return "contrast"
        preset = rec.get("preset")
        if preset:
            return preset
        if rec.get("few_shot_examples"):
            return "default"
        return "baseline"
    return None


def resolve_preset(label: str, answers_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    from_answers = infer_preset_from_answers(answers_path)
    if from_answers:
        return from_answers
    from_label = infer_preset_from_label(label)
    if from_label:
        return from_label
    return "baseline"


def _truncate(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_prompt_spec(preset: str) -> dict:
    if preset == "baseline":
        return {
            "preset": "baseline",
            "runner": "baseline_runner.py",
            "system_prompt": BASELINE_SYSTEM,
            "examples_file": None,
            "examples": [],
            "message_count": 2,
            "note": "System prompt + one user message per dev question.",
        }

    if preset == "rag-no-cite":
        from rag.generate import SYSTEM_NO_CITE

        return {
            "preset": "rag-no-cite",
            "runner": "rag_runner.py",
            "system_prompt": SYSTEM_NO_CITE,
            "examples_file": None,
            "examples": [],
            "message_count": 2,
            "note": (
                "Dense retrieve top-k (+ neighbor window) → context passages + question. "
                "Answer only from context; citations disabled."
            ),
        }

    if preset == "contrast":
        import contrast_runner as cr

        examples = cr.load_examples()
        sample_question = "[dev question from reference_questions.json]"
        messages = cr.build_messages(sample_question, examples)
        rendered_examples = []
        for ex in examples:
            rendered_examples.append(
                {
                    "id": ex.get("id"),
                    "question": ex["question"],
                    "assistant": json.dumps(ex["answer"], ensure_ascii=False, indent=2),
                    "assistant_dir": "ltr",
                }
            )
        return {
            "preset": "contrast",
            "runner": "contrast_runner.py",
            "system_prompt": cr.CONTRAST_SYSTEM,
            "examples_file": cr.DEFAULT_EXAMPLES,
            "examples": rendered_examples,
            "message_count": len(messages),
            "note": (
                "JSON schema (qualitative + specifics{best, alt, confident}); "
                "Python gate commits or hedges each specific."
            ),
        }

    examples_path = Path(PRESET_EXAMPLES.get(preset, PRESET_EXAMPLES["default"]))
    examples = load_examples(ROOT / examples_path)
    sample_question = "[dev question from reference_questions.json]"
    messages = build_messages(sample_question, examples, PRESETS[preset])
    rendered_examples = []
    for ex in examples:
        rendered_examples.append(
            {
                "id": ex.get("id"),
                "question": ex["question"],
                "assistant": format_assistant_answer(ex),
                "assistant_dir": "rtl",
            }
        )
    return {
        "preset": preset,
        "runner": "few_shot_runner.py",
        "system_prompt": PRESETS[preset],
        "examples_file": str(examples_path),
        "examples": rendered_examples,
        "message_count": len(messages),
        "note": f"{len(examples)} few-shot turns, then the dev question.",
    }


def _dedupe_specs(specs: list[tuple[str, dict]]) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    mappings: list[tuple[str, str]] = []
    unique: dict[str, dict] = {}
    for label, spec in specs:
        preset = spec["preset"]
        mappings.append((label, preset))
        if preset not in unique:
            unique[preset] = spec
    return mappings, unique


def _render_examples_compact(examples: list[dict]) -> str:
    if not examples:
        return ""

    rows = []
    for i, ex in enumerate(examples, 1):
        ex_id = html.escape(ex.get("id") or f"ex-{i}")
        q_prev = html.escape(_truncate(ex["question"], 90))
        a_prev = html.escape(_truncate(ex["assistant"], 90))
        adir = ex.get("assistant_dir", "rtl")
        rows.append(
            f"""
            <details class="prompt-example">
              <summary><span class="ex-num">{i}</span> {ex_id} · {q_prev}</summary>
              <div class="prompt-turn">
                <div class="prompt-role">User</div>
                <pre class="prompt-text hebrew" dir="rtl">{html.escape(ex['question'])}</pre>
              </div>
              <div class="prompt-turn">
                <div class="prompt-role">Assistant</div>
                <pre class="prompt-text{' hebrew' if adir == 'rtl' else ''}" dir="{adir}">{html.escape(ex['assistant'])}</pre>
              </div>
            </details>
            """
        )

    return f"""
    <details class="prompt-examples-wrap">
      <summary>{len(examples)} few-shot examples — previews; expand any row for full text</summary>
      <div class="prompt-examples-list">{''.join(rows)}</div>
    </details>
    """


def render_prompt_section(specs: list[tuple[str, dict]]) -> str:
    mappings, unique = _dedupe_specs(specs)

    map_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(preset)}</code></td></tr>"
        for label, preset in mappings
    )

    preset_blocks = []
    for preset, spec in unique.items():
        meta = [
            f"<span><code>{html.escape(spec['runner'])}</code></span>",
            f"<span>{spec['message_count']} msgs</span>",
        ]
        if spec["examples_file"]:
            meta.append(f"<span><code>{html.escape(spec['examples_file'])}</code></span>")
        meta_html = " · ".join(meta)

        preset_blocks.append(
            f"""
            <details class="prompt-preset">
              <summary><strong>{html.escape(preset)}</strong> — {meta_html}</summary>
              <p class="prompt-note">{html.escape(spec['note'])}</p>
              <details class="prompt-system">
                <summary>System prompt</summary>
                <pre class="prompt-text">{html.escape(spec['system_prompt'])}</pre>
              </details>
              {_render_examples_compact(spec['examples'])}
            </details>
            """
        )

    return f"""
    <section id="prompts">
      <h2>Prompt configurations</h2>
      <p class="section-note">{len(unique)} unique presets across {len(mappings)} runs. Expand for system prompts and examples.</p>
      <div class="prompt-map-wrap">
        <table class="prompt-map">
          <thead><tr><th>Run</th><th>Preset</th></tr></thead>
          <tbody>{map_rows}</tbody>
        </table>
      </div>
      <div class="prompt-list">{''.join(preset_blocks)}</div>
    </section>
    """
