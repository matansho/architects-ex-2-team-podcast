"""Resolve and render prompt configurations for dashboard reports."""

from __future__ import annotations

import html
import json
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
    if "baseline" in lower:
        return "baseline"
    if "contrast" in lower:  # check before few-shot: "Few-shot, contrast" contains both
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
        if rec.get("approach") == "contrast":
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


def build_prompt_spec(preset: str) -> dict:
    if preset == "baseline":
        return {
            "preset": "baseline",
            "runner": "baseline_runner.py",
            "system_prompt": BASELINE_SYSTEM,
            "examples_file": None,
            "examples": [],
            "message_count": 2,
            "note": "Each dev question is sent as a single user message after the system prompt.",
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
                    # assistant turn is the JSON object the model is trained to emit
                    "assistant": json.dumps(ex["answer"], ensure_ascii=False, indent=2),
                    "assistant_dir": "ltr",  # JSON, not Hebrew prose
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
                f"{len(examples)} few-shot user/assistant turns (JSON schema: qualitative + "
                "specifics{best, alt, confident}) precede each dev question; a Python gate then "
                "commits or hedges each specific."
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
        "note": (
            f"{len(examples)} few-shot user/assistant turns precede each dev question "
            "as the final user message."
        ),
    }


def render_prompt_section(specs: list[tuple[str, dict]]) -> str:
    blocks = []
    for label, spec in specs:
        example_blocks = []
        for i, ex in enumerate(spec["examples"], 1):
            ex_id = html.escape(ex.get("id") or f"example-{i}")
            adir = ex.get("assistant_dir", "rtl")
            acls = "prompt-text hebrew" if adir == "rtl" else "prompt-text"
            example_blocks.append(
                f"""
                <details class="prompt-example">
                  <summary>Example {i}{f' · {ex_id}' if ex.get('id') else ''}</summary>
                  <div class="prompt-turn">
                    <div class="prompt-role">User</div>
                    <pre class="prompt-text hebrew" dir="rtl">{html.escape(ex['question'])}</pre>
                  </div>
                  <div class="prompt-turn">
                    <div class="prompt-role">Assistant</div>
                    <pre class="{acls}" dir="{adir}">{html.escape(ex['assistant'])}</pre>
                  </div>
                </details>
                """
            )

        meta = [
            f"<div><span class=\"k\">Runner</span><span class=\"v\">{html.escape(spec['runner'])}</span></div>",
            f"<div><span class=\"k\">Preset</span><span class=\"v\">{html.escape(spec['preset'])}</span></div>",
            f"<div><span class=\"k\">Messages</span><span class=\"v\">{spec['message_count']}</span></div>",
        ]
        if spec["examples_file"]:
            meta.append(
                f"<div><span class=\"k\">Examples file</span>"
                f"<span class=\"v\"><code>{html.escape(spec['examples_file'])}</code></span></div>"
            )

        blocks.append(
            f"""
            <details class="prompt-run" open>
              <summary><strong>{html.escape(label)}</strong></summary>
              <div class="prompt-meta">{''.join(meta)}</div>
              <p class="prompt-note">{html.escape(spec['note'])}</p>
              <div class="prompt-turn">
                <div class="prompt-role">System</div>
                <pre class="prompt-text">{html.escape(spec['system_prompt'])}</pre>
              </div>
              {''.join(example_blocks) if example_blocks else ''}
            </details>
            """
        )

    return f"""
    <section id="prompts">
      <h2>Prompt configurations</h2>
      <p class="section-note">Full system prompts and few-shot examples used for each run.</p>
      <div class="prompt-list">{''.join(blocks)}</div>
    </section>
    """
