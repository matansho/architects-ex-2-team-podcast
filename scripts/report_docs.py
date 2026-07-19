"""Render deliverable markdown and compact prompt documentation for HTML reports."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES_DIR = ROOT / "deliverables"


def _truncate(text: str, limit: int = 72) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def markdown_to_html(md: str) -> str:
    """Minimal markdown → HTML for deliverable write-ups (headings + paragraphs)."""
    parts: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            parts.append(f"<p>{html.escape(' '.join(para))}</p>")
            para.clear()

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            flush_para()
            continue
        if line.startswith("## "):
            flush_para()
            parts.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("# "):
            flush_para()
            parts.append(f"<p class=\"doc-lead\">{html.escape(line[2:])}</p>")
        else:
            para.append(line)
    flush_para()
    return "\n".join(parts)


def load_deliverables(deliverables_dir: Path | None = None) -> list[tuple[str, str]]:
    base = deliverables_dir or DELIVERABLES_DIR
    if not base.is_dir():
        return []
    docs: list[tuple[str, str]] = []
    for path in sorted(base.glob("*.md")):
        docs.append((path.stem.replace("_", " ").title(), path.read_text(encoding="utf-8")))
    return docs


def render_deliverables_section(deliverables_dir: Path | None = None) -> str:
    docs = load_deliverables(deliverables_dir)
    if not docs:
        return ""

    blocks = []
    for title, md in docs:
        blocks.append(
            f"""
            <details class="doc-block" open>
              <summary><strong>{html.escape(title)}</strong></summary>
              <div class="doc-body">{markdown_to_html(md)}</div>
            </details>
            """
        )

    return f"""
    <section id="report">
      <h2>Stage 1 report</h2>
      <p class="section-note">Reflection write-up from <code>deliverables/</code>.</p>
      <div class="doc-list">{''.join(blocks)}</div>
    </section>
    """
