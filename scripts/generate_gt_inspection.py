#!/usr/bin/env python3
"""Generate an HTML ground-truth inspection report with source previews.

Example:
  python scripts/generate_gt_inspection.py \
    --input reference_questions.json \
    --output reports/exploration/gt_inspection.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = ROOT / "corpus"
DEFAULT_OUTPUT = ROOT / "reports" / "exploration" / "gt_inspection.html"


CSS = """
:root {
  --bg: #f7f6f2;
  --card: #ffffff;
  --ink: #1f2430;
  --muted: #5f6675;
  --line: #d7dbe3;
  --accent: #006d5b;
  --warn: #8f3b00;
  --err-bg: #fff0ec;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 25% -10%, #eef7ff 0%, transparent 45%),
    radial-gradient(circle at 115% 0%, #fff4e6 0%, transparent 45%),
    var(--bg);
  color: var(--ink);
  font-family: "Noto Sans Hebrew", "Assistant", "Rubik", sans-serif;
  line-height: 1.45;
}
header {
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(4px);
  background: rgba(247, 246, 242, 0.86);
  border-bottom: 1px solid var(--line);
  padding: 14px 18px;
}
header h1 {
  margin: 0;
  font-size: 22px;
}
header .meta {
  margin-top: 6px;
  color: var(--muted);
  font-size: 14px;
}
main {
  max-width: 1400px;
  margin: 0 auto;
  padding: 18px;
}
.toc {
  margin: 0 0 16px;
  padding: 12px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
}
.toc h2 {
  margin: 0 0 8px;
  font-size: 16px;
}
.toc ul {
  margin: 0;
  padding-left: 20px;
  columns: 2;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 18px;
  box-shadow: 0 4px 16px rgba(37, 42, 52, 0.05);
}
.q-title {
  margin: 0;
  font-size: 17px;
}
.badges {
  margin: 8px 0 12px;
  color: var(--muted);
  font-size: 13px;
}
.grid {
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(360px, 1.6fr);
  gap: 14px;
}
.answer {
  direction: rtl;
  text-align: right;
  white-space: pre-wrap;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
  background: #fdfdfd;
}
.source {
  border: 1px solid var(--line);
  border-radius: 10px;
  margin-bottom: 10px;
  padding: 10px;
  background: #fcfcff;
}
.source .label {
  font-size: 13px;
  color: var(--muted);
}
.source img.pdf-page {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 8px;
  background: #fff;
}
.source pre {
  white-space: pre-wrap;
  margin: 8px 0 0;
  max-height: 320px;
  overflow: auto;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
}
.warn {
  color: var(--warn);
  font-size: 13px;
}
.error {
  color: #8c2f20;
  background: var(--err-bg);
  border: 1px solid #ffd2c4;
  border-radius: 8px;
  padding: 8px;
  margin-top: 8px;
}
@media (max-width: 980px) {
  .grid {
    grid-template-columns: 1fr;
  }
  .toc ul {
    columns: 1;
  }
}
"""


def normalize_file_ref(file_ref: str) -> str:
  p = file_ref.strip().replace("\\", "/")
  if p.startswith("corpus/"):
    return p[len("corpus/") :]
  return p


def build_corpus_index(corpus_root: Path) -> dict[str, list[Path]]:
  index: dict[str, list[Path]] = {}
  for p in corpus_root.rglob("*"):
    if not p.is_file():
      continue
    if p.suffix.lower() not in {".pdf", ".txt"}:
      continue
    index.setdefault(p.name, []).append(p)
  return index


def resolve_source_file(
  file_ref: str,
  domain: str,
  corpus_root: Path,
  index: dict[str, list[Path]],
) -> tuple[Path | None, str | None]:
  rel = normalize_file_ref(file_ref)
  if not rel:
    return None, "missing file path"

  direct = corpus_root / rel
  if direct.is_file():
    return direct, None

  basename = Path(rel).name
  candidates = index.get(basename, [])
  if not candidates:
    return None, f"file not found in corpus: {file_ref}"

  if len(candidates) == 1:
    return candidates[0], f"resolved by basename: {basename}"

  domain_matches = [p for p in candidates if p.parts and domain in p.parts]
  if len(domain_matches) == 1:
    return domain_matches[0], (
      f"resolved by basename+domain among {len(candidates)} candidates: {basename}"
    )

  chosen = sorted(candidates)[0]
  return chosen, (
    f"ambiguous basename ({len(candidates)} candidates), picked: "
    f"{chosen.relative_to(corpus_root).as_posix()}"
  )


def load_questions(path: Path) -> list[dict[str, Any]]:
  data = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(data, dict):
    if "questions" in data and isinstance(data["questions"], list):
      return data["questions"]
    raise ValueError("JSON object must contain a 'questions' list")
  if isinstance(data, list):
    return data
  raise ValueError("Input JSON must be either a list or an object with 'questions'")


def esc(val: Any) -> str:
  return html.escape("" if val is None else str(val))


def to_rel_url(target: Path, out_path: Path) -> str:
  rel = Path(target).relative_to(out_path.parent).as_posix() if target.is_relative_to(out_path.parent) else Path(
    Path(os.path.relpath(target, out_path.parent))
  ).as_posix()
  posix = PurePosixPath(rel)
  return "/".join(quote(part) for part in posix.parts)


@lru_cache(maxsize=512)
def render_pdf_page_data_uri(pdf_path_str: str, page_num: int, scale: float) -> tuple[str | None, str | None]:
  pdf_path = Path(pdf_path_str)
  try:
    doc = pdfium.PdfDocument(str(pdf_path))
  except Exception as exc:
    return None, f"Failed opening PDF: {exc}"

  try:
    total_pages = len(doc)
    if page_num < 1 or page_num > total_pages:
      return None, f"page {page_num} out of range (1-{total_pages})"

    page = doc[page_num - 1]
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", None
  except Exception as exc:
    return None, f"Failed rendering page {page_num}: {exc}"
  finally:
    doc.close()


def render_source_block(
  source: dict[str, Any],
  domain: str,
  corpus_root: Path,
  out_path: Path,
  index: dict[str, list[Path]],
) -> str:
  file_ref = source.get("file", "")
  page = source.get("page")
  page_display = "null" if page is None else str(page)
  resolved_path, warning = resolve_source_file(file_ref, domain, corpus_root, index)

  head = (
    f"<div class=\"label\">file: {esc(file_ref)} | page: {esc(page_display)}</div>"
  )

  if warning:
    head += f"<div class=\"warn\">{esc(warning)}</div>"

  if resolved_path is None:
    return f"<div class=\"source\">{head}<div class=\"error\">Missing source file.</div></div>"

  rel_path = resolved_path.relative_to(ROOT).as_posix()
  path_meta = f"<div class=\"label\">resolved: {esc(rel_path)}</div>"
  suffix = resolved_path.suffix.lower()

  if suffix == ".pdf":
    if page is None:
      return (
        "<div class=\"source\">"
        f"{head}{path_meta}<div class=\"warn\">page is null in ground-truth; no page image rendered.</div>"
        "</div>"
      )

    if not isinstance(page, int) or page < 1:
      return (
        "<div class=\"source\">"
        f"{head}{path_meta}<div class=\"error\">Invalid page value: {esc(page)}</div>"
        "</div>"
      )

    img_data_uri, render_err = render_pdf_page_data_uri(str(resolved_path), page, 1.7)
    if render_err:
      return (
        "<div class=\"source\">"
        f"{head}{path_meta}<div class=\"error\">{esc(render_err)}</div>"
        "</div>"
      )

    return (
      "<div class=\"source\">"
      f"{head}{path_meta}"
      f"<img class=\"pdf-page\" loading=\"lazy\" src=\"{esc(img_data_uri)}\" alt=\"{esc(file_ref)} page {page}\">"
      "</div>"
    )

  if suffix == ".txt":
    text = resolved_path.read_text(encoding="utf-8", errors="replace")
    snippet = text[:6000]
    trimmed = "" if len(text) <= len(snippet) else "\n\n...[truncated]"
    return (
      "<div class=\"source\">"
      f"{head}{path_meta}"
      "<div class=\"label\">TXT preview</div>"
      f"<pre>{esc(snippet + trimmed)}</pre>"
      "</div>"
    )

  return (
    "<div class=\"source\">"
    f"{head}{path_meta}<div class=\"error\">Unsupported source type: {esc(suffix)}</div>"
    "</div>"
  )


def render_question(
  q: dict[str, Any],
  corpus_root: Path,
  out_path: Path,
  index: dict[str, list[Path]],
) -> str:
  qid = q.get("id", "")
  domain = q.get("domain", "")
  difficulty = q.get("difficulty", "")
  question = q.get("question", "")
  answer = q.get("ground_truth_answer", "")
  groups_raw = q.get("ground_truth_sources")
  if groups_raw is None:
    groups_raw = q.get("ground_truth_source")
  if groups_raw is None:
    groups_raw = q.get("references")
  if groups_raw is None:
    groups_raw = q.get("sources")

  if groups_raw is None:
    groups: list[Any] = []
  elif isinstance(groups_raw, list):
    groups = groups_raw
  else:
    groups = [groups_raw]

  source_blocks: list[str] = []
  for i, group in enumerate(groups, start=1):
    any_of: list[Any] = []
    if isinstance(group, dict):
      if "any_of" in group and isinstance(group["any_of"], list):
        any_of = group["any_of"]
      elif "all_of" in group and isinstance(group["all_of"], list):
        any_of = group["all_of"]
      elif "file" in group:
        any_of = [group]
    elif isinstance(group, list):
      any_of = group

    if len(any_of) > 1:
      source_blocks.append(
        f"<div class=\"label\">Group {i}: any of {len(any_of)} sources is acceptable.</div>"
      )
    for src in any_of:
      if isinstance(src, dict):
        source_blocks.append(render_source_block(src, domain, corpus_root, out_path, index))

  if not source_blocks:
    source_blocks.append("<div class=\"warn\">No references found for this question.</div>")

  return (
    f"<section class=\"card\" id=\"{esc(qid)}\">"
    f"<h2 class=\"q-title\">{esc(qid)} | {esc(question)}</h2>"
    f"<div class=\"badges\">domain: {esc(domain)} | difficulty: {esc(difficulty)} | source-groups: {esc(len(groups))}</div>"
    "<div class=\"grid\">"
    f"<div><h3>Ground-truth Answer</h3><div class=\"answer\">{esc(answer)}</div></div>"
    f"<div><h3>Referenced Sources</h3>{''.join(source_blocks)}</div>"
    "</div>"
    "</section>"
  )


def build_html(
  questions: list[dict[str, Any]],
  input_path: Path,
  corpus_root: Path,
  out_path: Path,
) -> str:
  index = build_corpus_index(corpus_root)

  toc_items = []
  for q in questions:
    qid = q.get("id", "")
    q_text = str(q.get("question", ""))
    short = (q_text[:80] + "...") if len(q_text) > 80 else q_text
    toc_items.append(f"<li><a href=\"#{esc(qid)}\">{esc(qid)}: {esc(short)}</a></li>")

  sections = [render_question(q, corpus_root, out_path, index) for q in questions]

  return (
    "<!doctype html>"
    "<html lang=\"he\">"
    "<head>"
    "<meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    "<title>Ground-Truth Inspection</title>"
    f"<style>{CSS}</style>"
    "</head>"
    "<body>"
    "<header>"
    "<h1>Ground-Truth Inspection</h1>"
    f"<div class=\"meta\">input: {esc(input_path.relative_to(ROOT).as_posix())} | corpus: {esc(corpus_root.relative_to(ROOT).as_posix())} | questions: {len(questions)}</div>"
    "</header>"
    "<main>"
    "<nav class=\"toc\"><h2>Questions</h2><ul>"
    f"{''.join(toc_items)}"
    "</ul></nav>"
    f"{''.join(sections)}"
    "</main>"
    "</body>"
    "</html>"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Generate an HTML report for manual ground-truth inspection with "
      "side-by-side source previews."
    )
  )
  parser.add_argument("--input", required=True, type=Path, help="JSON file with questions and ground_truth fields")
  parser.add_argument("--output", "--out", dest="output", type=Path, default=DEFAULT_OUTPUT, help=f"Output HTML path (default: {DEFAULT_OUTPUT})")
  parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT, help=f"Corpus root directory (default: {DEFAULT_CORPUS_ROOT})")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  input_path = args.input.resolve()
  out_path = args.output.resolve()
  corpus_root = args.corpus_root.resolve()

  if not input_path.exists():
    raise SystemExit(f"Input JSON not found: {input_path}")
  if not corpus_root.exists():
    raise SystemExit(f"Corpus directory not found: {corpus_root}")

  questions = load_questions(input_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  html_content = build_html(questions, input_path, corpus_root, out_path)
  out_path.write_text(html_content, encoding="utf-8")

  print(f"Wrote {len(questions)} questions to: {out_path}")


if __name__ == "__main__":
  main()
