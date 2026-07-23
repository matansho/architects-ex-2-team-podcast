#!/usr/bin/env python3
"""Generate an interactive HTML explorer for Docling parse output."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ITEM_COLORS = {
    "SectionHeaderItem": "#2563eb",
    "TextItem": "#374151",
    "ListItem": "#059669",
    "TableItem": "#d97706",
    "PictureItem": "#9ca3af",
    "TitleItem": "#7c3aed",
    "CaptionItem": "#6b7280",
    "FormulaItem": "#db2777",
    "CodeItem": "#0891b2",
    "GroupItem": "#64748b",
}


def esc(text: str) -> str:
    return html.escape(text or "")


def truncate(text: str, n: int = 500) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def item_page(item) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return prov[0].page_no
    return None


def item_text(item) -> str:
    text = getattr(item, "text", None)
    if text:
        return str(text)
    if hasattr(item, "export_to_markdown"):
        try:
            return item.export_to_markdown() or ""
        except Exception:
            pass
    data = getattr(item, "data", None)
    if data is not None:
        return str(data)
    return ""


def table_summary(item) -> str:
    data = getattr(item, "data", None)
    if data is None:
        return "(table — no preview)"
    num_rows = getattr(data, "num_rows", None)
    num_cols = getattr(data, "num_cols", None)
    if num_rows is not None and num_cols is not None:
        return f"Table: {num_rows} rows × {num_cols} cols"
    cells = getattr(data, "table_cells", None) or []
    return f"Table: {len(cells)} cells"


def extract_items(doc) -> list[dict]:
    rows: list[dict] = []
    for idx, (item, level) in enumerate(doc.iterate_items()):
        kind = type(item).__name__
        raw = item_text(item)
        if kind == "TableItem":
            display = table_summary(item)
            preview = truncate(raw, 800) if raw and not raw.startswith("table_cells") else display
        elif kind == "PictureItem":
            display = "(image / logo — no text)"
            preview = display
        else:
            display = truncate(raw, 400)
            preview = raw
        rows.append(
            {
                "idx": idx,
                "level": level,
                "kind": kind,
                "page": item_page(item),
                "display": display,
                "preview": preview,
                "chars": len(raw),
            }
        )
    return rows


def pypdf_page_text(pdf_path: Path, page_no: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if page_no < 1 or page_no > len(reader.pages):
        return f"(page {page_no} out of range; PDF has {len(reader.pages)} pages)"
    return reader.pages[page_no - 1].extract_text() or "(empty)"


def render_item_card(row: dict) -> str:
    kind = row["kind"]
    color = ITEM_COLORS.get(kind, "#64748b")
    page_badge = f'<span class="badge page">p.{row["page"]}</span>' if row["page"] else '<span class="badge muted">no page</span>'
    is_hebrew = kind != "TableItem" and bool(re.search(r"[\u0590-\u05FF]", row["preview"] or ""))
    rtl_attr = ' dir="rtl"' if is_hebrew else ""
    rtl_class = " hebrew" if is_hebrew else ""

    body = esc(row["preview"]) if row["kind"] != "PictureItem" else esc(row["display"])
    if row["kind"] == "TableItem" and len(body) > 1200:
        body = esc(truncate(row["preview"], 1200))

    return f"""
    <div class="item-card" data-kind="{esc(kind)}" data-page="{row['page'] or ''}">
      <div class="item-head">
        <span class="kind-pill" style="background:{color}">{esc(kind)}</span>
        <span class="badge">#{row['idx']}</span>
        <span class="badge">level {row['level']}</span>
        {page_badge}
        <span class="badge muted">{row['chars']} chars</span>
      </div>
      <pre class="item-body{rtl_class}"{rtl_attr}>{body}</pre>
    </div>
    """


def build_html(
    pdf_path: Path,
    txt_path: Path | None,
    pdf_items: list[dict],
    txt_items: list[dict],
    pdf_md: str,
    txt_md: str,
    pdf_pages: int,
) -> str:
    pdf_counts = Counter(r["kind"] for r in pdf_items)
    by_page: dict[int, list[dict]] = defaultdict(list)
    for row in pdf_items:
        if row["page"]:
            by_page[row["page"]].append(row)

    page_options = "".join(
        f'<option value="{p}">Page {p} ({len(by_page[p])} items)</option>' for p in sorted(by_page)
    )

    kind_legend = "".join(
        f'<span class="legend-item"><span class="dot" style="background:{ITEM_COLORS.get(k, "#64748b")}"></span>{esc(k)} ({n})</span>'
        for k, n in pdf_counts.most_common()
    )

    all_cards = "".join(render_item_card(r) for r in pdf_items)
    page_sections = []
    for p in sorted(by_page):
        cards = "".join(render_item_card(r) for r in by_page[p])
        pypdf_sample = esc(truncate(pypdf_page_text(pdf_path, p), 1500))
        page_sections.append(
            f"""
            <section class="page-section" data-page="{p}" id="page-{p}">
              <h3>Page {p} <span class="count">{len(by_page[p])} Docling items</span></h3>
              <details class="compare">
                <summary>Compare: raw pypdf text for page {p}</summary>
                <pre class="compare-body hebrew" dir="rtl">{pypdf_sample}</pre>
                <p class="hint">pypdf is what your citation judge uses today (<code>eval/corpus.py</code>). Docling adds structure on top.</p>
              </details>
              <div class="item-list">{cards}</div>
            </section>
            """
        )

    txt_section = ""
    if txt_path and txt_items:
        txt_cards = "".join(render_item_card(r) for r in txt_items)
        txt_section = f"""
        <section id="txt-sample">
          <h2>TXT sample — {esc(txt_path.name)}</h2>
          <p class="note warn">Docling sees this as <strong>{len(txt_items)} item(s)</strong> with no page numbers.
          Web scrapes need custom cleaning — Docling does not split nav/footer from content.</p>
          <div class="item-list">{txt_cards}</div>
          <details><summary>Exported markdown</summary><pre class="md-preview hebrew" dir="rtl">{esc(truncate(txt_md, 3000))}</pre></details>
        </section>
        """

    # Pick a messy markdown snippet (table with reversed Hebrew) for the "why md looks wrong" panel
    md_lines = pdf_md.splitlines()
    md_snippet_lines = md_lines[10:35] if len(md_lines) > 35 else md_lines[:25]
    md_snippet = esc("\n".join(md_snippet_lines))

    # Find dev-26 anchor items for callout
    dev26_items = [r for r in pdf_items if r["preview"] and ("3.1.1" in r["preview"] or "5.3" in r["preview"])]
    dev26_callout = ""
    if dev26_items:
        dev26_cards = "".join(render_item_card(r) for r in dev26_items[:3])
        dev26_callout = f"""
        <section id="dev26">
          <h2>Ground-truth anchors (dev-26)</h2>
          <p class="note">Search <code>3.1.1</code> (acupuncture) and <code>5.3</code> (12 treatments/year). Docling finds them as <strong>ListItem</strong> on page 2–3 — good chunk boundaries for RAG.</p>
          <div class="item-list">{dev26_cards}</div>
        </section>
        """

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Docling Explorer — {esc(pdf_path.name)}</title>
  <style>
    :root {{
      --bg: #f4f6f9; --card: #fff; --text: #111827; --muted: #6b7280;
      --border: #e5e7eb; --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ background: linear-gradient(135deg, #1e3a5f, #2563eb); color: white; padding: 1.25rem 1.5rem; }}
    header h1 {{ margin: 0 0 .35rem; font-size: 1.35rem; }}
    header p {{ margin: 0; opacity: .9; font-size: .9rem; }}
    nav.toc {{ margin-top: .75rem; display: flex; flex-wrap: wrap; gap: .4rem; }}
    nav.toc a {{ color: white; opacity: .88; text-decoration: none; font-size: .8rem; padding: .2rem .5rem; border: 1px solid rgba(255,255,255,.25); border-radius: 999px; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 1rem 1.25rem 2.5rem; }}
    section {{ background: var(--card); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ margin: 0 0 .65rem; font-size: 1.05rem; }}
    h3 {{ margin: 0 0 .55rem; font-size: .95rem; }}
    .count {{ color: var(--muted); font-weight: 500; font-size: .82rem; }}
    .note {{ background: #eff6ff; border-left: 3px solid var(--accent); padding: .6rem .75rem; border-radius: 8px; font-size: .88rem; margin: 0 0 .75rem; }}
    .note.warn {{ background: #fffbeb; border-color: #d97706; }}
    .hint {{ font-size: .78rem; color: var(--muted); margin: .35rem 0 0; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .5rem; margin-bottom: .75rem; }}
    .stat {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: .55rem .65rem; }}
    .stat .n {{ font-size: 1.25rem; font-weight: 700; color: var(--accent); }}
    .stat .l {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: .45rem .75rem; margin-bottom: .75rem; }}
    .legend-item {{ font-size: .78rem; display: flex; align-items: center; gap: .3rem; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .75rem; }}
    .toolbar label {{ font-size: .78rem; color: var(--muted); display: flex; flex-direction: column; gap: .15rem; }}
    select, input {{ padding: .35rem .5rem; border: 1px solid var(--border); border-radius: 8px; font-size: .85rem; }}
    .item-list {{ display: grid; gap: .55rem; }}
    .item-card {{ border: 1px solid var(--border); border-radius: 10px; padding: .55rem .65rem; background: #fcfdff; }}
    .item-head {{ display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; margin-bottom: .35rem; }}
    .kind-pill {{ color: white; font-size: .68rem; font-weight: 700; padding: .15rem .45rem; border-radius: 999px; }}
    .badge {{ font-size: .68rem; background: #eef2ff; color: #3730a3; padding: .12rem .38rem; border-radius: 999px; font-weight: 600; }}
    .badge.page {{ background: #dbeafe; color: #1d4ed8; }}
    .badge.muted {{ background: #f3f4f6; color: var(--muted); font-weight: 500; }}
    .item-body {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: .84rem; line-height: 1.55; background: white; border: 1px solid var(--border); border-radius: 8px; padding: .55rem .65rem; max-height: 280px; overflow: auto; }}
    .hebrew {{ unicode-bidi: plaintext; }}
    .flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .65rem; }}
    .flow-step {{ border: 1px solid var(--border); border-radius: 10px; padding: .65rem; background: #f8fafc; font-size: .84rem; }}
    .flow-step strong {{ display: block; margin-bottom: .25rem; color: var(--accent); }}
    .compare {{ margin-bottom: .65rem; }}
    .compare-body {{ font-size: .8rem; max-height: 200px; overflow: auto; background: #fefce8; border: 1px solid #fde68a; border-radius: 8px; padding: .55rem; }}
    .md-preview {{ font-size: .78rem; max-height: 240px; overflow: auto; background: #f9fafb; border-radius: 8px; padding: .55rem; }}
    .hidden {{ display: none !important; }}
    .highlight {{ outline: 2px solid #fbbf24; background: #fffbeb !important; }}
    code {{ background: #f3f4f6; padding: .08rem .25rem; border-radius: 4px; font-size: .85em; }}
  </style>
</head>
<body>
  <header>
    <h1>Docling Explorer</h1>
    <p>{esc(str(pdf_path.relative_to(ROOT)))} · {pdf_pages} pages · {len(pdf_items)} structured items · {generated}</p>
    <nav class="toc">
      <a href="#how">How to read this</a>
      <a href="#markdown">Markdown export</a>
      <a href="#dev26">dev-26 anchors</a>
      <a href="#stats">Stats</a>
      <a href="#pages">By page</a>
      <a href="#all">All items</a>
      {"<a href='#txt-sample'>TXT contrast</a>" if txt_section else ""}
    </nav>
  </header>
  <main>
    <section id="how">
      <h2>What Docling did</h2>
      <div class="flow">
        <div class="flow-step"><strong>1. Input</strong>PDF pages (images + text layout)</div>
        <div class="flow-step"><strong>2. Detect</strong>Headings, paragraphs, lists, tables, pictures</div>
        <div class="flow-step"><strong>3. OCR</strong>Hebrew text extracted (can reverse word order)</div>
        <div class="flow-step"><strong>4. Output</strong>Structured items + optional markdown export</div>
      </div>
      <p class="note">Each card below is one <em>item</em> from <code>doc.iterate_items()</code>.
      Use <strong>SectionHeaderItem</strong> + <strong>ListItem</strong> for structural chunking — not the markdown file directly.
      Section numbers like <code>3.1.1</code> (acupuncture) and <code>5.3</code> (12 treatments) appear in list items on pages 2–4.</p>
    </section>

    <section id="markdown">
      <h2>Why <code>pdf.md</code> looked confusing</h2>
      <p class="note warn">The markdown export flattens structure and OCR can reverse Hebrew word order in tables. Use the structured items above for chunking — not the raw markdown file.</p>
      <div class="flow">
        <div class="flow-step"><strong>Headings as ##</strong>Section numbers appear at end: <code>: מבוא . 1</code></div>
        <div class="flow-step"><strong>Tables garbled</strong>Cell text often reversed: <code>מ בע" לביטוח חברה הראל</code></div>
        <div class="flow-step"><strong>Lists OK-ish</strong>Bullet items keep section refs like <code>3.1.1</code></div>
        <div class="flow-step"><strong>Images dropped</strong>Logos become <code>&lt;!-- image --&gt;</code></div>
      </div>
      <details open>
        <summary>Sample from exported markdown (lines 11–35)</summary>
        <pre class="md-preview hebrew" dir="rtl">{md_snippet}</pre>
      </details>
    </section>

    {dev26_callout}

    <section id="stats">
      <h2>Item types</h2>
      <div class="stats">
        <div class="stat"><div class="n">{pdf_pages}</div><div class="l">Pages</div></div>
        <div class="stat"><div class="n">{len(pdf_items)}</div><div class="l">Items</div></div>
        <div class="stat"><div class="n">{pdf_counts.get('ListItem', 0)}</div><div class="l">List items</div></div>
        <div class="stat"><div class="n">{pdf_counts.get('TableItem', 0)}</div><div class="l">Tables</div></div>
        <div class="stat"><div class="n">{len(pdf_md):,}</div><div class="l">MD chars</div></div>
      </div>
      <div class="legend">{kind_legend}</div>
    </section>

    <section id="pages">
      <h2>By page</h2>
      <div class="toolbar">
        <label>Jump to page<select id="page-jump"><option value="">All pages shown</option>{page_options}</select></label>
        <label>Filter kind<select id="kind-filter"><option value="">All kinds</option>{"".join(f'<option value="{esc(k)}">{esc(k)}</option>' for k in pdf_counts)}</select></label>
        <label>Search<input id="search" type="search" placeholder="Hebrew or 3.1.1…" /></label>
      </div>
      {"".join(page_sections)}
    </section>

    <section id="all">
      <h2>All items (flat list)</h2>
      <div class="item-list" id="flat-list">{all_cards}</div>
    </section>

    {txt_section}
  </main>
  <script>
    const jump = document.getElementById('page-jump');
    const kindFilter = document.getElementById('kind-filter');
    const search = document.getElementById('search');

    function applyFilters() {{
      const page = jump.value;
      const kind = kindFilter.value.toLowerCase();
      const q = search.value.trim().toLowerCase();
      document.querySelectorAll('.page-section').forEach(sec => {{
        sec.classList.toggle('hidden', page && sec.dataset.page !== page);
      }});
      document.querySelectorAll('.item-card').forEach(card => {{
        const matchKind = !kind || (card.dataset.kind || '').toLowerCase() === kind;
        const text = (card.innerText || '').toLowerCase();
        const matchSearch = !q || text.includes(q);
        card.classList.toggle('hidden', !(matchKind && matchSearch));
        card.classList.toggle('highlight', q && matchSearch && q.length > 2);
      }});
    }}
    jump.addEventListener('change', () => {{ applyFilters(); if (jump.value) document.getElementById('page-' + jump.value)?.scrollIntoView({{behavior:'smooth'}}); }});
    kindFilter.addEventListener('change', applyFilters);
    search.addEventListener('input', applyFilters);
  </script>
</body>
</html>"""


def load_from_samples(samples_dir: Path) -> tuple[Path, Path | None, list[dict], list[dict], str, str, int]:
    pdf_meta = json.loads((samples_dir / "pdf_meta.json").read_text(encoding="utf-8"))
    txt_meta_path = samples_dir / "txt_meta.json"
    pdf_path = ROOT / pdf_meta["path"]
    pdf_md = (samples_dir / "pdf.md").read_text(encoding="utf-8")
    pdf_pages = pdf_meta["pages"]

    def meta_to_items(structure: list[dict]) -> list[dict]:
        rows = []
        for entry in structure:
            kind = entry["kind"]
            raw = entry.get("text") or ""
            if kind == "TableItem":
                display = table_summary_from_text(raw)
                preview = display if raw.startswith("table_cells") else raw
            elif kind == "PictureItem":
                display = "(image / logo — no text)"
                preview = display
            else:
                display = truncate(raw, 400)
                preview = raw
            rows.append(
                {
                    "idx": entry["i"],
                    "level": entry["level"],
                    "kind": kind,
                    "page": entry.get("page"),
                    "display": display,
                    "preview": preview,
                    "chars": len(raw),
                }
            )
        return rows

    pdf_items = meta_to_items(pdf_meta["structure_preview"])

    txt_path = txt_meta_path and ROOT / json.loads(txt_meta_path.read_text(encoding="utf-8"))["path"]
    txt_items: list[dict] = []
    txt_md = ""
    if txt_meta_path.exists():
        txt_meta = json.loads(txt_meta_path.read_text(encoding="utf-8"))
        txt_path = ROOT / txt_meta["path"]
        txt_items = meta_to_items(txt_meta["structure_preview"])
        txt_md = (samples_dir / "txt.md").read_text(encoding="utf-8")

    return pdf_path, txt_path, pdf_items, txt_items, pdf_md, txt_md, pdf_pages


def table_summary_from_text(raw: str) -> str:
    m = re.search(r"num_rows=(\d+).*num_cols=(\d+)", raw)
    if m:
        return f"Table: {m.group(1)} rows × {m.group(2)} cols"
    cells = re.search(r"table_cells=\[", raw)
    return "Table (cell dump)" if cells else "(table — no preview)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Docling explorer HTML")
    ap.add_argument(
        "--pdf",
        default="corpus/health/files/כתב-שירות-שירותי-רפואה-משלימה-אלטרנטיבית.pdf",
    )
    ap.add_argument(
        "--txt",
        default="corpus/apartment/pages/earthquak-insurance-coverage.txt",
        help="Optional TXT for contrast (set empty to skip)",
    )
    ap.add_argument(
        "--from-samples",
        default="",
        help="Use cached reports/docling_samples instead of re-running Docling",
    )
    ap.add_argument("--out", default="reports/docling_explorer.html")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_samples:
        samples_dir = ROOT / args.from_samples
        pdf_path, txt_path, pdf_items, txt_items, pdf_md, txt_md, pdf_pages = load_from_samples(
            samples_dir
        )
    else:
        from docling.document_converter import DocumentConverter

        pdf_path = ROOT / args.pdf
        txt_path = ROOT / args.txt if args.txt else None

        converter = DocumentConverter()
        print(f"Converting PDF: {pdf_path}", flush=True)
        pdf_doc = converter.convert(str(pdf_path)).document
        pdf_items = extract_items(pdf_doc)
        pdf_md = pdf_doc.export_to_markdown()
        pdf_pages = len(getattr(pdf_doc, "pages", {}) or {}) or max(
            (r["page"] or 0 for r in pdf_items), default=0
        )

        txt_items: list[dict] = []
        txt_md = ""
        if txt_path and txt_path.exists():
            print(f"Converting TXT: {txt_path}", flush=True)
            txt_doc = converter.convert(str(txt_path)).document
            txt_items = extract_items(txt_doc)
            txt_md = txt_doc.export_to_markdown()

    html_out = build_html(pdf_path, txt_path, pdf_items, txt_items, pdf_md, txt_md, pdf_pages)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out_path} ({len(pdf_items)} PDF items)")


if __name__ == "__main__":
    main()
