#!/usr/bin/env python3
"""Generate the single Stage-2 PDF chunking report (logic + stats + overlay)."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk import chunk_blocks, chunk_border, chunk_color  # noqa: E402
from rag.chunk_stats import summarize  # noqa: E402
from rag.parse import build_tree, docling_to_blocks  # noqa: E402

# Hierarchy token sizes from 12 policy-like PDFs (פוליס/תנאי/כתב-שירות), cl100k_base.
# Regenerated historically via structural size probe — embedded so the report
# stays one-file without re-parsing the full sample every time.
HIERARCHY_STATS = {
    "sample": "12 policy-like PDFs across domains (filename filter)",
    "encoding": "cl100k_base",
    "levels": [
        {
            "key": "full_doc",
            "title": "Full document",
            "desc": "Entire PDF as one blob",
            "n": 12,
            "mean": 58732.8,
            "min": 3901,
            "p50": 54358,
            "p75": 72760,
            "p90": 116666,
            "p95": 142854,
            "max": 169975,
        },
        {
            "key": "page",
            "title": "Page",
            "desc": "All Docling text on one page",
            "n": 466,
            "mean": 1512.0,
            "min": 25,
            "p50": 1756,
            "p75": 2212,
            "p90": 2400,
            "p95": 2529,
            "max": 3001,
        },
        {
            "key": "section",
            "title": "Section (§N)",
            "desc": "Depth-1 numbered unit + descendants",
            "n": 586,
            "mean": 1178.3,
            "min": 1,
            "p50": 147,
            "p75": 1002,
            "p90": 3162,
            "p95": 5448,
            "max": 44314,
        },
        {
            "key": "clause",
            "title": "Clause (§N.M)",
            "desc": "Depth-2 + descendants",
            "n": 478,
            "mean": 513.9,
            "min": 3,
            "p50": 228,
            "p75": 577,
            "p90": 1309,
            "p95": 1907,
            "max": 7092,
        },
        {
            "key": "subclause",
            "title": "Sub-clause (§N.M.K+)",
            "desc": "Depth ≥ 3 + descendants",
            "n": 319,
            "mean": 358.9,
            "min": 5,
            "p50": 161,
            "p75": 408,
            "p90": 885,
            "p95": 1251,
            "max": 5503,
        },
        {
            "key": "paragraph",
            "title": "Paragraph / item",
            "desc": "Single Docling text/list item",
            "n": 6063,
            "mean": 115.1,
            "min": 1,
            "p50": 85,
            "p75": 159,
            "p90": 255,
            "p95": 324,
            "max": 1762,
        },
    ],
    "outside_clause_note": (
        "~65% of text tokens sit at section-only or unnumbered depth "
        "(not under §N.M+), so clause-only indexing is not enough — "
        "section/orphan packing is required."
    ),
}

KIND_ORDER = [
    "subclause",
    "subclause_pack",
    "clause",
    "clause_pack",
    "section_pack",
    "orphan_pack",
    "split",
]

KIND_COLORS = {
    "subclause": "#2563eb",
    "subclause_pack": "#3b82f6",
    "clause": "#0891b2",
    "clause_pack": "#06b6d4",
    "section_pack": "#d97706",
    "orphan_pack": "#7c3aed",
    "split": "#dc2626",
}


def convert_with_images(pdf: Path, scale: float = 1.15):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.generate_page_images = True
    opts.images_scale = scale
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return conv.convert(str(pdf)).document


def page_image_jpeg(page_item, quality: int = 55) -> bytes:
    pil = page_item.image.pil_image
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def pdf_bbox_to_pct(bbox, page_size) -> tuple[float, float, float, float]:
    pw = float(page_size.width)
    ph = float(page_size.height)
    l, t, r, b = bbox
    top_y = ph - t
    bottom_y = ph - b
    x = min(l, r)
    w = abs(r - l)
    y = min(top_y, bottom_y)
    h = abs(bottom_y - top_y)
    return (100.0 * y / ph, 100.0 * x / pw, 100.0 * w / pw, 100.0 * h / ph)


def fmt(n) -> str:
    if n is None:
        return "—"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _chart_payload(chunks, target: int, max_tokens: int) -> dict:
    """JSON consumed by Plotly.js in the report."""
    levels = [
        lvl
        for lvl in HIERARCHY_STATS["levels"]
        if lvl["key"] != "full_doc"  # scale dwarfs the rest
    ]
    by_kind: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_kind[c.kind].append(c.tokens)
    toks = [c.tokens for c in chunks]

    return {
        "target": target,
        "max_tokens": max_tokens,
        "hierarchy": {
            "labels": [lvl["title"] for lvl in levels],
            "keys": [lvl["key"] for lvl in levels],
            "p50": [lvl["p50"] for lvl in levels],
            "mean": [lvl["mean"] for lvl in levels],
            "p95": [lvl["p95"] for lvl in levels],
            "n": [lvl["n"] for lvl in levels],
        },
        "vectors": {
            "tokens": toks,
            "kinds": [c.kind for c in chunks],
            "ids": [c.id for c in chunks],
            "labels": [c.label for c in chunks],
            "by_kind": {k: by_kind[k] for k in KIND_ORDER if k in by_kind},
            "kind_counts": dict(Counter(c.kind for c in chunks)),
            "kind_colors": KIND_COLORS,
        },
    }


def build_report(
    *,
    pdf_rel: str,
    pdf_name: str,
    chunks,
    blocks,
    block_to_chunk: dict[int, int],
    page_jpegs: dict[int, bytes],
    page_sizes: dict,
    target: int,
    max_tokens: int,
    max_pages: int | None,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    toks = sorted(c.tokens for c in chunks)
    vstats = summarize(toks)
    kinds = Counter(c.kind for c in chunks)
    chart_json = json.dumps(_chart_payload(chunks, target, max_tokens), ensure_ascii=False)

    # Hierarchy stats table
    h_rows = []
    for lvl in HIERARCHY_STATS["levels"]:
        h_rows.append(
            f"""<tr>
              <td><strong>{html.escape(lvl['title'])}</strong>
                <div class="desc">{html.escape(lvl['desc'])}</div></td>
              <td class="num">{fmt(lvl['n'])}</td>
              <td class="num">{fmt(lvl['mean'])}</td>
              <td class="num">{fmt(lvl['min'])}</td>
              <td class="num">{fmt(lvl['p50'])}</td>
              <td class="num">{fmt(lvl['p75'])}</td>
              <td class="num">{fmt(lvl['p90'])}</td>
              <td class="num">{fmt(lvl['p95'])}</td>
              <td class="num">{fmt(lvl['max'])}</td>
            </tr>"""
        )

    # Per-kind vector stats
    kind_stat_rows = []
    for k in KIND_ORDER:
        if k not in kinds:
            continue
        ks = summarize([c.tokens for c in chunks if c.kind == k])
        kind_stat_rows.append(
            f"""<tr>
              <td><span class="swatch" style="background:{KIND_COLORS.get(k, '#94a3b8')}"></span>
                {html.escape(k)}</td>
              <td class="num">{kinds[k]}</td>
              <td class="num">{fmt(ks['mean'])}</td>
              <td class="num">{fmt(ks['p50'])}</td>
              <td class="num">{fmt(ks['p95'])}</td>
              <td class="num">{fmt(ks['min'])}</td>
              <td class="num">{fmt(ks['max'])}</td>
            </tr>"""
        )

    # Sidebar + pages
    side_items = []
    for c in chunks:
        pages = ",".join(str(p) for p in c.pages[:5]) + ("…" if len(c.pages) > 5 else "")
        side_items.append(
            f"""<button type="button" class="chunk-item" data-chunk="{c.id}"
                style="border-left:4px solid {chunk_border(c.id)}; background:{chunk_color(c.id)}">
              <span class="cid">#{c.id}</span>
              <span class="lab">{html.escape(c.label)}</span>
              <span class="meta">{html.escape(c.kind)} · {c.tokens} tok · p{html.escape(pages)}</span>
            </button>"""
        )

    overlays: dict[int, list] = defaultdict(list)
    for b in blocks:
        cid = block_to_chunk.get(b.idx)
        if cid is None:
            continue
        for page, l, t, r, bot in b.bboxes:
            if page not in page_sizes:
                continue
            top, left, w, h = pdf_bbox_to_pct((l, t, r, bot), page_sizes[page])
            overlays[page].append((cid, top, left, w, h))

    page_nos = sorted(page_jpegs)
    if max_pages:
        page_nos = page_nos[:max_pages]

    page_sections = []
    for pno in page_nos:
        b64 = base64.b64encode(page_jpegs[pno]).decode("ascii")
        rects = []
        for cid, top, left, w, h in overlays.get(pno, []):
            rects.append(
                f"""<div class="ov" data-chunk="{cid}" title="#{cid}"
                  style="top:{top:.2f}%;left:{left:.2f}%;width:{w:.2f}%;height:{h:.2f}%;
                         background:{chunk_color(cid)};outline:1px solid {chunk_border(cid)}"></div>"""
            )
        page_sections.append(
            f"""<section class="page" id="page-{pno}">
              <h3>Page {pno}</h3>
              <div class="page-wrap">
                <img src="data:image/jpeg;base64,{b64}" alt="page {pno}" />
                <div class="overlay">{''.join(rects)}</div>
              </div>
            </section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PDF chunking report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {{ --bg:#f1f5f9; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --accent:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header.top {{ background:#0f172a; color:#fff; padding:1rem 1.25rem; }}
    header.top h1 {{ margin:0 0 .3rem; font-size:1.25rem; }}
    header.top p {{ margin:0; opacity:.88; font-size:.86rem; }}
    nav.toc {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.75rem; }}
    nav.toc a {{ color:#fff; text-decoration:none; font-size:.78rem; padding:.2rem .55rem;
                 border:1px solid rgba(255,255,255,.28); border-radius:999px; opacity:.9; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:1rem 1.25rem 2.5rem; }}
    section.card {{ background:var(--card); border-radius:12px; padding:1rem 1.1rem; margin-bottom:1rem;
                    box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ margin:0 0 .65rem; font-size:1.05rem; }}
    h3 {{ margin:1rem 0 .4rem; font-size:.92rem; }}
    h4 {{ margin:.85rem 0 .35rem; font-size:.82rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .note {{ background:#eff6ff; border-left:3px solid var(--accent); padding:.55rem .75rem;
             border-radius:8px; font-size:.86rem; margin:0 0 .75rem; }}
    .warn {{ background:#fffbeb; border-left-color:#d97706; }}
    ol.steps, ul {{ font-size:.88rem; line-height:1.55; }}
    code {{ background:#f1f5f9; padding:.08rem .28rem; border-radius:4px; font-size:.84em; }}
    table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
    th, td {{ padding:.4rem .45rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .desc {{ font-size:.72rem; color:var(--muted); margin-top:.12rem; }}
    .flow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.5rem; margin:.75rem 0; }}
    .flow div {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.55rem .65rem; font-size:.82rem; }}
    .flow strong {{ display:block; color:var(--accent); margin-bottom:.2rem; font-size:.78rem; }}
    .kpi {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:.5rem; margin:.5rem 0 .75rem; }}
    .kpi .box {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.5rem .6rem; }}
    .kpi .n {{ font-size:1.2rem; font-weight:700; color:var(--accent); }}
    .kpi .l {{ font-size:.68rem; color:var(--muted); text-transform:uppercase; }}
    .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin:.75rem 0; }}
    .charts .full {{ grid-column:1 / -1; }}
    .chart {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.35rem .5rem .15rem; min-height:280px; }}
    .chart-title {{ font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin:.25rem .35rem 0; }}
    .mermaid {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.75rem; margin:.5rem 0 .85rem; overflow-x:auto; }}
    .swatch {{ display:inline-block; width:.65rem; height:.65rem; border-radius:2px; margin-right:.35rem; vertical-align:middle; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:.85rem; }}
    .overlay-layout {{ display:grid; grid-template-columns:280px 1fr; gap:.75rem; min-height:70vh; }}
    aside.chunks {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; overflow:auto;
                    max-height:75vh; position:sticky; top:.5rem; padding:.5rem; }}
    aside.chunks h3 {{ margin:.2rem .3rem .45rem; font-size:.75rem; color:var(--muted); text-transform:uppercase; }}
    .chunk-item {{ display:block; width:100%; text-align:left; border:1px solid var(--border); border-radius:8px;
                   padding:.35rem .45rem; margin:0 0 .3rem; cursor:pointer; font:inherit; background:#fff; }}
    .chunk-item:hover, .chunk-item.active {{ outline:2px solid #0f172a; }}
    .chunk-item .cid {{ font-weight:700; font-size:.72rem; margin-right:.3rem; }}
    .chunk-item .lab {{ font-size:.78rem; }}
    .chunk-item .meta {{ display:block; font-size:.68rem; color:var(--muted); margin-top:.1rem; }}
    .page {{ margin-bottom:1rem; }}
    .page h3 {{ margin:0 0 .35rem; }}
    .page-wrap {{ position:relative; display:inline-block; max-width:100%; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
    .page-wrap img {{ display:block; max-width:100%; height:auto; }}
    .overlay {{ position:absolute; inset:0; pointer-events:none; }}
    .ov {{ position:absolute; border-radius:2px; pointer-events:auto; cursor:pointer; }}
    .ov.highlight {{ outline:3px solid #000 !important; z-index:5; }}
    @media (max-width:900px) {{
      .overlay-layout, .charts, .two-col {{ grid-template-columns:1fr; }}
      aside.chunks {{ position:relative; max-height:220px; }}
    }}
  </style>
</head>
<body>
  <header class="top">
    <h1>PDF chunking report</h1>
    <p>Stage 2 structural vectors · {html.escape(pdf_rel)} · {generated}</p>
    <nav class="toc">
      <a href="#logic">Logic</a>
      <a href="#hierarchy-stats">Hierarchy sizes</a>
      <a href="#vector-stats">Vector stats</a>
      <a href="#overlay">Overlay</a>
    </nav>
  </header>
  <div class="wrap">

    <section class="card" id="logic">
      <h2>1. Chunking logic</h2>
      <p class="note">Docling’s visual nesting is unreliable for Hebrew legal PDFs (e.g. §3 and §2.3 both as list level 2).
      We rebuild hierarchy from <strong>section numbers</strong> in item text, then cut vectors for embedding.</p>

      <h4>Pipeline</h4>
      <div class="mermaid">
flowchart LR
  A[PDF] --> B[Docling iterate_items]
  B --> C[Blocks + bbox]
  C --> D[Extract § refs from text]
  D --> E[Rebuild tree by number]
  E --> F[Group units]
  F --> G[Header attach]
  G --> H[Pack / split]
  H --> I[Vectors ~{target} tok]
      </div>

      <div class="flow">
        <div><strong>1. Parse</strong>Docling <code>iterate_items()</code> → blocks with page + bbox. Prefer structure over markdown export (RTL/OCR noise).</div>
        <div><strong>2. Number</strong>Extract § refs (<code>3.1.1</code>) with heuristics; ignore Docling <code>level</code>. Reject mid-text quantities, dates, plan codes.</div>
        <div><strong>3. Group</strong>Consecutive blocks with the same unit key: subclause → clause → section → orphan.</div>
        <div><strong>4. Pack</strong>Clause/subclause kept whole if ≤ max; section/orphan glued toward ~{target} tok (max {max_tokens}).</div>
        <div><strong>5. Headers</strong>Never end a vector on a title; peel trailing headers onto the next body.</div>
      </div>

      <h4>Decision: which unit becomes a vector?</h4>
      <div class="mermaid">
flowchart TD
  S[Block run] --> D{{Depth of § ref?}}
  D -->|≥3| SC[subclause — emit if ≤ max]
  D -->|2| CL[clause — emit if ≤ max]
  D -->|1| SP[section_pack toward target]
  D -->|none| OP[orphan_pack toward target]
  SC --> OV{{tokens &gt; max?}}
  CL --> OV
  OV -->|yes| SPL[split on block / char boundaries]
  OV -->|no| DONE[one vector]
  SP --> DONE
  OP --> DONE
  SPL --> DONE
      </div>

      <div class="two-col">
        <div>
          <h3>Unit priority</h3>
          <ol class="steps">
            <li><strong>Sub-clause</strong> (<code>§N.M.K+</code>) — one vector if ≤ max tokens</li>
            <li><strong>Clause</strong> (<code>§N.M</code>) — same; <em>fold in</em> consecutive descendants (<code>§N.M.*</code>) while the combined size stays ≤ max (keeps stems like §4.8 with §4.8.1/§4.8.2)</li>
            <li><strong>Small siblings</strong> — consecutive tiny units packed toward ~{target}: clauses/subclauses that share a parent (e.g. §2.1+§2.2), and top-level sections (e.g. §5+§6)</li>
            <li><strong>Section pack</strong> — remaining depth-1 / inherited §N body, glued until ~{target} tokens</li>
            <li><strong>Orphan pack</strong> — no section number; pack by page/order to ~{target}</li>
            <li><strong>Split</strong> — only if a single unit exceeds max ({max_tokens})</li>
          </ol>
        </div>
        <div>
          <h3>Packing rules (section / orphan)</h3>
          <ul>
            <li>Accumulate blocks until adding the next would exceed <code>target</code> <em>and</em> the buffer is already ≥ 40% of target.</li>
            <li>A header always starts a new pack (stays with the following body).</li>
            <li>Before flushing, peel any trailing header into the next pack.</li>
            <li>A single block &gt; max is hard-split by character length approx. to max tokens.</li>
          </ul>
        </div>
      </div>

      <h3>Header rule</h3>
      <ul>
        <li>A <code>SectionHeaderItem</code> (or short numbered title ≤ 90 chars) <em>starts</em> a pack — it is not appended onto the previous full pack.</li>
        <li>If a pack would end on a header, the header is peeled onto the next vector.</li>
        <li>A lone <code>§N</code> title merges into the following <code>§N.M</code> clause when present.</li>
      </ul>

      <h3>Section-number extraction (parse)</h3>
      <ul>
        <li>Prefer a dotted self-ref at the start/end of the line over cross-refs mid-sentence.</li>
        <li>Reject patterns that look like quantities (<code>3.5 טון</code>), dates, or plan codes.</li>
        <li>Tree parent/child comes from the numeric path (<code>2.3.1</code> under <code>2.3</code>), not Docling’s visual level.</li>
      </ul>

      <h3>Why not page-sized vectors?</h3>
      <p class="note warn">Pages are ~1.5–2.5k tokens (p50/p95). That buries fine clauses (e.g. treatment limits).
      Retrieve <strong>top‑k</strong> vectors (~400–800 tok each) so generation still sees enough context.
      Clause p50 (~228) is below target; section p50 (~147) is tiny but mean/p95 explode — packing fills the gap.</p>
    </section>

    <section class="card" id="hierarchy-stats">
      <h2>2. Hierarchy size stats (tokens)</h2>
      <p class="note">{html.escape(HIERARCHY_STATS['sample'])} · {html.escape(HIERARCHY_STATS['encoding'])}.
      These are <em>natural</em> cut sizes before packing — used to choose the ~{target} target.
      Full-document sizes are omitted from charts (scale ~50k+) but remain in the table.</p>

      <div class="charts">
        <div class="chart">
          <div class="chart-title">P50 / mean / P95 by hierarchy level</div>
          <div id="chart-hierarchy"></div>
        </div>
        <div class="chart">
          <div class="chart-title">Unit counts (log scale)</div>
          <div id="chart-hierarchy-n"></div>
        </div>
      </div>

      <table>
        <thead>
          <tr><th>Level</th><th>N</th><th>Mean</th><th>Min</th><th>P50</th><th>P75</th><th>P90</th><th>P95</th><th>Max</th></tr>
        </thead>
        <tbody>{''.join(h_rows)}</tbody>
      </table>
      <p class="note warn" style="margin-top:.75rem">{html.escape(HIERARCHY_STATS['outside_clause_note'])}</p>
    </section>

    <section class="card" id="vector-stats">
      <h2>3. Actual vectors on demo PDF</h2>
      <p class="note">After applying the packer to <code>{html.escape(pdf_name)}</code> (target {target}, max {max_tokens}).</p>
      <div class="kpi">
        <div class="box"><div class="n">{len(chunks)}</div><div class="l">Vectors</div></div>
        <div class="box"><div class="n">{fmt(vstats['mean'])}</div><div class="l">Mean tok</div></div>
        <div class="box"><div class="n">{fmt(vstats['p50'])}</div><div class="l">P50</div></div>
        <div class="box"><div class="n">{fmt(vstats['p75'])}</div><div class="l">P75</div></div>
        <div class="box"><div class="n">{fmt(vstats['p95'])}</div><div class="l">P95</div></div>
        <div class="box"><div class="n">{fmt(vstats['max'])}</div><div class="l">Max</div></div>
        <div class="box"><div class="n">{fmt(vstats['min'])}</div><div class="l">Min</div></div>
      </div>

      <div class="charts">
        <div class="chart full">
          <div class="chart-title">Token distribution of vectors (histogram)</div>
          <div id="chart-hist"></div>
        </div>
        <div class="chart">
          <div class="chart-title">Tokens by kind (box)</div>
          <div id="chart-box"></div>
        </div>
        <div class="chart">
          <div class="chart-title">Vector count by kind</div>
          <div id="chart-kind-counts"></div>
        </div>
      </div>

      <table>
        <thead>
          <tr><th>Kind</th><th>Count</th><th>Mean</th><th>P50</th><th>P95</th><th>Min</th><th>Max</th></tr>
        </thead>
        <tbody>{''.join(kind_stat_rows)}</tbody>
      </table>
    </section>

    <section class="card" id="overlay">
      <h2>4. Overlay — each vector a color</h2>
      <p class="note">Showing pages 1–{len(page_nos)} of {len(page_jpegs)}.
      Boxes are Docling bboxes painted by vector id. Click a chunk to highlight.</p>
      <div class="overlay-layout">
        <aside class="chunks">
          <h3>Chunks ({len(chunks)})</h3>
          {''.join(side_items)}
        </aside>
        <div class="pages">
          {''.join(page_sections)}
        </div>
      </div>
    </section>

  </div>
  <script id="chart-data" type="application/json">{chart_json}</script>
  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' }});

    const DATA = JSON.parse(document.getElementById('chart-data').textContent);
    const layoutBase = {{
      margin: {{ t: 24, r: 20, b: 48, l: 52 }},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{ size: 11, color: '#334155' }},
      legend: {{ orientation: 'h', y: 1.12 }},
    }};

    // Hierarchy percentile bars
    Plotly.newPlot('chart-hierarchy', [
      {{
        type: 'bar', name: 'P50', x: DATA.hierarchy.labels, y: DATA.hierarchy.p50,
        marker: {{ color: '#94a3b8' }},
      }},
      {{
        type: 'bar', name: 'Mean', x: DATA.hierarchy.labels, y: DATA.hierarchy.mean,
        marker: {{ color: '#2563eb' }},
      }},
      {{
        type: 'bar', name: 'P95', x: DATA.hierarchy.labels, y: DATA.hierarchy.p95,
        marker: {{ color: '#0f172a' }},
      }},
      {{
        type: 'scatter', mode: 'lines', name: 'target',
        x: DATA.hierarchy.labels,
        y: DATA.hierarchy.labels.map(() => DATA.target),
        line: {{ color: '#d97706', dash: 'dash', width: 2 }},
      }},
      {{
        type: 'scatter', mode: 'lines', name: 'max',
        x: DATA.hierarchy.labels,
        y: DATA.hierarchy.labels.map(() => DATA.max_tokens),
        line: {{ color: '#dc2626', dash: 'dot', width: 2 }},
      }},
    ], Object.assign({{}}, layoutBase, {{
      barmode: 'group',
      yaxis: {{ title: 'tokens', type: 'log' }},
      height: 300,
    }}), {{ responsive: true, displayModeBar: false }});

    Plotly.newPlot('chart-hierarchy-n', [{{
      type: 'bar',
      x: DATA.hierarchy.labels,
      y: DATA.hierarchy.n,
      marker: {{ color: '#0891b2' }},
      text: DATA.hierarchy.n,
      textposition: 'outside',
    }}], Object.assign({{}}, layoutBase, {{
      yaxis: {{ title: 'count', type: 'log' }},
      height: 300,
      showlegend: false,
    }}), {{ responsive: true, displayModeBar: false }});

    // Vector histogram
    Plotly.newPlot('chart-hist', [{{
      type: 'histogram',
      x: DATA.vectors.tokens,
      nbinsx: 30,
      marker: {{ color: '#2563eb', line: {{ color: '#fff', width: 1 }} }},
      hovertemplate: '%{{x}} tok · count %{{y}}<extra></extra>',
    }}, {{
      type: 'scatter', mode: 'lines', name: 'target',
      x: [DATA.target, DATA.target], y: [0, 1],
      yaxis: 'y2',
      line: {{ color: '#d97706', dash: 'dash', width: 2 }},
      showlegend: true,
    }}, {{
      type: 'scatter', mode: 'lines', name: 'max',
      x: [DATA.max_tokens, DATA.max_tokens], y: [0, 1],
      yaxis: 'y2',
      line: {{ color: '#dc2626', dash: 'dot', width: 2 }},
      showlegend: true,
    }}], Object.assign({{}}, layoutBase, {{
      xaxis: {{ title: 'tokens per vector' }},
      yaxis: {{ title: 'count' }},
      yaxis2: {{ overlaying: 'y', visible: false, range: [0, 1] }},
      height: 320,
      shapes: [
        {{ type: 'line', x0: DATA.target, x1: DATA.target, y0: 0, y1: 1, yref: 'paper',
           line: {{ color: '#d97706', dash: 'dash', width: 2 }} }},
        {{ type: 'line', x0: DATA.max_tokens, x1: DATA.max_tokens, y0: 0, y1: 1, yref: 'paper',
           line: {{ color: '#dc2626', dash: 'dot', width: 2 }} }},
      ],
    }}), {{ responsive: true, displayModeBar: false }});

    // Box by kind
    const boxTraces = Object.keys(DATA.vectors.by_kind).map(k => ({{
      type: 'box',
      name: k,
      y: DATA.vectors.by_kind[k],
      marker: {{ color: DATA.vectors.kind_colors[k] || '#64748b' }},
      boxpoints: 'outliers',
    }}));
    Plotly.newPlot('chart-box', boxTraces, Object.assign({{}}, layoutBase, {{
      yaxis: {{ title: 'tokens' }},
      height: 300,
      showlegend: false,
      shapes: [
        {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: DATA.target, y1: DATA.target,
           line: {{ color: '#d97706', dash: 'dash', width: 1.5 }} }},
        {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: DATA.max_tokens, y1: DATA.max_tokens,
           line: {{ color: '#dc2626', dash: 'dot', width: 1.5 }} }},
      ],
    }}), {{ responsive: true, displayModeBar: false }});

    const kindLabels = Object.keys(DATA.vectors.kind_counts);
    Plotly.newPlot('chart-kind-counts', [{{
      type: 'bar',
      x: kindLabels,
      y: kindLabels.map(k => DATA.vectors.kind_counts[k]),
      marker: {{ color: kindLabels.map(k => DATA.vectors.kind_colors[k] || '#94a3b8') }},
      text: kindLabels.map(k => DATA.vectors.kind_counts[k]),
      textposition: 'outside',
    }}], Object.assign({{}}, layoutBase, {{
      yaxis: {{ title: 'vectors' }},
      height: 300,
      showlegend: false,
    }}), {{ responsive: true, displayModeBar: false }});

    const items = document.querySelectorAll('.chunk-item');
    const ovs = document.querySelectorAll('.ov');
    function highlight(id) {{
      items.forEach(el => el.classList.toggle('active', el.dataset.chunk === String(id)));
      ovs.forEach(el => el.classList.toggle('highlight', el.dataset.chunk === String(id)));
      const first = document.querySelector('.ov.highlight');
      if (first) first.closest('.page')?.scrollIntoView({{behavior:'smooth', block:'start'}});
    }}
    items.forEach(el => el.addEventListener('click', () => highlight(el.dataset.chunk)));
    ovs.forEach(el => el.addEventListener('click', () => highlight(el.dataset.chunk)));
  </script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate unified PDF chunking report")
    ap.add_argument(
        "--pdf",
        default="corpus/business/files/פוליסת-מכלול-לחבר-מושב-מהדורת-ינואר-2023.pdf",
    )
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--scale", type=float, default=1.15)
    ap.add_argument("--max-pages", type=int, default=20, help="0 = all pages")
    ap.add_argument("--out", default="reports/stage2/chunking_report.html")
    args = ap.parse_args()

    pdf = ROOT / args.pdf
    rel = str(pdf.relative_to(ROOT))
    print(f"Converting {rel}…", flush=True)
    doc = convert_with_images(pdf, scale=args.scale)
    blocks = docling_to_blocks(doc)
    build_tree(blocks)
    print(f"  {len(blocks)} blocks, {len(doc.pages)} pages", flush=True)

    chunks = chunk_blocks(blocks, target_tokens=args.target, max_tokens=args.max_tokens)
    block_to_chunk = {bi: c.id for c in chunks for bi in c.block_indices}
    print(f"  {len(chunks)} vectors", flush=True)

    page_jpegs, page_sizes = {}, {}
    for pno, page in sorted(doc.pages.items(), key=lambda x: x[0]):
        if page.image is None:
            continue
        page_jpegs[int(pno)] = page_image_jpeg(page)
        page_sizes[int(pno)] = page.size

    max_pages = None if args.max_pages == 0 else args.max_pages
    html_out = build_report(
        pdf_rel=rel,
        pdf_name=pdf.name,
        chunks=chunks,
        blocks=blocks,
        block_to_chunk=block_to_chunk,
        page_jpegs=page_jpegs,
        page_sizes=page_sizes,
        target=args.target,
        max_tokens=args.max_tokens,
        max_pages=max_pages,
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
