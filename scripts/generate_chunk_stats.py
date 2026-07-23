#!/usr/bin/env python3
"""Generate HTML token-size stats across document hierarchy levels."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk_stats import LEVELS, chunks_for_document, summarize  # noqa: E402
from rag.parse import parse_pdf  # noqa: E402


import re

POLICY_KEEP = re.compile(r"(פוליס|תנאי|כתב[-_]?שירות|כתבשירות)", re.I)
POLICY_DROP = re.compile(
    r"(טופס|תעריפ|גילוי[-_]?נאות|דפי[-_]?עזר|הצהר|ויתור|בקש|אישור|עדכון|באנגלית|english)",
    re.I,
)


def pick_pdfs(n: int, seed: int = 42, mode: str = "naive") -> list[Path]:
    """Sample PDFs across domains.

    mode=naive: largest files per domain (byte size) — biased toward big blobs,
    includes forms/tariffs.
    mode=policies: Hebrew policy-like names (פוליס/תנאי/כתב-שירות), drop forms.
    """
    pdfs = list((ROOT / "corpus").rglob("*.pdf"))
    if mode == "policies":
        pdfs = [
            p
            for p in pdfs
            if POLICY_KEEP.search(p.name) and not POLICY_DROP.search(p.name)
        ]

    by_dom: dict[str, list[Path]] = defaultdict(list)
    for p in pdfs:
        dom = p.relative_to(ROOT / "corpus").parts[0]
        by_dom[dom].append(p)
    for dom in by_dom:
        by_dom[dom].sort(key=lambda p: p.stat().st_size, reverse=True)

    rng = random.Random(seed)
    picks: list[Path] = []
    domains = sorted(by_dom.keys())
    i = 0
    while len(picks) < n and domains:
        dom = domains[i % len(domains)]
        pool = by_dom[dom]
        if not pool:
            domains = [d for d in domains if by_dom[d]]
            if not domains:
                break
            i += 1
            continue
        top = pool[: max(1, len(pool) // 2)]
        choice = top[0] if i < len(domains) else rng.choice(top)
        if choice not in picks:
            picks.append(choice)
            pool.remove(choice)
        i += 1
    return picks[:n]


def box_figure(by_level: dict[str, list[int]]) -> go.Figure:
    fig = go.Figure()
    for key, title, _ in LEVELS:
        vals = by_level.get(key, [])
        if not vals:
            continue
        fig.add_trace(
            go.Box(
                y=vals,
                name=title,
                boxmean=True,
                marker_color="#2563eb",
                line_color="#1e3a5f",
            )
        )
    fig.update_layout(
        title="Token counts by hierarchy level (cl100k_base)",
        yaxis_title="Tokens",
        yaxis_type="log",
        height=480,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
        template="plotly_white",
    )
    return fig


def hist_figure(by_level: dict[str, list[int]], level_key: str, title: str) -> go.Figure:
    vals = by_level.get(level_key, [])
    fig = go.Figure(
        data=[
            go.Histogram(
                x=vals,
                nbinsx=40,
                marker_color="#3b82f6",
                opacity=0.85,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tokens",
        yaxis_title="Count",
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def build_html(
    chunks: list,
    files: list[str],
    generated: str,
) -> str:
    by_level: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_level[c.level].append(c.tokens)

    summaries = {key: summarize(by_level.get(key, [])) for key, _, _ in LEVELS}

    rows = []
    for key, title, desc in LEVELS:
        s = summaries[key]
        if s["n"] == 0:
            continue
        rows.append(
            f"""<tr>
              <td><strong>{html.escape(title)}</strong><div class="desc">{html.escape(desc)}</div></td>
              <td class="num">{s['n']:,}</td>
              <td class="num">{s['mean']:,}</td>
              <td class="num">{s['min']:,}</td>
              <td class="num">{s['p50']:,}</td>
              <td class="num">{s['p75']:,}</td>
              <td class="num">{s['p90']:,}</td>
              <td class="num">{s['p95']:,}</td>
              <td class="num">{s['p99']:,}</td>
              <td class="num">{s['max']:,}</td>
            </tr>"""
        )

    box_div = pio.to_html(box_figure(by_level), full_html=False, include_plotlyjs="cdn")
    hist_divs = []
    for key, title, _ in LEVELS:
        if key == "full_doc":
            continue
        if not by_level.get(key):
            continue
        hist_divs.append(
            pio.to_html(
                hist_figure(by_level, key, f"{title} — token distribution"),
                full_html=False,
                include_plotlyjs=False,
            )
        )

    # Extreme examples table (largest / smallest non-trivial)
    examples = []
    for key, title, _ in LEVELS:
        vals = [c for c in chunks if c.level == key]
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda c: c.tokens)
        small = vals_sorted[0]
        large = vals_sorted[-1]
        examples.append(
            f"""<tr>
              <td>{html.escape(title)}</td>
              <td class="num">{small.tokens:,}</td>
              <td>{html.escape(small.label[:80])} <span class="muted">{html.escape(Path(small.file).name[:40])}</span></td>
              <td class="num">{large.tokens:,}</td>
              <td>{html.escape(large.label[:80])} <span class="muted">{html.escape(Path(large.file).name[:40])}</span></td>
            </tr>"""
        )

    file_list = "".join(f"<li><code>{html.escape(f)}</code></li>" for f in files)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk token size statistics</title>
  <style>
    :root {{ --bg:#f4f6f9; --card:#fff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; --accent:#2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, system-ui, sans-serif; background:var(--bg); color:var(--text); }}
    header {{ background:#1e3a5f; color:#fff; padding:1.2rem 1.5rem; }}
    header h1 {{ margin:0 0 .35rem; font-size:1.3rem; }}
    header p {{ margin:0; opacity:.9; font-size:.88rem; }}
    main {{ max-width:1100px; margin:0 auto; padding:1rem 1.25rem 2.5rem; }}
    section {{ background:var(--card); border-radius:12px; padding:1rem 1.1rem; margin-bottom:1rem;
               box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ margin:0 0 .65rem; font-size:1.05rem; }}
    .note {{ background:#eff6ff; border-left:3px solid var(--accent); padding:.55rem .75rem;
             border-radius:8px; font-size:.86rem; margin:0 0 .75rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.84rem; }}
    th, td {{ padding:.45rem .5rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .desc {{ font-size:.72rem; color:var(--muted); font-weight:400; margin-top:.15rem; }}
    .muted {{ color:var(--muted); font-size:.75rem; }}
    code {{ font-size:.78rem; background:#f3f4f6; padding:.05rem .3rem; border-radius:4px; }}
    ul.files {{ font-size:.8rem; columns:2; gap:1.5rem; margin:.35rem 0 0; padding-left:1.1rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Chunk token size statistics</h1>
    <p>{len(files)} PDFs · {len(chunks):,} chunks · tiktoken <code>cl100k_base</code> · {generated}</p>
  </header>
  <main>
    <section>
      <h2>How levels are defined</h2>
      <p class="note">Hierarchy comes from Docling items + section-number parsing (<code>rag/parse.py</code>).
      Section/clause/subclause sizes include <em>all descendant text</em> under that node
      (i.e. “chunk at this cut”). Paragraph = one Docling text/list item.</p>
      <ul>
        <li><strong>Full document</strong> — whole PDF</li>
        <li><strong>Page</strong> — all items on one page</li>
        <li><strong>Section (§N)</strong> — depth-1 numbered heading + children</li>
        <li><strong>Clause (§N.M)</strong> — depth-2 + children</li>
        <li><strong>Sub-clause (§N.M.K+)</strong> — depth ≥ 3 + children</li>
        <li><strong>Paragraph / item</strong> — single Docling block</li>
      </ul>
    </section>

    <section>
      <h2>Summary (tokens)</h2>
      <table>
        <thead>
          <tr>
            <th>Level</th><th>N</th><th>Mean</th><th>Min</th>
            <th>P50</th><th>P75</th><th>P90</th><th>P95</th><th>P99</th><th>Max</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Distribution (log scale)</h2>
      {box_div}
    </section>

    <section>
      <h2>Histograms</h2>
      {''.join(f'<div style="margin-bottom:.75rem">{d}</div>' for d in hist_divs)}
    </section>

    <section>
      <h2>Extremes</h2>
      <table>
        <thead>
          <tr><th>Level</th><th>Smallest</th><th>Label</th><th>Largest</th><th>Label</th></tr>
        </thead>
        <tbody>{''.join(examples)}</tbody>
      </table>
    </section>

    <section>
      <h2>Sampled files</h2>
      <ul class="files">{file_list}</ul>
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="Number of PDFs to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--mode",
        choices=("naive", "policies"),
        default="naive",
        help="naive=largest/domain; policies=פוליס/תנאי/כתב-שירות filter",
    )
    ap.add_argument("--out", default="", help="Output HTML path (default by mode)")
    ap.add_argument("pdfs", nargs="*", help="Optional explicit PDF paths")
    args = ap.parse_args()

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    if not args.out:
        args.out = (
            "reports/chunk_size_stats_policies.html"
            if args.mode == "policies"
            else "reports/chunk_size_stats.html"
        )

    if args.pdfs:
        paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.pdfs]
    else:
        paths = pick_pdfs(args.n, args.seed, mode=args.mode)

    all_chunks = []
    files = []
    for i, path in enumerate(paths, 1):
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        print(f"[{i}/{len(paths)}] {rel}", flush=True)
        try:
            blocks, tree = parse_pdf(path)
            chunks = chunks_for_document(rel, blocks, tree, encoding)
            all_chunks.extend(chunks)
            files.append(rel)
            print(f"  → {len(blocks)} blocks, {len(chunks)} chunk measurements", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(all_chunks, files, generated), encoding="utf-8")

    # also dump json summary for reuse
    by_level: dict[str, list[int]] = defaultdict(list)
    for c in all_chunks:
        by_level[c.level].append(c.tokens)
    summary = {k: summarize(by_level.get(k, [])) for k, _, _ in LEVELS}
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps({"files": files, "summary": summary, "generated": generated}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    print(f"Wrote {json_path}")
    for key, title, _ in LEVELS:
        s = summary[key]
        if s["n"]:
            print(f"  {title:28s} n={s['n']:5d}  mean={s['mean']:8}  p50={s['p50']:6}  p95={s['p95']:6}  max={s['max']}")


if __name__ == "__main__":
    main()
