#!/usr/bin/env python3
"""Generate Stage-2 TXT (web scrape) cleaning + chunking report."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk_stats import summarize  # noqa: E402
from rag.txt import chunk_txt_file, parse_txt_file  # noqa: E402


def build_report(
    *,
    rows: list[dict],
    demos: list[dict],
    target: int,
    max_tokens: int,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = len(rows)
    low_n = sum(1 for r in rows if r["low_quality"])
    body_toks = [r["body_tokens"] for r in rows]
    chunk_toks = [t for r in rows for t in r["chunk_tokens"]]
    n_chunks = sum(r["n_chunks"] for r in rows)
    retain = [r["retain"] for r in rows]

    body_stats = summarize(body_toks)
    chunk_stats = summarize(chunk_toks)

    # retention as %
    retain_pct = sorted(int(round(x * 100)) for x in retain)
    retain_sum = summarize(retain_pct)

    multi = sum(1 for r in rows if r["n_chunks"] > 1)
    single = sum(1 for r in rows if r["n_chunks"] == 1)
    empty = sum(1 for r in rows if r["n_chunks"] == 0)

    domain_counts = Counter(r["domain"] for r in rows)
    domain_rows = "".join(
        f"<tr><td>{html.escape(d)}</td><td class='num'>{c}</td></tr>"
        for d, c in sorted(domain_counts.items())
    )

    # smallest / largest cleaned bodies
    by_body = sorted(rows, key=lambda r: r["body_tokens"])
    tiny_rows = "".join(
        f"<tr><td>{html.escape(r['rel'])}</td><td class='num'>{r['body_tokens']}</td>"
        f"<td class='num'>{r['n_chunks']}</td><td>{html.escape(r['note'] or '—')}</td></tr>"
        for r in by_body[:12]
    )
    big_rows = "".join(
        f"<tr><td>{html.escape(r['rel'])}</td><td class='num'>{r['body_tokens']}</td>"
        f"<td class='num'>{r['n_chunks']}</td><td class='num'>{r['raw_chars']}</td></tr>"
        for r in reversed(by_body[-12:])
    )

    demo_html = []
    for d in demos:
        chunk_items = "".join(
            f"""<div class="chunk">
              <div class="meta">#{c['id']} · {c['tokens']} tok · {html.escape(c['label'])}</div>
              <div class="body" dir="rtl">{html.escape(c['text'][:1200])}{'…' if len(c['text'])>1200 else ''}</div>
            </div>"""
            for c in d["chunks"]
        )
        demo_html.append(
            f"""<section class="card demo" id="demo-{html.escape(d['stem'])}">
              <h3 dir="rtl">{html.escape(d['title'] or d['rel'])}</h3>
              <p class="note">{html.escape(d['rel'])} · raw {d['raw_chars']:,} chars →
                body {d['body_chars']:,} chars ({d['retain']:.0%}) ·
                {d['body_tokens']} tok → {d['n_chunks']} vector(s)
                {'· <span class="bad">' + html.escape(d['note']) + '</span>' if d['low_quality'] else ''}
              </p>
              <div class="split">
                <div>
                  <h4>Cleaned body</h4>
                  <pre class="rtl" dir="rtl">{html.escape(d['body'][:2500])}{'…' if len(d['body'])>2500 else ''}</pre>
                </div>
                <div>
                  <h4>Chunks</h4>
                  {chunk_items or '<p class="note">No chunks (skipped / empty).</p>'}
                </div>
              </div>
            </section>"""
        )

    chart = {
        "body_tokens": body_toks,
        "chunk_tokens": chunk_toks,
        "retain_pct": retain_pct,
        "target": target,
        "max_tokens": max_tokens,
        "n_chunks_hist": list(Counter(r["n_chunks"] for r in rows).items()),
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TXT chunking report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ --bg:#f1f5f9; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --accent:#0f766e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header.top {{ background:#134e4a; color:#fff; padding:1rem 1.25rem; }}
    header.top h1 {{ margin:0 0 .3rem; font-size:1.25rem; }}
    header.top p {{ margin:0; opacity:.88; font-size:.86rem; }}
    nav.toc {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.75rem; }}
    nav.toc a {{ color:#fff; text-decoration:none; font-size:.78rem; padding:.2rem .55rem;
                 border:1px solid rgba(255,255,255,.28); border-radius:999px; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:1rem 1.25rem 2.5rem; }}
    section.card {{ background:var(--card); border-radius:12px; padding:1rem 1.1rem; margin-bottom:1rem;
                    box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ margin:0 0 .65rem; font-size:1.05rem; }}
    h3 {{ margin:0 0 .45rem; font-size:.95rem; }}
    h4 {{ margin:.2rem 0 .4rem; font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .note {{ background:#f0fdfa; border-left:3px solid var(--accent); padding:.55rem .75rem;
             border-radius:8px; font-size:.86rem; margin:0 0 .75rem; }}
    .warn {{ background:#fffbeb; border-left-color:#d97706; }}
    .bad {{ color:#b91c1c; font-weight:600; }}
    .kpi {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:.5rem; margin:.5rem 0 .75rem; }}
    .kpi .box {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.5rem .6rem; }}
    .kpi .n {{ font-size:1.15rem; font-weight:700; color:var(--accent); }}
    .kpi .l {{ font-size:.68rem; color:var(--muted); text-transform:uppercase; }}
    table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
    th, td {{ padding:.4rem .45rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }}
    th {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }}
    .chart {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; min-height:260px; padding:.25rem; }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }}
    pre.rtl {{ white-space:pre-wrap; background:#f8fafc; border:1px solid var(--border); border-radius:8px;
               padding:.6rem .7rem; font-size:.78rem; max-height:320px; overflow:auto; margin:0; }}
    .chunk {{ border:1px solid var(--border); border-radius:8px; padding:.45rem .55rem; margin:0 0 .4rem; background:#fff; }}
    .chunk .meta {{ font-size:.72rem; color:var(--muted); margin-bottom:.25rem; }}
    .chunk .body {{ font-size:.82rem; line-height:1.45; }}
    ul {{ font-size:.88rem; line-height:1.55; }}
    code {{ background:#f1f5f9; padding:.08rem .28rem; border-radius:4px; font-size:.84em; }}
    @media (max-width:900px) {{ .charts, .split {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header class="top">
    <h1>TXT chunking report</h1>
    <p>Stage 2 web scrapes · {n} files under <code>corpus/*/pages/*.txt</code> · {generated}</p>
    <nav class="toc">
      <a href="#logic">Logic</a>
      <a href="#corpus">Corpus stats</a>
      <a href="#demos">Examples</a>
    </nav>
  </header>
  <div class="wrap">

    <section class="card" id="logic">
      <h2>1. Why TXT ≠ PDF</h2>
      <p class="note warn">These are Harel website dumps (one long line each): title, skip-nav, breadcrumb,
      article body, then a huge repeated site footer. Docling yields a single TextItem with no pages/bboxes.
      There is almost no legal § structure — chunking is <strong>clean → pack by tokens</strong>.</p>
      <ul>
        <li><strong>Clean</strong> — strip <code>| הראל…</code> chrome, skip-nav, breadcrumb; cut at footer markers
          (<code>הצהרת נגישות</code> / <code>תנאי שימוש</code> / …).</li>
        <li><strong>Segment</strong> — soft splits on <code>?</code> / sentence ends.</li>
        <li><strong>Pack</strong> — toward ~{target} tok (max {max_tokens}), same budget as PDF vectors.</li>
        <li><strong>Citations</strong> — <code>page: null</code>; identity is the file path only.</li>
        <li><strong>Low quality</strong> — IE interstitial / aspx shells flagged ({low_n} files).</li>
      </ul>
    </section>

    <section class="card" id="corpus">
      <h2>2. Corpus stats</h2>
      <div class="kpi">
        <div class="box"><div class="n">{n}</div><div class="l">TXT files</div></div>
        <div class="box"><div class="n">{n_chunks}</div><div class="l">Vectors</div></div>
        <div class="box"><div class="n">{single}</div><div class="l">1-vector files</div></div>
        <div class="box"><div class="n">{multi}</div><div class="l">Multi-vector</div></div>
        <div class="box"><div class="n">{empty}</div><div class="l">Empty / skipped</div></div>
        <div class="box"><div class="n">{low_n}</div><div class="l">Low quality</div></div>
        <div class="box"><div class="n">{retain_sum['p50']}%</div><div class="l">Retain P50</div></div>
        <div class="box"><div class="n">{body_stats['p50']}</div><div class="l">Body tok P50</div></div>
        <div class="box"><div class="n">{chunk_stats['p50']}</div><div class="l">Chunk tok P50</div></div>
        <div class="box"><div class="n">{chunk_stats['p95']}</div><div class="l">Chunk tok P95</div></div>
      </div>

      <div class="charts">
        <div class="chart"><div id="chart-body"></div></div>
        <div class="chart"><div id="chart-chunks"></div></div>
        <div class="chart"><div id="chart-retain"></div></div>
        <div class="chart"><div id="chart-nchunks"></div></div>
      </div>

      <div class="split" style="margin-top:.75rem">
        <div>
          <h4>Domains</h4>
          <table><thead><tr><th>Domain</th><th>N</th></tr></thead><tbody>{domain_rows}</tbody></table>
        </div>
        <div>
          <h4>Smallest cleaned bodies</h4>
          <table><thead><tr><th>File</th><th>Tok</th><th>Chunks</th><th>Note</th></tr></thead>
          <tbody>{tiny_rows}</tbody></table>
        </div>
      </div>
      <h4 style="margin-top:.75rem">Largest cleaned bodies</h4>
      <table><thead><tr><th>File</th><th>Body tok</th><th>Chunks</th><th>Raw chars</th></tr></thead>
      <tbody>{big_rows}</tbody></table>
    </section>

    {''.join(demo_html)}

  </div>
  <script id="chart-data" type="application/json">{json.dumps(chart)}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('chart-data').textContent);
    const base = {{
      margin: {{ t: 28, r: 16, b: 40, l: 48 }},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{ size: 11, color: '#334155' }},
    }};
    Plotly.newPlot('chart-body', [{{
      type: 'histogram', x: DATA.body_tokens, nbinsx: 30,
      marker: {{ color: '#0f766e' }},
    }}], Object.assign({{}}, base, {{
      title: 'Cleaned body tokens / file',
      xaxis: {{ title: 'tokens' }}, height: 280,
      shapes: [
        {{ type:'line', x0:DATA.target, x1:DATA.target, y0:0, y1:1, yref:'paper',
           line:{{ color:'#d97706', dash:'dash', width:2 }} }},
        {{ type:'line', x0:DATA.max_tokens, x1:DATA.max_tokens, y0:0, y1:1, yref:'paper',
           line:{{ color:'#dc2626', dash:'dot', width:2 }} }},
      ],
    }}), {{responsive:true, displayModeBar:false}});

    Plotly.newPlot('chart-chunks', [{{
      type: 'histogram', x: DATA.chunk_tokens, nbinsx: 30,
      marker: {{ color: '#0891b2' }},
    }}], Object.assign({{}}, base, {{
      title: 'Vector token sizes',
      xaxis: {{ title: 'tokens' }}, height: 280,
      shapes: [
        {{ type:'line', x0:DATA.target, x1:DATA.target, y0:0, y1:1, yref:'paper',
           line:{{ color:'#d97706', dash:'dash', width:2 }} }},
        {{ type:'line', x0:DATA.max_tokens, x1:DATA.max_tokens, y0:0, y1:1, yref:'paper',
           line:{{ color:'#dc2626', dash:'dot', width:2 }} }},
      ],
    }}), {{responsive:true, displayModeBar:false}});

    Plotly.newPlot('chart-retain', [{{
      type: 'histogram', x: DATA.retain_pct, nbinsx: 20,
      marker: {{ color: '#7c3aed' }},
    }}], Object.assign({{}}, base, {{
      title: 'Chars retained after clean (%)',
      xaxis: {{ title: '%' }}, height: 280,
    }}), {{responsive:true, displayModeBar:false}});

    const nc = DATA.n_chunks_hist.sort((a,b)=>a[0]-b[0]);
    Plotly.newPlot('chart-nchunks', [{{
      type: 'bar', x: nc.map(x=>x[0]), y: nc.map(x=>x[1]),
      marker: {{ color: '#134e4a' }}, text: nc.map(x=>x[1]), textposition: 'outside',
    }}], Object.assign({{}}, base, {{
      title: 'Vectors per file',
      xaxis: {{ title: 'n chunks', dtick: 1 }}, height: 280, showlegend:false,
    }}), {{responsive:true, displayModeBar:false}});
  </script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TXT chunking report")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument(
        "--demo",
        action="append",
        default=[],
        help="Relative path under corpus/ (repeatable). Defaults to a few samples.",
    )
    ap.add_argument("--out", default="reports/stage2/txt_chunking_report.html")
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    paths = sorted(corpus.glob("*/pages/*.txt"))
    if not paths:
        raise SystemExit(f"No TXT files under {corpus}/*/pages/")

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    rows: list[dict] = []
    for path in paths:
        cleaned, chunks = chunk_txt_file(
            path, target_tokens=args.target, max_tokens=args.max_tokens
        )
        rel = str(path.relative_to(corpus))
        domain = path.relative_to(corpus).parts[0]
        body_tokens = len(enc.encode(cleaned.body)) if cleaned.body else 0
        retain = cleaned.body_chars / max(cleaned.raw_chars, 1)
        rows.append(
            {
                "rel": rel,
                "domain": domain,
                "title": cleaned.title,
                "raw_chars": cleaned.raw_chars,
                "body_chars": cleaned.body_chars,
                "body_tokens": body_tokens,
                "retain": retain,
                "n_chunks": len(chunks),
                "chunk_tokens": [c.tokens for c in chunks],
                "low_quality": cleaned.low_quality,
                "note": cleaned.note,
            }
        )

    demo_rels = args.demo or [
        "apartment/pages/earthquak-insurance-coverage.txt",
        "car/pages/faq.txt",
        "car/pages/comprehensive.txt",
        "health/pages/medications.txt",
        "apartment/pages/plumbers-find.aspx.txt",
    ]
    demos: list[dict] = []
    for rel in demo_rels:
        path = corpus / rel
        if not path.exists():
            print(f"skip missing demo {rel}", flush=True)
            continue
        cleaned, chunks = chunk_txt_file(
            path, target_tokens=args.target, max_tokens=args.max_tokens
        )
        body_tokens = len(enc.encode(cleaned.body)) if cleaned.body else 0
        demos.append(
            {
                "rel": rel,
                "stem": path.stem,
                "title": cleaned.title,
                "body": cleaned.body,
                "raw_chars": cleaned.raw_chars,
                "body_chars": cleaned.body_chars,
                "body_tokens": body_tokens,
                "retain": cleaned.body_chars / max(cleaned.raw_chars, 1),
                "n_chunks": len(chunks),
                "low_quality": cleaned.low_quality,
                "note": cleaned.note,
                "chunks": [
                    {"id": c.id, "tokens": c.tokens, "label": c.label, "text": c.text}
                    for c in chunks
                ],
            }
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_report(
            rows=rows, demos=demos, target=args.target, max_tokens=args.max_tokens
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {out} · {len(rows)} files · "
        f"{sum(r['n_chunks'] for r in rows)} vectors · "
        f"{sum(1 for r in rows if r['low_quality'])} low-quality"
    )


if __name__ == "__main__":
    main()
