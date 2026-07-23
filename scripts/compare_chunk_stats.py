#!/usr/bin/env python3
"""Compare naive vs policy-filtered chunk token stats."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]

LEVELS = [
    ("full_doc", "Full document"),
    ("page", "Page"),
    ("section", "Section (§N)"),
    ("clause", "Clause (§N.M)"),
    ("subclause", "Sub-clause (§N.M.K+)"),
    ("paragraph", "Paragraph / item"),
]

METRICS = ["n", "mean", "min", "p50", "p75", "p90", "p95", "p99", "max"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return f"{v:,}"


def delta(a, b) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    if a == 0:
        return f"{d:+,.1f}" if isinstance(d, float) else f"{d:+,}"
    pct = 100.0 * d / a
    if isinstance(a, float) or isinstance(b, float):
        return f"{d:+,.1f} ({pct:+.0f}%)"
    return f"{d:+,} ({pct:+.0f}%)"


def compare_bars(naive: dict, policies: dict) -> str:
    fig = go.Figure()
    metrics_show = ["mean", "p50", "p95"]
    for metric in metrics_show:
        fig.add_trace(
            go.Bar(
                name=f"Naive · {metric}",
                x=[t for _, t in LEVELS],
                y=[naive["summary"][k].get(metric) or 0 for k, _ in LEVELS],
                marker_color="#94a3b8",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"Policies · {metric}",
                x=[t for _, t in LEVELS],
                y=[policies["summary"][k].get(metric) or 0 for k, _ in LEVELS],
                marker_color="#2563eb",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Naive vs policies sample — mean / p50 / p95 tokens",
        yaxis_title="Tokens (log)",
        yaxis_type="log",
        height=480,
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=80),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def file_list(files: list[str], title: str) -> str:
    items = "".join(
        f"<li><code>{html.escape(Path(f).name)}</code> "
        f"<span class='muted'>{html.escape(str(Path(f).parts[1]) if len(Path(f).parts)>1 else '')}</span></li>"
        for f in files
    )
    return f"<h3>{html.escape(title)}</h3><ul class='files'>{items}</ul>"


def main() -> None:
    naive = load(ROOT / "reports/chunk_size_stats.json")
    policies = load(ROOT / "reports/chunk_size_stats_policies.json")

    rows = []
    for key, title in LEVELS:
        a = naive["summary"][key]
        b = policies["summary"][key]
        rows.append(
            f"""<tr>
              <td rowspan="2"><strong>{html.escape(title)}</strong></td>
              <td>Naive</td>
              {''.join(f'<td class="num">{fmt(a.get(m))}</td>' for m in METRICS)}
            </tr>
            <tr class="pol">
              <td>Policies</td>
              {''.join(f'<td class="num">{fmt(b.get(m))}</td>' for m in METRICS)}
            </tr>
            <tr class="delta">
              <td></td><td>Δ</td>
              {''.join(f'<td class="num">{html.escape(delta(a.get(m), b.get(m)))}</td>' for m in METRICS)}
            </tr>"""
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    chart = compare_bars(naive, policies)

    out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk stats comparison — naive vs policies</title>
  <style>
    :root {{ --bg:#f4f6f9; --card:#fff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; }}
    body {{ margin:0; font-family:Inter,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ background:#1e3a5f; color:#fff; padding:1.2rem 1.5rem; }}
    header h1 {{ margin:0 0 .35rem; font-size:1.25rem; }}
    header p {{ margin:0; opacity:.9; font-size:.88rem; }}
    main {{ max-width:1100px; margin:0 auto; padding:1rem 1.25rem 2.5rem; }}
    section {{ background:var(--card); border-radius:12px; padding:1rem 1.1rem; margin-bottom:1rem;
               box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    h2 {{ margin:0 0 .65rem; font-size:1.05rem; }}
    h3 {{ margin:.75rem 0 .35rem; font-size:.92rem; }}
    .note {{ background:#eff6ff; border-left:3px solid #2563eb; padding:.55rem .75rem;
             border-radius:8px; font-size:.86rem; margin:0 0 .75rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.8rem; }}
    th, td {{ padding:.4rem .45rem; border-bottom:1px solid var(--border); text-align:left; }}
    th {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    tr.pol td {{ background:#f8fafc; }}
    tr.delta td {{ color:#047857; font-size:.75rem; background:#f0fdf4; }}
    .muted {{ color:var(--muted); font-size:.75rem; }}
    ul.files {{ font-size:.78rem; columns:2; padding-left:1.1rem; margin:.25rem 0 0; }}
    code {{ font-size:.76rem; background:#f3f4f6; padding:.05rem .25rem; border-radius:4px; }}
    a {{ color:#2563eb; }}
  </style>
</head>
<body>
  <header>
    <h1>Chunk token stats — naive vs policies sample</h1>
    <p>12 PDFs each · tiktoken cl100k_base · {generated}</p>
  </header>
  <main>
    <section>
      <h2>Sampling difference</h2>
      <p class="note"><strong>Naive:</strong> largest files per domain (byte size) — includes forms, tariffs, help pages.<br/>
      <strong>Policies:</strong> filename matches פוליס / תנאי / כתב-שירות; drops טופס, תעריפון, גילוי נאות, דפי עזר, ויתור, באנגלית, etc.</p>
      <p>Full reports:
        <a href="chunk_size_stats.html">naive</a> ·
        <a href="chunk_size_stats_policies.html">policies</a>
      </p>
    </section>

    <section>
      <h2>Side-by-side (tokens)</h2>
      <table>
        <thead>
          <tr>
            <th>Level</th><th>Sample</th>
            <th>N</th><th>Mean</th><th>Min</th><th>P50</th><th>P75</th>
            <th>P90</th><th>P95</th><th>P99</th><th>Max</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Chart</h2>
      {chart}
    </section>

    <section>
      <h2>Files sampled</h2>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        {file_list(naive['files'], 'Naive')}
        {file_list(policies['files'], 'Policies')}
      </div>
    </section>
  </main>
</body>
</html>"""

    path = ROOT / "reports/chunk_size_stats_compare.html"
    path.write_text(out, encoding="utf-8")
    print(f"Wrote {path}")

    # console snapshot
    print(f"\n{'Level':28s} {'metric':6s} {'naive':>10s} {'policies':>10s} {'Δ':>14s}")
    for key, title in LEVELS:
        for m in ("n", "mean", "p50", "p95"):
            a = naive["summary"][key].get(m)
            b = policies["summary"][key].get(m)
            print(f"{title:28s} {m:6s} {fmt(a):>10s} {fmt(b):>10s} {delta(a,b):>14s}")


if __name__ == "__main__":
    main()
