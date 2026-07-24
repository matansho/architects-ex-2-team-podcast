#!/usr/bin/env python3
"""Generate an interactive Plotly HTML report for Exercise 2 data."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from pypdf import PdfReader
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS_PATH = ROOT / "reference_questions.json"
MANIFEST_PATH = CORPUS / "manifest.json"
OUT_PATH = ROOT / "reports" / "exploration" / "data_exploration.html"

DOMAIN_LABELS = {
    "apartment": "Apartment",
    "business": "Business",
    "car": "Car",
    "dental": "Dental",
    "diseases-disabilities": "Diseases & Disabilities",
    "health": "Health",
    "life": "Life",
    "long-term-care": "Long-Term Care",
    "loss-of-working-ability": "Loss of Working Ability",
    "mortgage": "Mortgage",
    "personal-accident": "Personal Accident",
    "travel": "Travel",
}

PDF_KEYWORDS = {
    "פוליס": "Policy (פוליס)",
    "תנאי": "Terms (תנאי)",
    "טופס": "Form (טופס)",
    "חוברת": "Booklet (חוברת)",
    "הודעה": "Notice (הודעה)",
    "מידע": "Material info (מידע)",
    "נוהל": "Procedure (נוהל)",
    "תצהיר": "Affidavit (תצהיר)",
    "כתב": "Deed/charter (כתב)",
}


def label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain)


def load_manifest_files() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = []
    for rel, url in manifest.items():
        path = CORPUS / rel
        ext = path.suffix.lower()
        kind = "pdf" if ext == ".pdf" else "txt" if ext == ".txt" else "other"
        host = (
            "media.harel-group.co.il"
            if "media.harel" in url
            else "www.harel-group.co.il"
            if "harel-group" in url
            else "other"
        )
        files.append(
            {
                "rel": rel,
                "domain": rel.split("/")[0],
                "subdir": rel.split("/")[1] if "/" in rel else "",
                "kind": kind,
                "size": path.stat().st_size,
                "url": url,
                "host": host,
                "name": path.name,
            }
        )
    return files


def analyze_pdfs(files: list[dict]) -> tuple[list[int], list[int]]:
    pages, sizes = [], []
    for f in files:
        if f["kind"] != "pdf":
            continue
        path = CORPUS / f["rel"]
        sizes.append(path.stat().st_size)
        try:
            pages.append(len(PdfReader(str(path)).pages))
        except Exception:
            pages.append(0)
    return pages, sizes


def analyze_txt(files: list[dict]) -> dict:
    lengths, hebrew_ratios = [], []
    for f in files:
        if f["kind"] != "txt":
            continue
        text = (CORPUS / f["rel"]).read_text(encoding="utf-8", errors="ignore")
        lengths.append(len(text))
        heb = len(re.findall(r"[\u0590-\u05FF]", text))
        lat = len(re.findall(r"[A-Za-z]", text))
        hebrew_ratios.append(heb / (heb + lat + 1))
    return {
        "count": len(lengths),
        "lengths": lengths,
        "hebrew_ratio_avg": sum(hebrew_ratios) / len(hebrew_ratios) if hebrew_ratios else 0,
    }


def analyze_questions(files: list[dict]) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    manifest = {f["rel"] for f in files}

    rows = []
    cited_files = set()
    for q in questions:
        source_kinds = []
        null_pages = 0
        page_numbers = []
        for group in q["ground_truth_sources"]:
            for src in group["any_of"]:
                cited_files.add(src["file"])
                source_kinds.append("pdf" if src["file"].endswith(".pdf") else "txt")
                if src["page"] is None:
                    null_pages += 1
                else:
                    page_numbers.append(src["page"])

        rows.append(
            {
                **q,
                "q_len": len(q["question"]),
                "a_len": len(q["ground_truth_answer"]),
                "num_groups": len(q["ground_truth_sources"]),
                "num_sources": sum(len(g["any_of"]) for g in q["ground_truth_sources"]),
                "has_number": bool(re.search(r"\d", q["ground_truth_answer"])),
                "source_kinds": source_kinds,
                "null_pages": null_pages,
                "page_numbers": page_numbers,
                "in_manifest": all(
                    any(s["file"] in manifest for s in g["any_of"])
                    for g in q["ground_truth_sources"]
                ),
            }
        )

    return {"questions": questions, "rows": rows, "cited_files": cited_files}


def fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_js)


def corpus_domain_bar(files: list[dict]) -> go.Figure:
    domains = sorted({f["domain"] for f in files}, key=lambda d: label(d))
    pdf = [sum(1 for f in files if f["domain"] == d and f["kind"] == "pdf") for d in domains]
    txt = [sum(1 for f in files if f["domain"] == d and f["kind"] == "txt") for d in domains]

    fig = go.Figure()
    fig.add_bar(name="PDF policy docs", x=[label(d) for d in domains], y=pdf, marker_color="#4C78A8")
    fig.add_bar(name="Web pages (.txt)", x=[label(d) for d in domains], y=txt, marker_color="#F58518")
    fig.update_layout(
        barmode="stack",
        title="Corpus documents by insurance domain",
        xaxis_title="Domain",
        yaxis_title="Document count",
        legend_title="Source type",
        height=480,
    )
    return fig


def corpus_size_scatter(files: list[dict]) -> go.Figure:
    fig = go.Figure()
    for kind, color in [("pdf", "#4C78A8"), ("txt", "#F58518")]:
        subset = [f for f in files if f["kind"] == kind]
        fig.add_scatter(
            x=[label(f["domain"]) for f in subset],
            y=[f["size"] / 1024 for f in subset],
            mode="markers",
            name=kind.upper(),
            marker=dict(color=color, size=8, opacity=0.65),
            text=[f["rel"] for f in subset],
            hovertemplate="%{text}<br>Size: %{y:.1f} KB<extra></extra>",
        )
    fig.update_layout(
        title="Per-document size distribution",
        xaxis_title="Domain",
        yaxis_title="File size (KB)",
        height=460,
    )
    return fig


def pdf_page_hist(pdf_pages: list[int]) -> go.Figure:
    fig = go.Figure(go.Histogram(x=pdf_pages, nbinsx=25, marker_color="#72B7B2"))
    fig.update_layout(
        title="PDF page-count distribution (350 documents)",
        xaxis_title="Pages per PDF",
        yaxis_title="Count",
        height=380,
    )
    return fig


def pdf_keyword_bar(files: list[dict]) -> go.Figure:
    counts = Counter()
    for f in files:
        if f["kind"] != "pdf":
            continue
        for needle, label_text in PDF_KEYWORDS.items():
            if needle in f["name"]:
                counts[label_text] += 1

    items = counts.most_common()
    fig = go.Figure(go.Bar(x=[c for _, c in items], y=[k for k, _ in items], orientation="h", marker_color="#B279A2"))
    fig.update_layout(title="PDF filename keyword patterns", xaxis_title="Count", height=420)
    return fig


def question_heatmap(rows: list[dict]) -> go.Figure:
    domains = sorted({r["domain"] for r in rows}, key=lambda d: label(d))
    diffs = ["easy", "medium", "hard"]
    z = []
    for diff in diffs:
        z.append([sum(1 for r in rows if r["domain"] == d and r["difficulty"] == diff) for d in domains])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[label(d) for d in domains],
            y=[d.title() for d in diffs],
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            hovertemplate="Domain: %{x}<br>Difficulty: %{y}<br>Questions: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Dev question set coverage (48 questions)", height=360)
    return fig


def question_lengths(rows: list[dict]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Question length (chars)", "Ground-truth answer length (chars)"))
    for col, field, color in [(1, "q_len", "#4C78A8"), (2, "a_len", "#F58518")]:
        for diff, dash in [("easy", "solid"), ("medium", "dot"), ("hard", "dash")]:
            vals = [r[field] for r in rows if r["difficulty"] == diff]
            fig.add_box(y=vals, name=diff, legendgroup=diff, showlegend=col == 1, marker_color=color, boxmean=True, row=1, col=col)
    fig.update_layout(title="Text length by difficulty", height=420)
    return fig


def citation_mix(rows: list[dict]) -> go.Figure:
    diffs = ["easy", "medium", "hard"]
    pdf_counts, txt_counts, multi_group = [], [], []
    for diff in diffs:
        subset = [r for r in rows if r["difficulty"] == diff]
        pdf_counts.append(sum(1 for r in subset for k in r["source_kinds"] if k == "pdf"))
        txt_counts.append(sum(1 for r in subset for k in r["source_kinds"] if k == "txt"))
        multi_group.append(sum(1 for r in subset if r["num_groups"] > 1))

    fig = go.Figure()
    fig.add_bar(name="PDF citations", x=diffs, y=pdf_counts, marker_color="#4C78A8")
    fig.add_bar(name="Web page citations", x=diffs, y=txt_counts, marker_color="#F58518")
    fig.add_scatter(
        name="Multi-document questions",
        x=diffs,
        y=multi_group,
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#E45756", width=3),
    )
    fig.update_layout(
        title="Citation source types by difficulty",
        yaxis_title="Citation count",
        yaxis2=dict(title="Questions needing 2+ source groups", overlaying="y", side="right"),
        barmode="group",
        height=420,
    )
    return fig


def domain_gap_chart(files: list[dict], rows: list[dict]) -> go.Figure:
    corpus_domains = sorted({f["domain"] for f in files}, key=lambda d: label(d))
    q_domains = {r["domain"] for r in rows}
    corpus_counts = [sum(1 for f in files if f["domain"] == d) for d in corpus_domains]
    has_dev = ["Yes" if d in q_domains else "No" for d in corpus_domains]

    fig = go.Figure(
        go.Bar(
            x=[label(d) for d in corpus_domains],
            y=corpus_counts,
            marker_color=["#54A24B" if d in q_domains else "#BAB0AC" for d in corpus_domains],
            text=has_dev,
            textposition="outside",
            hovertemplate="%{x}<br>Documents: %{y}<br>Dev questions: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Corpus size vs. dev-set coverage (green = has dev questions)",
        yaxis_title="Documents in corpus",
        height=460,
    )
    return fig


def build_summary(files: list[dict], txt_stats: dict, pdf_pages: list[int], qdata: dict) -> str:
    rows = qdata["rows"]
    total_mb = sum(f["size"] for f in files) / 1e6
    domains_all = sorted({f["domain"] for f in files})
    domains_dev = sorted({r["domain"] for r in rows})
    blind_only = [d for d in domains_all if d not in domains_dev]

    cited_missing = sorted(
        {
            src["file"]
            for r in rows
            for g in r["ground_truth_sources"]
            for src in g["any_of"]
            if src["file"] not in {f["rel"] for f in files}
        }
    )

    return f"""
    <div class="cards">
      <div class="card"><div class="num">{len(files)}</div><div class="lbl">Corpus documents</div></div>
      <div class="card"><div class="num">{len(rows)}</div><div class="lbl">Dev questions</div></div>
      <div class="card"><div class="num">{len(domains_all)}</div><div class="lbl">Insurance domains</div></div>
      <div class="card"><div class="num">{total_mb:.0f} MB</div><div class="lbl">Corpus size</div></div>
      <div class="card"><div class="num">{txt_stats['hebrew_ratio_avg']*100:.0f}%</div><div class="lbl">Avg Hebrew in web pages</div></div>
      <div class="card"><div class="num">{sum(p for p in pdf_pages if p) / max(1, sum(1 for p in pdf_pages if p)):.0f}</div><div class="lbl">Avg PDF pages</div></div>
    </div>
    <h2>Key findings</h2>
    <ul>
      <li><strong>Two document types:</strong> <code>files/*.pdf</code> are official policy PDFs (page numbers matter for citations); <code>pages/*.txt</code> are scraped marketing/info web pages (page is usually <code>null</code>).</li>
      <li><strong>Hebrew-first:</strong> Web pages are ~{txt_stats['hebrew_ratio_avg']*100:.0f}% Hebrew characters. PDFs use Hebrew filenames and legal language.</li>
      <li><strong>Dev set is a subset:</strong> 48 questions cover 8/12 domains. No dev questions yet for: <em>{', '.join(label(d) for d in blind_only)}</em> — blind eval will test generalization there.</li>
      <li><strong>Difficulty ramps citation complexity:</strong> Easy questions cite PDFs with explicit page numbers. Hard questions often require 2 source groups (multi-document synthesis); 10/16 hard questions need multiple documents.</li>
      <li><strong>Numeric answers are common:</strong> {sum(1 for r in rows if r['has_number'])}/{len(rows)} ground-truth answers contain digits (limits, percentages, waiting periods).</li>
      <li><strong>Citation schema:</strong> Each fact has an <code>any_of</code> group — citing any one acceptable source earns full credit. Cross-document questions need a hit from every group.</li>
      {f'<li><strong>Data gap:</strong> 1 cited file missing from corpus: <code>{cited_missing[0]}</code> (likely renamed page scrape).</li>' if cited_missing else ''}
    </ul>
    <h2>Implications for your system</h2>
    <ul>
      <li><strong>Chunking:</strong> Median PDF is {sorted([p for p in pdf_pages if p])[len([p for p in pdf_pages if p])//2]} pages; median web scrape is {sorted(txt_stats['lengths'])[len(txt_stats['lengths'])//2]:,} chars. Page-aligned chunks for PDFs; whole-page or section chunks for .txt.</li>
      <li><strong>Retrieval:</strong> Business domain has the largest corpus ({max(sum(1 for f in files if f['domain']==d) for d in domains_all)} docs) but same dev-question count as others — retrieval precision matters.</li>
      <li><strong>Evaluation:</strong> Track citation accuracy separately for PDF (file+page) vs web (file only). Hallucination risk is highest on medium/hard numeric questions.</li>
      <li><strong>Baseline trap:</strong> Models may answer easy Hebrew insurance questions plausibly without documents — your harness must measure groundedness, not fluency alone.</li>
    </ul>
    """


def build_html(files: list[dict], txt_stats: dict, pdf_pages: list[int], qdata: dict) -> str:
    figs = [
        corpus_domain_bar(files),
        domain_gap_chart(files, qdata["rows"]),
        corpus_size_scatter(files),
        pdf_page_hist(pdf_pages),
        pdf_keyword_bar(files),
        question_heatmap(qdata["rows"]),
        question_lengths(qdata["rows"]),
        citation_mix(qdata["rows"]),
    ]

    plot_divs = []
    for i, fig in enumerate(figs):
        plot_divs.append(fig_to_div(fig, include_js=(i == 0)))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = build_summary(files, txt_stats, pdf_pages, qdata)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>APEX Exercise 2 — Data Exploration Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f6f8fb; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #1e3a5f, #2563eb); color: white; padding: 2rem 2.5rem; }}
    header h1 {{ margin: 0 0 0.4rem; font-size: 1.8rem; }}
    header p {{ margin: 0; opacity: 0.9; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
    section {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }}
    h2 {{ margin-top: 0; font-size: 1.2rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin: 1rem 0 1.5rem; }}
    .card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; text-align: center; }}
    .num {{ font-size: 1.6rem; font-weight: 700; color: #1d4ed8; }}
    .lbl {{ font-size: 0.82rem; color: #6b7280; margin-top: 0.25rem; }}
    code {{ background: #eef2ff; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.9em; }}
    ul {{ line-height: 1.55; }}
    .plot {{ margin-top: 0.5rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 0.5rem 0.6rem; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <header>
    <h1>Harel Insurance Exercise — Data Exploration Report</h1>
    <p>Interactive analysis of <code>corpus/</code> and <code>reference_questions.json</code> · Generated {generated}</p>
  </header>
  <main>
    <section>{summary}</section>

    <section><h2>1. Corpus composition by domain</h2><div class="plot">{plot_divs[0]}</div></section>
    <section><h2>2. Dev-set coverage vs. full corpus</h2><div class="plot">{plot_divs[1]}</div></section>
    <section><h2>3. Document sizes</h2><div class="plot">{plot_divs[2]}</div></section>
    <section><h2>4. PDF structure</h2><div class="plot">{plot_divs[3]}</div><div class="plot">{plot_divs[4]}</div></section>
    <section><h2>5. Dev questions</h2><div class="plot">{plot_divs[5]}</div><div class="plot">{plot_divs[6]}</div></section>
    <section><h2>6. Citation patterns (what grading expects)</h2><div class="plot">{plot_divs[7]}</div></section>

    <section>
      <h2>Data schemas</h2>
      <table>
        <tr><th>Asset</th><th>Fields</th><th>Notes</th></tr>
        <tr><td><code>reference_questions.json</code></td><td>id, domain, difficulty, question, ground_truth_answer, ground_truth_sources[]</td><td>48 dev examples; sources use <code>any_of</code> groups</td></tr>
        <tr><td><code>ground_truth_sources</code></td><td>any_of: [{{file, page}}]</td><td>page is 1-based for PDFs; null for web .txt pages</td></tr>
        <tr><td><code>corpus/manifest.json</code></td><td>local_path → original URL</td><td>571 entries; frozen snapshot — do not re-scrape live site</td></tr>
        <tr><td><code>corpus/&lt;domain&gt;/files/</code></td><td>PDF policy documents</td><td>Hebrew filenames; legal/policy language; page citations required</td></tr>
        <tr><td><code>corpus/&lt;domain&gt;/pages/</code></td><td>TXT scraped web pages</td><td>~3–6K chars each; navigation boilerplate included; mostly Hebrew</td></tr>
      </table>
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    files = load_manifest_files()
    txt_stats = analyze_txt(files)
    pdf_pages, _ = analyze_pdfs(files)
    qdata = analyze_questions(files)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_html(files, txt_stats, pdf_pages, qdata), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
