#!/usr/bin/env python3
"""
Build an interactive HTML dashboard comparing multiple answer runs.

    python scripts/generate_compare_dashboard.py \
      --run "DeepSeek V4|baseline_answers.jsonl" \
      --eval "DeepSeek V4|reports/stage1/baseline_eval.json" \
      --out reports/stage1/compare_dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prompt_catalog import build_prompt_spec, render_prompt_section, resolve_preset
from report_docs import render_deliverables_section

DOMAIN_LABELS = {
    "apartment": "Apartment",
    "business": "Business",
    "car": "Car",
    "dental": "Dental",
    "health": "Health",
    "life": "Life",
    "mortgage": "Mortgage",
    "travel": "Travel",
}


def parse_labeled_arg(raw: str) -> tuple[str, Path]:
    if "|" not in raw:
        path = Path(raw)
        return path.stem, path
    label, path = raw.split("|", 1)
    return label.strip(), Path(path)


def load_questions(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["questions"]
    return {q["id"]: q for q in data}


def load_answers(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def load_eval(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in data.get("results", [])}


def avg(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def p50(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def esc(text: str) -> str:
    return html.escape(text or "")


def build_run_stats(
    label: str,
    answers: dict[str, dict],
    questions: dict[str, dict],
    eval_rows: dict[str, dict] | None,
    corpus_rows: dict[str, dict] | None = None,
) -> dict:
    rows = []
    for qid, q in questions.items():
        if qid not in answers:
            continue
        a = answers[qid]
        ev = (eval_rows or {}).get(qid, {})
        ce = (corpus_rows or {}).get(qid, {})
        cite_score = ev.get("citation_score")
        rows.append(
            {
                "id": qid,
                "domain": q["domain"],
                "difficulty": q["difficulty"],
                "citation_score": cite_score,
                "latency_ms": a.get("latency_ms"),
                "answer_len": len(a.get("answer", "")),
                "relevant": ev.get("relevant"),
                "hallucination": ev.get("hallucination"),
                "refusal": ev.get("refusal"),
                "gt_covered": ce.get("gt_covered"),
                "fully_supported": ce.get("fully_supported"),
                "corpus_ok": ce.get("ok"),
                "corpus_refusal": ce.get("refusal"),
            }
        )

    judged = [r for r in rows if r["relevant"] is not None]
    relevance_rate = (
        avg([1.0 if r["relevant"] else 0.0 for r in judged]) if judged else None
    )
    hallucination_rate = (
        avg([1.0 if r["hallucination"] else 0.0 for r in judged]) if judged else None
    )
    refusal_rate = (
        avg([1.0 if r["refusal"] else 0.0 for r in judged]) if judged else None
    )
    # Answer-quality PR (one operating point per run):
    #   recall    = fraction of questions with a relevant answer
    #   precision = relevant / (relevant + hallucinated)  — quality among assertive errors
    recall = relevance_rate
    precision = None
    if judged:
        n_rel = sum(1 for r in judged if r["relevant"])
        n_hall = sum(1 for r in judged if r["hallucination"])
        denom = n_rel + n_hall
        precision = (n_rel / denom) if denom else 1.0

    corpus_judged = [r for r in rows if r["corpus_ok"] is not None]
    gt_covered_rate = (
        avg([1.0 if r["gt_covered"] else 0.0 for r in corpus_judged])
        if corpus_judged
        else None
    )
    fully_supported_rate = (
        avg([1.0 if r["fully_supported"] else 0.0 for r in corpus_judged])
        if corpus_judged
        else None
    )
    corpus_ok_rate = (
        avg([1.0 if r["corpus_ok"] else 0.0 for r in corpus_judged])
        if corpus_judged
        else None
    )
    # Corpus PR: hit = both checks true (ok).
    #   recall    = fraction of questions with ok
    #   precision = ok / (ok + assertive failures)  — refusals excluded from denom
    corpus_recall = corpus_ok_rate
    corpus_precision = None
    if corpus_judged:
        n_ok = sum(1 for r in corpus_judged if r["corpus_ok"])
        n_fail = sum(
            1
            for r in corpus_judged
            if not r["corpus_ok"] and not r.get("corpus_refusal")
        )
        denom = n_ok + n_fail
        corpus_precision = (n_ok / denom) if denom else 1.0

    return {
        "label": label,
        "count": len(rows),
        "citation_accuracy": avg(
            [r["citation_score"] for r in rows if r["citation_score"] is not None]
        ),
        "latency_avg": avg([r["latency_ms"] for r in rows if r["latency_ms"]]),
        "latency_p50": p50([r["latency_ms"] for r in rows if r["latency_ms"]]),
        "answer_len_avg": avg([r["answer_len"] for r in rows]),
        "relevance_rate": relevance_rate,
        "hallucination_rate": hallucination_rate,
        "refusal_rate": refusal_rate,
        "recall": recall,
        "precision": precision,
        "gt_covered_rate": gt_covered_rate,
        "fully_supported_rate": fully_supported_rate,
        "corpus_ok_rate": corpus_ok_rate,
        "corpus_recall": corpus_recall,
        "corpus_precision": corpus_precision,
        "rows": rows,
    }


RUN_COLORS = ["#4C78A8", "#F58518", "#E45756", "#B279A2", "#54A24B", "#EECA3B", "#9D755D", "#FF9DA6"]
PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}
CHART_HEIGHT = 520
PR_CHART_HEIGHT = 340
CHART_MARGIN_TOP = 72
CHART_MARGIN_LEFT = 60
CHART_MARGIN_RIGHT = 35


def _run_colors(n: int) -> list[str]:
    return [RUN_COLORS[i % len(RUN_COLORS)] for i in range(n)]


def _y_headroom(vals: list[float], *, as_pct: bool = False) -> float:
    numeric = [v for v in vals if v is not None]
    if not numeric:
        return 15.0
    ymax = max(numeric)
    if as_pct or ymax <= 100:
        return min(100.0, ymax * 1.28 + 10.0)
    return ymax * 1.18 + max(ymax * 0.05, 50.0)


def _legend_bottom_margin(n_series: int) -> tuple[int, float]:
    """Reserve space below the plot for a horizontal legend (wraps ~4 items/row)."""
    rows = max(1, (n_series + 3) // 4)
    bottom = 48 + rows * 26
    y = -0.12 - (rows - 1) * 0.14
    return bottom, y


def _apply_chart_layout(
    fig: go.Figure,
    *,
    y_vals: list[float],
    as_pct: bool = False,
    legend: bool = False,
    n_series: int = 0,
    margin_bottom: int = 48,
    height: int | None = None,
) -> None:
    bottom = margin_bottom
    axis = dict(
        automargin=True,
        gridcolor="#2a3544",
        zerolinecolor="#2a3544",
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#94a3b8"),
    )
    layout: dict = dict(
        height=CHART_HEIGHT if height is None else height,
        autosize=True,
        paper_bgcolor="#151b24",
        plot_bgcolor="#151b24",
        font=dict(color="#e5e7eb"),
        title_font=dict(color="#f1f5f9"),
        margin=dict(
            t=CHART_MARGIN_TOP,
            r=CHART_MARGIN_RIGHT,
            b=bottom,
            l=CHART_MARGIN_LEFT,
        ),
        xaxis=dict(**axis),
        yaxis=dict(
            **axis,
            range=[0, _y_headroom(y_vals, as_pct=as_pct)],
        ),
    )
    if legend:
        legend_bottom, legend_y = _legend_bottom_margin(n_series)
        layout["margin"]["b"] = max(bottom, legend_bottom)
        layout["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#cbd5e1"),
            bgcolor="rgba(0,0,0,0)",
            tracegroupgap=6,
        )
    fig.update_layout(**layout)


def _chart_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)


def fig_metric_bars(runs: list[dict], metric: str, title: str, as_pct: bool = False, color: str = "#4C78A8") -> str:
    labels = [r["label"] for r in runs]
    raw_vals = [r[metric] for r in runs]
    if as_pct:
        text = [f"{v:.0%}" if v is not None else "n/a" for v in raw_vals]
        vals = [v * 100 if v is not None else 0 for v in raw_vals]
        ytitle = "%"
    else:
        text = [f"{v:.0f}" if v is not None else "n/a" for v in raw_vals]
        vals = raw_vals
        ytitle = "ms" if "latency" in metric else "chars" if "len" in metric else ""

    fig = go.Figure(go.Bar(x=labels, y=vals, text=text, textposition="outside", marker_color=color))
    fig.update_layout(title=title, yaxis_title=ytitle)
    bottom = 96 if len(labels) > 4 else 48
    if len(labels) > 4:
        fig.update_layout(xaxis_tickangle=-25)
    _apply_chart_layout(fig, y_vals=vals, as_pct=as_pct, margin_bottom=bottom)
    return _chart_html(fig)


def _pr_axis_range(xy: list[tuple[float, float]], *, pad: float = 6.0) -> list[float]:
    """Tight axis window around points (still within 0–100)."""
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    lo = max(0.0, min(xs + ys) - pad)
    hi = min(100.0, max(xs + ys) + pad)
    if hi - lo < 20:
        mid = (lo + hi) / 2
        lo = max(0.0, mid - 10)
        hi = min(100.0, mid + 10)
    return [lo, hi]


def _fig_pr_scatter(
    pts: list[dict],
    *,
    xy: list[tuple[float, float]],
    title: str,
    hover_lines: list[str],
) -> str:
    """Shared PR scatter: xy are (recall%, precision%) in [0,100].

    Coincident points stay stacked at the true coordinates; hover lists every
    run that shares the point.
    """
    if not pts:
        return ""

    by_coord: dict[tuple[float, float], list[str]] = defaultdict(list)
    for r, (rec, prec) in zip(pts, xy):
        by_coord[(round(rec, 3), round(prec, 3))].append(r["label"])

    colors = _run_colors(len(pts))
    fig = go.Figure()
    for r, color, (rec, prec), hover in zip(pts, colors, xy, hover_lines):
        peers = by_coord[(round(rec, 3), round(prec, 3))]
        if len(peers) > 1:
            peer_note = "<br><br><b>Runs at this point:</b><br>• " + "<br>• ".join(
                peers
            )
            hover = hover.replace("<extra></extra>", peer_note + "<extra></extra>")
        fig.add_trace(
            go.Scatter(
                x=[rec],
                y=[prec],
                mode="markers",
                name=r["label"],
                marker=dict(
                    size=11,
                    color=color,
                    line=dict(width=1, color="#e2e8f0"),
                ),
                hovertemplate=hover,
            )
        )

    axis_range = _pr_axis_range(xy)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Recall (%)",
        yaxis_title="Precision (%)",
        showlegend=True,
    )
    # Side legend — bottom legend eats the plot when many runs share a compact chart.
    legend_w = 168 if len(pts) <= 8 else 200
    _apply_chart_layout(
        fig,
        y_vals=axis_range,
        as_pct=True,
        legend=False,
        margin_bottom=40,
        height=PR_CHART_HEIGHT,
    )
    fig.update_layout(
        margin=dict(t=56, r=legend_w, b=40, l=48),
        xaxis=dict(range=axis_range),
        yaxis=dict(range=axis_range, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=9, color="#cbd5e1"),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            tracegroupgap=2,
        ),
    )
    return _chart_html(fig)


def fig_precision_recall_scatter(runs: list[dict]) -> str:
    """One point per run: recall (x) vs precision (y) from answer-judge metrics."""
    pts = [
        r
        for r in runs
        if r.get("precision") is not None and r.get("recall") is not None
    ]
    if not pts:
        return ""
    xy = [(float(r["recall"]) * 100, float(r["precision"]) * 100) for r in pts]
    hovers = [
        (
            f"<b>{r['label']}</b><br>"
            f"Recall: {xy[i][0]:.1f}%<br>"
            f"Precision: {xy[i][1]:.1f}%<br>"
            f"Relevance: {(r['relevance_rate'] or 0)*100:.1f}%<br>"
            f"Hallucination: {(r['hallucination_rate'] or 0)*100:.1f}%"
            "<extra></extra>"
        )
        for i, r in enumerate(pts)
    ]
    return _fig_pr_scatter(
        pts,
        xy=xy,
        title=(
            "Precision–recall (answer judge)<br>"
            "<sup>Recall = % relevant · Precision = relevant / (relevant + hallucinated)</sup>"
        ),
        hover_lines=hovers,
    )


def fig_corpus_precision_recall_scatter(runs: list[dict]) -> str:
    """PR from corpus judge: hit = both GT covered and cite-supported (ok)."""
    pts = [
        r
        for r in runs
        if r.get("corpus_recall") is not None and r.get("corpus_precision") is not None
    ]
    if not pts:
        return ""
    xy = [
        (float(r["corpus_recall"]) * 100, float(r["corpus_precision"]) * 100)
        for r in pts
    ]
    hovers = [
        (
            f"<b>{r['label']}</b><br>"
            f"Recall: {xy[i][0]:.1f}%<br>"
            f"Precision: {xy[i][1]:.1f}%<br>"
            f"Corpus OK (hit): {(r.get('corpus_ok_rate') or 0)*100:.1f}%<br>"
            f"GT covered: {(r.get('gt_covered_rate') or 0)*100:.1f}%<br>"
            f"Cite-supported: {(r.get('fully_supported_rate') or 0)*100:.1f}%"
            "<extra></extra>"
        )
        for i, r in enumerate(pts)
    ]
    return _fig_pr_scatter(
        pts,
        xy=xy,
        title=(
            "Precision–recall (corpus judge)<br>"
            "<sup>Hit = GT covered ∧ cite-supported · "
            "Recall = % hit · Precision = hit / (hit + assertive miss)</sup>"
        ),
        hover_lines=hovers,
    )


def fig_grouped_metric_by_difficulty(runs: list[dict], metric: str, title: str) -> str:
    diffs = ["easy", "medium", "hard"]
    fig = go.Figure()
    all_y: list[float] = []
    for run, color in zip(runs, _run_colors(len(runs))):
        by_diff: dict[str, list[float]] = {d: [] for d in diffs}
        for row in run["rows"]:
            val = row.get(metric)
            if val is not None:
                by_diff[row["difficulty"]].append(1.0 if val else 0.0)
        y_vals = [avg(by_diff[d]) * 100 for d in diffs]
        all_y.extend(y_vals)
        fig.add_bar(
            name=run["label"],
            x=[d.title() for d in diffs],
            y=y_vals,
            text=[f"{avg(by_diff[d]):.0%}" for d in diffs],
            textposition="outside",
            marker_color=color,
        )
    fig.update_layout(title=title, barmode="group", yaxis_title="%")
    _apply_chart_layout(fig, y_vals=all_y, as_pct=True, legend=True, n_series=len(runs))
    return _chart_html(fig)


def summary_cards(runs: list[dict]) -> str:
    cards = []
    for run in runs:
        rel = run["relevance_rate"]
        hall = run["hallucination_rate"]
        ref = run["refusal_rate"]
        rel_s = f"{rel:.0%}" if rel is not None else "—"
        hall_s = f"{hall:.0%}" if hall is not None else "—"
        ref_s = f"{ref:.0%}" if ref is not None else "—"
        rel_cls = "good" if rel and rel >= 0.5 else "bad" if rel is not None else ""
        hall_cls = "bad" if hall and hall >= 0.3 else "good" if hall is not None else ""
        cok = run.get("corpus_ok_rate")
        cov = run.get("gt_covered_rate")
        supp = run.get("fully_supported_rate")
        corpus_metrics = ""
        if cok is not None:
            corpus_metrics = f"""
                <div><span class="k">GT covered</span><span class="v">{cov:.0%}</span></div>
                <div><span class="k">Cite-supported</span><span class="v">{supp:.0%}</span></div>
                <div><span class="k">Corpus OK</span><span class="v">{cok:.0%}</span></div>
            """
        cards.append(
            f"""
            <div class="run-card">
              <h3>{esc(run['label'])}</h3>
              <div class="hero-metrics">
                <div class="hero {rel_cls}"><span class="n">{rel_s}</span><span class="l">Relevance</span></div>
                <div class="hero {hall_cls}"><span class="n">{hall_s}</span><span class="l">Hallucination</span></div>
              </div>
              <div class="metrics">
                <div><span class="k">Refusal</span><span class="v">{ref_s}</span></div>
                <div><span class="k">Citation</span><span class="v">{run['citation_accuracy']:.0%}</span></div>
                {corpus_metrics}
                <div><span class="k">Latency avg</span><span class="v">{run['latency_avg']:.0f} ms</span></div>
                <div><span class="k">Latency p50</span><span class="v">{run['latency_p50']:.0f} ms</span></div>
                <div><span class="k">Answer len</span><span class="v">{run['answer_len_avg']:.0f}</span></div>
                <div><span class="k">Questions</span><span class="v">{run['count']}</span></div>
              </div>
            </div>
            """
        )
    return '<div class="run-cards">' + "".join(cards) + "</div>"


def question_payload(
    questions: dict[str, dict],
    run_answers: list[tuple[str, dict[str, dict]]],
    run_evals: list[tuple[str, dict[str, dict] | None]],
    run_corpus: list[tuple[str, dict[str, dict] | None]] | None = None,
) -> str:
    if run_corpus is None:
        run_corpus = [(label, None) for label, _ in run_answers]
    payload: dict[str, dict] = {}
    for qid, q in questions.items():
        runs = []
        for (label, answers), (_, evmap), (_, cemap) in zip(
            run_answers, run_evals, run_corpus
        ):
            a = answers.get(qid, {})
            ev = (evmap or {}).get(qid, {})
            ce = (cemap or {}).get(qid, {})
            cite_score = ev.get("citation_score")
            runs.append(
                {
                    "label": label,
                    "answer": a.get("answer", "(missing)"),
                    "citations": a.get("citations", []),
                    "latency_ms": a.get("latency_ms"),
                    "tokens": a.get("tokens"),
                    "citation_score": cite_score,
                    "relevant": ev.get("relevant"),
                    "hallucination": ev.get("hallucination"),
                    "refusal": ev.get("refusal"),
                    "judge_reasoning": ev.get("judge_reasoning"),
                    "citation_reasoning": ev.get("citation_reasoning"),
                    "gt_covered": ce.get("gt_covered"),
                    "fully_supported": ce.get("fully_supported"),
                    "corpus_ok": ce.get("ok"),
                    "corpus_judge_reasoning": ce.get("judge_reasoning"),
                }
            )
        payload[qid] = {
            "id": qid,
            "domain": q["domain"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "ground_truth": q["ground_truth_answer"],
            "sources": q["ground_truth_sources"],
            "runs": runs,
        }
    return json.dumps(payload, ensure_ascii=False)


def table_headers(runs: list[dict], has_eval: bool, has_corpus: bool) -> str:
    # sticky-c0..c3 = question columns pinned on horizontal scroll
    fixed = (
        '<th class="sortable sticky-col sticky-c0" data-key="id">ID</th>'
        '<th class="sortable sticky-col sticky-c1" data-key="domain">Domain</th>'
        '<th class="sortable sticky-col sticky-c2" data-key="difficulty">Diff</th>'
        '<th class="sortable sticky-col sticky-c3" data-key="outcome">Outcome</th>'
    )
    # Second header row needs one cell per sticky col (colspan breaks position:sticky).
    fixed_sub = (
        '<th class="sub sticky-col sticky-c0"></th>'
        '<th class="sub sticky-col sticky-c1"></th>'
        '<th class="sub sticky-col sticky-c2"></th>'
        '<th class="sub sticky-col sticky-c3"></th>'
    )
    per_run = []
    for i, r in enumerate(runs):
        label = esc(r["label"])
        if has_eval and has_corpus:
            per_run.append(f'<th colspan="9" class="run-group" data-run="{i}">{label}</th>')
        elif has_eval:
            per_run.append(f'<th colspan="6" class="run-group" data-run="{i}">{label}</th>')
        else:
            per_run.append(f'<th colspan="2" class="run-group" data-run="{i}">{label}</th>')
    sub = []
    for i in range(len(runs)):
        if has_eval:
            sub.extend([
                f'<th class="sub sortable" data-key="relevant" data-run="{i}">Rel</th>',
                f'<th class="sub sortable" data-key="hallucination" data-run="{i}">Hall</th>',
                f'<th class="sub sortable" data-key="refusal" data-run="{i}">Ref</th>',
            ])
            if has_corpus:
                sub.extend([
                    f'<th class="sub sortable" data-key="gt_covered" data-run="{i}" title="GT facts covered by answer">Cov</th>',
                    f'<th class="sub sortable" data-key="fully_supported" data-run="{i}" title="Answer supported by citations">Supp</th>',
                    f'<th class="sub sortable" data-key="corpus_ok" data-run="{i}" title="GT covered and cite-supported">COk</th>',
                ])
            sub.extend([
                f'<th class="sub sortable" data-key="citation" data-run="{i}">Cite</th>',
                f'<th class="sub sortable" data-key="latency" data-run="{i}">Lat</th>',
                f'<th class="sub sortable" data-key="tokens" data-run="{i}">Tok</th>',
            ])
        else:
            sub.extend([
                f'<th class="sub sortable" data-key="citation" data-run="{i}">Cite</th>',
                f'<th class="sub sortable" data-key="latency" data-run="{i}">Lat</th>',
            ])
    return f"<tr>{fixed}{''.join(per_run)}</tr><tr>{fixed_sub}{''.join(sub)}</tr>"


def build_html(
    runs: list[dict],
    questions: dict[str, dict],
    run_answers: list[tuple[str, dict[str, dict]]],
    run_evals: list[tuple[str, dict[str, dict] | None]],
    run_corpus: list[tuple[str, dict[str, dict] | None]] | None = None,
    title: str = "Run Comparison Dashboard",
    prompt_specs: list[tuple[str, dict]] | None = None,
    deliverables_html: str = "",
) -> str:
    if run_corpus is None:
        run_corpus = [(label, None) for label, _ in run_answers]
    has_eval = any(r["relevance_rate"] is not None for r in runs)
    has_corpus = any(r.get("corpus_ok_rate") is not None for r in runs)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    charts = []
    if has_eval:
        pr_scatter = fig_precision_recall_scatter(runs)
        if pr_scatter:
            charts.append(("pr", pr_scatter))
        charts.extend([
            ("", fig_metric_bars(runs, "relevance_rate", "Relevance rate", as_pct=True, color="#54A24B")),
            ("", fig_metric_bars(runs, "hallucination_rate", "Hallucination rate", as_pct=True, color="#E45756")),
            ("", fig_metric_bars(runs, "refusal_rate", "Refusal rate", as_pct=True, color="#F58518")),
            ("", fig_grouped_metric_by_difficulty(runs, "relevant", "Relevance by difficulty")),
            ("", fig_grouped_metric_by_difficulty(runs, "hallucination", "Hallucination by difficulty")),
        ])
    if has_corpus:
        corpus_pr = fig_corpus_precision_recall_scatter(runs)
        if corpus_pr:
            charts.append(("pr", corpus_pr))
        charts.extend([
            ("", fig_metric_bars(runs, "gt_covered_rate", "Corpus judge: GT covered", as_pct=True, color="#54A24B")),
            ("", fig_metric_bars(runs, "fully_supported_rate", "Corpus judge: cite-supported", as_pct=True, color="#72B7B2")),
            ("", fig_metric_bars(runs, "corpus_ok_rate", "Corpus judge: OK (both)", as_pct=True, color="#B279A2")),
        ])
    charts.extend([
        ("", fig_metric_bars(runs, "latency_avg", "Average latency (ms)", color="#4C78A8")),
        ("", fig_metric_bars(runs, "citation_accuracy", "Citation accuracy", as_pct=True, color="#72B7B2")),
    ])

    def _chart_class(kind: str) -> str:
        if kind == "wide":
            return "chart chart-wide"
        if kind == "pr":
            return "chart chart-pr"
        return "chart"

    chart_sections = "".join(
        f'<div class="{_chart_class(kind)}">{c}</div>' for kind, c in charts
    )
    qdata = question_payload(questions, run_answers, run_evals, run_corpus)
    run_labels = json.dumps([r["label"] for r in runs], ensure_ascii=False)
    run_toggles = (
        '<div class="run-toggle" id="run-toggle"><span class="run-toggle-label">Runs</span>'
        + "".join(
            f'<label class="run-check"><input type="checkbox" class="run-toggle-box" '
            f'data-run="{i}" checked /> {esc(r["label"])}</label>'
            for i, r in enumerate(runs)
        )
        + "</div>"
        if len(runs) > 1
        else ""
    )
    prompts_html = render_prompt_section(prompt_specs) if prompt_specs else ""
    prompts_nav = '<a href="#prompts">Prompts</a>' if prompt_specs else ""
    report_nav = '<a href="#report">Report</a>' if deliverables_html else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{esc(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #151b24;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #60a5fa;
      --border: #2a3544;
      --good: #34d399;
      --good-bg: #064e3b;
      --bad: #f87171;
      --bad-bg: #7f1d1d;
      --warn: #fbbf24;
      --warn-bg: #78350f;
      --neutral-bg: #1e293b;
      --surface: #1a222d;
      --input: #0f141c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }}
    header {{ background: linear-gradient(135deg, #0a1628, #1e3a5f); color: white; padding: 1.5rem 2rem 1.2rem; border-bottom: 1px solid var(--border); }}
    header h1 {{ margin: 0 0 .3rem; font-size: 1.55rem; font-weight: 700; }}
    header p {{ margin: 0; opacity: .88; font-size: .92rem; }}
    nav.toc {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .9rem; }}
    nav.toc a {{ color: white; opacity: .85; text-decoration: none; font-size: .82rem; padding: .25rem .55rem; border: 1px solid rgba(255,255,255,.25); border-radius: 999px; }}
    nav.toc a:hover {{ opacity: 1; background: rgba(255,255,255,.12); }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 1.1rem 1.25rem 2.5rem; }}
    section {{ background: var(--card); border-radius: 14px; border: 1px solid var(--border); box-shadow: 0 1px 4px rgba(0,0,0,.35); padding: 1rem 1.15rem; margin-bottom: 1rem; }}
    h2 {{ margin: 0 0 .75rem; font-size: 1.05rem; font-weight: 650; color: #f1f5f9; }}
    .run-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .75rem; }}
    .run-card {{ border: 1px solid var(--border); border-radius: 12px; padding: .85rem; background: var(--surface); }}
    .run-card h3 {{ margin: 0 0 .55rem; color: var(--accent); font-size: .95rem; }}
    .hero-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: .45rem; margin-bottom: .55rem; }}
    .hero {{ border-radius: 10px; padding: .45rem .55rem; text-align: center; background: var(--neutral-bg); }}
    .hero.good {{ background: var(--good-bg); }}
    .hero.bad {{ background: var(--bad-bg); }}
    .hero .n {{ display: block; font-size: 1.35rem; font-weight: 700; }}
    .hero .l {{ display: block; font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: .3rem .5rem; font-size: .8rem; }}
    .metrics .k {{ color: var(--muted); }}
    .metrics .v {{ font-weight: 600; text-align: right; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 1rem; align-items: start; }}
    .chart {{ min-width: 0; overflow: visible; min-height: {CHART_HEIGHT}px; }}
    .chart-wide {{ grid-column: 1 / -1; }}
    .chart-pr {{
      max-width: 560px;
      min-height: {PR_CHART_HEIGHT}px;
      justify-self: start;
    }}
    .chart > div {{ overflow: visible !important; height: {CHART_HEIGHT}px !important; }}
    .chart .plotly-graph-div {{ overflow: visible !important; height: {CHART_HEIGHT}px !important; }}
    .chart-pr > div,
    .chart-pr .plotly-graph-div {{ height: {PR_CHART_HEIGHT}px !important; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .75rem; }}
    .controls label {{ font-size: .78rem; color: var(--muted); display: flex; flex-direction: column; gap: .15rem; }}
    select, input {{ padding: .4rem .55rem; border: 1px solid var(--border); border-radius: 8px; font-size: .88rem; background: var(--input); color: var(--text); }}
    #q-select {{ min-width: 340px; }}
    .badge {{ display: inline-block; padding: .12rem .42rem; border-radius: 999px; font-size: .72rem; font-weight: 600; }}
    .badge.easy {{ background: var(--good-bg); color: #6ee7b7; }}
    .badge.medium {{ background: var(--warn-bg); color: #fcd34d; }}
    .badge.hard {{ background: var(--bad-bg); color: #fca5a5; }}
    .badge.domain {{ background: #1e3a5f; color: #93c5fd; }}
    .panel {{ display: grid; gap: .65rem; }}
    .block {{ border: 1px solid var(--border); border-radius: 10px; padding: .75rem .85rem; background: var(--surface); }}
    .block h4 {{ margin: 0 0 .45rem; font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .hebrew {{ line-height: 1.6; font-size: .95rem; }}
    .runs-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: .65rem; }}
    .run-answer {{ border: 1px solid var(--border); border-radius: 10px; padding: .75rem; background: var(--surface); }}
    .run-answer.fail {{ border-color: #7f1d1d; background: #1a1010; }}
    .run-answer.ok {{ border-color: #065f46; background: #0f1a16; }}
    .run-answer h5 {{ margin: 0 0 .35rem; color: var(--accent); font-size: .88rem; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .45rem; }}
    .pill {{ font-size: .7rem; font-weight: 600; padding: .12rem .4rem; border-radius: 999px; }}
    .pill.yes {{ background: var(--good-bg); color: var(--good); }}
    .pill.no {{ background: var(--bad-bg); color: var(--bad); }}
    .pill.neutral {{ background: var(--neutral-bg); color: var(--muted); }}
    .pill.warn {{ background: var(--warn-bg); color: var(--warn); }}
    .judge-box {{ margin-top: .5rem; padding: .45rem .55rem; background: var(--input); border-radius: 8px; font-size: .8rem; color: #cbd5e1; border-left: 3px solid var(--accent); }}
    .sources {{ font-size: .78rem; color: var(--muted); margin-top: .35rem; }}
    .sources code {{ background: #1e3a5f; padding: .08rem .28rem; border-radius: 4px; font-size: .74rem; color: #93c5fd; }}
    .table-toolbar {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .6rem; }}
    .table-wrap {{ max-height: 620px; overflow: auto; border: 1px solid var(--border); border-radius: 10px; }}
    table {{ width: max-content; min-width: 100%; border-collapse: separate; border-spacing: 0; font-size: .78rem; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: .38rem .45rem; text-align: center; vertical-align: middle; }}
    th {{ background: #1a222d; position: sticky; top: 0; z-index: 2; color: #cbd5e1; }}
    th.run-group {{ background: #1e3a5f; color: #93c5fd; font-size: .76rem; border-left: 2px solid #3b82f6; }}
    th.sub {{ top: 28px; font-size: .7rem; color: var(--muted); font-weight: 500; }}
    th.sortable {{ cursor: pointer; user-select: none; }}
    th.sortable:hover {{ background: #243044; }}
    /* Pin question columns while scrolling run metrics horizontally */
    .sticky-col {{ position: sticky; z-index: 1; background: var(--bg); }}
    th.sticky-col {{ z-index: 5; background: #1a222d; }}
    th.sticky-col.sub {{ z-index: 4; top: 28px; background: #1a222d; }}
    .sticky-c0 {{ left: 0; min-width: 210px; max-width: 240px; }}
    .sticky-c1 {{ left: 210px; min-width: 110px; }}
    .sticky-c2 {{ left: 320px; min-width: 72px; }}
    .sticky-c3 {{ left: 392px; min-width: 100px; box-shadow: 2px 0 0 #2a3544; }}
    td.id-cell {{ text-align: left; font-weight: 500; color: #e2e8f0; white-space: nowrap; }}
    td.domain-cell {{ text-align: left; color: var(--muted); }}
    tr.data-row {{ cursor: pointer; }}
    tr.data-row td.sticky-col {{ background: var(--bg); }}
    tr.data-row:nth-child(even) td.sticky-col {{ background: #121820; }}
    tr.data-row:hover td {{ background: #1e293b; }}
    tr.data-row:hover td.sticky-col {{ background: #1e293b; }}
    tr.data-row.active td {{ background: #1e3a5f; }}
    tr.data-row.active td.sticky-col {{ background: #1e3a5f; }}
    tr.data-row:nth-child(even) td:not(.sticky-col) {{ background: #121820; }}
    tr.data-row:nth-child(even):hover td {{ background: #1e293b; }}
    .cell-yes {{ color: var(--good); font-weight: 700; }}
    .cell-no {{ color: var(--bad); font-weight: 700; }}
    .cell-dash {{ color: #64748b; }}
    .outcome-pill {{ display: inline-block; padding: .1rem .38rem; border-radius: 999px; font-size: .68rem; font-weight: 700; text-transform: uppercase; }}
    .outcome-pill.relevant {{ background: var(--good-bg); color: var(--good); }}
    .outcome-pill.hallucination {{ background: var(--bad-bg); color: var(--bad); }}
    .outcome-pill.refusal {{ background: var(--warn-bg); color: var(--warn); }}
    .outcome-pill.wrong {{ background: var(--neutral-bg); color: #94a3b8; }}
    .outcome-pill.missing {{ background: #1e293b; color: #64748b; }}
    .count-label {{ font-size: .78rem; color: var(--muted); margin-left: auto; }}
    .run-toggle {{ display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .6rem; padding: .3rem .5rem; border: 1px solid var(--border); border-radius: 8px; background: var(--input); }}
    .run-toggle-label {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .run-check {{ display: inline-flex; align-items: center; gap: .3rem; font-size: .78rem; color: var(--text); cursor: pointer; }}
    .run-check input {{ accent-color: var(--accent); cursor: pointer; }}
    .section-note {{ margin: 0 0 .85rem; color: var(--muted); font-size: .88rem; }}
    .prompt-list {{ display: grid; gap: .45rem; }}
    .prompt-meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .35rem .75rem; font-size: .8rem; margin-bottom: .55rem; }}
    .prompt-meta .k {{ color: var(--muted); margin-right: .35rem; }}
    .prompt-meta .v {{ font-weight: 600; }}
    .prompt-note {{ margin: 0 0 .65rem; font-size: .82rem; color: var(--muted); }}
    .prompt-turn {{ margin-bottom: .65rem; }}
    .prompt-role {{ font-size: .72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: .2rem; }}
    .prompt-text {{ margin: 0; padding: .65rem .75rem; background: var(--input); border: 1px solid var(--border); border-radius: 8px; white-space: pre-wrap; word-break: break-word; font-size: .84rem; line-height: 1.55; overflow-x: auto; color: var(--text); }}
    .prompt-example {{ border: 1px solid var(--border); border-radius: 8px; padding: .35rem .5rem; margin-bottom: .35rem; background: var(--input); }}
    .prompt-example > summary {{ cursor: pointer; font-size: .78rem; color: #cbd5e1; }}
    .prompt-example .ex-num {{ color: var(--muted); font-weight: 600; }}
    .prompt-preset {{ border: 1px solid var(--border); border-radius: 10px; padding: .55rem .65rem; margin-bottom: .55rem; background: var(--surface); }}
    .prompt-preset > summary {{ cursor: pointer; font-size: .88rem; color: var(--accent); }}
    .prompt-system {{ margin: .45rem 0; }}
    .prompt-system > summary {{ cursor: pointer; font-size: .8rem; color: var(--muted); }}
    .prompt-examples-wrap {{ margin-top: .45rem; }}
    .prompt-examples-wrap > summary {{ cursor: pointer; font-size: .8rem; font-weight: 600; color: #cbd5e1; }}
    .prompt-examples-list {{ margin-top: .4rem; }}
    .prompt-map-wrap {{ overflow-x: auto; margin-bottom: .75rem; border: 1px solid var(--border); border-radius: 10px; }}
    .prompt-map {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
    .prompt-map th, .prompt-map td {{ padding: .4rem .55rem; border-bottom: 1px solid var(--border); text-align: left; }}
    .prompt-map th {{ background: #1a222d; color: var(--muted); font-weight: 600; }}
    .doc-list {{ display: grid; gap: .65rem; }}
    .doc-block {{ border: 1px solid var(--border); border-radius: 10px; padding: .65rem .75rem; background: var(--surface); }}
    .doc-block > summary {{ cursor: pointer; font-size: .92rem; color: var(--accent); margin-bottom: .35rem; }}
    .doc-body {{ font-size: .9rem; line-height: 1.6; color: #e2e8f0; }}
    .doc-body h3 {{ margin: .85rem 0 .4rem; font-size: .92rem; color: #f1f5f9; }}
    .doc-body p {{ margin: 0 0 .65rem; }}
    .doc-body .doc-lead {{ color: var(--muted); font-size: .88rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <p>{len(runs)} run(s) · {len(questions)} questions · generated {generated}</p>
    <nav class="toc">
      <a href="#summary">Summary</a>
      {report_nav}
      {prompts_nav}
      <a href="#charts">Charts</a>
      <a href="#explorer">Explorer</a>
      <a href="#table">All questions</a>
    </nav>
  </header>
  <main>
    <section id="summary">
      <h2>Run summary</h2>
      {summary_cards(runs)}
    </section>

    {deliverables_html}

    {prompts_html}

    <section id="charts">
      <h2>Metrics comparison</h2>
      <div class="charts">{chart_sections}</div>
    </section>

    <section id="explorer">
      <h2>Question explorer</h2>
      <div class="controls">
        <label>Question<select id="q-select"></select></label>
        <label>Domain<select id="domain-filter"><option value="">All</option>
          {''.join(f'<option value="{d}">{DOMAIN_LABELS.get(d,d)}</option>' for d in sorted({q['domain'] for q in questions.values()}))}
        </select></label>
        <label>Difficulty<select id="difficulty-filter"><option value="">All</option>
          <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
        </select></label>
        <label>Search<input id="search" type="search" placeholder="Question text or ID..." /></label>
      </div>
      <div class="panel">
        <div class="block">
          <h4>Question <span id="q-badges"></span></h4>
          <div id="q-text" class="hebrew" dir="rtl"></div>
        </div>
        <div class="block">
          <h4>Ground truth</h4>
          <div id="gt-text" class="hebrew" dir="rtl"></div>
          <div id="gt-sources" class="sources"></div>
        </div>
        <div class="block">
          <h4>Answers by run</h4>
          <div id="runs-grid" class="runs-grid"></div>
        </div>
      </div>
    </section>

    <section id="table">
      <h2>All questions table</h2>
      <div class="table-toolbar">
        <label>Filter<select id="table-filter">
          <option value="all">All questions</option>
          <option value="hallucination">Hallucinations only</option>
          <option value="relevant">Relevant only</option>
          <option value="refusal">Refusals only</option>
          <option value="wrong">Wrong (not relevant, not refusal)</option>
          <option value="corpus_fail">Corpus judge fail</option>
          <option value="corpus_ok">Corpus judge OK</option>
        </select></label>
        <label>Domain<select id="table-domain"><option value="">All</option>
          {''.join(f'<option value="{d}">{DOMAIN_LABELS.get(d,d)}</option>' for d in sorted({q['domain'] for q in questions.values()}))}
        </select></label>
        <label>Difficulty<select id="table-diff"><option value="">All</option>
          <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
        </select></label>
        {run_toggles}
        <span class="count-label" id="table-count"></span>
      </div>
      <div class="table-wrap">
        <table id="cmp-table">
          <thead>{table_headers(runs, has_eval, has_corpus)}</thead>
          <tbody id="cmp-body"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const QDATA = {qdata};
    const RUN_LABELS = {run_labels};
    const HAS_EVAL = {str(has_eval).lower()};
    const HAS_CORPUS = {str(has_corpus).lower()};
    let sortKey = 'id';
    let sortAsc = true;
    let activeRowId = null;
    const visibleRuns = new Set(RUN_LABELS.map((_, i) => i));

    function applyRunVisibility() {{
      document.querySelectorAll('#cmp-table [data-run]').forEach(el => {{
        el.style.display = visibleRuns.has(Number(el.dataset.run)) ? '' : 'none';
      }});
    }}

    function esc(s) {{ return (s ?? '').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
    function hebrewHtml(s) {{ return esc(s).replace(/\\n/g, '<br>'); }}

    function outcome(run) {{
      if (run.relevant == null) return 'missing';
      if (run.refusal) return 'refusal';
      if (run.hallucination) return 'hallucination';
      if (run.relevant) return 'relevant';
      return 'wrong';
    }}

    function primaryOutcome(q) {{
      const r = q.runs[0];
      return outcome(r);
    }}

    function primaryCorpusOk(q) {{
      for (const r of q.runs) {{
        if (r.corpus_ok != null) return r.corpus_ok;
      }}
      return null;
    }}

    function yn(val) {{
      if (val == null) return '<span class="cell-dash">—</span>';
      return val ? '<span class="cell-yes">✓</span>' : '<span class="cell-no">✗</span>';
    }}

    function pct(v) {{
      if (v == null) return '<span class="cell-dash">—</span>';
      const cls = v >= 1 ? 'cell-yes' : 'cell-no';
      return `<span class="${{cls}}">${{Math.round(v*100)}}%</span>`;
    }}

    function pill(label, val) {{
      if (val == null) return `<span class="pill neutral">${{label}} —</span>`;
      const cls = val ? 'yes' : 'no';
      return `<span class="pill ${{cls}}">${{label}} ${{val ? '✓' : '✗'}}</span>`;
    }}

    function renderQuestion(qid) {{
      const q = QDATA[qid];
      if (!q) return;
      activeRowId = qid;
      document.querySelectorAll('tr.data-row').forEach(tr => tr.classList.toggle('active', tr.dataset.id === qid));
      document.getElementById('q-badges').innerHTML =
        `<span class="badge ${{q.difficulty}}">${{q.difficulty}}</span> <span class="badge domain">${{q.domain}}</span>`;
      document.getElementById('q-text').innerHTML = hebrewHtml(q.question);
      document.getElementById('gt-text').innerHTML = hebrewHtml(q.ground_truth);
      document.getElementById('gt-sources').innerHTML = q.sources.map((g, i) => {{
        const opts = g.any_of.map(s => `<code>${{esc(s.file)}}${{s.page ? ' p.'+s.page : ''}}</code>`).join(' OR ');
        return `Group ${{i+1}}: ${{opts}}`;
      }}).join('<br>');

      document.getElementById('runs-grid').innerHTML = q.runs.map(r => {{
        const oc = outcome(r);
        const cardCls = oc === 'relevant' ? 'ok' : (oc === 'hallucination' || oc === 'wrong') ? 'fail' : '';
        const latency = r.latency_ms != null ? `${{Math.round(r.latency_ms)}} ms` : '—';
        const tokens = r.tokens ? `${{r.tokens.prompt}}+${{r.tokens.completion}}` : '—';
        const corpusPills = HAS_CORPUS
          ? pill('Cov', r.gt_covered) + pill('Supp', r.fully_supported) + pill('COk', r.corpus_ok)
          : '';
        const corpusJudge = r.corpus_judge_reasoning
          ? `<div class="judge-box"><strong>Corpus judge:</strong> ${{esc(r.corpus_judge_reasoning)}}</div>`
          : '';
        return `<div class="run-answer ${{cardCls}}">
          <h5>${{esc(r.label)}}</h5>
          <div class="pills">
            ${{HAS_EVAL ? pill('Rel', r.relevant) + pill('Hall', r.hallucination) + pill('Ref', r.refusal) : ''}}
            ${{corpusPills}}
            <span class="pill neutral">Cite ${{Math.round((r.citation_score||0)*100)}}%</span>
            <span class="pill neutral">${{latency}}</span>
            <span class="pill neutral">${{tokens}} tok</span>
          </div>
          <div class="hebrew" dir="rtl">${{hebrewHtml(r.answer)}}</div>
          ${{r.judge_reasoning ? `<div class="judge-box"><strong>Judge:</strong> ${{esc(r.judge_reasoning)}}</div>` : ''}}
          ${{corpusJudge}}
        </div>`;
      }}).join('');
    }}

    function filteredIds(source='explorer') {{
      const dom = document.getElementById(source === 'table' ? 'table-domain' : 'domain-filter').value;
      const diff = document.getElementById(source === 'table' ? 'table-diff' : 'difficulty-filter').value;
      const search = document.getElementById('search').value.trim().toLowerCase();
      const tableFilter = source === 'table' ? document.getElementById('table-filter').value : 'all';
      return Object.keys(QDATA).filter(id => {{
        const q = QDATA[id];
        const oc = primaryOutcome(q);
        const cok = primaryCorpusOk(q);
        if (dom && q.domain !== dom) return false;
        if (diff && q.difficulty !== diff) return false;
        if (search && !q.question.toLowerCase().includes(search) && !id.toLowerCase().includes(search)) return false;
        if (tableFilter === 'hallucination' && oc !== 'hallucination') return false;
        if (tableFilter === 'relevant' && oc !== 'relevant') return false;
        if (tableFilter === 'refusal' && oc !== 'refusal') return false;
        if (tableFilter === 'wrong' && oc !== 'wrong') return false;
        if (tableFilter === 'corpus_fail' && cok !== false) return false;
        if (tableFilter === 'corpus_ok' && cok !== true) return false;
        return true;
      }});
    }}

    function refreshSelect() {{
      const sel = document.getElementById('q-select');
      const current = sel.value;
      const ids = filteredIds('explorer');
      sel.innerHTML = ids.map(id => {{
        const q = QDATA[id];
        return `<option value="${{id}}">${{id}} · ${{q.difficulty}} · ${{q.domain}}</option>`;
      }}).join('');
      if (ids.includes(current)) sel.value = current;
      else if (ids.length) {{ sel.value = ids[0]; renderQuestion(ids[0]); }}
    }}

    function sortValue(q, key) {{
      const r = q.runs[0] || {{}};
      if (key === 'id') return q.id;
      if (key === 'domain') return q.domain;
      if (key === 'difficulty') return ['easy','medium','hard'].indexOf(q.difficulty);
      if (key === 'outcome') return primaryOutcome(q);
      if (key === 'relevant') return r.relevant ? 1 : 0;
      if (key === 'hallucination') return r.hallucination ? 1 : 0;
      if (key === 'refusal') return r.refusal ? 1 : 0;
      if (key === 'gt_covered' || key === 'fully_supported' || key === 'corpus_ok') {{
        for (const rr of q.runs) {{
          if (rr[key] != null) return rr[key] ? 1 : 0;
        }}
        return -1;
      }}
      if (key === 'citation') return r.citation_score || 0;
      if (key === 'latency') return r.latency_ms || 0;
      if (key === 'tokens') return (r.tokens?.prompt || 0) + (r.tokens?.completion || 0);
      return q.id;
    }}

    function buildTable() {{
      const ids = filteredIds('table').sort((a, b) => {{
        const qa = QDATA[a], qb = QDATA[b];
        const va = sortValue(qa, sortKey), vb = sortValue(qb, sortKey);
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
      }});
      document.getElementById('table-count').textContent = `${{ids.length}} / ${{Object.keys(QDATA).length}} shown`;
      const body = document.getElementById('cmp-body');
      body.innerHTML = ids.map(id => {{
        const q = QDATA[id];
        const oc = primaryOutcome(q);
        const cells = q.runs.map((r, ri) => {{
          const latency = r.latency_ms != null ? Math.round(r.latency_ms) : '<span class="cell-dash">—</span>';
          const tokens = r.tokens ? `${{r.tokens.prompt}}+${{r.tokens.completion}}` : '<span class="cell-dash">—</span>';
          if (HAS_EVAL) {{
            const corpusCells = HAS_CORPUS
              ? `<td data-run="${{ri}}">${{yn(r.gt_covered)}}</td><td data-run="${{ri}}">${{yn(r.fully_supported)}}</td><td data-run="${{ri}}">${{yn(r.corpus_ok)}}</td>`
              : '';
            return `<td data-run="${{ri}}">${{yn(r.relevant)}}</td><td data-run="${{ri}}">${{yn(r.hallucination)}}</td><td data-run="${{ri}}">${{yn(r.refusal)}}</td>${{corpusCells}}<td data-run="${{ri}}">${{pct(r.citation_score)}}</td><td data-run="${{ri}}">${{latency}}</td><td data-run="${{ri}}">${{tokens}}</td>`;
          }}
          return `<td data-run="${{ri}}">${{pct(r.citation_score)}}</td><td data-run="${{ri}}">${{latency}}</td>`;
        }}).join('');
        return `<tr class="data-row${{activeRowId===id?' active':''}}" data-id="${{id}}" data-domain="${{q.domain}}" data-diff="${{q.difficulty}}">
          <td class="id-cell sticky-col sticky-c0">${{id}}</td>
          <td class="domain-cell sticky-col sticky-c1">${{q.domain}}</td>
          <td class="sticky-col sticky-c2"><span class="badge ${{q.difficulty}}">${{q.difficulty}}</span></td>
          <td class="sticky-col sticky-c3"><span class="outcome-pill ${{oc}}">${{oc}}</span></td>
          ${{cells}}
        </tr>`;
      }}).join('');
      body.querySelectorAll('tr.data-row').forEach(tr => {{
        tr.addEventListener('click', () => {{
          document.getElementById('q-select').value = tr.dataset.id;
          renderQuestion(tr.dataset.id);
          document.getElementById('explorer').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }});
      }});
      applyRunVisibility();
    }}

    document.getElementById('q-select').addEventListener('change', e => renderQuestion(e.target.value));
    ['domain-filter','difficulty-filter','search'].forEach(id => {{
      const el = document.getElementById(id);
      el.addEventListener('input', () => {{ refreshSelect(); }});
      el.addEventListener('change', () => {{ refreshSelect(); }});
    }});
    ['table-filter','table-domain','table-diff'].forEach(id => {{
      document.getElementById(id).addEventListener('change', buildTable);
    }});
    document.querySelectorAll('.run-toggle-box').forEach(cb => {{
      cb.addEventListener('change', () => {{
        const ri = Number(cb.dataset.run);
        if (cb.checked) visibleRuns.add(ri); else visibleRuns.delete(ri);
        applyRunVisibility();
      }});
    }});
    document.querySelectorAll('th.sortable').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        if (sortKey === key) sortAsc = !sortAsc; else {{ sortKey = key; sortAsc = true; }}
        buildTable();
      }});
    }});

    function resizeCharts() {{
      document.querySelectorAll('.plotly-graph-div').forEach(el => {{
        if (el.offsetParent !== null) Plotly.Plots.resize(el);
      }});
    }}

    refreshSelect();
    buildTable();
    if (document.getElementById('q-select').value) renderQuestion(document.getElementById('q-select').value);
    window.addEventListener('load', resizeCharts);
    window.addEventListener('resize', resizeCharts);
  </script>
</body>
</html>"""


class _TabAction(argparse.Action):
    """``--tab`` starts a dataset; following --run/--eval/--corpus-eval/--prompt attach to it."""

    def __call__(self, parser, namespace, value, option_string=None):
        tabs = getattr(namespace, "tabs", None)
        if tabs is None:
            tabs = []
            setattr(namespace, "tabs", tabs)
        name, qpath = parse_labeled_arg(value)
        tabs.append(
            {"name": name, "questions": qpath, "run": [], "eval": [], "corpus": [], "prompt": []}
        )


def _current_bucket(namespace) -> dict:
    """The tab most recently opened by --tab, or a lazily-made default bucket."""
    tabs = getattr(namespace, "tabs", None)
    if tabs:
        return tabs[-1]
    bucket = getattr(namespace, "_default_bucket", None)
    if bucket is None:
        bucket = {"name": None, "questions": None, "run": [], "eval": [], "corpus": [], "prompt": []}
        setattr(namespace, "_default_bucket", bucket)
    return bucket


class _BucketAction(argparse.Action):
    """Append a ``Label|path`` value to the current tab (or the default bucket)."""

    def __call__(self, parser, namespace, value, option_string=None):
        _current_bucket(namespace)[self.dest].append(value)


def build_dataset_doc(ds: dict, args, deliverables_html: str) -> tuple[str, int]:
    """Run the load+stats pipeline for one dataset → (full HTML doc, n questions)."""
    qpath = ds["questions"] or Path(args.questions)
    questions = load_questions(ROOT / qpath)
    run_specs = [parse_labeled_arg(r) for r in ds["run"]]
    eval_map = {label: path for label, path in (parse_labeled_arg(e) for e in ds["eval"])}
    corpus_map = {label: path for label, path in (parse_labeled_arg(e) for e in ds["corpus"])}
    prompt_map: dict[str, str] = {}
    for raw in ds["prompt"]:
        label, preset = parse_labeled_arg(raw)
        prompt_map[label] = str(preset)

    run_answers: list[tuple[str, dict[str, dict]]] = []
    run_evals: list[tuple[str, dict[str, dict] | None]] = []
    run_corpus: list[tuple[str, dict[str, dict] | None]] = []
    runs: list[dict] = []
    prompt_specs: list[tuple[str, dict]] = []
    for label, path in run_specs:
        answers = load_answers(ROOT / path)
        ev_path = eval_map.get(label)
        ev_rows = load_eval(ROOT / ev_path) if ev_path and (ROOT / ev_path).exists() else None
        ce_path = corpus_map.get(label)
        ce_rows = (
            load_eval(ROOT / ce_path) if ce_path and (ROOT / ce_path).exists() else None
        )
        run_answers.append((label, answers))
        run_evals.append((label, ev_rows))
        run_corpus.append((label, ce_rows))
        runs.append(build_run_stats(label, answers, questions, ev_rows, ce_rows))
        preset = resolve_preset(label, ROOT / path, prompt_map.get(label))
        prompt_specs.append((label, build_prompt_spec(preset)))

    doc = build_html(
        runs,
        questions,
        run_answers,
        run_evals,
        run_corpus=run_corpus,
        title=ds["name"] or args.title,
        prompt_specs=prompt_specs,
        deliverables_html=deliverables_html,
    )
    return doc, len(questions)


def build_tab_shell(title: str, tabs: list[tuple[str, str, int]]) -> str:
    """Wrap N full dashboard documents in one page with top-level tabs.

    Each dashboard is a self-contained document rendered into its own
    ``<iframe srcdoc>`` — full CSS/JS isolation, so the per-dataset dashboards
    (which share global element ids) can't collide.
    """
    import html as _html

    buttons: list[str] = []
    frames: list[str] = []
    for i, (name, doc, nq) in enumerate(tabs):
        active = " active" if i == 0 else ""
        label = _html.escape(name or f"Tab {i + 1}")
        buttons.append(
            f'<button class="tabbtn{active}" data-i="{i}">{label}'
            f'<span class="cnt">{nq}Q</span></button>'
        )
        frames.append(
            f'<iframe class="tabframe{active}" data-i="{i}" '
            f'srcdoc="{_html.escape(doc, quote=True)}"></iframe>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0d0f14; color: #e6e6e6;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  .tabbar {{ display: flex; gap: 6px; align-items: center; padding: 8px 12px;
            background: #14171f; border-bottom: 1px solid #262b36;
            position: sticky; top: 0; z-index: 10; }}
  .tabbar .brand {{ font-weight: 600; font-size: 14px; margin-right: 12px; color: #9aa4b2; }}
  .tabbtn {{ padding: 7px 16px; cursor: pointer; border: 1px solid #2a2f3a;
            background: #1b1f28; color: #b9c1cc; border-radius: 8px; font-size: 13px;
            display: inline-flex; align-items: center; gap: 7px; }}
  .tabbtn:hover {{ background: #222735; }}
  .tabbtn.active {{ background: #2f6feb; border-color: #2f6feb; color: #fff; }}
  .tabbtn .cnt {{ font-size: 11px; opacity: .8;
                 background: rgba(255,255,255,.14); padding: 1px 6px; border-radius: 999px; }}
  .tabframe {{ width: 100%; height: calc(100vh - 49px); border: 0; display: none;
              background: #0d0f14; }}
  .tabframe.active {{ display: block; }}
</style>
</head>
<body>
  <div class="tabbar">
    <span class="brand">{_html.escape(title)}</span>
    {''.join(buttons)}
  </div>
  {''.join(frames)}
<script>
  const btns = [...document.querySelectorAll('.tabbtn')];
  const frames = [...document.querySelectorAll('.tabframe')];
  btns.forEach(b => b.addEventListener('click', () => {{
    const i = +b.dataset.i;
    btns.forEach(x => x.classList.toggle('active', x === b));
    frames.forEach(f => f.classList.toggle('active', +f.dataset.i === i));
  }}));
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate run comparison dashboard")
    ap.add_argument(
        "--questions",
        default="reference_questions.json",
        help="Questions file for the default (no --tab) view",
    )
    ap.add_argument(
        "--tab",
        action=_TabAction,
        help='Open a dataset tab: "Name|questions.json". Runs/evals/prompts that '
        "follow attach to it. Repeat for multiple tabs; omit for a single view.",
    )
    ap.add_argument("--run", action=_BucketAction, help='Label|path.jsonl')
    ap.add_argument("--eval", action=_BucketAction, help='Optional eval JSON: "Label|path.json"')
    ap.add_argument(
        "--corpus-eval",
        dest="corpus",
        action=_BucketAction,
        help='Optional corpus-judge JSON joined by label: "Label|path.json"',
    )
    ap.add_argument("--out", default="reports/stage1/compare_dashboard.html")
    ap.add_argument("--title", default="Run Comparison Dashboard", help="Page/browser title")
    ap.add_argument(
        "--prompt",
        action=_BucketAction,
        help='Optional preset per run: "Label|baseline|default|concise|no-cite"',
    )
    ap.add_argument(
        "--deliverables-dir",
        default="deliverables",
        help="Directory of *.md deliverables to embed (set empty to skip)",
    )
    args = ap.parse_args()

    tabs = getattr(args, "tabs", None)
    if tabs:
        datasets = tabs
    else:
        bucket = getattr(args, "_default_bucket", None) or {
            "name": None, "questions": None, "run": [], "eval": [], "prompt": []
        }
        bucket["questions"] = Path(args.questions)
        datasets = [bucket]

    if not any(ds["run"] for ds in datasets):
        ap.error("at least one --run is required")

    deliverables_dir = args.deliverables_dir.strip()
    deliverables_html = (
        render_deliverables_section(ROOT / deliverables_dir) if deliverables_dir else ""
    )

    rendered: list[tuple[str, str, int]] = []  # (name, doc, n_questions)
    for ds in datasets:
        doc, nq = build_dataset_doc(ds, args, deliverables_html)
        rendered.append((ds["name"], doc, nq))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(rendered) == 1:
        final_html = rendered[0][1]  # single dataset → unchanged single-page output
    else:
        final_html = build_tab_shell(args.title, rendered)
    out.write_text(final_html, encoding="utf-8")
    print(f"Wrote {out} ({len(rendered)} tab(s))")


if __name__ == "__main__":
    main()
