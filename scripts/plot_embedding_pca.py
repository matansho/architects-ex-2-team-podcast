#!/usr/bin/env python3
"""2D scatter of corpus embeddings + dev questions / GT answers (PCA or t-SNE).

    python scripts/plot_embedding_pca.py
    → reports/stage2/embedding_pca.html

    python scripts/plot_embedding_pca.py --method tsne
    → reports/stage2/embedding_tsne.html
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.embed import Embedder  # noqa: E402
from rag.index_store import load_embeddings, load_vectors  # noqa: E402

# Distinct domain colors (12 domains)
DOMAIN_COLORS = {
    "apartment": "#2563eb",
    "business": "#0891b2",
    "car": "#059669",
    "dental": "#65a30d",
    "diseases-disabilities": "#ca8a04",
    "health": "#ea580c",
    "life": "#dc2626",
    "long-term-care": "#db2777",
    "loss-of-working-ability": "#9333ea",
    "mortgage": "#7c3aed",
    "personal-accident": "#4f46e5",
    "travel": "#0d9488",
}


def _truncate(text: str, n: int = 400) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _looks_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def _wrap_lines(text: str, width: int = 52) -> list[str]:
    """Word-wrap a single paragraph for hover readability."""
    words = (text or "").split()
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and n + add > width:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def _hover_html(meta: str, body: str, *, max_chars: int = 420) -> str:
    """Readable Plotly hover: muted meta + wrapped RTL-aware body."""
    body = _truncate(body, max_chars)
    lines = _wrap_lines(body, width=48)
    rtl = _looks_hebrew(body) or _looks_hebrew(meta)
    dir_attr = ' dir="rtl"' if rtl else ""
    align = "right" if rtl else "left"
    meta_e = html.escape(meta)
    body_e = "<br>".join(html.escape(line) for line in lines)
    return (
        f'<span style="font-size:11px;color:#64748b;font-family:ui-sans-serif,system-ui,sans-serif">'
        f"{meta_e}</span><br>"
        f'<span{dir_attr} style="display:inline-block;text-align:{align};'
        f"max-width:340px;margin-top:4px;font-size:13px;line-height:1.5;"
        f'color:#0f172a;font-family:Segoe UI,Tahoma,Arial Hebrew,sans-serif">'
        f"{body_e}</span>"
    )


def _gt_files(q: dict) -> set[str]:
    files: set[str] = set()
    for block in q.get("ground_truth_sources") or []:
        for src in block.get("any_of") or []:
            f = src.get("file")
            if f:
                files.add(f)
    return files


def _file_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if path else path


def _file_color(idx: int) -> str:
    """Stable distinct hex color for source-document coloring."""
    h = (idx * 0.61803398875) % 1.0
    s = 0.55 + (idx % 3) * 0.12
    l = 0.40 + (idx % 4) * 0.06
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _pca_project(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Center + SVD PCA. Returns (coords, explained_variance_ratio for kept comps)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, vt.shape[0], X.shape[0] - 1)
    comps = vt[:k].T
    coords = Xc @ comps
    var = (s**2) / max(len(X) - 1, 1)
    total = var.sum() or 1.0
    explained = var[:k] / total
    return coords.astype(np.float32), explained


def reduce_2d(
    X: np.ndarray,
    method: str,
    *,
    perplexity: float = 30.0,
    pca_dims: int = 50,
) -> tuple[np.ndarray, str, str, str]:
    """Return (n,2) coords, method title, note HTML, axis label prefix."""
    if method == "pca":
        coords, explained = _pca_project(X, 2)
        ev0, ev1 = 100 * explained[0], 100 * explained[1]
        note = (
            "PCA fitted jointly on corpus chunks + question texts + ground-truth answers "
            f"(PC1 {ev0:.1f}% · PC2 {ev1:.1f}% variance)."
        )
        return coords, "2D PCA", note, "PC"

    if method == "tsne":
        n = X.shape[0]
        n_pca = min(pca_dims, n - 1, X.shape[1])
        Xp, _ = _pca_project(X, n_pca)
        perp = min(perplexity, max(5.0, (n - 1) / 3))
        print(f"t-SNE: PCA→{n_pca}d, perplexity={perp:.1f}, n={n}…", flush=True)
        coords = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(Xp).astype(np.float32)
        note = (
            f"t-SNE after PCA→{n_pca}d (perplexity={perp:.0f}), fitted jointly on "
            "corpus + questions + GT. Axes are not interpretable — look at local neighborhoods."
        )
        return coords, "2D t-SNE", note, "t-SNE"

    raise SystemExit(f"Unknown method: {method}")


def build_html(
    *,
    corpus: dict,
    domain_colors: dict[str, str],
    question_trace: dict,
    gt_answer_trace: dict,
    gt_chunk_trace: dict | None,
    method_title: str,
    note: str,
    axis: str,
    model: str,
    n_corpus: int,
    n_questions: int,
    n_files: int,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        "corpus": corpus,
        "domain_colors": domain_colors,
        "questions": question_trace,
        "gt_answers": gt_answer_trace,
        "gt_chunks": gt_chunk_trace,
    }
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Embedding {html.escape(method_title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ --bg:#f8fafc; --text:#0f172a; --muted:#64748b; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ background:#0f172a; color:#fff; padding:1rem 1.25rem; }}
    header h1 {{ margin:0 0 .35rem; font-size:1.2rem; }}
    header p {{ margin:0; opacity:.88; font-size:.85rem; }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:1rem 1.25rem 2rem; }}
    .note {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:.75rem 1rem;
             font-size:.86rem; color:var(--muted); margin-bottom:1rem; line-height:1.5; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:.75rem 1.25rem; align-items:center;
                 margin-bottom:1rem; background:#fff; border:1px solid #e2e8f0; border-radius:10px;
                 padding:.65rem 1rem; font-size:.86rem; }}
    .controls label {{ color:var(--muted); margin-right:.35rem; }}
    .seg {{ display:inline-flex; border:1px solid #cbd5e1; border-radius:8px; overflow:hidden; }}
    .seg button {{ border:0; background:#f8fafc; color:#334155; padding:.35rem .85rem;
                   font:inherit; cursor:pointer; }}
    .seg button + button {{ border-left:1px solid #cbd5e1; }}
    .seg button.active {{ background:#0f172a; color:#fff; }}
    .controls select {{ max-width:min(420px, 70vw); font:inherit; padding:.3rem .5rem;
                        border:1px solid #cbd5e1; border-radius:6px; background:#fff; }}
    #plot {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; min-height:720px; }}
    .legend-hint span {{ display:inline-block; margin-right:1rem; font-size:.8rem; }}
    .sw {{ display:inline-block; width:.7rem; height:.7rem; border-radius:2px; margin-right:.3rem; vertical-align:middle; }}
    .hoverlayer .hovertext path {{ opacity: 0.96 !important; }}
  </style>
</head>
<body>
  <header>
    <h1>Corpus embeddings — {html.escape(method_title)}</h1>
    <p>{html.escape(model)} · {n_corpus} corpus vectors · {n_files} source docs · {n_questions} dev questions · {generated}</p>
  </header>
  <div class="wrap">
    <div class="note">
      {html.escape(note)}
      <br/>
      <span class="legend-hint">
        <span><i class="sw" style="background:#fbbf24;border-radius:50%;border:1px solid #0f172a"></i><strong>★ Questions</strong></span>
        <span><i class="sw" style="background:#b91c1c"></i><strong>■ GT answers</strong></span>
        <span><i class="sw" style="background:#fff;border:2px solid #0f172a"></i><strong>○ GT source chunks</strong></span>
        <span id="color-hint">Colored dots = corpus by domain</span>
      </span>
      Hover any point for text.
    </div>
    <div class="controls">
      <div>
        <label>Color by</label>
        <div class="seg" id="color-mode">
          <button type="button" data-mode="domain" class="active">Domain</button>
          <button type="button" data-mode="document">Source document</button>
        </div>
      </div>
      <div id="doc-filter-wrap" style="display:none">
        <label for="doc-filter">Highlight doc</label>
        <select id="doc-filter"><option value="">All documents</option></select>
      </div>
    </div>
    <div id="plot"></div>
  </div>
  <script id="data" type="application/json">{json.dumps(payload, ensure_ascii=False)}</script>
  <script>
    const D = JSON.parse(document.getElementById('data').textContent);
    const C = D.corpus;
    const hoverlabel = {{
      bgcolor: '#ffffff',
      bordercolor: '#94a3b8',
      align: 'left',
      namelength: -1,
      font: {{
        size: 13,
        color: '#0f172a',
        family: 'Segoe UI, Tahoma, Arial Hebrew, sans-serif',
      }},
    }};

    const layout = {{
      margin: {{ t: 24, r: 20, b: 48, l: 56 }},
      height: 740,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#fff',
      font: {{ size: 12, color: '#334155' }},
      xaxis: {{ title: '{axis} 1', zeroline: true, gridcolor: '#f1f5f9' }},
      yaxis: {{ title: '{axis} 2', zeroline: true, gridcolor: '#f1f5f9' }},
      legend: {{ orientation: 'h', y: 1.12, font: {{ size: 11 }} }},
      hovermode: 'closest',
      hoverlabel,
      uirevision: 'embedding',
    }};
    const config = {{ responsive: true, displayModeBar: true }};

    // Populate document filter (basename → full path; disambiguate collisions)
    const fileByShort = {{}};
    const shorts = [];
    for (let i = 0; i < C.file.length; i++) {{
      const f = C.file[i];
      const short = C.file_short[i];
      if (!fileByShort[short]) {{
        fileByShort[short] = f;
        shorts.push(short);
      }} else if (fileByShort[short] !== f) {{
        // rare basename collision — keep first mapping; hover still has full path
      }}
    }}
    shorts.sort((a, b) => a.localeCompare(b, 'he'));
    const sel = document.getElementById('doc-filter');
    for (const s of shorts) {{
      const opt = document.createElement('option');
      opt.value = fileByShort[s];
      opt.textContent = s;
      sel.appendChild(opt);
    }}

    function overlayTraces() {{
      const out = [];
      if (D.gt_chunks && D.gt_chunks.x.length) {{
        out.push({{
          type: 'scatter',
          mode: 'markers',
          name: 'GT source chunk',
          x: D.gt_chunks.x, y: D.gt_chunks.y,
          text: D.gt_chunks.hover,
          customdata: D.gt_chunks.ids,
          hoverlabel,
          marker: {{
            size: 14,
            symbol: 'circle-open',
            color: '#0f172a',
            line: {{ width: 2.5, color: '#0f172a' }},
          }},
          hovertemplate: '%{{text}}<extra></extra>',
        }});
      }}
      out.push({{
        type: 'scatter',
        mode: 'markers+text',
        name: 'GT answer',
        x: D.gt_answers.x, y: D.gt_answers.y,
        text: D.gt_answers.labels,
        customdata: D.gt_answers.hover,
        textposition: 'top center',
        textfont: {{ size: 8, color: '#7f1d1d' }},
        hoverlabel,
        marker: {{
          size: 12,
          symbol: 'square',
          color: '#dc2626',
          line: {{ width: 1, color: '#7f1d1d' }},
        }},
        hovertemplate: '%{{customdata}}<extra></extra>',
      }});
      out.push({{
        type: 'scatter',
        mode: 'markers+text',
        name: 'Question',
        x: D.questions.x, y: D.questions.y,
        text: D.questions.labels,
        customdata: D.questions.hover,
        textposition: 'bottom center',
        textfont: {{ size: 8, color: '#0f172a' }},
        hoverlabel,
        marker: {{
          size: 14,
          symbol: 'star',
          color: '#fbbf24',
          line: {{ width: 1.5, color: '#0f172a' }},
        }},
        hovertemplate: '%{{customdata}}<extra></extra>',
      }});
      return out;
    }}

    function corpusTracesDomain() {{
      const by = {{}};
      for (let i = 0; i < C.x.length; i++) {{
        const d = C.domain[i];
        if (!by[d]) by[d] = {{ x: [], y: [], hover: [] }};
        by[d].x.push(C.x[i]);
        by[d].y.push(C.y[i]);
        by[d].hover.push(C.hover[i]);
      }}
      return Object.keys(by).sort().map(d => ({{
        type: 'scattergl',
        mode: 'markers',
        name: d,
        x: by[d].x, y: by[d].y,
        text: by[d].hover,
        hoverlabel,
        marker: {{
          size: 7,
          color: D.domain_colors[d] || '#64748b',
          opacity: 0.55,
          line: {{ width: 0 }},
        }},
        hovertemplate: '%{{text}}<extra></extra>',
      }}));
    }}

    function corpusTracesDocument(highlightFile) {{
      const opacity = C.file.map(f =>
        (!highlightFile || f === highlightFile) ? 0.7 : 0.06
      );
      const size = C.file.map(f =>
        (!highlightFile || f === highlightFile) ? 7 : 5
      );
      return [{{
        type: 'scattergl',
        mode: 'markers',
        name: 'corpus (by document)',
        x: C.x, y: C.y,
        text: C.hover,
        hoverlabel,
        showlegend: false,
        marker: {{
          size,
          color: C.file_color,
          opacity,
          line: {{ width: 0 }},
        }},
        hovertemplate: '%{{text}}<extra></extra>',
      }}];
    }}

    let colorMode = 'domain';
    let plotted = false;

    function render() {{
      const hint = document.getElementById('color-hint');
      const docWrap = document.getElementById('doc-filter-wrap');
      let corpus;
      if (colorMode === 'document') {{
        hint.textContent = 'Colored dots = corpus by source document (' + {n_files} + ' files; hover for name)';
        docWrap.style.display = '';
        corpus = corpusTracesDocument(sel.value || '');
        layout.showlegend = true;
      }} else {{
        hint.textContent = 'Colored dots = corpus by domain';
        docWrap.style.display = 'none';
        corpus = corpusTracesDomain();
        layout.showlegend = true;
      }}
      const traces = corpus.concat(overlayTraces());
      if (!plotted) {{
        Plotly.newPlot('plot', traces, layout, config);
        plotted = true;
      }} else {{
        Plotly.react('plot', traces, layout, config);
      }}
    }}

    document.getElementById('color-mode').addEventListener('click', (e) => {{
      const btn = e.target.closest('button[data-mode]');
      if (!btn) return;
      colorMode = btn.dataset.mode;
      for (const b of document.querySelectorAll('#color-mode button')) {{
        b.classList.toggle('active', b === btn);
      }}
      render();
    }});
    sel.addEventListener('change', () => {{
      if (colorMode === 'document') render();
    }});

    render();
  </script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="2D scatter of embeddings + questions/GT")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--method", choices=("pca", "tsne"), default="pca")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument(
        "--out",
        default=None,
        help="Default: reports/stage2/embedding_{method}.html",
    )
    ap.add_argument("--hover-chars", type=int, default=420)
    ap.add_argument(
        "--gt-top-k",
        type=int,
        default=2,
        help="Mark only top-k chunks per question (cosine to GT answer) "
        "from GT-cited files. 0 = mark every chunk from those files.",
    )
    args = ap.parse_args()

    out_rel = args.out or f"reports/stage2/embedding_{args.method}.html"

    index_dir = ROOT / args.index
    vectors = load_vectors(index_dir)
    emb = load_embeddings(index_dir)
    if len(vectors) != len(emb):
        raise SystemExit(f"Mismatch: {len(vectors)} vectors vs {len(emb)} embeddings")

    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    model_name = cfg.get("model", "intfloat/multilingual-e5-base")

    questions = json.loads((ROOT / args.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]

    print(f"Embedding {len(questions)} questions + GT answers…", flush=True)
    embedder = Embedder(model_name=model_name)
    q_texts = [q["question"] for q in questions]
    gt_texts = [q["ground_truth_answer"] for q in questions]
    q_emb = embedder.embed_queries(q_texts)
    # GT answers are passages for E5
    gt_emb = embedder.embed_passages(gt_texts)

    # Joint reduction so questions/GT sit in same plane as corpus
    X = np.vstack([emb, q_emb, gt_emb])
    coords, method_title, note, axis = reduce_2d(
        X, args.method, perplexity=args.perplexity
    )
    n_c, n_q = len(vectors), len(questions)
    corpus_xy = coords[:n_c]
    q_xy = coords[n_c : n_c + n_q]
    gt_xy = coords[n_c + n_q :]

    # GT source chunks: top-k by cosine to GT answer (avoids marking whole PDFs)
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(vectors):
        by_file[v.location.file].append(i)

    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    gt_chunk_idx: list[int] = []
    seen: set[int] = set()
    for qi, q in enumerate(questions):
        files = _gt_files(q)
        idxs = [i for f in files for i in by_file.get(f, [])]
        if not idxs:
            continue
        if args.gt_top_k <= 0:
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    gt_chunk_idx.append(i)
            continue
        gt_n = gt_emb[qi] / (np.linalg.norm(gt_emb[qi]) + 1e-9)
        sims = emb_n[idxs] @ gt_n
        order = np.argsort(-sims)[: args.gt_top_k]
        for j in order:
            i = idxs[int(j)]
            if i not in seen:
                seen.add(i)
                gt_chunk_idx.append(i)
    print(
        f"GT markers: {len(gt_chunk_idx)} chunks"
        + (f" (top-{args.gt_top_k} per question)" if args.gt_top_k > 0 else " (all from GT files)"),
        flush=True,
    )

    gt_set = set(gt_chunk_idx)
    # Corpus cloud excludes GT-source markers (they get their own layer).
    cloud_idxs = [i for i in range(n_c) if i not in gt_set]

    unique_files = sorted({vectors[i].location.file for i in cloud_idxs})
    file_to_color = {f: _file_color(j) for j, f in enumerate(unique_files)}
    domain_colors = {
        d: DOMAIN_COLORS.get(d, "#64748b")
        for d in sorted({vectors[i].location.domain or "unknown" for i in cloud_idxs})
    }

    corpus: dict = {
        "x": corpus_xy[cloud_idxs, 0].tolist(),
        "y": corpus_xy[cloud_idxs, 1].tolist(),
        "domain": [],
        "file": [],
        "file_short": [],
        "file_color": [],
        "hover": [],
    }
    for i in cloud_idxs:
        v = vectors[i]
        domain = v.location.domain or "unknown"
        fpath = v.location.file
        meta = f"{domain} · {fpath}"
        if v.location.page:
            meta += f" · p{v.location.page}"
        if v.location.label:
            meta += f" · {v.location.label}"
        corpus["domain"].append(domain)
        corpus["file"].append(fpath)
        corpus["file_short"].append(_file_basename(fpath))
        corpus["file_color"].append(file_to_color[fpath])
        corpus["hover"].append(
            _hover_html(meta, v.text, max_chars=args.hover_chars)
        )

    q_labels = [q["id"].replace("dev-", "") for q in questions]
    question_trace = {
        "x": q_xy[:, 0].tolist(),
        "y": q_xy[:, 1].tolist(),
        "labels": q_labels,
        "hover": [
            _hover_html(
                f"Question · {q['id']} · {q['domain']} · {q['difficulty']}",
                q["question"],
                max_chars=args.hover_chars,
            )
            for q in questions
        ],
    }
    gt_answer_trace = {
        "x": gt_xy[:, 0].tolist(),
        "y": gt_xy[:, 1].tolist(),
        "labels": q_labels,
        "hover": [
            _hover_html(
                f"GT answer · {q['id']}",
                q["ground_truth_answer"],
                max_chars=args.hover_chars,
            )
            for q in questions
        ],
    }

    gt_chunk_trace = None
    if gt_chunk_idx:
        gt_chunk_trace = {
            "x": corpus_xy[gt_chunk_idx, 0].tolist(),
            "y": corpus_xy[gt_chunk_idx, 1].tolist(),
            "ids": [vectors[i].id for i in gt_chunk_idx],
            "hover": [
                _hover_html(
                    f"GT source · {vectors[i].location.file}"
                    + (
                        f" · p{vectors[i].location.page}"
                        if vectors[i].location.page
                        else ""
                    ),
                    vectors[i].text,
                    max_chars=args.hover_chars,
                )
                for i in gt_chunk_idx
            ],
        }

    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(
            corpus=corpus,
            domain_colors=domain_colors,
            question_trace=question_trace,
            gt_answer_trace=gt_answer_trace,
            gt_chunk_trace=gt_chunk_trace,
            method_title=method_title,
            note=note,
            axis=axis,
            model=model_name,
            n_corpus=n_c,
            n_questions=n_q,
            n_files=len(unique_files),
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {out} · {method_title} · {len(unique_files)} docs · "
        f"{len(gt_chunk_idx)} GT source chunks"
    )


if __name__ == "__main__":
    main()
