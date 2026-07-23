#!/usr/bin/env python3
"""Evaluate dense retrieval on reference_questions.json.

    python scripts/eval_retrieval.py
    → prints summary + reports/stage2/retrieval_eval.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.embed import Embedder  # noqa: E402
from rag.index_store import load_embeddings, load_vectors_meta  # noqa: E402
from rag.retrieve import (  # noqa: E402
    build_id_index,
    flatten_expanded,
    groups_hit,
    hit_matches_source,
    search_expanded,
)


def _truncate(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_html(
    *,
    rows: list[dict],
    ks: list[int],
    summary: dict,
    model: str,
    n_corpus: int,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {"rows": rows, "ks": ks, "summary": summary}
    metric_cells = "".join(
        f"<td><strong>@{k}</strong><br/>"
        f"file {summary['file_hit'][str(k)]*100:.0f}% · "
        f"page {summary['page_group'][str(k)]*100:.0f}% · "
        f"full {summary['full_group'][str(k)]*100:.0f}%</td>"
        for k in ks
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Retrieval eval</title>
  <style>
    :root {{ --bg:#f8fafc; --text:#0f172a; --muted:#64748b; --ok:#15803d; --bad:#b91c1c; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ background:#0f172a; color:#fff; padding:1rem 1.25rem; }}
    header h1 {{ margin:0 0 .35rem; font-size:1.2rem; }}
    header p {{ margin:0; opacity:.88; font-size:.85rem; }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:1rem 1.25rem 2rem; }}
    .note {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:.75rem 1rem;
             font-size:.86rem; color:var(--muted); margin-bottom:1rem; line-height:1.5; }}
    table.metrics {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #e2e8f0;
                    border-radius:10px; overflow:hidden; margin-bottom:1rem; }}
    table.metrics td {{ padding:.75rem 1rem; border-right:1px solid #e2e8f0; font-size:.9rem; }}
    .q {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.1rem; margin-bottom:.75rem; }}
    .q h3 {{ margin:0 0 .35rem; font-size:.95rem; }}
    .meta {{ color:var(--muted); font-size:.8rem; margin-bottom:.5rem; }}
    .he {{ direction:rtl; text-align:right; font-family:Segoe UI,Tahoma,Arial Hebrew,sans-serif; line-height:1.45; }}
    .badge {{ display:inline-block; font-size:.72rem; padding:.15rem .45rem; border-radius:999px;
              margin-right:.35rem; border:1px solid #e2e8f0; }}
    .badge.ok {{ background:#dcfce7; color:var(--ok); border-color:#86efac; }}
    .badge.bad {{ background:#fee2e2; color:var(--bad); border-color:#fecaca; }}
    .hits {{ margin-top:.6rem; font-size:.82rem; }}
    .hit {{ padding:.35rem 0; border-top:1px solid #f1f5f9; }}
    .hit .score {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
    details {{ margin-top:.4rem; }}
    summary {{ cursor:pointer; color:var(--muted); font-size:.8rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Retrieval eval — dense cosine (E5)</h1>
    <p>{html.escape(model)} · {n_corpus} corpus vectors · {len(rows)} questions · {generated}</p>
  </header>
  <div class="wrap">
    <div class="note">
      <strong>file</strong> = any retrieved chunk file matches a GT pointer ·
      <strong>page</strong> = fraction of <code>any_of</code> groups satisfied (file + page when set) ·
      <strong>full</strong> = all groups satisfied.
      Each hit includes ±{summary.get('window', 1)} neighboring chunks from the same file.
    </div>
    <table class="metrics"><tr>{metric_cells}</tr></table>
    <div id="list"></div>
  </div>
  <script id="data" type="application/json">{json.dumps(payload, ensure_ascii=False)}</script>
  <script>
    const D = JSON.parse(document.getElementById('data').textContent);
    const root = document.getElementById('list');
    for (const r of D.rows) {{
      const el = document.createElement('div');
      el.className = 'q';
      const badges = D.ks.map(k => {{
        const g = r.by_k[String(k)];
        const cls = g.full ? 'ok' : (g.file ? 'ok' : 'bad');
        const label = g.full ? `full@${{k}}` : (g.file ? `file@${{k}}` : `miss@${{k}}`);
        return `<span class="badge ${{cls}}">${{label}}</span>`;
      }}).join('');
      const hits = (r.hits || []).map(h => `
        <div class="hit">
          <span class="score">#${{h.rank}} · ${{h.score.toFixed(3)}} · ${{h.role}}</span>
          · <code>${{h.file}}</code>${{h.page != null ? ' p'+h.page : ''}}
          ${{h.match ? ' <span class="badge ok">GT</span>' : ''}}
          <div class="he">${{h.preview}}</div>
        </div>`).join('');
      el.innerHTML = `
        <h3>${{r.id}} <span class="meta">${{r.domain}} · ${{r.difficulty}}</span></h3>
        <div>${{badges}}</div>
        <div class="he" style="margin-top:.5rem">${{r.question}}</div>
        <details>
          <summary>top hits + GT groups</summary>
          <div class="meta">${{(r.group_details || []).join(' · ')}}</div>
          <div class="hits">${{hits}}</div>
        </details>`;
      root.appendChild(el);
    }}
  </script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval retrieval on reference questions")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--out", default="reports/stage2/retrieval_eval.html")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--window",
        type=int,
        default=1,
        help="Include ±N neighboring chunks (same file) around each hit",
    )
    ap.add_argument("--ks", default="1,3,5,10", help="Comma-separated cutoffs to report")
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    max_k = max(max(ks), args.top_k)

    index_dir = ROOT / args.index
    print("Loading index…", flush=True)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    if len(vectors) != len(emb):
        raise SystemExit(f"Mismatch: {len(vectors)} vectors vs {len(emb)} embeddings")
    id_to_idx = build_id_index(vectors)
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    model = cfg.get("model", "intfloat/multilingual-e5-base")

    questions = json.loads((ROOT / args.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]

    print(
        f"Embedding {len(questions)} queries with {model} "
        f"(top_k={max_k}, window=±{args.window})…",
        flush=True,
    )
    embedder = Embedder(model_name=model)
    q_emb = embedder.embed_queries([q["question"] for q in questions])

    file_hit = {k: 0 for k in ks}
    page_group = {k: 0.0 for k in ks}
    full_group = {k: 0 for k in ks}
    rows: list[dict] = []

    for qi, q in enumerate(questions):
        expanded = search_expanded(
            q_emb[qi],
            emb,
            vectors,
            top_k=max_k,
            window=args.window,
            id_to_idx=id_to_idx,
        )
        by_k: dict[str, dict] = {}
        for k in ks:
            # Score using primary hits + their neighbors (window may recover page)
            sub = flatten_expanded(expanded[:k])
            sat, total, details = groups_hit(q.get("ground_truth_sources") or [], sub)
            from eval.corpus import normalize_path

            gt_files = {
                normalize_path(o["file"])
                for g in (q.get("ground_truth_sources") or [])
                for o in (g.get("any_of") or [])
                if o.get("file")
            }
            retrieved_files = {normalize_path(h.vector.location.file) for h in sub}
            f_ok = bool(gt_files & retrieved_files)
            g_frac = sat / total if total else 1.0
            full = sat == total and total > 0
            if f_ok:
                file_hit[k] += 1
            page_group[k] += g_frac
            if full:
                full_group[k] += 1
            by_k[str(k)] = {
                "file": f_ok,
                "group_frac": g_frac,
                "full": full,
                "sat": sat,
                "total": total,
            }

        flat_all = flatten_expanded(expanded)
        _, _, details_all = groups_hit(q.get("ground_truth_sources") or [], flat_all)
        hit_rows = []
        for ex in expanded[: max(ks)]:
            bundle = [
                ("match", ex.match, ex.score, ex.rank),
                *[(nb.role, nb.vector, ex.score, ex.rank) for nb in ex.neighbors],
            ]
            for role, vec, score, rank in bundle:
                loc = vec.location
                match = any(
                    hit_matches_source(loc, opt)
                    for g in q.get("ground_truth_sources") or []
                    for opt in g.get("any_of") or []
                )
                hit_rows.append(
                    {
                        "rank": rank,
                        "score": score,
                        "role": role,
                        "file": loc.file,
                        "page": loc.page,
                        "match": match,
                        "preview": html.escape(_truncate(vec.text)),
                    }
                )

        rows.append(
            {
                "id": q["id"],
                "domain": q.get("domain", ""),
                "difficulty": q.get("difficulty", ""),
                "question": html.escape(q["question"]),
                "by_k": by_k,
                "hits": hit_rows,
                "group_details": details_all,
            }
        )
        top = expanded[0]
        status = "FULL" if by_k[str(max(ks))]["full"] else (
            "FILE" if by_k[str(max(ks))]["file"] else "MISS"
        )
        print(
            f"  {q['id']:28s} {status:4s}  "
            f"groups={by_k[str(max(ks))]['sat']}/{by_k[str(max(ks))]['total']}  "
            f"top={top.score:.3f} {top.match.location.file[:50]}",
            flush=True,
        )

    n = len(questions)
    summary = {
        "file_hit": {str(k): file_hit[k] / n for k in ks},
        "page_group": {str(k): page_group[k] / n for k in ks},
        "full_group": {str(k): full_group[k] / n for k in ks},
        "n": n,
        "window": args.window,
    }

    print("\n=== Summary ===")
    print(f"  window=±{args.window}")
    for k in ks:
        print(
            f"  @{k:2d}  file-hit {file_hit[k]}/{n} ({100*file_hit[k]/n:.1f}%)  "
            f"group-recall {100*page_group[k]/n:.1f}%  "
            f"full {full_group[k]}/{n} ({100*full_group[k]/n:.1f}%)"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(
            rows=rows,
            ks=ks,
            summary=summary,
            model=model,
            n_corpus=len(vectors),
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
