#!/usr/bin/env python3
"""Chunk a big PDF and render an HTML overlay (each vector a different color)."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk import chunk_blocks, chunk_border, chunk_color  # noqa: E402
from rag.parse import docling_to_blocks, build_tree  # noqa: E402


def convert_with_images(pdf: Path, scale: float = 1.25):
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
    """Convert Docling BOTTOMLEFT bbox to CSS % (top, left, width, height)."""
    # page_size: width/height in PDF points (same space as bbox)
    pw = float(page_size.width)
    ph = float(page_size.height)
    l, t, r, b = bbox  # t is top in BOTTOMLEFT (larger y), b is bottom
    # In BOTTOMLEFT: y increases upward; image y increases downward
    top_y = ph - t  # distance from top of page
    bottom_y = ph - b
    x = min(l, r)
    w = abs(r - l)
    y = min(top_y, bottom_y)
    h = abs(bottom_y - top_y)
    return (
        100.0 * y / ph,
        100.0 * x / pw,
        100.0 * w / pw,
        100.0 * h / ph,
    )


def build_html(
    *,
    pdf_name: str,
    chunks,
    block_to_chunk: dict[int, int],
    blocks,
    page_jpegs: dict[int, bytes],
    page_sizes: dict[int, object],
    stats: dict,
    max_pages: int | None,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Sidebar chunk list
    side_items = []
    for c in chunks:
        col = chunk_border(c.id)
        bg = chunk_color(c.id)
        pages = ",".join(str(p) for p in c.pages[:5]) + ("…" if len(c.pages) > 5 else "")
        side_items.append(
            f"""<button type="button" class="chunk-item" data-chunk="{c.id}"
                style="border-left:4px solid {col}; background:{bg}">
              <span class="cid">#{c.id}</span>
              <span class="lab">{html.escape(c.label)}</span>
              <span class="meta">{html.escape(c.kind)} · {c.tokens} tok · p{html.escape(pages)}</span>
            </button>"""
        )

    # Pages with overlays
    page_nos = sorted(page_jpegs)
    if max_pages:
        page_nos = page_nos[:max_pages]

    # Map chunk -> list of overlay rects per page
    overlays: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for b in blocks:
        cid = block_to_chunk.get(b.idx)
        if cid is None:
            continue
        for page, l, t, r, bot in b.bboxes:
            if page not in page_sizes:
                continue
            top, left, w, h = pdf_bbox_to_pct((l, t, r, bot), page_sizes[page])
            overlays[page].append((cid, top, left, w, h))

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

    kind_counts = defaultdict(int)
    for c in chunks:
        kind_counts[c.kind] += 1
    kind_s = ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items()))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk overlay — {html.escape(pdf_name)}</title>
  <style>
    :root {{ --bg:#f1f5f9; --panel:#fff; --text:#0f172a; --muted:#64748b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ background:#0f172a; color:#fff; padding:.9rem 1.1rem; }}
    header h1 {{ margin:0 0 .25rem; font-size:1.15rem; }}
    header p {{ margin:0; font-size:.82rem; opacity:.85; }}
    .layout {{ display:grid; grid-template-columns:300px 1fr; min-height:calc(100vh - 70px); }}
    aside {{ background:var(--panel); border-right:1px solid #e2e8f0; overflow:auto; max-height:calc(100vh - 70px);
             position:sticky; top:0; padding:.6rem; }}
    aside h2 {{ margin:.2rem .3rem .5rem; font-size:.85rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .chunk-item {{ display:block; width:100%; text-align:left; border:1px solid #e2e8f0; border-radius:8px;
                   padding:.4rem .5rem; margin:0 0 .35rem; cursor:pointer; font:inherit; }}
    .chunk-item:hover, .chunk-item.active {{ outline:2px solid #0f172a; }}
    .chunk-item .cid {{ font-weight:700; font-size:.75rem; margin-right:.35rem; }}
    .chunk-item .lab {{ font-size:.82rem; }}
    .chunk-item .meta {{ display:block; font-size:.7rem; color:var(--muted); margin-top:.15rem; }}
    main {{ padding:1rem 1.1rem 2rem; overflow:auto; }}
    .stats {{ background:#eff6ff; border-radius:10px; padding:.65rem .8rem; font-size:.84rem; margin-bottom:1rem; }}
    .page {{ margin-bottom:1.25rem; }}
    .page h3 {{ margin:0 0 .4rem; font-size:.95rem; }}
    .page-wrap {{ position:relative; display:inline-block; max-width:100%; box-shadow:0 1px 4px rgba(0,0,0,.12); background:#fff; }}
    .page-wrap img {{ display:block; max-width:100%; height:auto; }}
    .overlay {{ position:absolute; inset:0; pointer-events:none; }}
    .ov {{ position:absolute; border-radius:2px; pointer-events:auto; cursor:pointer; }}
    .ov.highlight {{ outline:3px solid #000 !important; z-index:5; }}
    .note {{ font-size:.8rem; color:var(--muted); margin:0 0 .75rem; }}
    @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ position:relative; max-height:240px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Vector chunk overlay</h1>
    <p>{html.escape(pdf_name)} · {len(chunks)} vectors · target ~{stats['target']} tok · {generated}</p>
  </header>
  <div class="layout">
    <aside>
      <h2>Chunks ({len(chunks)})</h2>
      {''.join(side_items)}
    </aside>
    <main>
      <div class="stats">
        <strong>Strategy:</strong> subclause/clause when numbered depth≥2; else pack section/orphan text to ~{stats['target']} tok (max {stats['max']}).<br/>
        <strong>Kinds:</strong> {html.escape(kind_s)} ·
        <strong>Token p50/mean:</strong> {stats['p50']} / {stats['mean']:.0f} ·
        <strong>Pages shown:</strong> {len(page_nos)}{(' of '+str(stats['n_pages'])) if max_pages else ''}
      </div>
      <p class="note">Colored boxes = Docling item bounding boxes assigned to each vector. Click a chunk in the sidebar to highlight it.</p>
      {''.join(page_sections)}
    </main>
  </div>
  <script>
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pdf",
        default="corpus/business/files/פוליסת-מכלול-לחבר-מושב-מהדורת-ינואר-2023.pdf",
    )
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=1.15)
    ap.add_argument("--max-pages", type=int, default=20, help="0 = all pages (large HTML)")
    ap.add_argument("--out", default="reports/chunk_overlay.html")
    args = ap.parse_args()

    pdf = ROOT / args.pdf
    print(f"Converting {pdf.relative_to(ROOT)} (page images scale={args.scale})…", flush=True)
    doc = convert_with_images(pdf, scale=args.scale)
    blocks = docling_to_blocks(doc)
    _tree = build_tree(blocks)
    print(f"  {len(blocks)} blocks, {len(doc.pages)} pages", flush=True)

    chunks = chunk_blocks(blocks, target_tokens=args.target, max_tokens=args.max_tokens)
    block_to_chunk = {}
    for c in chunks:
        for bi in c.block_indices:
            block_to_chunk[bi] = c.id

    toks = sorted(c.tokens for c in chunks)
    p50 = toks[len(toks) // 2] if toks else 0
    mean = sum(toks) / len(toks) if toks else 0
    print(f"  {len(chunks)} vectors · token p50={p50} mean={mean:.0f}", flush=True)

    page_jpegs = {}
    page_sizes = {}
    for pno, page in sorted(doc.pages.items(), key=lambda x: x[0]):
        if page.image is None:
            continue
        page_jpegs[int(pno)] = page_image_jpeg(page)
        page_sizes[int(pno)] = page.size

    max_pages = None if args.max_pages == 0 else args.max_pages
    html_out = build_html(
        pdf_name=pdf.name,
        chunks=chunks,
        block_to_chunk=block_to_chunk,
        blocks=blocks,
        page_jpegs=page_jpegs,
        page_sizes=page_sizes,
        stats={
            "target": args.target,
            "max": args.max_tokens,
            "p50": p50,
            "mean": mean,
            "n_pages": len(page_jpegs),
        },
        max_pages=max_pages,
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    meta = {
        "pdf": str(pdf.relative_to(ROOT)),
        "n_chunks": len(chunks),
        "token_p50": p50,
        "token_mean": round(mean, 1),
        "kinds": {k: sum(1 for c in chunks if c.kind == k) for k in {c.kind for c in chunks}},
        "chunks": [
            {
                "id": c.id,
                "kind": c.kind,
                "label": c.label,
                "tokens": c.tokens,
                "pages": c.pages,
                "section_ref": c.section_ref,
            }
            for c in chunks
        ],
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
