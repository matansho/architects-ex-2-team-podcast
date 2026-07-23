#!/usr/bin/env python3
"""Chunk + embed the corpus into structured vectors (text + embedding + location).

    python scripts/embed_corpus.py --txt-only
    python scripts/embed_corpus.py --pdf-limit 3

Each line in data/index/vectors.jsonl:

    {
      "id": "txt:car/pages/comprehensive.txt#0",
      "text": "...",
      "tokens": 485,
      "embedding": [0.01, ...],
      "location": {
        "file": "car/pages/comprehensive.txt",
        "page": null,
        "pages": [],
        "domain": "car",
        "source_type": "txt",
        "section_ref": null,
        "kind": "txt_pack",
        "label": "..."
      }
    }
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.corpus_chunks import collect_pdf_chunks, collect_txt_chunks  # noqa: E402
from rag.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from rag.index_store import chunk_to_indexed, save_index  # noqa: E402
from rag.table_cache import (  # noqa: E402
    DEFAULT_TABLE_DESC_DIR,
    apply_cached_table_descriptions,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed corpus into structured vectors")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--txt-only", action="store_true", help="Skip PDFs")
    ap.add_argument("--pdf-only", action="store_true", help="Skip TXT")
    ap.add_argument("--pdf-limit", type=int, default=None, help="Max PDFs to parse")
    ap.add_argument(
        "--pdf-cache",
        default="data/cache/pdf_chunks",
        help="Per-PDF chunk cache (resume-friendly). Empty string disables.",
    )
    ap.add_argument(
        "--keep-low-quality-txt",
        action="store_true",
        help="Include IE/aspx thin TXT pages",
    )
    ap.add_argument(
        "--llm-tables",
        action="store_true",
        help="Use LLM to write table descriptions during parse (slow; prefer shipped cache)",
    )
    ap.add_argument(
        "--no-table-descriptions",
        action="store_true",
        help="Skip heuristic table description enrichment during parse",
    )
    ap.add_argument(
        "--table-desc-cache",
        default=str(DEFAULT_TABLE_DESC_DIR),
        help=(
            "Apply shipped LLM table descriptions from this dir after chunking "
            f"(default: {DEFAULT_TABLE_DESC_DIR}). Empty string disables."
        ),
    )
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    out_dir = ROOT / args.out
    cache_dir = (ROOT / args.pdf_cache) if args.pdf_cache else None
    table_desc_dir = (
        (ROOT / args.table_desc_cache) if args.table_desc_cache else None
    )

    chunks = []
    if not args.pdf_only:
        print("Collecting TXT chunks…", flush=True)
        chunks.extend(
            collect_txt_chunks(
                corpus,
                target_tokens=args.target,
                max_tokens=args.max_tokens,
                skip_low_quality=not args.keep_low_quality_txt,
            )
        )
        print(f"  {len(chunks)} TXT chunks", flush=True)

    if not args.txt_only:
        print("Collecting PDF chunks (Docling)…", flush=True)
        if cache_dir is not None:
            print(f"  cache: {cache_dir}", flush=True)
        n_before = len(chunks)
        chunks.extend(
            collect_pdf_chunks(
                corpus,
                target_tokens=args.target,
                max_tokens=args.max_tokens,
                limit=args.pdf_limit,
                cache_dir=cache_dir,
                describe_tables=not args.no_table_descriptions,
                use_llm_tables=args.llm_tables,
            )
        )
        print(f"  {len(chunks) - n_before} PDF chunks", flush=True)

    if not chunks:
        raise SystemExit("No chunks collected")

    if table_desc_dir is not None:
        n_applied = apply_cached_table_descriptions(chunks, table_desc_dir)
        print(
            f"Applied {n_applied} table descriptions from {table_desc_dir}",
            flush=True,
        )

    print(f"Embedding {len(chunks)} chunks with {args.model}…", flush=True)
    embedder = Embedder(model_name=args.model, batch_size=args.batch_size)
    matrix = embedder.embed_passages([c.text_for_embedding() for c in chunks])
    assert matrix.shape == (len(chunks), embedder.dim), matrix.shape

    vectors = [chunk_to_indexed(c, matrix[i]) for i, c in enumerate(chunks)]
    n_table_desc = sum(1 for c in chunks if c.embed_text)
    save_index(
        out_dir,
        vectors,
        model=args.model,
        dim=embedder.dim,
        extra_config={
            "created": datetime.now(timezone.utc).isoformat(),
            "target_tokens": args.target,
            "max_tokens": args.max_tokens,
            "table_descriptions": n_table_desc,
            "table_desc_cache": str(table_desc_dir) if table_desc_dir else None,
        },
    )

    # Remove legacy split files if present
    for legacy in ("chunks.jsonl",):
        p = out_dir / legacy
        if p.exists():
            p.unlink()

    size_mb = (out_dir / "vectors.jsonl").stat().st_size / 1e6
    print(
        f"Wrote {out_dir}/vectors.jsonl ({size_mb:.1f} MB) · "
        f"{len(vectors)} vectors · dim={embedder.dim}"
    )


if __name__ == "__main__":
    main()
