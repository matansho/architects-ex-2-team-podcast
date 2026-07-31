#!/usr/bin/env python3
"""Build data/catalog.json from the index, corpus pages, and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.catalog import (
    DEFAULT_CATALOG_PATH,
    build_catalog,
    catalog_summary_for_prompt,
    list_products,
    save_catalog,
    search_catalog,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=Path("data/index"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--manifest", type=Path, default=Path("corpus/manifest.json"))
    ap.add_argument("--out", type=Path, default=DEFAULT_CATALOG_PATH)
    args = ap.parse_args()

    cat = build_catalog(
        index_dir=args.index, corpus_dir=args.corpus, manifest_path=args.manifest
    )
    save_catalog(cat, args.out)

    by_type: dict[str, int] = {}
    for e in cat.entries:
        by_type[e.doc_type] = by_type.get(e.doc_type, 0) + 1
    products = list_products(cat)
    print(f"wrote {args.out} · {cat.n_entries} entries · {len(products)} products")
    print("doc_types:", dict(sorted(by_type.items(), key=lambda x: -x[1])))
    print()
    print(catalog_summary_for_prompt(cat))
    print()
    print("smoke search 'סוויץ':")
    for e, score in search_catalog(cat, "סוויץ", limit=5):
        print(f"  {score:4.1f}  [{e.doc_type:14s}] {e.domain:12s}  {e.title[:50]}  ← {e.file}")
    print("smoke search 'משכנתא':")
    for e, score in search_catalog(cat, "משכנתא חיים", domain="mortgage", limit=5):
        print(f"  {score:4.1f}  [{e.doc_type:14s}] {e.title[:55]}")


if __name__ == "__main__":
    main()
