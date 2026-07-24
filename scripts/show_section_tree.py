#!/usr/bin/env python3
"""Print section hierarchy rebuilt from Docling + section-number parsing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.parse import format_tree, parse_pdf, section_depth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pdf",
        nargs="?",
        default="corpus/health/files/כתב-שירות-שירותי-רפואה-משלימה-אלטרנטיבית.pdf",
    )
    ap.add_argument("--flat", action="store_true", help="Also print flat block list")
    ap.add_argument("--max-depth", type=int, default=0, help="Truncate tree display (0=all)")
    args = ap.parse_args()

    path = ROOT / args.pdf
    print(f"Parsing {path.relative_to(ROOT)} …", flush=True)
    blocks, tree = parse_pdf(path)

    numbered = [b for b in blocks if b.section_ref and not b.inherited]
    print(f"\n{len(blocks)} blocks · {len(numbered)} with own section_ref\n")
    print("=== Hierarchy (section numbers, not Docling level) ===\n")
    print(format_tree(tree))

    # Spotlight the bug case
    print("\n=== Spotlight: §2.3 vs §3 (Docling put both at level 2) ===\n")
    for b in blocks:
        if b.section_ref in ("2.2", "2.3", "3", "3.1", "3.1.1", "5.3") and not b.inherited:
            print(
                f"  idx={b.idx:3d}  Docling L{b.docling_level}  "
                f"→ semantic depth {section_depth(b.section_ref)}  "
                f"§{b.section_ref}  p{b.page}  {b.kind}"
            )
            print(f"           {b.text[:90]!r}")

    if args.flat:
        print("\n=== Flat blocks ===\n")
        for b in blocks:
            flag = " (inherited)" if b.inherited else ""
            ref = b.section_ref or "-"
            print(
                f"{b.idx:3d} L{b.docling_level} §{ref:8s} d={b.section_depth} "
                f"p{b.page} {b.kind:18s}{flag}  {b.text[:60]!r}"
            )


if __name__ == "__main__":
    main()
