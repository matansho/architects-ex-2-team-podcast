#!/usr/bin/env python3
"""Quality gate for LLM table descriptions.

Exit 0 if cache looks healthy; exit 1 if we should STOP the re-describe.

    python scripts/check_table_desc_quality.py
    python scripts/check_table_desc_quality.py --cache artifacts/table_descriptions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.table_cache import DEFAULT_TABLE_DESC_DIR  # noqa: E402

# Fail the batch if these are breached (pilot + full run).
MAX_HEURISTIC_RATE = 0.05
MAX_EMPTY_PASSAGE_RATE = 0.02
MAX_VAGUE_OPENER_RATE = 0.25
MIN_MEDIAN_EMBED_LEN = 120
# Keyword probes: if sketch has term, embed_text should usually too.
PROBE_TERMS = ("נשך", "רחפן")
MIN_PROBE_HIT_RATE = 0.5  # of sketches that contain the term


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_TABLE_DESC_DIR))
    ap.add_argument("--sample", type=int, default=12, help="print N example passages")
    args = ap.parse_args()

    cache = ROOT / args.cache
    files = sorted(cache.glob("*.json"))
    if not files:
        print(f"FAIL: no descriptions in {cache}")
        return 1

    n = len(files)
    n_heur = 0
    n_empty = 0
    n_vague = 0
    lens: list[int] = []
    probe_denom = {t: 0 for t in PROBE_TERMS}
    probe_hit = {t: 0 for t in PROBE_TERMS}
    examples: list[dict] = []

    for path in files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FAIL: unreadable JSON {path.name}: {e}")
            return 1
        src = d.get("source") or "?"
        embed = (d.get("embed_text") or "").strip()
        passage = ((d.get("description") or {}).get("embed_passage") or "").strip()
        sketch = d.get("sketch") or ""
        if src == "heuristic":
            n_heur += 1
        if not passage:
            n_empty += 1
        if embed.startswith("מגוון") or "מגוון רחב" in embed[:80]:
            n_vague += 1
        lens.append(len(embed))
        for t in PROBE_TERMS:
            if t in sketch:
                probe_denom[t] += 1
                if t in embed:
                    probe_hit[t] += 1
        if len(examples) < args.sample:
            examples.append(
                {
                    "id": d.get("id"),
                    "source": src,
                    "embed": embed[:280],
                }
            )

    lens_sorted = sorted(lens)
    median = lens_sorted[len(lens_sorted) // 2]
    heur_rate = n_heur / n
    empty_rate = n_empty / n
    vague_rate = n_vague / n

    print(f"cache={cache}  n={n}")
    print(f"  heuristic: {n_heur} ({heur_rate:.1%})  max {MAX_HEURISTIC_RATE:.0%}")
    print(f"  empty passage: {n_empty} ({empty_rate:.1%})  max {MAX_EMPTY_PASSAGE_RATE:.0%}")
    print(f"  vague opener: {n_vague} ({vague_rate:.1%})  max {MAX_VAGUE_OPENER_RATE:.0%}")
    print(f"  embed_text median len: {median}  min {MIN_MEDIAN_EMBED_LEN}")
    for t in PROBE_TERMS:
        den = probe_denom[t]
        if den:
            rate = probe_hit[t] / den
            print(
                f"  probe '{t}': {probe_hit[t]}/{den} in embed ({rate:.0%})  "
                f"min {MIN_PROBE_HIT_RATE:.0%}"
            )
        else:
            print(f"  probe '{t}': (not in any sketch yet)")

    print("\n--- samples ---")
    for ex in examples:
        print(f"[{ex['source']}] {ex['id']}")
        print(f"  {ex['embed']}")
        print()

    failures: list[str] = []
    if heur_rate > MAX_HEURISTIC_RATE:
        failures.append(f"heuristic rate {heur_rate:.1%} > {MAX_HEURISTIC_RATE:.0%}")
    if empty_rate > MAX_EMPTY_PASSAGE_RATE:
        failures.append(f"empty passage {empty_rate:.1%} > {MAX_EMPTY_PASSAGE_RATE:.0%}")
    if vague_rate > MAX_VAGUE_OPENER_RATE:
        failures.append(f"vague opener {vague_rate:.1%} > {MAX_VAGUE_OPENER_RATE:.0%}")
    if median < MIN_MEDIAN_EMBED_LEN:
        failures.append(f"median embed len {median} < {MIN_MEDIAN_EMBED_LEN}")
    for t in PROBE_TERMS:
        den = probe_denom[t]
        if den >= 1:
            rate = probe_hit[t] / den
            if rate < MIN_PROBE_HIT_RATE:
                failures.append(
                    f"probe {t} hit rate {rate:.0%} < {MIN_PROBE_HIT_RATE:.0%} "
                    f"({probe_hit[t]}/{den})"
                )

    if failures:
        print("STOP — quality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — quality gate OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
