#!/usr/bin/env python3
"""Second-pass LLM table descriptions on an existing index (no Docling re-parse).

Uses compact table sketches (headers + sample rows). Updates embed_text, then
re-embeds only those vectors and rewrites data/index.

    set -a && source .env && set +a
    python scripts/llm_describe_tables.py
    python scripts/llm_describe_tables.py --limit 50 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rag.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from rag.index_store import (  # noqa: E402
    load_embeddings,
    load_vectors_meta,
    save_index,
)
from rag.table_describe import (  # noqa: E402
    describe_table_with_llm,
    extract_table_body,
    format_embed_text,
    format_payload,
    sketch_table,
)


def _cache_path(cache_dir: Path, vid: str) -> Path:
    safe = vid.replace("/", "__").replace(":", "_")
    return cache_dir / f"{safe}.json"


def _neighbor_text(vectors, idx: int, file: str, window: int = 2) -> tuple[str, str]:
    before: list[str] = []
    after: list[str] = []
    for j in range(idx - 1, max(-1, idx - 8), -1):
        if vectors[j].location.file != file:
            break
        if vectors[j].embed_text:
            continue  # skip other tables' payloads for context
        before.append(vectors[j].text[:400])
        if len(before) >= window:
            break
    for j in range(idx + 1, min(len(vectors), idx + 8)):
        if vectors[j].location.file != file:
            break
        if vectors[j].embed_text:
            continue
        after.append(vectors[j].text[:400])
        if len(after) >= window:
            break
    return "\n".join(reversed(before)), "\n".join(after)


def _describe_one(args) -> dict:
    (
        vid,
        payload,
        file,
        page,
        domain,
        before,
        after,
        model,
        cache_dir,
    ) = args
    cache_path = _cache_path(cache_dir, vid)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("embed_text") and cached.get("source") == "llm":
                return {
                    "id": vid,
                    "embed_text": cached["embed_text"],
                    "payload": cached.get("payload") or payload,
                    "cached": True,
                    "ok": True,
                }
        except Exception:
            pass

    body = extract_table_body(payload)
    desc = describe_table_with_llm(
        body,
        file=file,
        page=page,
        domain=domain,
        context_before=before,
        context_after=after,
        model=model,
        use_sketch=True,
    )
    embed = format_embed_text(desc, file=file, page=page)
    new_payload = format_payload(desc, body)
    # Heuristic fallback if LLM failed silently (no embed_passage and generic title)
    source = "llm" if (desc.embed_passage or "").strip() else "heuristic"
    cache_path.write_text(
        json.dumps(
            {
                "id": vid,
                "source": source,
                "description": desc.to_dict(),
                "sketch": sketch_table(body),
                "embed_text": embed,
                "payload": new_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "id": vid,
        "embed_text": embed,
        "payload": new_payload,
        "cached": False,
        "ok": True,
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-describe table chunks in existing index")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--cache", default="data/cache/table_descriptions")
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe only; do not rewrite index",
    )
    ap.add_argument(
        "--keep-payload",
        action="store_true",
        help="Only update embed_text (leave generator payload as-is)",
    )
    args = ap.parse_args()

    index_dir = ROOT / args.index
    cache_dir = ROOT / args.cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading index…", flush=True)
    vectors = load_vectors_meta(index_dir)
    matrix = load_embeddings(index_dir)
    assert len(vectors) == len(matrix), (len(vectors), matrix.shape)

    jobs = []
    for i, v in enumerate(vectors):
        if not v.embed_text:
            continue
        before, after = _neighbor_text(vectors, i, v.location.file)
        jobs.append(
            (
                v.id,
                v.text,
                v.location.file,
                v.location.page,
                v.location.domain,
                before,
                after,
                args.model,
                cache_dir,
            )
        )
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"Tables to describe: {len(jobs)} (workers={args.workers})", flush=True)

    results: dict[str, dict] = {}
    t0 = time.time()
    done = 0
    n_cached = 0
    n_llm = 0
    n_heur = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_describe_one, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            vid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  FAIL {vid}: {e!r}", flush=True)
                continue
            results[r["id"]] = r
            done += 1
            if r.get("cached"):
                n_cached += 1
            elif r.get("source") == "heuristic":
                n_heur += 1
            else:
                n_llm += 1
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - t0)
                print(
                    f"  [{done}/{len(jobs)}] "
                    f"llm={n_llm} cached={n_cached} heur={n_heur} "
                    f"({rate:.2f}/s)",
                    flush=True,
                )

    if args.dry_run:
        print("Dry run — index not updated", flush=True)
        return

    # Apply descriptions
    changed_idx: list[int] = []
    id_to_i = {v.id: i for i, v in enumerate(vectors)}
    for vid, r in results.items():
        i = id_to_i[vid]
        vectors[i].embed_text = r["embed_text"]
        if not args.keep_payload and r.get("payload"):
            vectors[i].text = r["payload"]
        changed_idx.append(i)

    print(f"Re-embedding {len(changed_idx)} table vectors…", flush=True)
    embedder = Embedder(model_name=DEFAULT_MODEL, batch_size=32)
    texts = [
        (vectors[i].embed_text or vectors[i].text or "") for i in changed_idx
    ]
    new_vecs = embedder.embed_passages(texts)
    for row, i in enumerate(changed_idx):
        matrix[i] = new_vecs[row]
        vectors[i].embedding = matrix[i].tolist()

    # Fill embeddings for unchanged rows (save_index expects them on objects)
    for i, v in enumerate(vectors):
        if not v.embedding:
            v.embedding = matrix[i].tolist()

    cfg_path = index_dir / "config.json"
    extra = {}
    if cfg_path.exists():
        extra = json.loads(cfg_path.read_text(encoding="utf-8"))
    extra.update(
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "table_descriptions": sum(1 for v in vectors if v.embed_text),
            "table_descriptions_llm_pass": {
                "model": args.model,
                "n_updated": len(changed_idx),
                "n_llm": n_llm,
                "n_cached": n_cached,
                "n_heuristic_fallback": n_heur,
            },
        }
    )
    # drop keys that save_index sets
    for k in ("model", "dim", "n_vectors", "n_txt", "n_pdf", "normalize", "schema", "files"):
        extra.pop(k, None)

    save_index(
        index_dir,
        vectors,
        model=DEFAULT_MODEL,
        dim=int(matrix.shape[1]),
        extra_config=extra,
    )
    print(
        f"Updated {index_dir} · {len(changed_idx)} tables · "
        f"elapsed {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
