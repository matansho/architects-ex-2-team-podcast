"""On-disk table description cache (git-tracked under artifacts/).

Each file is one vector id, keyed by a filesystem-safe name:

    artifacts/table_descriptions/<safe_id>.json

    {
      "id": "pdf:health/files/….pdf#42",
      "source": "llm" | "heuristic",
      "description": {…},
      "sketch": "…",
      "embed_text": "…",          # what E5 / rerank use
      "payload": "…"              # what the generator sees
    }

Teammates can rebuild the index without calling Gemma by loading this store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol

# Repo-relative default — committed, not under gitignored data/
DEFAULT_TABLE_DESC_DIR = Path("artifacts/table_descriptions")


class _HasTableFields(Protocol):
    id: str
    text: str
    embed_text: str | None
    tokens: int


def cache_path_for_id(cache_dir: Path, vid: str) -> Path:
    safe = vid.replace("/", "__").replace(":", "_")
    return cache_dir / f"{safe}.json"


def load_cached_description(cache_dir: Path, vid: str) -> dict[str, Any] | None:
    path = cache_path_for_id(cache_dir, vid)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not (data.get("embed_text") or "").strip():
        return None
    return data


def save_cached_description(cache_dir: Path, record: dict[str, Any]) -> Path:
    """Write one description record. Requires record['id'] and embed_text."""
    vid = record["id"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_id(cache_dir, vid)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def apply_cached_table_descriptions(
    chunks: Iterable[_HasTableFields],
    cache_dir: Path | None,
    *,
    update_payload: bool = True,
) -> int:
    """Override chunk embed_text (+ optional payload) from the shipped cache.

    Returns number of chunks updated. Missing cache entries are left unchanged.
    """
    if cache_dir is None or not cache_dir.is_dir():
        return 0

    from rag.chunk_stats import count_tokens

    n = 0
    for chunk in chunks:
        cached = load_cached_description(cache_dir, chunk.id)
        if not cached:
            continue
        chunk.embed_text = cached["embed_text"]
        if update_payload and cached.get("payload"):
            chunk.text = cached["payload"]
            chunk.tokens = count_tokens(chunk.text)
        n += 1
    return n
