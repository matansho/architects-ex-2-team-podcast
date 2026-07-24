"""On-disk corpus index: each vector holds text + embedding + location."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rag.corpus_chunks import CorpusChunk


@dataclass
class Location:
    """Where this vector came from (citation + routing metadata)."""

    file: str  # relative to corpus/, e.g. car/pages/comprehensive.txt
    page: int | None  # primary page for citations; null for TXT
    pages: list[int] = field(default_factory=list)
    domain: str = ""
    source_type: str = ""  # pdf | txt
    section_ref: str | None = None
    kind: str = ""
    label: str = ""


@dataclass
class IndexedVector:
    """One retrieval unit: content, vector, and location together.

    text:       passed to the generator (table payload = full CSV + abstract).
    embed_text: optional; stored for debugging. Embedding was computed from
                embed_text if set, else text (see CorpusChunk.text_for_embedding).
    """

    id: str
    text: str
    tokens: int
    embedding: list[float]
    location: Location
    embed_text: str | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "text": self.text,
            "tokens": self.tokens,
            "embedding": self.embedding,
            "location": asdict(self.location),
        }
        if self.embed_text:
            d["embed_text"] = self.embed_text
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IndexedVector:
        loc = d["location"]
        return cls(
            id=d["id"],
            text=d["text"],
            tokens=int(d["tokens"]),
            embedding=[float(x) for x in d.get("embedding") or []],
            location=Location(**{k: loc[k] for k in loc if k in Location.__dataclass_fields__}),
            embed_text=d.get("embed_text"),
        )


def chunk_to_indexed(chunk: CorpusChunk, embedding: np.ndarray) -> IndexedVector:
    return IndexedVector(
        id=chunk.id,
        text=chunk.text,
        tokens=chunk.tokens,
        embedding=np.asarray(embedding, dtype=np.float32).reshape(-1).tolist(),
        location=Location(
            file=chunk.file,
            page=chunk.page,
            pages=list(chunk.pages),
            domain=chunk.domain,
            source_type=chunk.source_type,
            section_ref=chunk.section_ref,
            kind=chunk.kind,
            label=chunk.label,
        ),
        embed_text=chunk.embed_text,
    )


def save_index(
    out_dir: Path,
    vectors: list[IndexedVector],
    *,
    model: str,
    dim: int,
    extra_config: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vectors.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for v in vectors:
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    # Dense matrix for fast search (same order as vectors.jsonl)
    matrix = (
        np.asarray([v.embedding for v in vectors], dtype=np.float32)
        if vectors
        else np.zeros((0, dim), dtype=np.float32)
    )
    np.save(out_dir / "embeddings.npy", matrix)

    config = {
        "model": model,
        "dim": dim,
        "n_vectors": len(vectors),
        "n_txt": sum(1 for v in vectors if v.location.source_type == "txt"),
        "n_pdf": sum(1 for v in vectors if v.location.source_type == "pdf"),
        "normalize": True,
        "schema": {
            "record": ["id", "text", "tokens", "embedding", "location"],
            "location": [
                "file",
                "page",
                "pages",
                "domain",
                "source_type",
                "section_ref",
                "kind",
                "label",
            ],
        },
        "files": {
            "vectors": path.name,
            "embeddings": "embeddings.npy",
        },
    }
    if extra_config:
        config.update(extra_config)
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_vectors(index_dir: Path) -> list[IndexedVector]:
    path = index_dir / "vectors.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    out: list[IndexedVector] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(IndexedVector.from_dict(json.loads(line)))
    return out


def load_vectors_meta(index_dir: Path) -> list[IndexedVector]:
    """Like load_vectors but drops embedding payloads (use with embeddings.npy)."""
    path = index_dir / "vectors.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    out: list[IndexedVector] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["embedding"] = []
            out.append(IndexedVector.from_dict(d))
    return out


def load_embeddings(index_dir: Path) -> np.ndarray:
    path = index_dir / "embeddings.npy"
    if path.exists():
        return np.load(path)
    vectors = load_vectors(index_dir)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray([v.embedding for v in vectors], dtype=np.float32)
