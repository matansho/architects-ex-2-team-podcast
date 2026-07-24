"""BM25 (sparse/keyword) indexes over corpus chunks and files."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.index_store import IndexedVector

# Hebrew + Latin word tokens (digits kept for policy numbers / amounts).
_TOKEN_RE = re.compile(r"[\w\u0590-\u05FF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class FileBM25Index:
    """BM25 over one bag-of-words document per file path."""

    files: list[str]
    bm25: BM25Okapi
    file_to_idxs: dict[str, list[int]]  # corpus vector indices per file

    def search_files(self, query: str, *, top_n: int = 20) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens or not self.files:
            return []
        scores = self.bm25.get_scores(tokens)
        n = min(top_n, len(self.files))
        if n <= 0:
            return []
        idx = sorted(range(len(self.files)), key=lambda i: scores[i], reverse=True)[:n]
        out = [(self.files[i], float(scores[i])) for i in idx if scores[i] > 0]
        return out


@dataclass
class ChunkBM25Index:
    """BM25 over individual chunks (same order as corpus vectors)."""

    bm25: BM25Okapi
    n: int

    def search(self, query: str, *, top_n: int = 50) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self.n == 0:
            return []
        scores = self.bm25.get_scores(tokens)
        n = min(top_n, self.n)
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def build_file_bm25(vectors: list[IndexedVector]) -> FileBM25Index:
    """Concatenate chunk texts per file and fit BM25Okapi."""
    texts: dict[str, list[str]] = defaultdict(list)
    file_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(vectors):
        f = v.location.file
        file_to_idxs[f].append(i)
        if v.text:
            texts[f].append(v.text)

    files = sorted(file_to_idxs.keys())
    corpus = [tokenize("\n".join(texts[f])) for f in files]
    bm25 = BM25Okapi(corpus if corpus else [[]])
    return FileBM25Index(files=files, bm25=bm25, file_to_idxs=dict(file_to_idxs))


def build_chunk_bm25(vectors: list[IndexedVector]) -> ChunkBM25Index:
    corpus = [tokenize(v.text) for v in vectors]
    bm25 = BM25Okapi(corpus if corpus else [[]])
    return ChunkBM25Index(bm25=bm25, n=len(vectors))
