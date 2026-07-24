"""Cross-encoder reranking over dense candidate hits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Multilingual retrieval reranker (100+ languages, incl. Hebrew).
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class Reranker:
    model_name: str = DEFAULT_RERANK_MODEL
    batch_size: int = 16
    device: str | None = None
    max_length: int = 512

    _model: object = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        device = self.device
        if device is None:
            try:
                import torch

                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = None
        kwargs: dict = {"device": device} if device else {}
        # max_length caps long insurance chunks for the cross-encoder
        self._model = CrossEncoder(self.model_name, max_length=self.max_length, **kwargs)
        return self._model

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Return relevance scores, one per passage (higher = better)."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        model = self._load()
        pairs = [(query, p) for p in passages]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
