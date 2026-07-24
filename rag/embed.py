"""Local embedding via sentence-transformers (multilingual E5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Strong default for Hebrew + English RAG; passage/query prefixes required.
DEFAULT_MODEL = "intfloat/multilingual-e5-base"


@dataclass
class Embedder:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    device: str | None = None
    normalize: bool = True

    _model: object = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

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
        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed retrieval documents. Shape (N, dim), L2-normalized by default."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payloads = [f"passage: {t}" if self._is_e5() else t for t in texts]
        return self._encode(payloads)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed search queries. Shape (N, dim), L2-normalized by default."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payloads = [f"query: {t}" if self._is_e5() else t for t in texts]
        return self._encode(payloads)

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vecs, dtype=np.float32)
