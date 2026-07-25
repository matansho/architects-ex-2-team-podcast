"""Cross-encoder reranking over dense candidate hits."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

# Multilingual retrieval reranker (100+ languages, incl. Hebrew).
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class Reranker:
    model_name: str = DEFAULT_RERANK_MODEL
    batch_size: int = 8
    device: str | None = None
    max_length: int = 512

    _model: object = None

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cannot allocate memory" in msg

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        # Hugging Face Hub may route large downloads through Xet; in some
        # environments this transport fails while regular HTTPS works.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

        device = self.device or "cpu"
        if device == "mps":
            print("  [rerank] MPS disabled by config; forcing CPU.", flush=True)
            device = "cpu"
            self.device = "cpu"
        kwargs: dict = {"device": device} if device else {}
        # max_length caps long insurance chunks for the cross-encoder
        try:
            print(
                f"  [rerank] loading CrossEncoder model={self.model_name} "
                f"device={device} batch={self.batch_size}",
                flush=True,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load reranker model from Hugging Face Hub. "
                "Set HF_HUB_DISABLE_XET=1 (or keep default), verify network, "
                "or use a pre-downloaded local reranker path via --rerank-model. "
                f"model={self.model_name}"
            ) from exc
        return self._model

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Return relevance scores, one per passage (higher = better)."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        model = self._load()
        pairs = [(query, p) for p in passages]
        batch = self.batch_size
        print(
            f"  [rerank] score start pairs={len(pairs)} initial_batch={batch}",
            flush=True,
        )
        while True:
            try:
                scores = model.predict(
                    pairs,
                    batch_size=batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self.batch_size = batch
                break
            except RuntimeError as exc:
                if not self._is_oom(exc):
                    raise
                if batch > 1:
                    next_batch = max(1, batch // 2)
                    if next_batch == batch:
                        next_batch = batch - 1
                    print(
                        "  [rerank] OOM on CPU; retrying with smaller "
                        f"batch ({batch} -> {next_batch}).",
                        flush=True,
                    )
                    batch = next_batch
                    continue
                print(
                    "  [rerank] OOM on CPU even at batch=1; cannot continue.",
                    flush=True,
                )
                raise

        print(f"  [rerank] scoring on CPU batch={self.batch_size}", flush=True)
        return np.asarray(scores, dtype=np.float32).reshape(-1)
