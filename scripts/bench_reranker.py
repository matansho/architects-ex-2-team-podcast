"""Benchmark cross-encoder reranking latency against top-20 agreement.

Cross-encoder scoring is ~70% of end-to-end RAG latency (11.5s of 16.5s p50), so
this sweeps the knobs that affect it — device, batch size, sequence cap, and
candidate count — and reports both the speedup and how much the top-20 changes
versus the current production config. A config is only useful if it keeps the
same passages in the top 20.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np

from rag.embed import DEFAULT_MODEL, Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import Reranker
from rag.retrieve import build_id_index, idxs_excluding_faqs, search

REF = {"device": "mps", "batch_size": 16, "max_length": 512, "candidate_n": 100}

CONFIGS = [
    REF,
    {"device": "mps", "batch_size": 32, "max_length": 512, "candidate_n": 100},
    {"device": "mps", "batch_size": 64, "max_length": 512, "candidate_n": 100},
    {"device": "mps", "batch_size": 32, "max_length": 384, "candidate_n": 100},
    {"device": "mps", "batch_size": 32, "max_length": 256, "candidate_n": 100},
    {"device": "mps", "batch_size": 32, "max_length": 192, "candidate_n": 100},
    {"device": "mps", "batch_size": 32, "max_length": 512, "candidate_n": 60},
    {"device": "mps", "batch_size": 32, "max_length": 512, "candidate_n": 40},
    {"device": "mps", "batch_size": 32, "max_length": 256, "candidate_n": 60},
    {"device": "cpu", "batch_size": 32, "max_length": 512, "candidate_n": 100},
    {"device": "cpu", "batch_size": 32, "max_length": 256, "candidate_n": 100},
]


def label(cfg: dict) -> str:
    return (
        f"{cfg['device']}/bs{cfg['batch_size']}/len{cfg['max_length']}"
        f"/n{cfg['candidate_n']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--n-queries", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="reports/stage2/rerank_bench.json")
    args = ap.parse_args()

    try:
        import torch

        print(f"torch {torch.__version__} · mps={torch.backends.mps.is_available()}")
    except Exception as e:  # noqa: BLE001
        print(f"torch unavailable: {e}")

    index_dir = Path(args.index)
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    vectors = load_vectors_meta(index_dir)
    corpus_emb = load_embeddings(index_dir)
    build_id_index(vectors)
    non_faq = idxs_excluding_faqs(vectors)
    print(f"index: {len(vectors)} vectors · {len(non_faq)} non-FAQ")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions["questions"]
    questions = questions[: args.n_queries]
    texts = [q["question"] for q in questions]
    print(f"embedding {len(texts)} queries…")
    embedder = Embedder(model_name=config.get("model", DEFAULT_MODEL))
    q_embs = embedder.embed_queries(texts)

    # Fixed candidate pools per query: reranker configs must see identical input.
    pools: list[list[str]] = []
    pool_ids: list[list[str]] = []
    for qe in q_embs:
        hits = search(np.asarray(qe), corpus_emb, vectors, top_k=max(c["candidate_n"] for c in CONFIGS), candidate_idxs=non_faq)
        pools.append([(h.vector.embed_text or h.vector.text or "") for h in hits])
        pool_ids.append([h.vector.id for h in hits])

    ref_top: list[list[str]] = []
    results = []

    for cfg in CONFIGS:
        rr = Reranker(
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            max_length=cfg["max_length"],
        )
        try:
            rr._load()
        except Exception as e:  # noqa: BLE001
            print(f"{label(cfg):<32} SKIP ({type(e).__name__}: {str(e)[:60]})")
            continue

        n = cfg["candidate_n"]
        # Warm up so kernel compilation is not charged to the first timing.
        rr.score(texts[0], pools[0][:n])

        per_query_ms: list[float] = []
        tops: list[list[str]] = []
        for _ in range(args.repeats):
            for qi, (q, pool) in enumerate(zip(texts, pools)):
                sub = pool[:n]
                t0 = time.perf_counter()
                scores = rr.score(q, sub)
                per_query_ms.append((time.perf_counter() - t0) * 1000)
                if _ == 0:
                    order = np.argsort(-scores)[: args.top_k]
                    tops.append([pool_ids[qi][int(j)] for j in order])

        if not ref_top:
            ref_top = tops

        overlaps = [
            len(set(a) & set(b)) / max(1, len(set(b)))
            for a, b in zip(tops, ref_top)
        ]
        p50 = st.median(per_query_ms)
        row = {
            **cfg,
            "p50_ms": round(p50, 1),
            "max_ms": round(max(per_query_ms), 1),
            "top20_overlap": round(float(np.mean(overlaps)), 3),
        }
        results.append(row)
        speed = results[0]["p50_ms"] / p50 if results else 1.0
        print(
            f"{label(cfg):<32} p50={p50 / 1000:6.2f}s  "
            f"speedup={speed:4.2f}x  top{args.top_k}_overlap={row['top20_overlap']:.0%}"
        )

        del rr

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
