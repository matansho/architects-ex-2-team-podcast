"""Measure the real size of the stage-2 shortlist (MiniLM top-40 ∪ dense top-40)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import Reranker
from rag.retrieve import idxs_excluding_faqs, search

QUESTIONS = [
    "כמה זמן יש לי להגיש תביעה לתגמולי ביטוח אחרי שקרה הנזק?",
    "מה מספר הטלפון של המוקד לתביעות ביטוח?",
    "קניתי חבילת esim מהספק ולא קיבלתי את ההנחה, למה?",
    "מה ההשתתפות העצמית בביטוח דירה במקרה של נזקי מים?",
    "האם ביטוח נסיעות מכסה ספורט אתגרי?",
    "מהי תקופת האכשרה בביטוח שיניים?",
    "האם צריך ביטוח לניסוי קליני שאושר בוועדת הלסינקי?",
    "מה קורה לפוליסה אם לא שילמתי פרמיה?",
]
CANDIDATE_N, REFINE_N = 100, 40

vectors = load_vectors_meta(Path("data/index"))
emb = load_embeddings(Path("data/index"))
non_faq = idxs_excluding_faqs(vectors)
embedder = Embedder(model_name="intfloat/multilingual-e5-base")
mini = Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

out = []
for q in QUESTIONS:
    q_emb = embedder.embed_queries([q])[0]
    cands = search(q_emb, emb, vectors, top_k=CANDIDATE_N, candidate_idxs=non_faq)
    scores = mini.score(q, [(h.vector.embed_text or h.vector.text or "") for h in cands])

    keep = min(REFINE_N, len(cands))
    mini_top = [int(j) for j in np.argsort(-scores)[:keep]]
    dense_top = list(range(keep))
    union = set(mini_top) | set(dense_top)
    rescued = [j for j in dense_top if j not in set(mini_top)]

    out.append({
        "q": q,
        "candidates": len(cands),
        "mini_top": keep,
        "union": len(union),
        "rescued_by_dense": len(rescued),
        "worst_mini_rank_rescued": (
            max(int(np.where(np.argsort(-scores) == j)[0][0]) + 1 for j in rescued) if rescued else None
        ),
    })
    print(f"union={len(union):>3}  rescued_by_dense={len(rescued):>3}   {q[:46]}", flush=True)

u = [r["union"] for r in out]
r = [r["rescued_by_dense"] for r in out]
print(f"\nshortlist size: min={min(u)} max={max(u)} mean={sum(u)/len(u):.1f}")
print(f"rescued by dense: min={min(r)} max={max(r)} mean={sum(r)/len(r):.1f}")
Path("reports/stage2/shortlist_demo_data.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("DEMO_DONE")
