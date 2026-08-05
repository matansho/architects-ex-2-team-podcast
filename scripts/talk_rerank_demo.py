"""Throwaway: capture a real dense -> MiniLM -> BGE reordering for the talk deck."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import Reranker
from rag.retrieve import idxs_excluding_faqs, search

QUESTION = "כמה זמן יש לי להגיש תביעה לתגמולי ביטוח אחרי שקרה הנזק?"

vectors = load_vectors_meta(Path("data/index"))
emb = load_embeddings(Path("data/index"))
non_faq = idxs_excluding_faqs(vectors)
print(f"index: {len(vectors)} chunks, {len(non_faq)} after FAQ filter", flush=True)

q_emb = Embedder(model_name="intfloat/multilingual-e5-base").embed_queries([QUESTION])[0]
dense = search(q_emb, emb, vectors, top_k=40, candidate_idxs=non_faq)
print(f"dense candidates: {len(dense)}", flush=True)

passages = [h.vector.text or "" for h in dense]
mini_scores = Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").score(QUESTION, passages)
print("minilm done", flush=True)
bge_scores = Reranker(model_name="BAAI/bge-reranker-v2-m3").score(QUESTION, passages)
print("bge done", flush=True)

rows = []
for i, h in enumerate(dense):
    loc = h.vector.location
    rows.append({
        "id": h.vector.id,
        "file": (loc.file or "").split("/")[-1],
        "page": loc.page,
        "dense_rank": i + 1,
        "dense": round(float(h.score), 4),
        "mini": round(float(mini_scores[i]), 4),
        "bge": round(float(bge_scores[i]), 4),
        "snippet": (h.vector.text or "").replace("\n", " ")[:160],
    })

for key in ("mini", "bge"):
    for rank, r in enumerate(sorted(rows, key=lambda r: -r[key]), 1):
        r[f"{key}_rank"] = rank

Path("/tmp/talk_rerank_demo.json").write_text(
    json.dumps({"question": QUESTION, "rows": rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nQUESTION:", QUESTION)
print(f"{'dense':>5} {'mini':>5} {'bge':>4}  file / page")
for r in sorted(rows, key=lambda r: r["bge_rank"])[:12]:
    print(f"{r['dense_rank']:>5} {r['mini_rank']:>5} {r['bge_rank']:>4}  {r['file'][:54]} p{r['page']}")
print("DEMO_DONE")
