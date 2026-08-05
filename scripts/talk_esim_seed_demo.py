"""Reproduce the seed passages the agent saw for the eSIM question (route = business)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import Reranker
from rag.retrieve import idxs_excluding_faqs, search_reranked

QUESTION = 'קניתי חבילת esim מהספק ולא קיבלתי את ההנחה, למה?'
ROUTED = {"business"}  # what the router actually returned for this question

vectors = load_vectors_meta(Path("data/index"))
emb = load_embeddings(Path("data/index"))
non_faq = idxs_excluding_faqs(vectors)
route_idxs = [i for i in non_faq if (vectors[i].location.domain or "") in ROUTED]
print(f"routed pool (business): {len(route_idxs)} of {len(non_faq)}", flush=True)

q_emb = Embedder(model_name="intfloat/multilingual-e5-base").embed_queries([QUESTION])[0]
res = search_reranked(
    QUESTION, q_emb, emb, vectors,
    Reranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
    refine_reranker=Reranker(model_name="BAAI/bge-reranker-v2-m3"),
    candidate_n=100, top_k=25, window=2,
    candidate_idxs=non_faq, route_idxs=route_idxs,
    route_n=80, route_global_n=20, refine_n=40,
)

print(f"\nTop 5 — this is the 'brief' the agent read before its first move:\n")
for h in res.expanded[:5]:
    loc = h.match.location
    print(f"[{h.rank}] {loc.domain:10} {(loc.file or '').split('/')[-1][:58]} p{loc.page}")
    print(f"     {(h.match.text or '')[:150].replace(chr(10), ' ')}")

print("\nDomain spread of all 25:")
from collections import Counter
print(" ", dict(Counter((h.match.location.domain or '?') for h in res.expanded)))

joined = " ".join((h.match.text or "") for h in res.expanded)
for term in ("esim", "eSIM", "גלוקל", "glocal", "נסיעות לחו", "הטבה"):
    print(f"  {term!r:12} present in the 25 seed passages: {term.lower() in joined.lower()}")
print("SEED_DONE")
