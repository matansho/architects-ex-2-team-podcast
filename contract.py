"""
Exercise 2 API contract -- your system MUST expose exactly this interface.

The blind evaluation calls POST /ask on your endpoint with an AskRequest and
expects an AskResponse. Fields you don't fill (e.g. cost_usd) simply score
worse on the efficiency component; fields with wrong types fail validation.

Run:

    uvicorn contract:app --port 8000
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "האם הביטוח מכסה נזק מפגיעת ברק?"}'

Do not change the request/response models.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.agent import run_agent
from rag.agent_tools import RetrievalBundle
from rag.catalog import DEFAULT_CATALOG_PATH, load_catalog
from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import DEFAULT_RERANK_MODEL, Reranker
from rag.retrieve import build_id_index, idxs_excluding_faqs
from rag_runner import resolve_model


class AskRequest(BaseModel):
    question: str = Field(..., description="Customer question, usually Hebrew")
    session_id: Optional[str] = Field(None, description="For multi-turn context (optional)")


class Citation(BaseModel):
    file: str = Field(..., description="Source document path or URL")
    page: Optional[int] = Field(None, description="1-based page number for PDFs")
    quote: Optional[str] = Field(None, description="The supporting passage (optional but persuasive)")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The answer, in the language of the question")
    citations: List[Citation] = Field(default_factory=list)
    domain: Optional[str] = Field(None, description="Routed insurance domain, e.g. 'travel'")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = Field(None, description="Estimated $ cost of answering this question")


app = FastAPI(title="APEX Exercise 2 -- Harel Support Agent")

_BUNDLE: RetrievalBundle | None = None
_LLM_MODEL: str | None = None
_LLM_KWARGS: dict = {}
_LLM_EXTRA: dict = {}


def _load_bundle() -> RetrievalBundle:
    global _BUNDLE, _LLM_MODEL, _LLM_KWARGS, _LLM_EXTRA
    if _BUNDLE is not None:
        return _BUNDLE

    index_dir = Path(os.environ.get("RAG_INDEX", "data/index"))
    catalog_path = Path(os.environ.get("RAG_CATALOG", str(DEFAULT_CATALOG_PATH)))
    model_name = os.environ.get("RAG_MODEL", "moonshotai/Kimi-K3")
    rerank_model = os.environ.get("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL)

    print(f"[contract] Loading index from {index_dir}…", flush=True)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    id_to_idx = build_id_index(vectors)
    catalog = load_catalog(catalog_path)
    embedder = Embedder()
    print(f"[contract] Loading reranker {rerank_model}…", flush=True)
    reranker = Reranker(model_name=rerank_model)
    reranker._load()

    _BUNDLE = RetrievalBundle(
        vectors=vectors,
        emb=emb,
        id_to_idx=id_to_idx,
        embedder=embedder,
        reranker=reranker,
        catalog=catalog,
        non_faq_idxs=idxs_excluding_faqs(vectors),
        catalog_min_score=float(os.environ.get("RAG_CATALOG_MIN_SCORE", "8.0")),
        top_k=int(os.environ.get("RAG_TOP_K", "20")),
        window=int(os.environ.get("RAG_WINDOW", "2")),
        candidate_n=int(os.environ.get("RAG_CANDIDATE_N", "100")),
        route_n=int(os.environ.get("RAG_ROUTE_N", "80")),
        route_global_n=int(os.environ.get("RAG_ROUTE_GLOBAL_N", "20")),
        use_route=True,
        use_catalog=True,
        enable_grep=os.environ.get("RAG_AGENT_GREP", "0").lower()
        in ("1", "true", "yes"),
    )
    if _BUNDLE.enable_grep:
        from rag.agent_tools import ensure_chunk_bm25

        print("[contract] Building chunk BM25 for grep …", flush=True)
        ensure_chunk_bm25(_BUNDLE)
    _LLM_MODEL, _LLM_KWARGS = resolve_model(model_name)
    effort = os.environ.get("RAG_REASONING_EFFORT", "low")
    if effort:
        _LLM_EXTRA = {"extra_body": {"reasoning_effort": effort}}
    print(
        f"[contract] Ready · {len(vectors)} chunks · catalog={catalog.n_entries} · "
        f"model={_LLM_MODEL}"
        + (" · grep=ON" if _BUNDLE.enable_grep else ""),
        flush=True,
    )
    return _BUNDLE


@app.on_event("startup")
def _startup() -> None:
    # Warm index so the first /ask is not a cold load during blind eval.
    if os.environ.get("RAG_LAZY_LOAD", "").lower() in ("1", "true", "yes"):
        return
    _load_bundle()


@app.get("/health")
def health():
    return {"status": "ok", "ready": _BUNDLE is not None}


def answer_question(question: str) -> AskResponse:
    bundle = _load_bundle()
    assert _LLM_MODEL is not None
    t0 = time.perf_counter()
    # Default OFF: gated verify stripped GT-critical facts on Train-48Q
    # (ok 42/48 → 33/48). Opt in with RAG_VERIFY=1 after the stripper is safer.
    verify_on = os.environ.get("RAG_VERIFY", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    result = run_agent(
        bundle,
        question,
        model=_LLM_MODEL,
        model_kwargs=_LLM_KWARGS,
        max_tool_rounds=int(os.environ.get("RAG_AGENT_MAX_ROUNDS", "2")),
        llm_timeout=float(os.environ.get("RAG_LLM_TIMEOUT", "300")),
        llm_extra=_LLM_EXTRA,
        max_citations=5,
        verbose=False,
        verify=verify_on,
        verify_model=os.environ.get("RAG_VERIFY_MODEL", "google/gemma-3-27b-it"),
        verify_timeout=float(os.environ.get("RAG_VERIFY_TIMEOUT", "15")),
        verify_force=os.environ.get("RAG_VERIFY_FORCE", "").lower()
        in ("1", "true", "yes"),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    citations = [
        Citation(
            file=c.get("file", ""),
            page=c.get("page"),
            quote=c.get("quote"),
        )
        for c in result.citations
        if c.get("file")
    ]
    # Crude confidence: more tools without fetches → lower; fetched evidence helps.
    n_fetch = len(result.state.fetched_ids)
    n_pass = len(result.state.passages)
    confidence = min(0.95, 0.45 + 0.03 * n_pass + 0.05 * n_fetch)
    if not result.answer or "אין בידי" in result.answer or "לא מספיק" in result.answer:
        confidence = min(confidence, 0.35)
    return AskResponse(
        answer=result.answer,
        citations=citations,
        domain=result.domain,
        confidence=confidence,
        latency_ms=round(latency_ms, 1),
        cost_usd=result.cost_usd,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return answer_question(req.question)
