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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.agent import run_agent
from rag.agent_tools import RetrievalBundle
from rag.catalog import DEFAULT_CATALOG_PATH, load_catalog
from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import Reranker
from rag.retrieve import build_id_index, idxs_excluding_faqs
from rag_runner import resolve_model

# Defaults match measured champion: MiniLM→BGE@40 · top-k 25 · LLM verify.
# Override any knob with RAG_* env vars (see scripts/run_gui.sh).
CHAMPION_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHAMPION_REFINE_MODEL = "BAAI/bge-reranker-v2-m3"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _env_optional_model(name: str, default: str) -> str | None:
    """Return model id, or None if env disables refine (0/none/off/empty)."""
    raw = os.environ.get(name, default)
    if raw is None:
        return None
    val = raw.strip()
    if not val or val.lower() in ("0", "none", "off", "false", "no"):
        return None
    return val


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

# Local GUI + optional cross-origin demos. Does not alter AskRequest/AskResponse.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).resolve().parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")

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
    rerank_model = os.environ.get("RAG_RERANK_MODEL", CHAMPION_RERANK_MODEL)
    refine_model = _env_optional_model(
        "RAG_RERANK_REFINE_MODEL", CHAMPION_REFINE_MODEL
    )
    refine_n = int(os.environ.get("RAG_RERANK_REFINE_N", "40"))

    print(f"[contract] Loading index from {index_dir}…", flush=True)
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    id_to_idx = build_id_index(vectors)
    catalog = load_catalog(catalog_path)
    embedder = Embedder()
    print(f"[contract] Loading stage-1 reranker {rerank_model}…", flush=True)
    reranker = Reranker(model_name=rerank_model)
    reranker._load()

    refine_reranker = None
    if refine_model:
        print(
            f"[contract] Loading refine reranker {refine_model} (top-{refine_n})…",
            flush=True,
        )
        refine_reranker = Reranker(model_name=refine_model)
        refine_reranker._load()

    _BUNDLE = RetrievalBundle(
        vectors=vectors,
        emb=emb,
        id_to_idx=id_to_idx,
        embedder=embedder,
        reranker=reranker,
        refine_reranker=refine_reranker,
        refine_n=refine_n,
        catalog=catalog,
        non_faq_idxs=idxs_excluding_faqs(vectors),
        catalog_min_score=float(os.environ.get("RAG_CATALOG_MIN_SCORE", "8.0")),
        top_k=int(os.environ.get("RAG_TOP_K", "25")),
        window=int(os.environ.get("RAG_WINDOW", "2")),
        candidate_n=int(os.environ.get("RAG_CANDIDATE_N", "100")),
        route_n=int(os.environ.get("RAG_ROUTE_N", "80")),
        route_global_n=int(os.environ.get("RAG_ROUTE_GLOBAL_N", "20")),
        route_preview_n=int(os.environ.get("RAG_ROUTE_PREVIEW_N", "20")),
        use_route=_env_bool("RAG_ROUTE", "1"),
        use_catalog=_env_bool("RAG_CATALOG_ON", "1"),
        enable_grep=_env_bool("RAG_AGENT_GREP", "0"),
    )
    if _BUNDLE.enable_grep:
        from rag.agent_tools import ensure_chunk_bm25

        print("[contract] Building chunk BM25 for grep …", flush=True)
        ensure_chunk_bm25(_BUNDLE)
    _LLM_MODEL, _LLM_KWARGS = resolve_model(model_name)
    effort = os.environ.get("RAG_REASONING_EFFORT", "low")
    if effort:
        _LLM_EXTRA = {"extra_body": {"reasoning_effort": effort}}
    refine_label = (
        f" · refine={refine_model}@{refine_n}" if refine_reranker else " · refine=OFF"
    )
    print(
        f"[contract] Ready · {len(vectors)} chunks · catalog={catalog.n_entries} · "
        f"top_k={_BUNDLE.top_k} · rerank={rerank_model}{refine_label} · "
        f"model={_LLM_MODEL}"
        + (" · grep=ON" if _BUNDLE.enable_grep else "")
        + (" · verify=ON" if _env_bool("RAG_VERIFY", "1") else " · verify=OFF"),
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
    cfg = None
    if _BUNDLE is not None:
        cfg = {
            "top_k": _BUNDLE.top_k,
            "window": _BUNDLE.window,
            "candidate_n": _BUNDLE.candidate_n,
            "route_n": _BUNDLE.route_n,
            "route_global_n": _BUNDLE.route_global_n,
            "refine": _BUNDLE.refine_reranker is not None,
            "refine_n": _BUNDLE.refine_n,
            "verify": _env_bool("RAG_VERIFY", "1"),
            "model": _LLM_MODEL,
        }
    return {"status": "ok", "ready": _BUNDLE is not None, "config": cfg}


@app.get("/")
def gui_root():
    """Browser chat UI (voice + text). Blind eval uses POST /ask only."""
    index = _UI_DIR / "index.html"
    if not index.is_file():
        return {"status": "ok", "ui": "missing", "ask": "/ask"}
    return FileResponse(index)


def answer_question(question: str) -> AskResponse:
    bundle = _load_bundle()
    assert _LLM_MODEL is not None
    t0 = time.perf_counter()
    # Default ON — measured Train 91.7% / Eval 83.6%. Disable with RAG_VERIFY=0.
    verify_on = _env_bool("RAG_VERIFY", "1")
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
        verify_model=os.environ.get("RAG_VERIFY_MODEL", "moonshotai/Kimi-K3"),
        verify_timeout=float(os.environ.get("RAG_VERIFY_TIMEOUT", "45")),
        verify_force=_env_bool("RAG_VERIFY_FORCE", "0"),
        verify_mode=os.environ.get("RAG_VERIFY_MODE", "extras"),
        verify_deterministic_only=_env_bool("RAG_VERIFY_DETERMINISTIC_ONLY", "0"),
        verify_apply_deterministic=_env_bool("RAG_VERIFY_APPLY_DETERMINISTIC", "0"),
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
