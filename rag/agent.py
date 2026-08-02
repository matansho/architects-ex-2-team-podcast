"""Hybrid ReAct agent: seed retrieve, then up to N tool rounds, then answer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import litellm

from rag.agent_tools import (
    BRIEF_BATCH,
    AgentState,
    RetrievalBundle,
    ToolTraceEntry,
    brief_page_footer,
    dispatch_tool,
    format_passages_brief,
    format_unread_seed_map,
    looks_multi_target_question,
    passages_for_answer,
    seed_retrieve,
    tool_more_passages,
    tool_schemas,
    trace_as_json,
)
from rag.generate import (
    SYSTEM_PASSAGE_CITE,
    build_context,
    build_messages,
    citations_from_passage_indices,
    parse_answer_with_passages,
)
from rag.retrieve import citations_from_hits
from rag.verify import (
    DEFAULT_VERIFY_MODEL,
    DEFAULT_VERIFY_TIMEOUT_S,
    verify_and_strip,
    verify_meta,
)

# Generic lexical few-shots (synthetic — not from eval / corpus questions).
_LEXICAL_FEWSHOT_CATALOG = """
Lexical / catalog examples (patterns only — invent queries from the customer
question; do not memorize these scenarios):

Example A — wrong product family in seed:
  Customer asks about a named rider the seed never titles.
  Good: catalog_lookup query="שם המוצר או סוג הכיסוי המדויק מהשאלה"
  Then: search with use_catalog=true, or fetch_chunks on catalog hits.
  Bad: final_answer from a related-but-different policy in seed.

Example B — need a form / schedule by type:
  Customer mentions submitting a change/request form; seed is general T&Cs.
  Good: catalog_lookup query="טופס בקשה [סוג השינוי] [סוג הפוליסה]"
  Bad: paraphrasing the legal conclusion you hope to find.
"""

_LEXICAL_FEWSHOT_GREP = """
Lexical search examples (patterns only — invent queries from the customer
question; do not memorize these scenarios):

Example A — exact contact / code missing from seed:
  Need a fax, email, shortcode, or product code the snippets never show.
  Good: grep query="03-1234567" or grep query="*1234" or the exact mailbox
  token from the question/domain docs you expect.
  Then: fetch_chunks on promising ids.
  Bad: grep query="לאן שולחים את המסמך ואיך יוצרים קשר" (full-sentence paraphrase).

Example B — form / endorsement title words:
  Customer asks about a procedural request (add a beneficiary, update
  address, cancel a rider); seed titles are unrelated general conditions.
  Good: grep query="טופס בקשה להוספת מוטב" or the distinctive noun phrase
  from the question (document-type + action), mode=bm25.
  Bad: grep query="השינוי ייכנס לתוקף רק לאחר אישור בכתב של החברה"
  (answer-shaped prose — lexical search will miss the form).

Example C — rare statute / clause label:
  Good: grep query="סעיף 14" plus a unique nearby term, or the Hebrew name of
  the rider/endorsement.
  Bad: grep with only stopwords or a whole rewritten answer.

Example D — catalog then narrow:
  Good: catalog_lookup query="[שם מוצר] [סוג מסמך]" → then grep/search inside
  those files (use_catalog=true on search).
  Bad: dense search alone when seed filenames clearly mismatch the ask.

Rule of thumb: lexical query = short distinctive tokens likely to appear
verbatim in a filename, form title, phone, or clause heading — not a
restatement of the answer you want.
"""

AGENT_SYSTEM = """You are a retrieval controller for a Harel Insurance support system.
You do NOT answer the customer yourself. You gather evidence with tools, then call final_answer.

Seed passages include primary hits AND their attached neighbors (window expand).
Neighbors are already in the answer context. Titles marked ★ look like
exclusions/limits/extensions.

Policy:
1. Only a first page of primaries is shown in full. Unread seed titles are
   listed below the page — scan them. For compare / both-products / multi-part
   questions, call more_passages before final_answer or search (paging is free
   vs tool-round budget). [N] labels match the answer context.
2. Read primaries, neighbor titles/snippets, AND the unread title map before
   any tool call.
3. If a neighbor or unread title already has the needed fact (cap, exclusion,
   waiting period, amount, second product) → more_passages if you need the
   snippet, else final_answer. Do not search/grep for text already listed.
4. If you need fuller text for an id that is listed but truncated → fetch_chunks
   on that id (skip if observation says already_in_context).
5. If titles near a hit look promising but you lack the right id → peek_titles,
   then fetch_chunks for interesting titles not yet listed.
6. Wrong product → catalog_lookup with product keywords, then search with a
   tighter query and use_catalog=true (reuses those files; search does not
   re-run catalog on its own). Prefer paging unread seed over search when the
   unread map already names the missing product/file.
7. Prefer few tool calls. Never invent policy facts.
""" + _LEXICAL_FEWSHOT_CATALOG

AGENT_SYSTEM_GREP = """You are a retrieval controller for a Harel Insurance support system.
You do NOT answer the customer yourself. You gather evidence with tools, then call final_answer.

Seed passages include primary hits AND their attached neighbors (window expand).
Neighbors are already in the answer context. Titles marked ★ look like
exclusions/limits/extensions.

Policy:
1. Only a first page of primaries is shown in full. Unread seed titles are
   listed below the page — scan them. For compare / both-products / multi-part
   questions, call more_passages before final_answer or search (paging is free
   vs tool-round budget). [N] labels match the answer context.
2. Read primaries, neighbor titles/snippets, AND the unread title map before
   any tool call.
3. If a neighbor or unread title already has the needed fact (cap, exclusion,
   waiting period, amount, second product) → more_passages if you need the
   snippet, else final_answer. Do not search/grep for text already listed.
4. If you need fuller text for an id that is listed but truncated → fetch_chunks
   on that id (skip if observation says already_in_context).
5. If titles near a hit look promising but you lack the right id → peek_titles,
   then fetch_chunks for interesting titles not yet listed.
6. Wrong product → catalog_lookup with product keywords, then search with a
   tighter query and use_catalog=true (reuses those files; search does not
   re-run catalog on its own). Prefer paging unread seed over search when the
   unread map already names the missing product/file.
7. Exact tokens missing from seed (product codes, phones, statute names,
   form-title phrases) → grep then fetch_chunks on new hits. Prefer grep over
   search only for distinctive wording likely to appear verbatim.
8. Prefer few tool calls. Never invent policy facts.
""" + _LEXICAL_FEWSHOT_GREP


@dataclass
class AgentResult:
    state: AgentState
    answer: str
    citations: list[dict[str, Any]]
    passage_meta: dict[str, Any] | None
    messages_agent: list[dict[str, Any]]
    latency_ms: dict[str, float]
    tokens: dict[str, int]
    cost_usd: float | None
    domain: str | None
    verify: dict[str, Any] | None = None


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _assistant_tool_message(msg: Any) -> dict[str, Any]:
    """Normalize litellm/OpenAI assistant message with tool_calls for history."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    serialized = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        serialized.append(
            {
                "id": getattr(tc, "id", "") or "",
                "type": "function",
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                },
            }
        )
    content = getattr(msg, "content", None)
    out: dict[str, Any] = {"role": "assistant", "content": content}
    if serialized:
        out["tool_calls"] = serialized
    return out


def run_agent_loop(
    bundle: RetrievalBundle,
    question: str,
    *,
    model: str,
    model_kwargs: dict | None = None,
    max_tool_rounds: int = 2,
    llm_timeout: float = 90.0,
    llm_extra: dict | None = None,
    verbose: bool = False,
) -> AgentState:
    """Seed retrieve + tool rounds. Does not generate the customer answer."""
    state = seed_retrieve(bundle, question)
    if verbose:
        print(
            f"    agent seed → {len(state.passages)} passages"
            f" · route={state.route_domains}"
            f" · catalog={len(state.catalog_files)} files",
            flush=True,
        )

    total_passages = len(state.passages)
    # Multi-part / compare asks: auto-reveal page 2 so the controller brief
    # already covers a second product/file (no Nebius round spent).
    seed_n = min(BRIEF_BATCH, total_passages)
    state.brief_shown = seed_n
    if looks_multi_target_question(question) and total_passages > BRIEF_BATCH:
        tool_more_passages(state, batch=BRIEF_BATCH)
        seed_n = state.brief_shown
    seed_block = format_passages_brief(
        state.passages, offset=0, max_n=seed_n
    )
    unread = format_unread_seed_map(
        state.passages,
        shown=state.brief_shown,
        seed_n=state.seed_passage_n or total_passages,
    )
    page_note = brief_page_footer(
        shown=state.brief_shown,
        total=total_passages,
        batch=BRIEF_BATCH,
        unread_map=unread,
    )
    cat_note = ""
    if state.catalog_matched:
        titles = ", ".join(m["title"][:40] for m in state.catalog_matched[:3])
        cat_note = f"\nCatalog matched: {titles}\n"

    tools = tool_schemas(enable_grep=bundle.enable_grep)
    tool_hint = (
        "more_passages / peek_titles / fetch_chunks / catalog_lookup / "
        "search / final_answer"
    )
    if bundle.enable_grep:
        tool_hint = (
            "more_passages / peek_titles / fetch_chunks / catalog_lookup / "
            "search / grep / final_answer"
        )
    system = AGENT_SYSTEM_GREP if bundle.enable_grep else AGENT_SYSTEM

    user = (
        f"Customer question:\n{question}\n\n"
        f"Route domains: {state.route_domains}\n"
        f"{cat_note}\n"
        f"Seed passages (first page; neighbors already in answer context):\n"
        f"{seed_block}\n\n"
        f"{page_note}\n"
        f"If neighbors / unread titles already cover the ask, prefer "
        f"final_answer (or more_passages for a second product on multi-part "
        f"questions) over search.\n"
        f"Decide: {tool_hint}."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    state.messages = messages

    kwargs = dict(model_kwargs or {})
    extra = dict(llm_extra or {})
    t_loop = time.perf_counter()
    agent_prompt_tok = 0
    agent_comp_tok = 0

    # more_passages-only turns do not consume tool-round budget. Cap total LLM
    # iterations so a stuck pager cannot loop forever.
    budget_rounds = 0
    llm_iters = 0
    max_llm_iters = max_tool_rounds + 1 + max(
        8, (total_passages + BRIEF_BATCH - 1) // max(BRIEF_BATCH, 1) + 2
    )

    while llm_iters < max_llm_iters:
        # Last budgeted iteration forces final if model keeps calling tools.
        tool_choice: Any = "auto"
        if budget_rounds >= max_tool_rounds:
            tool_choice = {
                "type": "function",
                "function": {"name": "final_answer"},
            }

        t0 = time.perf_counter()
        resp = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            timeout=llm_timeout,
            num_retries=0,
            **kwargs,
            **extra,
        )
        state.timings_ms[f"agent_llm_round{llm_iters}_ms"] = round(
            (time.perf_counter() - t0) * 1000, 1
        )
        usage = getattr(resp, "usage", None)
        if usage:
            agent_prompt_tok += int(getattr(usage, "prompt_tokens", 0) or 0)
            agent_comp_tok += int(getattr(usage, "completion_tokens", 0) or 0)

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        messages.append(_assistant_tool_message(msg))
        llm_iters += 1

        if not tool_calls:
            # Model answered in prose — treat as done.
            state.done = True
            if verbose:
                print("    agent → no tool_calls (stop)", flush=True)
            break

        called: list[str] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args = _parse_tool_args(getattr(fn, "arguments", "") if fn else "")
            tc_id = getattr(tc, "id", "") or name
            called.append(name)
            if verbose:
                print(f"    agent tool → {name} {args}", flush=True)
            obs = dispatch_tool(name, args, state, bundle)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": obs,
                }
            )
            if state.done:
                break
        if state.done:
            break

        # Pagination-only turns are free vs max_tool_rounds.
        if called and all(n == "more_passages" for n in called):
            continue
        budget_rounds += 1
        if budget_rounds > max_tool_rounds:
            break

    if not state.done:
        state.done = True
        state.tool_trace.append(
            ToolTraceEntry(
                name="final_answer",
                arguments={"reason": "max_rounds"},
                ok=True,
                summary="forced after max tool rounds",
            )
        )

    state.timings_ms["agent_loop_ms"] = round(
        (time.perf_counter() - t_loop) * 1000, 1
    )
    state.timings_ms["agent_prompt_tokens"] = float(agent_prompt_tok)
    state.timings_ms["agent_completion_tokens"] = float(agent_comp_tok)
    state.messages = messages
    return state


def generate_answer(
    state: AgentState,
    *,
    model: str,
    model_kwargs: dict | None = None,
    llm_timeout: float = 90.0,
    llm_extra: dict | None = None,
    max_citations: int = 5,
    system_prompt: str = SYSTEM_PASSAGE_CITE,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, dict[str, int], float | None]:
    """Final grounded answer from the full working passage set (cite path)."""
    answer_passages = passages_for_answer(state)
    context = build_context(answer_passages)
    messages = build_messages(
        state.question, context, system_prompt=system_prompt
    )
    kwargs = dict(model_kwargs or {})
    extra = dict(llm_extra or {})
    resp = litellm.completion(
        model=model,
        messages=messages,
        timeout=llm_timeout,
        num_retries=0,
        **kwargs,
        **extra,
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    tokens = {
        "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    cost = None
    try:
        from tf_client import EST_PRICE

        cost = (
            tokens["prompt"] * EST_PRICE[0] + tokens["completion"] * EST_PRICE[1]
        ) / 1e6
    except Exception:
        pass

    parsed = parse_answer_with_passages(content)
    answer = parsed.answer
    passage_meta = {
        "cite_source": "passages" if parsed.parse_ok else "hits_fallback",
        "passage_indices": parsed.passage_indices,
        "raw_list": parsed.used_passages_raw,
        "parse_ok": parsed.parse_ok,
        "n_answer_passages": len(answer_passages),
        "n_working_passages": len(state.passages),
    }
    if parsed.parse_ok and parsed.passage_indices:
        citations = citations_from_passage_indices(
            answer_passages,
            parsed.passage_indices,
            max_citations=max_citations,
        )
    elif parsed.parse_ok and not parsed.passage_indices:
        # Explicit USED_PASSAGES: none — do not invent cites from top hits.
        citations = []
        passage_meta["cite_source"] = "none"
    else:
        # Unparseable footer: still fall back, but mark it for diagnostics.
        citations = citations_from_hits(
            answer_passages, max_citations=max_citations
        )
        passage_meta["cite_source"] = "hits_fallback"
    return answer, citations, passage_meta, tokens, cost


def run_agent(
    bundle: RetrievalBundle,
    question: str,
    *,
    model: str,
    model_kwargs: dict | None = None,
    max_tool_rounds: int = 2,
    llm_timeout: float = 90.0,
    llm_extra: dict | None = None,
    max_citations: int = 5,
    verbose: bool = False,
    verify: bool = False,
    verify_model: str = DEFAULT_VERIFY_MODEL,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT_S,
    verify_force: bool = False,
    verify_mode: str = "extras",
    verify_deterministic_only: bool = False,
    verify_apply_deterministic: bool = False,
) -> AgentResult:
    """Full hybrid agent: seed → tools → grounded answer → optional verify/strip."""
    t0 = time.perf_counter()
    state = run_agent_loop(
        bundle,
        question,
        model=model,
        model_kwargs=model_kwargs,
        max_tool_rounds=max_tool_rounds,
        llm_timeout=llm_timeout,
        llm_extra=llm_extra,
        verbose=verbose,
    )
    t_ans = time.perf_counter()
    answer, citations, passage_meta, tokens, cost = generate_answer(
        state,
        model=model,
        model_kwargs=model_kwargs,
        llm_timeout=llm_timeout,
        llm_extra=llm_extra,
        max_citations=max_citations,
    )
    state.timings_ms["answer_llm_ms"] = round(
        (time.perf_counter() - t_ans) * 1000, 1
    )

    verify_info: dict[str, Any] | None = None
    if verify:
        used_tools = any(
            t.name != "final_answer" for t in state.tool_trace
        )
        v = verify_and_strip(
            question,
            answer,
            passages_for_answer(state),
            passage_indices=(passage_meta or {}).get("passage_indices"),
            n_citations=len(citations),
            used_tools=used_tools,
            model=verify_model,
            timeout_s=verify_timeout,
            force=verify_force,
            mode=verify_mode if verify_mode in ("extras", "strict") else "extras",
            deterministic_only=bool(verify_deterministic_only),
            apply_deterministic=bool(verify_apply_deterministic),
        )
        verify_info = verify_meta(v)
        state.timings_ms["verify_ms"] = v.latency_ms
        if v.gated and verbose:
            print(f"    verify gated ({v.gate_reason})", flush=True)
        elif v.skipped and verbose:
            print(
                f"    verify skipped → answer as-is ({v.skip_reason})",
                flush=True,
            )
        elif verbose and (
            v.ran or v.deterministic_stripped or v.reverted
        ):
            print(
                f"    verify → mode={v.mode} unsupported={len(v.unsupported)} "
                f"stripped={len(v.stripped)} "
                f"det={len(v.deterministic_stripped)}"
                + (f" reverted={v.revert_reason}" if v.reverted else "")
                + f" ({v.latency_ms:.0f}ms)",
                flush=True,
            )
        if v.answer_after and v.answer_after != answer:
            answer = v.answer_after
        if v.tokens:
            tokens["verify_prompt"] = v.tokens.get("prompt", 0)
            tokens["verify_completion"] = v.tokens.get("completion", 0)
        if cost is not None and v.cost_usd is not None:
            cost += v.cost_usd

    state.timings_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Fold agent-loop tokens into totals when present.
    agent_p = int(state.timings_ms.get("agent_prompt_tokens", 0))
    agent_c = int(state.timings_ms.get("agent_completion_tokens", 0))
    ans_p = tokens["prompt"]
    ans_c = tokens["completion"]
    ver_p = int(tokens.get("verify_prompt", 0))
    ver_c = int(tokens.get("verify_completion", 0))
    tokens = {
        "prompt": ans_p + agent_p + ver_p,
        "completion": ans_c + agent_c + ver_c,
        "answer_prompt": ans_p,
        "answer_completion": ans_c,
        "agent_prompt": agent_p,
        "agent_completion": agent_c,
        "verify_prompt": ver_p,
        "verify_completion": ver_c,
    }
    if cost is not None:
        try:
            from tf_client import EST_PRICE

            cost += (
                agent_p * EST_PRICE[0] + agent_c * EST_PRICE[1]
            ) / 1e6
        except Exception:
            pass

    domain = None
    if state.route_domains:
        domain = state.route_domains[0] if len(state.route_domains) == 1 else ",".join(
            state.route_domains
        )

    return AgentResult(
        state=state,
        answer=answer,
        citations=citations,
        passage_meta=passage_meta,
        messages_agent=getattr(state, "messages", []),
        latency_ms=dict(state.timings_ms),
        tokens=tokens,
        cost_usd=cost,
        domain=domain,
        verify=verify_info,
    )


def agent_meta(result: AgentResult) -> dict[str, Any]:
    """JSON-serializable agent diagnostics for answer records /ask."""
    st = result.state
    return {
        "max_tool_rounds": None,
        "tool_trace": trace_as_json(st),
        "n_tools": len(st.tool_trace),
        "fetched_ids": list(st.fetched_ids),
        "n_passages": len(st.passages),
        "n_answer_passages": len(passages_for_answer(st)),
        "brief_shown": st.brief_shown,
        "brief_batch": BRIEF_BATCH,
        "seed_passage_n": st.seed_passage_n,
        "route_domains": st.route_domains,
        "catalog_files": st.catalog_files[:12],
        "catalog_matched": st.catalog_matched,
        "title_map_n": len(st.title_map),
        "verify": result.verify,
    }
