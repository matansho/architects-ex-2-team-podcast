#!/usr/bin/env bash
# Apply LLM verify v3 on frozen champion Eval-61Q answers, then corpus-judge.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
source scripts/activate.sh 2>/dev/null || true
set -a && source .env && set +a
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.tokenfactory.nebius.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$NEBIUS_API_KEY}"

SRC=${SRC:-reports/rag_answers_agent_v2_hybrid_topk25_eval61q.jsonl}
OUT=${OUT:-reports/rag_answers_agent_v2_hybrid_topk25_verify_llm_eval61q.jsonl}
EVAL=${EVAL:-reports/stage2/rag_agent_v2_hybrid_topk25_verify_llm_eval61q_corpus_judge_eval.json}
QUESTIONS=${QUESTIONS:-eval_questions2.json}
MODEL=${MODEL:-moonshotai/Kimi-K3}
JUDGE_MODEL=${JUDGE_MODEL:-$MODEL}
BASELINE=${BASELINE:-reports/stage2/rag_agent_v2_hybrid_topk25_eval61q_corpus_judge_eval.json}

echo "=== LLM VERIFY postprocess $SRC → $OUT === $(date +%T)"
.venv/bin/python - <<PY
import json
from pathlib import Path

from rag.index_store import load_vectors
from rag.retrieve import ExpandedHit, normalize_cite_file
from rag.verify import verify_and_strip, verify_meta

src = Path("$SRC")
out = Path("$OUT")
model = "$MODEL"
qpath = Path("$QUESTIONS")
qs = json.loads(qpath.read_text(encoding="utf-8"))
if isinstance(qs, dict):
    qs = qs["questions"]
qmap = {q["id"]: q["question"] for q in qs}
order = [q["id"] for q in qs]

print("loading index …", flush=True)
vectors = load_vectors(Path("data/index"))
print(f"  {len(vectors)} vectors", flush=True)

by_fp: dict[tuple[str, int | None], object] = {}
for v in vectors:
    f = normalize_cite_file(v.location.file)
    pages = set(v.location.pages or [])
    if v.location.page is not None:
        pages.add(v.location.page)
    if not pages:
        by_fp.setdefault((f, None), v)
    for p in pages:
        by_fp.setdefault((f, p), v)


def passages_from_cites(cites: list[dict], *, limit: int = 8) -> list[ExpandedHit]:
    out: list[ExpandedHit] = []
    seen: set[str] = set()
    for c in cites or []:
        f = normalize_cite_file(c.get("file") or "")
        p = c.get("page") if isinstance(c.get("page"), int) else None
        v = by_fp.get((f, p)) or by_fp.get((f, None))
        if v is None or v.id in seen:
            continue
        seen.add(v.id)
        out.append(ExpandedHit(rank=len(out) + 1, score=1.0, match=v, neighbors=[]))
        if len(out) >= limit:
            break
    return out

by_id = {}
with src.open(encoding="utf-8") as fin:
    for line in fin:
        if line.strip():
            rec = json.loads(line)
            by_id[rec["id"]] = rec

n = changed = gated = skipped = reverted = missing = 0
with out.open("w", encoding="utf-8") as fout:
    for i, qid in enumerate(order, 1):
        rec = by_id.get(qid)
        if not rec:
            missing += 1
            print(f"  [{i}/{len(order)}] {qid} MISSING", flush=True)
            continue
        n += 1
        q = qmap.get(qid, rec.get("question", ""))
        ans = rec.get("answer") or ""
        passages = passages_from_cites(rec.get("citations") or [])
        print(f"  [{i}/{len(order)}] {qid} …", flush=True)
        v = verify_and_strip(
            q,
            ans,
            passages,
            passage_indices=list(range(1, len(passages) + 1)) or None,
            n_citations=len(rec.get("citations") or []),
            force=False,
            model=model,
            mode="extras",
            apply_deterministic=False,
            deterministic_only=False,
        )
        meta = verify_meta(v)
        if v.gated:
            gated += 1
        if v.skipped:
            skipped += 1
        if v.reverted:
            reverted += 1
        if meta["changed"]:
            changed += 1
            rec = dict(rec)
            rec["answer"] = v.answer_after
            print(
                f"    changed drop={meta.get('drop_ids')} ({v.latency_ms:.0f}ms)",
                flush=True,
            )
        else:
            print(
                f"    keep gated={v.gated} skipped={v.skipped} ({v.latency_ms:.0f}ms)",
                flush=True,
            )
        retr = rec.setdefault("retrieval", {})
        agent = retr.setdefault("agent", {})
        agent["verify"] = meta
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(
    f"wrote {out} n={n} missing={missing} gated={gated} skipped={skipped} "
    f"changed={changed} reverted={reverted}",
    flush=True,
)
PY
pp_rc=$?
[ "$pp_rc" -eq 0 ] || exit "$pp_rc"

echo "=== JUDGE 61Q === $(date +%T)"
.venv/bin/python run_eval.py \
  --answers "$OUT" \
  --questions "$QUESTIONS" \
  --corpus-judge \
  --judge-model "$JUDGE_MODEL" \
  --judge-reasoning-effort low \
  --label agent-v2-hybrid-topk25-verify-llm-eval61q \
  --out "$EVAL"
echo "=== DONE rc=$? === $(date +%T)"

.venv/bin/python - <<PY
import json
from pathlib import Path
oldp = Path("$BASELINE")
newp = Path("$EVAL")
if oldp.exists() and newp.exists():
    old = json.loads(oldp.read_text())
    new = json.loads(newp.read_text())
    print(f"champion61: ok={old['ok_rate']:.1%} gt={old['gt_covered_rate']:.1%} supp={old['fully_supported_rate']:.1%}")
    print(f"verifyllm61: ok={new['ok_rate']:.1%} gt={new['gt_covered_rate']:.1%} supp={new['fully_supported_rate']:.1%}")
    ob = {r['id']: r for r in old['results']}
    nb = {r['id']: r for r in new['results']}
    fixed = sorted(i for i in ob if not ob[i]['ok'] and nb[i]['ok'])
    reg = sorted(i for i in ob if ob[i]['ok'] and not nb[i]['ok'])
    print("fixed", fixed)
    print("regressed", reg)
PY
