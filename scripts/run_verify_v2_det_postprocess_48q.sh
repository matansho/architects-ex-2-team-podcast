#!/usr/bin/env bash
# Apply verify v2 *deterministic* aside strip on the champion 48Q answers,
# then corpus-judge. No agent regen / no verify LLM.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
source scripts/activate.sh 2>/dev/null || true

SRC=${SRC:-reports/rag_answers_agent_v2_hybrid_topk25_48q.jsonl}
OUT=${OUT:-reports/rag_answers_agent_v2_hybrid_topk25_verify_det_48q.jsonl}
EVAL=${EVAL:-reports/stage2/rag_agent_v2_hybrid_topk25_verify_det_48q_corpus_judge_eval.json}
MODEL=${MODEL:-moonshotai/Kimi-K3}

echo "=== POSTPROCESS deterministic verify on $SRC === $(date +%T)"
.venv/bin/python - <<PY
import json
from pathlib import Path
from rag.verify import should_verify, strip_unasked_asides, covers_question, verify_meta, VerifyResult

src = Path("$SRC")
out = Path("$OUT")
qs = json.loads(Path("reference_questions.json").read_text(encoding="utf-8"))
if isinstance(qs, dict):
    qs = qs["questions"]
qmap = {q["id"]: q["question"] for q in qs}

n = changed = gated = 0
with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        n += 1
        q = qmap.get(rec["id"], rec.get("question", ""))
        ans = rec.get("answer") or ""
        ok, reason = should_verify(q, ans)
        v = VerifyResult(ran=False, gated=not ok, gate_reason=reason, answer_before=ans, answer_after=ans, mode="extras")
        if ok:
            new, removed = strip_unasked_asides(q, ans)
            if removed and covers_question(q, ans, new)[0]:
                v.ran = True
                v.deterministic_stripped = removed
                v.stripped = removed
                v.answer_after = new
                rec["answer"] = new
                changed += 1
            elif removed:
                v.reverted = True
                v.revert_reason = "covers_question"
        else:
            gated += 1
        # stash meta under retrieval.agent.verify for dashboard compatibility
        retr = rec.setdefault("retrieval", {})
        agent = retr.setdefault("agent", {})
        agent["verify"] = verify_meta(v)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {out} n={n} gated={gated} changed={changed}")
PY

echo "=== JUDGE === $(date +%T)"
.venv/bin/python run_eval.py \
  --answers "$OUT" \
  --questions reference_questions.json \
  --corpus-judge \
  --judge-model "$MODEL" \
  --judge-reasoning-effort low \
  --label agent-v2-hybrid-topk25-verify-det-48q \
  --out "$EVAL"
echo "=== DONE rc=$? === $(date +%T)"
.venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("$EVAL")
if p.exists():
    ev = json.loads(p.read_text())
    print(f"OK {ev.get('ok_rate',0):.1%}  GT {ev.get('gt_covered_rate',0):.1%}  supp {ev.get('fully_supported_rate',0):.1%}")
PY
