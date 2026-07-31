"""Does a declarative rewrite retrieve better than the raw question?

Our embedder is e5, which takes a "query: " prefix precisely so question-shaped
text lands near declarative passages. So rewriting questions into statements may
be redundant here, or may help — this measures it instead of assuming.

Method: for each question, take dense candidates from both query forms, score
the union with the cross-encoder using the ORIGINAL question (the same arbiter
for both forms), and treat the CE top-10 as the relevant set. Then ask which
form's dense stage actually surfaces those passages, since feeding the CE is the
only job the dense stage has.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rag.embed import Embedder
from rag.index_store import load_embeddings, load_vectors_meta
from rag.rerank import DEFAULT_RERANK_MODEL, Reranker
from rag.retrieve import search

REWRITE_SYSTEM = """\
You rewrite Hebrew insurance questions as the sentence a policy document would
use to ANSWER them.

Keep EVERY condition, amount, product name, timeframe and qualifier from the
question. Those details are what identify the correct clause — dropping one
makes the search match a different, more general clause. Do not summarise into
a topic label. The statement should be about as detailed as the question.

Remove ONLY: the question word, and first-person framing such as
"יש לי" / "אני" / "אם אני". Keep the situation itself.

Never reinterpret a word you are unsure about — copy the original term verbatim.
For example "המשרד שלי יושבת" means the office was SHUT DOWN (הושבת), not a
meeting; if unsure, keep the original wording rather than paraphrasing it.

Examples:
- "יש לי ביטוח למחזיקי אקדח ברישיון. אם המבטח לא העמיד לי עורך דין ופניתי
  לעורך דין לפי בחירתי להגנה בהליך הפלילי, עד איזה סכום ישפו אותי על שכר
  הטרחה וההוצאות?"
  → "בביטוח למחזיקי אקדח ברישיון, כאשר המבטח לא העמיד עורך דין והמבוטח פנה
  לעורך דין לפי בחירתו להגנה בהליך פלילי, המבטח משפה את המבוטח על שכר טרחה
  והוצאות משפט עד לסכום"
- "אני חוקר שמתכנן ניסוי קליני שאושר על ידי ועדת הלסינקי. האם אני חייב לרכוש
  ביטוח לניסוי?"
  → "חובת רכישת פוליסת ביטוח עבור ניסוי קליני שאושר על ידי ועדת הלסינקי"

Return ONLY valid JSON: {"statement": "..."}
"""


def rewrite(question: str, model: str) -> str:
    import litellm

    import os

    kwargs: dict = {}
    base = os.environ.get("OPENAI_BASE_URL")
    m = model
    if base:
        kwargs["api_base"] = base
        m = f"openai/{model.removeprefix('openai/')}"
    resp = litellm.completion(
        model=m,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question.strip()},
        ],
        temperature=0,
        timeout=90,
        **kwargs,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    if "{" in raw:
        raw = raw[raw.index("{") : raw.rindex("}") + 1]
    return str(json.loads(raw)["statement"]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--questions", default="reference_questions.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--candidate-n", type=int, default=100)
    ap.add_argument("--gold-k", type=int, default=10)
    ap.add_argument("--rewrite-model", default="google/gemma-3-27b-it")
    ap.add_argument("--out", default="reports/stage2/query_form_bench.json")
    args = ap.parse_args()

    from pathlib import Path

    index_dir = Path(args.index)
    config = json.loads((index_dir / "config.json").read_text())
    vectors = load_vectors_meta(index_dir)
    emb = load_embeddings(index_dir)
    embedder = Embedder(model_name=config.get("model"))
    reranker = Reranker(model_name=DEFAULT_RERANK_MODEL)
    print(f"index: {len(vectors)} vectors · embed={config.get('model')}", flush=True)

    questions = json.load(open(args.questions))[: args.n]
    rows = []
    for q in questions:
        qt = q["question"]
        t0 = time.perf_counter()
        try:
            stmt = rewrite(qt, args.rewrite_model)
        except Exception as e:  # noqa: BLE001
            print(f"  {q['id']}: rewrite failed ({e!r}) — skipped", flush=True)
            continue
        rewrite_ms = (time.perf_counter() - t0) * 1000

        forms = {"question": qt, "statement": stmt}
        cand: dict[str, list] = {}
        for name, text in forms.items():
            vec = embedder.embed_queries([text])[0]
            hits = search(vec, emb, vectors, top_k=args.candidate_n)
            cand[name] = hits

        # Union, scored by the CE against the ORIGINAL question so neither form
        # gets a home-field advantage in defining what counts as relevant.
        pool: dict[str, object] = {}
        for hits in cand.values():
            for h in hits:
                pool[h.vector.id] = h.vector
        ids = list(pool)
        texts = [
            getattr(pool[i], "embed_text", None) or getattr(pool[i], "text", "")
            for i in ids
        ]
        scores = reranker.score(qt, texts)
        gold = {
            ids[i]
            for i in np.argsort(scores)[::-1][: args.gold_k]
        }

        row = {"id": q["id"], "question": qt, "statement": stmt,
               "rewrite_ms": round(rewrite_ms, 1), "pool": len(ids)}
        for name, hits in cand.items():
            got = [h.vector.id for h in hits]
            found = [g for g in gold if g in got]
            ranks = [got.index(g) + 1 for g in found]
            row[f"{name}_recall"] = len(found) / max(len(gold), 1)
            row[f"{name}_mean_rank"] = (
                round(sum(ranks) / len(ranks), 1) if ranks else None
            )
        rows.append(row)
        print(
            f"  {q['id']:26s} recall Q={row['question_recall']:.2f} "
            f"S={row['statement_recall']:.2f}  |  {stmt[:52]}",
            flush=True,
        )

    if rows:
        qr = sum(r["question_recall"] for r in rows) / len(rows)
        sr = sum(r["statement_recall"] for r in rows) / len(rows)
        wins = sum(1 for r in rows if r["statement_recall"] > r["question_recall"])
        loss = sum(1 for r in rows if r["statement_recall"] < r["question_recall"])
        print(f"\n{'=' * 62}")
        print(f"questions            : {len(rows)}")
        print(f"recall@{args.candidate_n} question  : {qr:.3f}")
        print(f"recall@{args.candidate_n} statement : {sr:.3f}  ({sr - qr:+.3f})")
        print(f"statement better/worse/tie: {wins}/{loss}/{len(rows) - wins - loss}")
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
