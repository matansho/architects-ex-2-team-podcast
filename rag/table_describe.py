"""Table retrieval contract: embed a description, return the full table body.

Design
------
Each table becomes one retrieval unit with two strings:

1. **embed_text** (description) — what we embed / search against.
   Built from a structured TableDescription so questions about prices,
   age bands, riders, etc. can match without needing the raw CSV in the vector.

2. **text** (payload) — what we put in the LLM context on a hit.
   Caption/summary + full table (CSV from Docling). Never truncate.

Index-time flow
---------------
    TableItem → CSV body (rag.parse._table_text)
             → TableDescription (heuristic or LLM)
             → embed_text = format_embed_text(desc)
             → text       = format_payload(desc, body)

Query-time flow
---------------
    dense/rerank hits embed_text → generator receives text (full table).

LLM description (optional)
--------------------------
Use DESCRIBE_SYSTEM + DESCRIBE_USER with forced JSON matching TableDescription.
Offline / no-API path: describe_table_heuristic().
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


DESCRIBE_SYSTEM = """\
You describe insurance-policy tables for a Hebrew/English retrieval index.
You receive a SKETCH (headers + a few sample rows), not necessarily the full table —
infer what the table is about; only cite numbers that appear in the sketch.

Return ONLY valid JSON with these keys:
  title: short name of the table (Hebrew ok)
  summary: 1-2 sentences — what the table is for and what you look up
  embed_passage: 2-3 fluent sentences IN THE TABLE'S LANGUAGE (usually Hebrew) for
                  embedding/search. Encyclopedia blurb a customer query could match.
                  Include product name, lookup axes (e.g. age→price), units, and
                  1-3 example values FROM THE SKETCH. No bullet lists, no labels like
                  "ממדים:" / "ערכים:".
  row_axis: what each row represents
  column_axis: what each column represents
  units: list of units (e.g. ["₪/חודש", "$/יום"])
  notable_values: up to 5 short "label: value" facts grounded in the sketch
  product_hints: product / plan / rider names (for search)

Rules:
- Product name is critical. Prefer brand/plan from the File name and surrounding text
  over garbled RTL headers. Only use a name that actually appears there (do not copy
  example brands from these instructions). Put it in product_hints AND embed_passage.
- Do not invent cells, waiting periods, or legal conclusions.
- If the sketch is a form / contact block, say so briefly.
- Keep Latin product names as written when they appear in the File/context.
- In JSON string values never put an ASCII " inside the text
  (write רו״ח / בע״מ, not רו"ח / בע"מ).
"""

DESCRIBE_USER = """\
File: {file}
Page: {page}
Domain: {domain}

Preceding text (context only):
```
{context_before}
```

Table SKETCH (headers + sample rows — PRIMARY source):
```
{table_body}
```

Trailing text (context only):
```
{context_after}
```

Return the JSON object now.
"""


@dataclass
class TableDescription:
    """Structured abstract of a table — source of embed_text."""

    title: str = ""
    summary: str = ""
    embed_passage: str = ""  # fluent sentences for the embedding model
    row_axis: str = ""
    column_axis: str = ""
    units: list[str] = field(default_factory=list)
    notable_values: list[str] = field(default_factory=list)
    product_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> TableDescription:
        if not d:
            return cls()
        return cls(
            title=str(d.get("title") or ""),
            summary=str(d.get("summary") or ""),
            embed_passage=str(d.get("embed_passage") or ""),
            row_axis=str(d.get("row_axis") or ""),
            column_axis=str(d.get("column_axis") or ""),
            units=[str(x) for x in (d.get("units") or [])],
            notable_values=[str(x) for x in (d.get("notable_values") or [])],
            product_hints=[str(x) for x in (d.get("product_hints") or [])],
        )


def format_embed_text(
    desc: TableDescription,
    *,
    file: str = "",
    page: int | None = None,
) -> str:
    """Natural-language passage for E5 (prefer LLM embed_passage; else compose prose)."""
    if (desc.embed_passage or "").strip():
        return desc.embed_passage.strip()

    # Fallback: turn structured fields into a few sentences (no labeled dump).
    products = ", ".join(desc.product_hints[:4]) if desc.product_hints else ""
    units = " או ".join(desc.units) if desc.units else ""
    examples = desc.notable_values[:5]

    bits: list[str] = []
    subject = desc.title or products or "טבלת ביטוח"
    bits.append(f"{subject}.")
    if desc.summary:
        bits.append(desc.summary.rstrip(".") + ".")
    elif products:
        bits.append(f"הטבלה שייכת למוצר {products}.")
    if desc.row_axis and desc.column_axis:
        bits.append(
            f"ניתן לאתר ערכים לפי {desc.row_axis} ולקבל {desc.column_axis}"
            + (f" ב{units}." if units else ".")
        )
    elif units:
        bits.append(f"היחידות בטבלה: {units}.")
    if examples:
        # Keep examples inside a sentence, not a labeled list.
        bits.append("לדוגמה: " + "; ".join(examples) + ".")
    where = ""
    if file:
        where = f" (מתוך {file}" + (f", עמוד {page}" if page is not None else "") + ")"
        bits[0] = bits[0].rstrip(".") + where + "."
    return " ".join(bits).strip()


def format_payload(desc: TableDescription, table_body: str) -> str:
    """LLM context: short abstract + full table body."""
    head: list[str] = []
    if desc.title:
        head.append(f"טבלה: {desc.title}")
    if desc.summary:
        head.append(desc.summary)
    if desc.row_axis or desc.column_axis:
        head.append(
            "ממדים: "
            + ", ".join(x for x in (desc.row_axis, desc.column_axis) if x)
        )
    body = (table_body or "").strip()
    if head and body:
        return "\n".join(head) + "\n\n" + body
    return body or "\n".join(head)


def extract_table_body(payload: str) -> str:
    """Strip optional abstract prefix from format_payload output."""
    t = (payload or "").strip()
    if t.startswith("טבלה:") and "\n\n" in t:
        return t.split("\n\n", 1)[1].strip()
    return t


def sketch_table(
    table_body: str,
    *,
    max_rows: int = 8,
    max_chars: int = 1800,
) -> str:
    """Compact view for LLM describe: header + sample rows (not the full grid)."""
    body = extract_table_body(table_body)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) <= max_rows:
        sketch_lines = lines
    else:
        head_n = max(3, max_rows // 2)
        tail_n = max(2, max_rows - head_n)
        sketch_lines = lines[:head_n] + ["…"] + lines[-tail_n:]
    text = "\n".join(sketch_lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "\n…"
    return text


def looks_like_table(text: str) -> bool:
    """Heuristic: CSV-ish or markdown table."""
    t = (text or "").strip()
    if not t or t == "(table)":
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if sum(1 for ln in lines if ln.strip().startswith("|")) >= 2:
        return True
    comma_lines = sum(1 for ln in lines if ln.count(",") >= 2)
    return comma_lines >= 2


def _guess_units(text: str) -> list[str]:
    units: list[str] = []
    if re.search(r"₪|ש[\"״]?ח|NIS", text):
        units.append("₪")
    if re.search(r"\$|USD|דולר", text):
        units.append("$")
    if re.search(r"ליום|/יום|daily", text, re.I):
        units.append("ליום")
    if re.search(r"לחודש|/חודש|monthly", text, re.I):
        units.append("לחודש")
    if re.search(r"%|אחוז", text):
        units.append("%")
    return units


def _sample_numeric_cells(text: str, limit: int = 10) -> list[str]:
    """Pull a few number-bearing lines as notable_values seeds."""
    scored: list[tuple[int, str]] = []
    for ln in text.splitlines():
        ln = ln.strip().strip('"')
        if not ln or not re.search(r"\d", ln):
            continue
        if re.fullmatch(r"[\d\.,]+", ln):
            continue
        score = 0
        if re.search(r"₪|\$|ש[\"״]?ח", ln):
            score += 3
        if re.search(r"גיל|חודש|יום|פרמיה|תעריף|השתתפות", ln):
            score += 2
        if re.search(r"\d+\.\d{2}", ln):  # money-like decimals
            score += 2
        if len(ln) > 160:
            ln = ln[:157] + "…"
        scored.append((score, ln))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    # de-dupe preserving order
    out: list[str] = []
    seen: set[str] = set()
    for _, ln in scored:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
        if len(out) >= limit:
            break
    return out


def describe_table_heuristic(
    table_body: str,
    *,
    file: str = "",
    page: int | None = None,
    domain: str = "",
) -> TableDescription:
    """No-LLM description: enough for a first embed_text."""
    body = (table_body or "").strip()
    fname = file.rsplit("/", 1)[-1] if file else ""
    title = fname.replace(".pdf", "").replace("-", " ")[:80] if fname else "טבלת פוליסה"
    if page is not None:
        title = f"{title} (עמוד {page})"

    lines = [ln for ln in body.splitlines() if ln.strip()]
    header = lines[0][:120] if lines else ""
    summary_bits = [
        f"טבלה מתוך מסמך ביטוח{(' בתחום ' + domain) if domain else ''}.",
    ]
    if header:
        summary_bits.append(f"כותרת/שורה ראשונה: {header}")
    summary_bits.append(f"כוללת {max(0, len(lines) - 1)} שורות נתונים בקירוב.")

    return TableDescription(
        title=title,
        summary=" ".join(summary_bits),
        row_axis="שורות הטבלה (לרוב מסלול / גיל / כיסוי)",
        column_axis="עמודות הטבלה (לרוב מחיר / תנאי / שם כיסוי)",
        units=_guess_units(body),
        notable_values=_sample_numeric_cells(body),
        product_hints=[p for p in (domain, fname.replace(".pdf", "")) if p],
    )


def _strip_json_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Sometimes model prepends prose before {
    if "{" in text and not text.lstrip().startswith("{"):
        text = text[text.index("{") :]
    if "}" in text:
        text = text[: text.rindex("}") + 1]
    return text.strip()


def _escape_inner_double_quotes(s: str) -> str:
    """Escape " that appear inside JSON string values (e.g. רו\"ח)."""
    out: list[str] = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and in_string and i + 1 < n:
            out.append(ch)
            out.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                # Closing quote if next non-space is structural
                if j >= n or s[j] in ",}]:":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_llm_json(text: str) -> str:
    """Best-effort fixes for common Gemma JSON mistakes."""
    s = text
    # Hebrew abbrevs that use ASCII " as geresh
    for bad, good in (
        ('רו"ח', "רו״ח"),
        ('עו"ד', "עו״ד"),
        ('בע"מ', "בע״מ"),
        ('וכו"', "וכו׳"),
        ("וכו'", "וכו׳"),
    ):
        s = s.replace(bad, good)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = _escape_inner_double_quotes(s)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # again after quote fixes
    return s


def parse_description_json(raw: str) -> TableDescription:
    """Parse LLM JSON (fences + common repairs for trailing commas / inner quotes)."""
    text = _strip_json_fences(raw)
    last_err: Exception | None = None
    for candidate in (text, _repair_llm_json(text)):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("description JSON must be an object")
            return TableDescription.from_dict(data)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def describe_table_with_llm(
    table_body: str,
    *,
    file: str = "",
    page: int | None = None,
    domain: str = "",
    context_before: str = "",
    context_after: str = "",
    model: str = "google/gemma-3-27b-it",
    complete=None,
    use_sketch: bool = True,
) -> TableDescription:
    """Call an LLM to fill TableDescription. `complete(messages)->str` injectable.

    Default uses litellm with OPENAI_BASE_URL / API key (same as runners).
    By default sends a compact sketch (headers + sample rows), not the full table.
    Falls back to heuristic on any failure.
    """
    raw_body = (table_body or "").strip()
    body = sketch_table(raw_body) if use_sketch else raw_body
    if not use_sketch and len(body) > 4000:
        body = body[:2000] + "\n…\n" + body[-2000:]

    def _clip(s: str, n: int = 800) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n // 2] + "\n…\n" + s[-n // 2 :]

    user = DESCRIBE_USER.format(
        file=file or "?",
        page=page if page is not None else "?",
        domain=domain or "?",
        context_before=_clip(context_before) or "(none)",
        table_body=body or "(empty)",
        context_after=_clip(context_after) or "(none)",
    )
    messages = [
        {"role": "system", "content": DESCRIBE_SYSTEM},
        {"role": "user", "content": user},
    ]

    def _call(timeout: float) -> str:
        import os

        import litellm

        kwargs: dict = {}
        base = os.environ.get("OPENAI_BASE_URL")
        m = model
        if base:
            kwargs["api_base"] = base
            m = f"openai/{model.removeprefix('openai/')}"
        elif "/" not in model:
            m = f"openai/{model}"
        resp = litellm.completion(
            model=m, messages=messages, temperature=0, timeout=timeout, **kwargs
        )
        return resp.choices[0].message.content or ""

    try:
        raw = complete(messages) if complete is not None else _call(90)
        return parse_description_json(raw)
    except Exception as e:
        err = repr(e)
        if complete is None and ("Timeout" in err or "timeout" in err or "RateLimit" in err):
            try:
                import time

                time.sleep(1.5)
                return parse_description_json(_call(150))
            except Exception as e2:
                print(
                    f"      LLM table describe failed after retry ({e2!r}); using heuristic",
                    flush=True,
                )
        else:
            print(f"      LLM table describe failed ({err}); using heuristic", flush=True)
        return describe_table_heuristic(
            table_body, file=file, page=page, domain=domain
        )


def enrich_table_strings(
    table_body: str,
    *,
    file: str = "",
    page: int | None = None,
    domain: str = "",
    context_before: str = "",
    context_after: str = "",
    use_llm: bool = False,
    model: str = "google/gemma-3-27b-it",
) -> tuple[str, str, TableDescription]:
    """Return (embed_text, payload_text, description)."""
    if use_llm:
        desc = describe_table_with_llm(
            table_body,
            file=file,
            page=page,
            domain=domain,
            context_before=context_before,
            context_after=context_after,
            model=model,
        )
    else:
        desc = describe_table_heuristic(
            table_body, file=file, page=page, domain=domain
        )
    embed = format_embed_text(desc, file=file, page=page)
    payload = format_payload(desc, table_body)
    return embed, payload, desc
