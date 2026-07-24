"""Clean and chunk Harel web-scrape TXT pages (corpus/*/pages/*.txt).

Unlike policy PDFs, these are single-line HTML dumps: title | brand, skip-nav,
breadcrumb, body, then a huge site footer. There is no § hierarchy and no page
numbers (citations use file only, page=null).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from rag.chunk import VectorChunk
from rag.chunk_stats import count_tokens

# Site chrome after the skip-nav anchors
_CHROME_PREFIXES = (
    "דלג לתוכן ראשי",
    "דלג לתפריט תחתון",
    "פעולות נפוצות",
    "ביטוח חיסכון ופנסיה",
    "הראל פיננסים",
    "משכנתא והלוואה",
    "חיפוש",
    "אזור אישי",
)

# Footer / secondary-nav starts (present on nearly all scrapes)
_FOOTER_MARKERS = (
    "הצהרת נגישות",
    "תנאי שימוש",
    "פורטלים מקצועיים",
    "המרכז לתכנון כלכלי מתקדם פיננסים והשקעות",
)

# Leading breadcrumb: "ביטוח ביטוח רכב המוצרים שלנו …"
_BREADCRUMB_RE = re.compile(
    r"^(?:ביטוח\s+){1,3}"
    r"(?:דירה ורכוש|רכב|בריאות|שיניים|נסיעות לחו\"?ל|סיעודי|חיים|"
    r"עסק|משכנתא|תאונות אישיות|מחלות|אובדן כושר|חיסכון)?"
    r"\s*(?:כדאי לדעת|המוצרים שלנו|שירות למבוטחים[^\s]*)?\s*"
)

_IE_JUNK = "דפדפן ה- Explorer כבר לא עובד"

# Soft split points when packing long pages (used via re.split lookbehind)


@dataclass
class CleanedTxt:
    path: Path
    title: str
    body: str
    raw_chars: int
    body_chars: int
    low_quality: bool
    note: str = ""


def clean_web_txt(raw: str) -> tuple[str, str, bool, str]:
    """Return (title, body, low_quality, note)."""
    raw_chars = len(raw or "")
    text = html.unescape(raw or "")
    for ch in ("\xa0", "\u200b", "\u200f", "\u200e", "\u200c"):
        text = text.replace(ch, " " if ch == "\xa0" else "")
    text = re.sub(r"\s+", " ", text).strip()

    title = ""
    if "|" in text[:200]:
        title = text.split("|", 1)[0].strip()
        text = text.split("|", 1)[1]
        for m in ("דלג לתוכן ראשי", "דלג לתפריט תחתון"):
            if m in text:
                text = text.split(m, 1)[-1]
                break

    body = text.lstrip()
    changed = True
    while changed:
        changed = False
        for prefix in _CHROME_PREFIXES:
            if body.startswith(prefix):
                body = body[len(prefix) :].lstrip()
                changed = True

    cut = len(body)
    for marker in _FOOTER_MARKERS:
        i = body.find(marker)
        if i != -1 and i > 80:
            cut = min(cut, i)
    body = body[:cut].strip()
    body = _BREADCRUMB_RE.sub("", body).lstrip()

    # Collapse duplicated H1 (title repeated twice at start)
    if title and body.startswith(title):
        rest = body[len(title) :].lstrip()
        if rest.startswith(title):
            body = title + " " + rest[len(title) :].lstrip()

    low = False
    note = ""
    if _IE_JUNK in body or _IE_JUNK in (raw or ""):
        low = True
        note = "IE deprecation interstitial / thin aspx page"
    elif len(body) < 120:
        low = True
        note = "very little body after chrome strip"

    return title, body, low, note


def parse_txt_file(path: Path) -> CleanedTxt:
    raw = path.read_text(encoding="utf-8")
    title, body, low, note = clean_web_txt(raw)
    return CleanedTxt(
        path=path,
        title=title,
        body=body,
        raw_chars=len(raw),
        body_chars=len(body),
        low_quality=low,
        note=note,
    )


def _segment_body(body: str) -> list[str]:
    """Split cleaned body into soft segments at sentence/question boundaries."""
    body = (body or "").strip()
    if not body:
        return []
    parts = re.split(r"(?<=[?!.])\s+", body)
    return [p.strip() for p in parts if p.strip()]


def chunk_txt(
    cleaned: CleanedTxt,
    *,
    target_tokens: int = 500,
    max_tokens: int = 700,
    encoding=None,
    skip_low_quality: bool = False,
) -> list[VectorChunk]:
    """Pack cleaned web text toward target_tokens (no § hierarchy, no pages)."""
    if skip_low_quality and cleaned.low_quality:
        return []
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

    segments = _segment_body(cleaned.body)
    if not segments:
        return []

    pieces: list[tuple[str, list[int]]] = []  # text, segment indices
    buf: list[str] = []
    buf_idx: list[int] = []
    buf_tok = 0

    def flush() -> None:
        nonlocal buf, buf_idx, buf_tok
        if not buf:
            return
        pieces.append((" ".join(buf).strip(), list(buf_idx)))
        buf, buf_idx, buf_tok = [], [], 0

    for i, seg in enumerate(segments):
        st = count_tokens(seg, encoding)
        if st > max_tokens:
            flush()
            # hard-split oversized segment; shrink slice until under max
            start = 0
            while start < len(seg):
                room = max_tokens
                # estimate chars for this budget, then back off if still over
                approx = max(80, int(len(seg) * room / max(st, 1)))
                end = min(len(seg), start + approx)
                piece = seg[start:end].strip()
                while piece and count_tokens(piece, encoding) > max_tokens and end > start + 40:
                    end = start + max(40, int((end - start) * 0.85))
                    piece = seg[start:end].strip()
                if piece:
                    pieces.append((piece, [i]))
                if end <= start:
                    break
                start = end
            continue
        if buf and (
            (buf_tok + st > target_tokens and buf_tok >= target_tokens * 0.4)
            or buf_tok + st > max_tokens
        ):
            flush()
        buf.append(seg)
        buf_idx.append(i)
        buf_tok += st
    flush()

    label_base = cleaned.title or cleaned.path.stem
    chunks: list[VectorChunk] = []
    for n, (text, idxs) in enumerate(pieces):
        if not text:
            continue
        chunks.append(
            VectorChunk(
                id=n,
                kind="txt_pack",
                label=label_base if n == 0 else f"{label_base} #{n + 1}",
                text=text,
                tokens=count_tokens(text, encoding),
                section_ref=None,
                block_indices=idxs,
                pages=[],
            )
        )
    return chunks


def chunk_txt_file(
    path: Path,
    *,
    target_tokens: int = 500,
    max_tokens: int = 700,
    skip_low_quality: bool = False,
) -> tuple[CleanedTxt, list[VectorChunk]]:
    cleaned = parse_txt_file(path)
    return cleaned, chunk_txt(
        cleaned,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        skip_low_quality=skip_low_quality,
    )
