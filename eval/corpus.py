"""Resolve cited corpus files/pages to text for citation judging."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

MAX_PAGE_CHARS = 6000
MAX_TABLE_CHARS_PER_PAGE = 8000


def normalize_path(path: str) -> str:
    p = path.strip().replace("\\", "/")
    if p.startswith("corpus/"):
        p = p[len("corpus/") :]
    p = p.replace(".aspx.txt", ".txt")
    return p


@dataclass
class ResolvedCitation:
    file: str
    page: int | None
    text: str | None
    error: str | None = None
    table_payloads: list[str] | None = None

    @property
    def ok(self) -> bool:
        if self.error is not None:
            return False
        if (self.text or "").strip():
            return True
        return bool(self.table_payloads and any((t or "").strip() for t in self.table_payloads))

    def text_for_judge(self) -> str:
        """Page extract plus any index table payloads for this location."""
        parts: list[str] = []
        if self.text and self.text.strip():
            parts.append(self.text.strip())
        if self.table_payloads:
            joined = "\n\n".join(t.strip() for t in self.table_payloads if t.strip())
            if joined:
                parts.append("--- Tables from index for this page ---\n" + joined)
        return "\n\n".join(parts)

@lru_cache(maxsize=512)
def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=256)
def _pdf_page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def _read_pdf_page(path: Path, page: int | None) -> tuple[str | None, str | None]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return None, f"unreadable PDF: {exc}"

    if not reader.pages:
        return None, "PDF has no pages"

    if page is None:
        chunks: list[str] = []
        for i, pg in enumerate(reader.pages[:5], start=1):
            chunks.append(f"[page {i}]\n{pg.extract_text() or ''}")
        text = "\n\n".join(chunks)
        if len(reader.pages) > 5:
            text += f"\n\n[truncated: {len(reader.pages)} pages total]"
        return text[:MAX_PAGE_CHARS], None

    if page < 1 or page > len(reader.pages):
        return None, f"page {page} out of range (1-{len(reader.pages)})"

    text = reader.pages[page - 1].extract_text() or ""
    if not text.strip():
        # Still return a handle so index table payloads can rescue the page.
        return "", f"page {page} is empty"
    return text[:MAX_PAGE_CHARS], None


def resolve_citation(corpus_root: Path, citation: dict) -> ResolvedCitation:
    raw_file = citation.get("file", "")
    page = citation.get("page")
    rel = normalize_path(raw_file)
    if not rel:
        return ResolvedCitation(raw_file, page, None, "missing file path")

    path = corpus_root / rel
    if not path.exists():
        return ResolvedCitation(rel, page, None, f"file not found: {rel}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _read_txt(path)
        if not text.strip():
            return ResolvedCitation(rel, page, None, "empty text file")
        return ResolvedCitation(rel, page, text[:MAX_PAGE_CHARS])

    if suffix == ".pdf":
        text, err = _read_pdf_page(path, page)
        if err and "empty" not in err:
            return ResolvedCitation(rel, page, None, err)
        return ResolvedCitation(rel, page, text or "", err)

    return ResolvedCitation(rel, page, None, f"unsupported file type: {suffix}")


def resolve_citations(corpus_root: Path, citations: list[dict] | None) -> list[ResolvedCitation]:
    return [resolve_citation(corpus_root, c) for c in (citations or [])]


def load_table_payloads_by_location(
    index_dir: Path | str = "data/index",
) -> dict[tuple[str, int | None], list[str]]:
    """Map (normalized file, page) → table chunk payloads from the index.

    Uses `text` (generator payload), not `embed_text`.
    """
    path = Path(index_dir) / "vectors.jsonl"
    if not path.is_file():
        return {}

    by_loc: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            loc = row.get("location") or {}
            kind = str(loc.get("kind") or "")
            if not kind.startswith("table"):
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            file = normalize_path(str(loc.get("file") or ""))
            if not file:
                continue
            page = loc.get("page")
            if page is not None:
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = None
            by_loc[(file, page)].append(text)
    return dict(by_loc)


def enrich_resolved_with_tables(
    resolved: list[ResolvedCitation],
    table_by_loc: dict[tuple[str, int | None], list[str]] | None,
    *,
    max_chars_per_page: int = MAX_TABLE_CHARS_PER_PAGE,
) -> list[ResolvedCitation]:
    """Attach index table payloads to resolved citations (mutates and returns)."""
    if not table_by_loc:
        return resolved
    for r in resolved:
        if r.error and "empty" not in (r.error or ""):
            continue
        payloads = table_by_loc.get((normalize_path(r.file), r.page)) or []
        if not payloads:
            continue
        kept: list[str] = []
        used = 0
        for p in payloads:
            if used >= max_chars_per_page:
                break
            chunk = p[: max_chars_per_page - used]
            kept.append(chunk)
            used += len(chunk)
        r.table_payloads = kept
        # Table payloads can stand in for an empty pypdf extract.
        if r.error and "empty" in r.error and kept:
            r.error = None
    return resolved
