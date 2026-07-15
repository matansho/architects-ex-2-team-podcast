"""Resolve cited corpus files/pages to text for citation judging."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

MAX_PAGE_CHARS = 6000


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

    @property
    def ok(self) -> bool:
        return self.error is None and bool((self.text or "").strip())


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
        return None, f"page {page} is empty"
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
        if err:
            return ResolvedCitation(rel, page, None, err)
        return ResolvedCitation(rel, page, text)

    return ResolvedCitation(rel, page, None, f"unsupported file type: {suffix}")


def resolve_citations(corpus_root: Path, citations: list[dict] | None) -> list[ResolvedCitation]:
    return [resolve_citation(corpus_root, c) for c in (citations or [])]
