"""Estimate token sizes of corpus text at different structural granularities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag.parse import Block, TreeNode, section_depth


@dataclass
class Chunk:
    level: str
    tokens: int
    chars: int
    file: str
    label: str
    page: int | None = None


def count_tokens(text: str, encoding=None) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _node_text(node: TreeNode) -> str:
    parts: list[str] = []
    for b in node.blocks:
        if b.kind == "TableItem":
            parts.append(b.text[:2000])
        else:
            parts.append(b.text)
    for child in node.children:
        parts.append(_node_text(child))
    return "\n".join(p for p in parts if p)


def _pages_of(node: TreeNode) -> list[int]:
    pages: set[int] = set()
    for b in node.blocks:
        if b.page:
            pages.add(b.page)
    for child in node.children:
        pages.update(_pages_of(child))
    return sorted(pages)


def _walk(node: TreeNode):
    yield node
    for child in node.children:
        yield from _walk(child)


def chunks_for_document(
    file_rel: str,
    blocks: list[Block],
    tree: TreeNode,
    encoding=None,
) -> list[Chunk]:
    """Emit estimated chunks at several hierarchy levels for one document."""
    out: list[Chunk] = []

    def add(level: str, text: str, label: str, page: int | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        out.append(
            Chunk(
                level=level,
                tokens=count_tokens(text, encoding),
                chars=len(text),
                file=file_rel,
                label=label,
                page=page,
            )
        )

    # full document
    full = "\n".join(b.text for b in blocks if b.kind != "TableItem" or b.text)
    add("full_doc", full, file_rel)

    # page
    by_page: dict[int, list[str]] = defaultdict(list)
    for b in blocks:
        if b.page is None:
            continue
        by_page[b.page].append(b.text)
    for page, parts in sorted(by_page.items()):
        add("page", "\n".join(parts), f"{file_rel}#p{page}", page=page)

    # numbered hierarchy from tree
    for node in _walk(tree):
        if node.ref is None:
            continue
        text = _node_text(node)
        pages = _pages_of(node)
        page = pages[0] if pages else None
        depth = section_depth(node.ref)
        if depth == 1:
            add("section", text, f"§{node.ref}", page=page)
        elif depth == 2:
            add("clause", text, f"§{node.ref}", page=page)
        elif depth >= 3:
            add("subclause", text, f"§{node.ref}", page=page)

    # paragraph / atomic Docling block (skip huge table dumps)
    for b in blocks:
        if b.kind == "TableItem":
            continue
        label = f"§{b.section_ref}" if b.section_ref else f"#{b.idx}"
        add("paragraph", b.text, label, page=b.page)

    return out


LEVELS = [
    ("full_doc", "Full document", "Entire PDF as one chunk"),
    ("page", "Page", "All Docling text on a single page"),
    ("section", "Section (§N)", "Top-level numbered section + descendants (depth 1)"),
    ("clause", "Clause (§N.M)", "Second-level section + descendants (depth 2)"),
    ("subclause", "Sub-clause (§N.M.K+)", "Depth ≥ 3 numbered unit + descendants"),
    ("paragraph", "Paragraph / item", "Single Docling text/list item"),
]


def summarize(values: list[int]) -> dict:
    if not values:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    xs = sorted(values)
    n = len(xs)

    def pct(p: float) -> int:
        if n == 1:
            return xs[0]
        i = (n - 1) * p / 100.0
        lo = int(i)
        hi = min(lo + 1, n - 1)
        frac = i - lo
        return int(round(xs[lo] * (1 - frac) + xs[hi] * frac))

    return {
        "n": n,
        "min": xs[0],
        "max": xs[-1],
        "mean": round(sum(xs) / n, 1),
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }
