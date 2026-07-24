"""Parse Docling items into a section-number hierarchy.

Docling's visual ``level`` is unreliable for Hebrew legal PDFs (e.g. §3 and
§2.3 both appear as ListItem level 2). We rebuild hierarchy from section
numbers extracted from item text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Numbers often appear at end after OCR/RTL: "השירותים . 3" or mid: ". 3.1.1 . בגוף"
SECTION_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)*)(?![\d.])")
# Compact list markers with no space after the dot: "; .2.1" / ".2.1" (Docling)
COMPACT_SECTION_RE = re.compile(r"(?:^|[\s;:\-])\.(\d+(?:\.\d+)*)(?![\d.])")

# Cross-ref phrases that mention other sections (prefer not to treat as self-ref)
CROSS_REF_HINTS = ("עד", "סעיפים", "כמפורט", "לעיל", "להלן")

# Docling often merges a numbered title with the next body line:
# ".  סייגים מיוחדים לפרק ראשון 2 ואלא המבטח..."
_MERGED_HEADER_START = re.compile(r"^[\.\:\-]\s+\S")


@dataclass
class Block:
    idx: int
    kind: str
    page: int | None
    text: str
    docling_level: int
    section_ref: str | None = None
    section_depth: int = 0
    inherited: bool = False  # True if section_ref came from previous block
    # PDF coords (Docling BOTTOMLEFT): (page, l, t, r, b)
    bboxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


@dataclass
class TreeNode:
    ref: str | None
    depth: int
    blocks: list[Block] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)

    def title(self) -> str:
        if not self.blocks:
            return self.ref or "(root)"
        first = self.blocks[0].text.strip()
        first = re.sub(r"\s+", " ", first)
        return first[:80] + ("…" if len(first) > 80 else "")


def section_depth(ref: str | None) -> int:
    if not ref:
        return 0
    return len(ref.split("."))


def parent_ref(ref: str) -> str | None:
    parts = ref.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def extract_section_ref(text: str) -> str | None:
    """Pick the item's own section number from OCR'd Hebrew text.

    Heuristics (Hebrew legal OCR):
    - Prefer dotted refs (``3.1.1``) over bare quantities (``12`` treatments).
    - Require section punctuation (``. 3`` / ``: 5``) *or* a clear end/start
      marker — rejects mid-text quantities like ``טון, 3.5 לרכב``.
    - Accept Docling merges: ``.  TITLE N body...`` where N sits mid-line.
    - Accept compact markers: ``.2.1`` (dot abutting the number).
    - Reject plan codes (``524``), dates (``9.2.2023``), leading-zero junk.
    """
    text = (text or "").strip()
    if not text:
        return None

    # (start, end, ref) — compact matches win over a bare digit at the same span
    found: dict[tuple[int, int], str] = {}
    for m in SECTION_RE.finditer(text):
        found[m.span()] = m.group(1)
    for m in COMPACT_SECTION_RE.finditer(text):
        # span of the digits only (group 1), not the leading list-dot
        found[m.span(1)] = m.group(1)

    if not found:
        return None

    scored: list[tuple[float, str]] = []
    first_dotted_punct_seen = False
    line_is_merged_header = bool(_MERGED_HEADER_START.match(text))

    for (start, end), ref in found.items():
        parts = ref.split(".")
        root = parts[0]
        dotted = "." in ref

        # Plan numbers / years / phone fragments / dates (9.2.2023)
        if len(root) >= 3 or any(len(p) >= 4 for p in parts):
            continue
        # OCR junk like ". 00"
        if root.startswith("0"):
            continue
        if section_depth(ref) > 5:
            continue

        before = text[max(0, start - 4) : start]
        after = text[end : min(len(text), end + 4)]
        has_section_punct = bool(re.search(r"[\.\:]\s*$", before))
        comma_before = bool(re.search(r",\s*$", before))
        at_start = start == 0 or not text[:start].strip()
        at_end = end >= len(text) - 3
        # Short OCR tail after number: ". 3.1.1 . בגוף" (punct already required)
        # or bare ref at EOL with a few trailing punctuation chars only
        tail = text[end:].strip()
        end_like = at_end or (not tail) or (
            has_section_punct
            and len(tail) <= 12
            and bool(re.match(r"^[\.\:\-\s\u0590-\u05FF]*$", tail))
        )
        # Quantity phrases: "עד 3.5 טון"
        qty_before = bool(re.search(r"עד\s*$", text[max(0, start - 4) : start]))

        # Merged title+body: ".  <Hebrew title> N <body...>"
        merged_header = (
            line_is_merged_header
            and not dotted
            and 8 <= start <= 100
            and len(tail) >= 12
            and bool(re.search(r"[\u0590-\u05FF]", text[:start]))
            and not comma_before
            and not qty_before
        )

        if comma_before or qty_before:
            continue

        # Section marker shape: punct before, start/end, or merged header.
        # Reject bare mid-text dotted numbers (tonnage, etc.).
        if not (has_section_punct or at_start or end_like or merged_header):
            continue

        # Mid-text without section punct still rejected even if short Hebrew tail
        if not has_section_punct and not at_start and not at_end and not merged_header:
            continue

        score = 0.0
        if dotted:
            score += 4.0
        if has_section_punct:
            score += 3.0
        if merged_header:
            score += 5.0

        # Self-number is usually the first dotted ". N.N" marker; later ones
        # are often cross-refs ("כמפורט בסעיף 5.3", "עד 3.1.6").
        if dotted and has_section_punct and not first_dotted_punct_seen:
            score += 6.0
            first_dotted_punct_seen = True

        if end_like:
            score += 2.0
        elif re.match(r"\s*[\.\:\,]\s*", after):
            score += 1.5
        else:
            score += 0.5

        # Cross-ref windows — demote (but not for merged headers: body often
        # continues with "ואלא" / "להלן" right after the number)
        if not merged_header:
            window = text[max(0, start - 30) : end + 20]
            if any(h in window for h in CROSS_REF_HINTS):
                score -= 5.0

        scored.append((score, ref))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _item_page(item) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return prov[0].page_no
    return None


def _item_bboxes(item) -> list[tuple[int, float, float, float, float]]:
    """Return (page, l, t, r, b) in Docling BOTTOMLEFT PDF coords."""
    out: list[tuple[int, float, float, float, float]] = []
    for p in getattr(item, "prov", None) or []:
        bb = getattr(p, "bbox", None)
        if bb is None:
            continue
        out.append((p.page_no, float(bb.l), float(bb.t), float(bb.r), float(bb.b)))
    return out


def _item_text(item, doc=None) -> str:
    text = getattr(item, "text", None)
    if text:
        return str(text)
    if hasattr(item, "export_to_markdown"):
        try:
            if doc is not None:
                return item.export_to_markdown(doc=doc) or ""
            return item.export_to_markdown() or ""
        except TypeError:
            try:
                return item.export_to_markdown() or ""
            except Exception:
                return ""
        except Exception:
            pass
    return ""


def _dataframe_to_text(df) -> str:
    """Compact table text for embedding / LLM context (prefer CSV over padded MD)."""
    if df is None:
        return ""
    try:
        # Drop fully empty columns/rows from sparse Docling grids
        cleaned = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if cleaned.empty:
            cleaned = df
        # Prefer CSV: Hebrew RTL markdown tables are often huge padded pipes
        csv = cleaned.to_csv(index=False).strip()
        if csv:
            return csv
    except Exception:
        pass
    try:
        return df.to_string(index=False).strip()
    except Exception:
        return str(df)


def _table_grid_text(item) -> str:
    """Last-resort dump of Docling TableData grid cells."""
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if not grid:
        return ""
    rows: list[str] = []
    try:
        for row in grid:
            cells = []
            for cell in row:
                val = getattr(cell, "text", None)
                if val is None:
                    val = str(cell) if cell is not None else ""
                val = str(val).strip()
                if val:
                    cells.append(val)
            if cells:
                rows.append(" | ".join(cells))
    except Exception:
        return ""
    return "\n".join(rows)


def _table_text(item, doc=None) -> str:
    """Full table serialization — never truncate; avoid empty '(table)' stubs."""
    caption = ""
    if hasattr(item, "caption_text"):
        try:
            caption = (item.caption_text(doc=doc) if doc is not None else item.caption_text()) or ""
        except TypeError:
            try:
                caption = item.caption_text() or ""
            except Exception:
                caption = ""
        except Exception:
            caption = ""
    caption = str(caption).strip()

    # 1) DataFrame → CSV (usually denser / more complete than markdown)
    df = None
    if hasattr(item, "export_to_dataframe"):
        try:
            df = item.export_to_dataframe(doc=doc) if doc is not None else item.export_to_dataframe()
        except TypeError:
            try:
                df = item.export_to_dataframe()
            except Exception:
                df = None
        except Exception:
            df = None
    body = _dataframe_to_text(df)

    # 2) Markdown export
    if not body.strip() and hasattr(item, "export_to_markdown"):
        try:
            body = (
                item.export_to_markdown(doc=doc) if doc is not None else item.export_to_markdown()
            ) or ""
        except TypeError:
            try:
                body = item.export_to_markdown() or ""
            except Exception:
                body = ""
        except Exception:
            body = ""

    # 3) Raw text / grid
    if not body.strip():
        body = getattr(item, "text", None) or ""
    if not str(body).strip():
        body = _table_grid_text(item)

    body = str(body).strip()
    if caption and body:
        return f"{caption}\n\n{body}"
    if caption:
        return caption
    return body


def docling_to_blocks(doc) -> list[Block]:
    """Convert Docling document items into Blocks with section refs."""
    blocks: list[Block] = []
    for idx, (item, level) in enumerate(doc.iterate_items()):
        kind = type(item).__name__
        if kind == "PictureItem":
            continue
        if kind == "TableItem":
            text = _table_text(item, doc=doc)
            if not text.strip():
                # Skip empty tables rather than indexing a useless stub
                continue
            blocks.append(
                Block(
                    idx=idx,
                    kind=kind,
                    page=_item_page(item),
                    text=text,
                    docling_level=level,
                    section_ref=None,
                    section_depth=0,
                    bboxes=_item_bboxes(item),
                )
            )
            continue

        text = _item_text(item, doc=doc).strip()
        if not text:
            continue

        ref = extract_section_ref(text)
        blocks.append(
            Block(
                idx=idx,
                kind=kind,
                page=_item_page(item),
                text=text,
                docling_level=level,
                section_ref=ref,
                section_depth=section_depth(ref),
                bboxes=_item_bboxes(item),
            )
        )

    # Inherit section for orphans that sit between numbered items
    _inherit_orphans(blocks)
    return blocks


def _inherit_orphans(blocks: list[Block]) -> None:
    """Attach unnumbered blocks to the current section context."""
    current: str | None = None
    for b in blocks:
        if b.kind == "TableItem":
            if current and not b.section_ref:
                b.section_ref = current
                b.section_depth = section_depth(current)
                b.inherited = True
            continue
        if b.section_ref:
            current = b.section_ref
        elif current:
            b.section_ref = current
            b.section_depth = section_depth(current)
            b.inherited = True


def build_tree(blocks: list[Block]) -> TreeNode:
    """Build a section tree from numbered blocks (order preserved)."""
    root = TreeNode(ref=None, depth=0)
    # Stack of (node) where node.depth == section_depth(node.ref)
    stack: list[TreeNode] = [root]

    for b in blocks:
        ref = b.section_ref
        if not ref:
            stack[-1].blocks.append(b)
            continue

        depth = section_depth(ref)

        # If this is a new numbered heading (not inherited), open/close nodes
        if b.inherited:
            # Content belonging to current section — stay under matching node
            _ensure_path(stack, root, ref)
            stack[-1].blocks.append(b)
            continue

        # Pop until parent can accept this depth
        while len(stack) > 1 and stack[-1].depth >= depth:
            # Same ref continuing? append to existing
            if stack[-1].ref == ref:
                break
            stack.pop()

        if stack[-1].ref == ref:
            stack[-1].blocks.append(b)
            continue

        # Ensure ancestors exist (e.g. 3.1.1 before we've seen 3.1 as own block)
        parent = parent_ref(ref)
        if parent:
            _ensure_path(stack, root, parent)

        # Pop again so we attach under correct parent
        while len(stack) > 1 and stack[-1].depth >= depth:
            stack.pop()
        while len(stack) > 1 and stack[-1].ref and not (
            ref == stack[-1].ref or ref.startswith(stack[-1].ref + ".")
        ):
            stack.pop()

        node = TreeNode(ref=ref, depth=depth, blocks=[b])
        stack[-1].children.append(node)
        stack.append(node)

    return root


def _ensure_path(stack: list[TreeNode], root: TreeNode, ref: str) -> None:
    """Make sure stack ends at a node for ``ref``, creating placeholders if needed."""
    # Find existing node for ref in tree
    found = _find_node(root, ref)
    if found:
        # Rebuild stack to that node
        path = _path_to(root, found)
        stack.clear()
        stack.extend(path)
        return

    # Create missing ancestors then the node
    parts = ref.split(".")
    for i in range(1, len(parts) + 1):
        partial = ".".join(parts[:i])
        existing = _find_node(root, partial)
        if existing:
            path = _path_to(root, existing)
            stack.clear()
            stack.extend(path)
            continue
        depth = i
        while len(stack) > 1 and stack[-1].depth >= depth:
            stack.pop()
        node = TreeNode(ref=partial, depth=depth)
        stack[-1].children.append(node)
        stack.append(node)


def _find_node(root: TreeNode, ref: str) -> TreeNode | None:
    if root.ref == ref:
        return root
    for child in root.children:
        hit = _find_node(child, ref)
        if hit:
            return hit
    return None


def _path_to(root: TreeNode, target: TreeNode) -> list[TreeNode]:
    path: list[TreeNode] = []

    def dfs(node: TreeNode) -> bool:
        path.append(node)
        if node is target:
            return True
        for child in node.children:
            if dfs(child):
                return True
        path.pop()
        return False

    dfs(root)
    return path


def format_tree(node: TreeNode, indent: int = 0) -> str:
    lines: list[str] = []
    if node.ref is not None:
        pages = sorted({b.page for b in node.blocks if b.page})
        page_s = f"p{','.join(map(str, pages))}" if pages else "p?"
        kinds = {b.kind for b in node.blocks}
        title = node.title()
        # Prefer a short label: ref + first non-inherited text snippet
        label = title
        for b in node.blocks:
            if not b.inherited:
                label = re.sub(r"\s+", " ", b.text.strip())[:70]
                break
        pad = "  " * indent
        lines.append(f"{pad}§{node.ref}  [{page_s}]  {label}")
    for child in node.children:
        lines.append(format_tree(child, indent + (0 if node.ref is None else 1)))
    return "\n".join(lines)


def parse_pdf(path: str | Path):
    """Run Docling + hierarchy rebuild. Returns (blocks, tree)."""
    from docling.document_converter import DocumentConverter

    path = Path(path)
    doc = DocumentConverter().convert(str(path)).document
    blocks = docling_to_blocks(doc)
    tree = build_tree(blocks)
    return blocks, tree
