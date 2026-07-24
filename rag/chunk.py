"""Structural chunking aimed at ~target_tokens per vector."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.chunk_stats import count_tokens
from rag.parse import Block, parent_ref, section_depth


@dataclass
class VectorChunk:
    id: int
    kind: str  # subclause | clause | clause_pack | subclause_pack | section_pack | orphan_pack | split
    label: str
    text: str
    tokens: int
    section_ref: str | None
    block_indices: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def _is_header(block: Block) -> bool:
    """Docling section headers (and short heading-like lines) should lead a chunk."""
    if block.kind == "SectionHeaderItem":
        return True
    text = (block.text or "").strip()
    if not text or len(text) > 90:
        return False
    # Own numbered title (not inherited body under a parent ref)
    if (
        block.kind in ("ListItem", "TextItem")
        and block.section_ref
        and not block.inherited
        and section_depth(block.section_ref) <= 3
    ):
        # Title shape: begins with ".  …" bullet. End-numbered body lines
        # ("נזק…; .2.1") are content — peeling them cascades clauses together.
        if re.match(r"^[\.\:\-]\s+\S", text):
            return True
        return False
    return False


def _unit_key(block: Block) -> tuple:
    """Grouping key before packing."""
    if block.kind == "TableItem":
        # keep with surrounding section context
        if block.section_ref and section_depth(block.section_ref) >= 2:
            return ("clause", block.section_ref)
        if block.section_ref:
            return ("section", block.section_ref)
        return ("orphan", block.page)
    ref = block.section_ref
    if not ref:
        return ("orphan", block.page)
    d = section_depth(ref)
    if d >= 3:
        return ("subclause", ref)
    if d == 2:
        return ("clause", ref)
    return ("section", ref)


def _is_descendant_ref(parent: str, child: str | None) -> bool:
    return bool(child and child.startswith(parent + "."))


def _group_token_count(blocks: list[Block], encoding) -> int:
    text = "\n".join(b.text for b in blocks if b.text)
    return count_tokens(text, encoding)


def _own_refs(blocks: list[Block]) -> set[str]:
    """Section numbers this group introduced (excludes inherited body)."""
    return {b.section_ref for b in blocks if b.section_ref and not b.inherited}


def _is_atomic_unit(blocks: list[Block]) -> bool:
    """True if the group is a single numbered unit (not an already-folded family)."""
    return len(_own_refs(blocks)) <= 1


def _fold_parent_children(
    groups: list[tuple[str, str | None, list[Block]]],
    max_tokens: int,
    encoding,
) -> list[tuple[str, str | None, list[Block]]]:
    """Absorb descendant clause/subclause groups into their parent when they fit.

    Keeps stems like ``§4.8`` ("…שנגרם:") together with ``§4.8.1`` / ``§4.8.2``.
    Greedy: take as many consecutive descendants as fit under ``max_tokens``.
    """
    out: list[tuple[str, str | None, list[Block]]] = []
    i = 0
    while i < len(groups):
        kind, ref, blist = groups[i]
        if kind in ("clause", "subclause") and ref and blist:
            combined = list(blist)
            combined_tok = _group_token_count(combined, encoding)
            j = i + 1
            while j < len(groups):
                ck, cr, cb = groups[j]
                if ck not in ("clause", "subclause") or not _is_descendant_ref(ref, cr):
                    break
                child_tok = _group_token_count(cb, encoding)
                if combined_tok + child_tok > max_tokens:
                    break
                combined.extend(cb)
                combined_tok += child_tok
                j += 1
            if j > i + 1:
                out.append((kind, ref, combined))
                i = j
                continue
        out.append((kind, ref, blist))
        i += 1
    return out


def _pack_small_siblings(
    groups: list[tuple[str, str | None, list[Block]]],
    target: int,
    encoding,
) -> list[tuple[str, str | None, list[Block]]]:
    """Pack consecutive small siblings toward ``target``.

    - Clauses/subclauses that share a parent (e.g. ``§2.1``+``§2.2``+``§2.3``)
    - Top-level sections (e.g. ``§5``+``§6``) when both are small

    Skips units already ≥ 50% of target, and already-folded parent+child families.
    """
    small_ceiling = max(1, int(target * 0.5))
    out: list[tuple[str, str | None, list[Block]]] = []
    i = 0
    while i < len(groups):
        kind, ref, blist = groups[i]
        if kind not in ("clause", "subclause", "section") or not ref or not blist:
            out.append(groups[i])
            i += 1
            continue

        tok = _group_token_count(blist, encoding)
        if tok >= small_ceiling or not _is_atomic_unit(blist):
            out.append(groups[i])
            i += 1
            continue

        if kind in ("clause", "subclause"):
            parent = parent_ref(ref)
            if parent is None:
                out.append(groups[i])
                i += 1
                continue

            def can_join(ck: str, cr: str | None, cb: list[Block]) -> bool:
                return (
                    ck in ("clause", "subclause")
                    and bool(cr)
                    and parent_ref(cr) == parent
                    and _is_atomic_unit(cb)
                )

            emit_kind = kind + "_pack"
        else:
            # Top-level §N siblings under the document root
            def can_join(ck: str, cr: str | None, cb: list[Block]) -> bool:
                return ck == "section" and bool(cr) and _is_atomic_unit(cb)

            # Keep-together kind so flush won't re-split on the next § header
            emit_kind = "section_pack"

        combined = list(blist)
        combined_tok = tok
        j = i + 1
        while j < len(groups):
            ck, cr, cb = groups[j]
            if not can_join(ck, cr, cb):
                break
            child_tok = _group_token_count(cb, encoding)
            if child_tok >= small_ceiling:
                break
            if combined_tok + child_tok > target:
                break
            combined.extend(cb)
            combined_tok += child_tok
            j += 1

        if j > i + 1:
            out.append((emit_kind, ref, combined))
            i = j
        else:
            out.append(groups[i])
            i += 1
    return out


def _attach_leading_headers(
    groups: list[tuple[str, str | None, list[Block]]],
) -> list[tuple[str, str | None, list[Block]]]:
    """Keep headers with the body that follows them.

    1. Lone ``§N`` header group → merge into following ``§N.M`` clause.
    2. Any group that *ends* on a header → move that header to the next group
       (fixes pack splits and section→section boundaries).
    """
    if not groups:
        return groups

    # Pass 1: merge header-only section groups into following clause/subclause
    merged: list[tuple[str, str | None, list[Block]]] = []
    i = 0
    while i < len(groups):
        kind, ref, blist = groups[i]
        if (
            i + 1 < len(groups)
            and kind == "section"
            and ref
            and blist
            and all(_is_header(b) for b in blist)
        ):
            nkind, nref, nlist = groups[i + 1]
            if nkind in ("clause", "subclause") and nref and (
                nref == ref or nref.startswith(ref + ".")
            ):
                merged.append((nkind, nref, blist + nlist))
                i += 2
                continue
        merged.append((kind, ref, blist))
        i += 1

    # Pass 2: peel trailing headers onto the next group
    for i in range(len(merged) - 1):
        kind, ref, blist = merged[i]
        moved: list[Block] = []
        while blist and _is_header(blist[-1]):
            moved.insert(0, blist.pop())
        if moved:
            nkind, nref, nlist = merged[i + 1]
            merged[i] = (kind, ref, blist)
            merged[i + 1] = (nkind, nref, moved + nlist)

    return [(k, r, b) for k, r, b in merged if b]


def _flush_group(
    kind: str,
    ref: str | None,
    blocks: list[Block],
    target: int,
    max_tokens: int,
    encoding,
) -> list[tuple[str, str | None, list[Block]]]:
    """Split a consecutive group into pack-sized pieces."""
    if not blocks:
        return []
    # Clause/subclause/sibling packs: keep together unless over max
    if kind in ("clause", "subclause", "clause_pack", "subclause_pack", "section_pack"):
        text = "\n".join(b.text for b in blocks)
        tok = count_tokens(text, encoding)
        if tok <= max_tokens:
            return [(kind, ref, blocks)]
        return _split_blocks(kind, ref, blocks, max_tokens, encoding)

    # Section / orphan (not yet sibling-packed): pack up to target
    return _pack_blocks(kind + "_pack", ref, blocks, target, max_tokens, encoding)


def _emit_pack(
    kind: str,
    ref: str | None,
    buf: list[Block],
) -> tuple[list[Block], list[Block]]:
    """Emit buf but keep a trailing header for the next pack."""
    if not buf:
        return [], []
    carry: list[Block] = []
    while buf and _is_header(buf[-1]):
        carry.insert(0, buf.pop())
    return buf, carry


def _pack_blocks(
    kind: str,
    ref: str | None,
    blocks: list[Block],
    target: int,
    max_tokens: int,
    encoding,
) -> list[tuple[str, str | None, list[Block]]]:
    out: list[tuple[str, str | None, list[Block]]] = []
    buf: list[Block] = []
    buf_tok = 0
    for b in blocks:
        bt = count_tokens(b.text, encoding)
        # Headers start a new pack (stay with following body)
        if buf and _is_header(b):
            emitted, carry = _emit_pack(kind, ref, buf)
            if emitted:
                out.append((kind, ref, emitted))
            buf = carry
            buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
        elif buf and buf_tok + bt > target and buf_tok >= target * 0.4:
            emitted, carry = _emit_pack(kind, ref, buf)
            if emitted:
                out.append((kind, ref, emitted))
            buf = carry
            buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
        if bt > max_tokens:
            if buf:
                emitted, carry = _emit_pack(kind, ref, buf)
                if emitted:
                    out.append((kind, ref, emitted))
                buf = carry
                buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
            out.extend(_split_blocks("split", ref, [b], max_tokens, encoding))
            continue
        buf.append(b)
        buf_tok += bt
    if buf:
        # Final pack may still end on a header if it's the last block — unavoidable
        out.append((kind, ref, buf))
    return out


def _split_blocks(
    kind: str,
    ref: str | None,
    blocks: list[Block],
    max_tokens: int,
    encoding,
) -> list[tuple[str, str | None, list[Block]]]:
    out: list[tuple[str, str | None, list[Block]]] = []
    buf: list[Block] = []
    buf_tok = 0
    for b in blocks:
        bt = count_tokens(b.text, encoding)
        if bt > max_tokens:
            if buf:
                emitted, carry = _emit_pack(kind, ref, buf)
                if emitted:
                    out.append((kind, ref, emitted))
                buf = carry
                buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
            # hard-split single oversized block by characters
            text = b.text
            approx = max(200, int(len(text) * max_tokens / max(bt, 1)))
            for i in range(0, len(text), approx):
                piece = Block(
                    idx=b.idx,
                    kind=b.kind,
                    page=b.page,
                    text=text[i : i + approx],
                    docling_level=b.docling_level,
                    section_ref=b.section_ref,
                    section_depth=b.section_depth,
                    inherited=b.inherited,
                    bboxes=b.bboxes if i == 0 else [],
                )
                out.append(("split", ref, [piece]))
            continue
        if buf and _is_header(b):
            emitted, carry = _emit_pack(kind, ref, buf)
            if emitted:
                out.append((kind, ref, emitted))
            buf = carry
            buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
        elif buf and buf_tok + bt > max_tokens:
            emitted, carry = _emit_pack(kind, ref, buf)
            if emitted:
                out.append((kind, ref, emitted))
            buf = carry
            buf_tok = sum(count_tokens(x.text, encoding) for x in buf)
        buf.append(b)
        buf_tok += bt
    if buf:
        out.append((kind, ref, buf))
    return out


def chunk_blocks(
    blocks: list[Block],
    *,
    target_tokens: int = 500,
    max_tokens: int = 700,
    encoding=None,
) -> list[VectorChunk]:
    """Build retrieval vectors: clause/subclause first (fold descendants under
    max; pack small siblings — including top-level sections — toward target),
    else pack section/orphan text."""
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

    # Group consecutive blocks with the same unit key
    groups: list[tuple[str, str | None, list[Block]]] = []
    cur_key = None
    cur_blocks: list[Block] = []
    for b in blocks:
        key = _unit_key(b)
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            kind, ref = cur_key[0], cur_key[1] if len(cur_key) > 1 else None
            if kind in ("orphan",):
                ref = None
            groups.append((kind, ref if kind != "orphan" else None, cur_blocks))
            cur_blocks = []
            cur_key = key
        cur_blocks.append(b)
    if cur_blocks and cur_key is not None:
        kind = cur_key[0]
        ref = cur_key[1] if kind != "orphan" and len(cur_key) > 1 else None
        groups.append((kind, ref, cur_blocks))

    groups = _attach_leading_headers(groups)
    groups = _fold_parent_children(groups, max_tokens, encoding)
    groups = _pack_small_siblings(groups, target_tokens, encoding)

    pieces: list[tuple[str, str | None, list[Block]]] = []
    for kind, ref, blist in groups:
        pieces.extend(_flush_group(kind, ref, blist, target_tokens, max_tokens, encoding))

    chunks: list[VectorChunk] = []
    for i, (kind, ref, blist) in enumerate(pieces):
        text = "\n".join(b.text for b in blist if b.text)
        if not text.strip():
            continue
        pages = sorted({b.page for b in blist if b.page})
        label = f"§{ref}" if ref else f"p{pages[0] if pages else '?'}"
        chunks.append(
            VectorChunk(
                id=i,
                kind=kind,
                label=label,
                text=text,
                tokens=count_tokens(text, encoding),
                section_ref=ref,
                block_indices=[b.idx for b in blist],
                pages=pages,
            )
        )
    return chunks


def chunk_color(chunk_id: int) -> str:
    """Distinct pastel-ish HSL color for overlays."""
    # Golden-angle spacing
    h = (chunk_id * 137.508) % 360
    return f"hsla({h:.1f}, 70%, 55%, 0.35)"


def chunk_border(chunk_id: int) -> str:
    h = (chunk_id * 137.508) % 360
    return f"hsl({h:.1f}, 70%, 40%)"
