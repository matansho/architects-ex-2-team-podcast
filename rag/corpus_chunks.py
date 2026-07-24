"""Collect VectorChunks from the Harel corpus (PDF policies + TXT pages)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.chunk import chunk_blocks
from rag.chunk_stats import count_tokens
from rag.parse import build_tree, docling_to_blocks
from rag.table_describe import enrich_table_strings, looks_like_table
from rag.txt import chunk_txt_file


@dataclass
class CorpusChunk:
    """One indexable unit with stable id + citation metadata.

    text:       generation payload (full table body for tables).
    embed_text: optional retrieval string; when set, embed this instead of text
                (used for table descriptions — see rag.table_describe).
    """

    id: str
    source_type: str  # pdf | txt
    domain: str
    file: str  # path relative to corpus/, e.g. car/pages/faq.txt
    page: int | None
    kind: str
    label: str
    text: str
    tokens: int
    section_ref: str | None = None
    pages: list[int] = field(default_factory=list)
    embed_text: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def text_for_embedding(self) -> str:
        return (self.embed_text if self.embed_text else self.text) or ""


def _maybe_enrich_table(
    chunk: CorpusChunk,
    *,
    use_llm_tables: bool = False,
    context_before: str = "",
    context_after: str = "",
) -> CorpusChunk:
    """If chunk looks like a table, set embed_text (description) + payload text."""
    if chunk.embed_text or not looks_like_table(chunk.text):
        return chunk
    embed, payload, _desc = enrich_table_strings(
        chunk.text,
        file=chunk.file,
        page=chunk.page,
        domain=chunk.domain,
        context_before=context_before,
        context_after=context_after,
        use_llm=use_llm_tables,
    )
    chunk.embed_text = embed
    chunk.text = payload
    chunk.tokens = count_tokens(payload)
    if chunk.kind and "table" not in chunk.kind:
        chunk.kind = f"table_{chunk.kind}"
    return chunk


def _enrich_file_tables(
    file_chunks: list[CorpusChunk],
    *,
    use_llm_tables: bool = False,
) -> None:
    """Describe tables using neighboring chunk text (pre-enrichment bodies)."""
    raw = [c.text for c in file_chunks]
    for i, cc in enumerate(file_chunks):
        if cc.embed_text or not looks_like_table(cc.text):
            continue
        before = "\n".join(raw[j] for j in range(max(0, i - 2), i))
        after = "\n".join(raw[j] for j in range(i + 1, min(len(raw), i + 3)))
        if use_llm_tables:
            print(f"    describe table {cc.id} (p={cc.page})…", flush=True)
        _maybe_enrich_table(
            cc,
            use_llm_tables=use_llm_tables,
            context_before=before,
            context_after=after,
        )


def _domain_of(rel: Path) -> str:
    return rel.parts[0] if rel.parts else ""


def collect_txt_chunks(
    corpus: Path,
    *,
    target_tokens: int = 500,
    max_tokens: int = 700,
    skip_low_quality: bool = True,
) -> list[CorpusChunk]:
    out: list[CorpusChunk] = []
    for path in sorted(corpus.glob("*/pages/*.txt")):
        rel = path.relative_to(corpus)
        cleaned, chunks = chunk_txt_file(
            path,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            skip_low_quality=skip_low_quality,
        )
        if skip_low_quality and cleaned.low_quality:
            continue
        for c in chunks:
            out.append(
                CorpusChunk(
                    id=f"txt:{rel.as_posix()}#{c.id}",
                    source_type="txt",
                    domain=_domain_of(rel),
                    file=rel.as_posix(),
                    page=None,
                    kind=c.kind,
                    label=c.label,
                    text=c.text,
                    tokens=c.tokens,
                    section_ref=c.section_ref,
                    pages=[],
                )
            )
    return out


def _pdf_cache_path(cache_dir: Path, rel: Path) -> Path:
    # Keep one file per PDF; nested dirs mirror corpus layout.
    return cache_dir / rel.with_suffix(".chunks.json")


def collect_pdf_chunks(
    corpus: Path,
    *,
    target_tokens: int = 500,
    max_tokens: int = 700,
    limit: int | None = None,
    paths: list[Path] | None = None,
    cache_dir: Path | None = None,
    describe_tables: bool = True,
    use_llm_tables: bool = False,
) -> list[CorpusChunk]:
    """Parse PDFs with Docling and chunk. Slow — caches per-PDF when cache_dir set.

    describe_tables: attach heuristic (or LLM) table descriptions → embed_text.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )

    opts = PdfPipelineOptions()
    opts.generate_page_images = False
    opts.do_ocr = False  # digital PDFs; OCR is slow and rarely needed here
    # Apple Silicon: use MPS when available; use all logical cores for CPU parts.
    import os

    import torch

    n_threads = os.cpu_count() or 4
    device = AcceleratorDevice.AUTO
    if torch.backends.mps.is_available():
        device = AcceleratorDevice.MPS
    elif torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
    opts.accelerator_options = AcceleratorOptions(
        num_threads=n_threads,
        device=device,
    )
    print(
        f"  Docling accelerator: device={device.value} threads={n_threads}",
        flush=True,
    )
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    pdfs = paths if paths is not None else sorted(corpus.glob("*/files/*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out: list[CorpusChunk] = []
    n_cached = 0
    for i, path in enumerate(pdfs):
        rel = path.relative_to(corpus)
        cache_path = _pdf_cache_path(cache_dir, rel) if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    payload.get("target_tokens") == target_tokens
                    and payload.get("max_tokens") == max_tokens
                    and bool(payload.get("describe_tables", False)) == describe_tables
                    and bool(payload.get("use_llm_tables", False)) == use_llm_tables
                ):
                    file_chunks = [CorpusChunk(**row) for row in payload["chunks"]]
                    out.extend(file_chunks)
                    n_cached += 1
                    print(
                        f"  [{i + 1}/{len(pdfs)}] {rel} (cache, {len(file_chunks)} chunks)",
                        flush=True,
                    )
                    continue
            except Exception as e:
                print(f"  [{i + 1}/{len(pdfs)}] {rel} cache unreadable ({e}), reparse", flush=True)

        print(f"  [{i + 1}/{len(pdfs)}] {rel}", flush=True)
        try:
            doc = conv.convert(str(path)).document
        except Exception as e:
            print(f"    FAIL: {e}", flush=True)
            continue
        blocks = docling_to_blocks(doc)
        build_tree(blocks)
        chunks = chunk_blocks(blocks, target_tokens=target_tokens, max_tokens=max_tokens)
        file_chunks: list[CorpusChunk] = []
        for c in chunks:
            page = c.pages[0] if c.pages else None
            cc = CorpusChunk(
                id=f"pdf:{rel.as_posix()}#{c.id}",
                source_type="pdf",
                domain=_domain_of(rel),
                file=rel.as_posix(),
                page=page,
                kind=c.kind,
                label=c.label,
                text=c.text,
                tokens=c.tokens,
                section_ref=c.section_ref,
                pages=list(c.pages),
            )
            file_chunks.append(cc)
        if describe_tables:
            _enrich_file_tables(file_chunks, use_llm_tables=use_llm_tables)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "target_tokens": target_tokens,
                        "max_tokens": max_tokens,
                        "describe_tables": describe_tables,
                        "use_llm_tables": use_llm_tables,
                        "chunks": [c.to_dict() for c in file_chunks],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        out.extend(file_chunks)
        print(f"    → {len(file_chunks)} chunks", flush=True)

    if n_cached:
        print(f"  reused {n_cached}/{len(pdfs)} PDFs from cache", flush=True)
    return out
