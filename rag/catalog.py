"""Corpus catalog: product/document map for agent navigation.

Layer 1 of the "scheme of the database": one row per indexed file with domain,
doc type, human title, aliases, URL, and chunk count. Agents call
`list_products` / `search_catalog` instead of guessing paths; filtered
retrieval can restrict to `entry.file` (and later related PDFs).

Built deterministically from the index + corpus pages + manifest — no LLM
required. Enrich aliases later if needed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rag.route import CORPUS_DOMAINS

# Prefer local rebuild under data/; fall back to the shipped artifact.
_DEFAULT_LOCAL = Path("data/catalog.json")
_DEFAULT_SHIPPED = Path("artifacts/catalog.json")
DEFAULT_CATALOG_PATH = (
    _DEFAULT_LOCAL if _DEFAULT_LOCAL.is_file() else _DEFAULT_SHIPPED
)

# URL path segment → coarse doc role (pages only; PDFs use filename heuristics).
_URL_ROLE = (
    ("/policies/", "product"),
    ("/claim/", "claim"),
    ("/services/", "service"),
    ("/information/", "info"),
    ("/calculators/", "tool"),
)


@dataclass
class CatalogEntry:
    id: str
    domain: str
    doc_type: str
    title: str
    aliases: list[str] = field(default_factory=list)
    file: str = ""
    url: str | None = None
    n_chunks: int = 0
    pages: list[int | None] = field(default_factory=list)
    blurb: str = ""
    related_files: list[str] = field(default_factory=list)


@dataclass
class Catalog:
    version: int
    generated: str
    n_entries: int
    domains: list[str]
    entries: list[CatalogEntry]

    def by_id(self) -> dict[str, CatalogEntry]:
        return {e.id: e for e in self.entries}

    def by_file(self) -> dict[str, CatalogEntry]:
        return {e.file: e for e in self.entries if e.file}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slug_aliases(stem: str) -> list[str]:
    """Filename / URL slug → searchable aliases."""
    raw = stem.replace(".aspx", "").replace("_", "-")
    parts = [p for p in re.split(r"[-_]+", raw) if p and p.lower() not in {"txt", "pdf", "pages", "files"}]
    joined = " ".join(parts)
    out = []
    if joined:
        out.append(joined)
        out.append(joined.replace(" ", "-"))
    # common product tokens worth keeping alone
    for tok in parts:
        if len(tok) >= 4 and tok.isascii():
            out.append(tok)
    return out


def _title_from_page_text(text: str, fallback: str) -> str:
    t = _norm(text)
    if not t:
        return fallback
    # Scraped Harel pages: "Title | הראל ביטוח…"
    if "|" in t:
        left = t.split("|", 1)[0].strip()
        if 3 <= len(left) <= 120:
            return left
    # Otherwise first ~80 chars of readable text
    return t[:80].rstrip(" ,.-")


def _title_from_pdf_name(name: str) -> str:
    stem = Path(name).stem
    # Drop very common edition suffixes for display, keep them in aliases via slug.
    stem = re.sub(
        r"-?(מהדורת|מהדורה|בתוקף|באנגלית).*$",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    title = stem.replace("-", " ").replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title or Path(name).stem


def _doc_type(file_rel: str, url: str | None) -> str:
    base = Path(file_rel).name.lower()
    path = file_rel.replace("\\", "/")
    if path.endswith("/faq.txt") or base == "faq.txt":
        return "faq"
    if "/pages/" in path:
        u = (url or "").lower()
        for needle, role in _URL_ROLE:
            if needle in u:
                return role
        # policies.txt / hub pages
        if base in {"policies.txt", "information.txt", "services.txt"}:
            return "hub"
        if base.endswith(".aspx.txt"):
            return "tool"
        return "page"
    # PDFs / files
    if base.startswith("טופס") or "טופס-" in base:
        return "form"
    if any(k in base for k in ("פוליס", "תנאי-פוליס", "תנאי_פוליס")):
        return "policy"
    if "כתב-שירות" in base or "כתב_שירות" in base:
        return "service_letter"
    if "חוברת-כללים" in base or "מערכת-כללים" in base:
        return "claims_rules"
    if base.startswith("גילוי-נאות") or "מידע-מהותי" in base:
        return "disclosure"
    if base.startswith("הודעה-על") or "תביע" in base:
        return "claim_form"
    return "document"


def _entry_id(domain: str, file_rel: str) -> str:
    stem = Path(file_rel).stem.replace(".aspx", "")
    # stable, path-unique
    safe = re.sub(r"[^\w\-]+", "-", stem, flags=re.UNICODE).strip("-").lower()
    return f"{domain}/{safe}"[:120]


def _blurb(text: str, title: str, limit: int = 180) -> str:
    t = _norm(text)
    if title and t.startswith(title):
        t = t[len(title) :].lstrip(" |,-")
    # Drop nav chrome if still present
    for junk in ("דלג לתוכן ראשי", "דלג לתפריט", "פעולות נפוצות"):
        if junk in t:
            t = t.split(junk, 1)[-1]
    t = _norm(t)
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _related_by_token(
    entries: list[CatalogEntry],
) -> None:
    """Attach PDF files to product pages when they share a distinctive token.

    Conservative: only link when a product page alias token (≥5 chars, not a
    generic word) appears in a same-domain PDF title/aliases.
    """
    generic = {
        "harel",
        "ביטוח",
        "פוליסה",
        "פוליסת",
        "תנאי",
        "טופס",
        "מהדורת",
        "רכב",
        "דירה",
        "בריאות",
        "חיים",
        "pages",
        "files",
        "insurance",
        "policy",
        "comprehensive",
        "third",
        "party",
    }
    products = [e for e in entries if e.doc_type == "product"]
    docs = [e for e in entries if e.doc_type in {"policy", "form", "service_letter", "disclosure", "document", "claims_rules", "claim_form"}]
    for prod in products:
        tokens = []
        for a in [prod.title, *prod.aliases]:
            for tok in re.split(r"[\s\-_/]+", a.lower()):
                tok = tok.strip("'\"")
                if len(tok) >= 5 and tok not in generic and not tok.isdigit():
                    tokens.append(tok)
        tokens = list(dict.fromkeys(tokens))[:6]
        if not tokens:
            continue
        related = []
        for doc in docs:
            if doc.domain != prod.domain or doc.file == prod.file:
                continue
            hay = " ".join([doc.title, *doc.aliases, doc.file]).lower()
            if any(tok in hay for tok in tokens):
                related.append(doc.file)
        prod.related_files = related[:12]


def build_catalog(
    *,
    index_dir: Path = Path("data/index"),
    corpus_dir: Path = Path("corpus"),
    manifest_path: Path = Path("corpus/manifest.json"),
) -> Catalog:
    from rag.index_store import load_vectors_meta

    manifest: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    vectors = load_vectors_meta(index_dir)
    # file → chunk ids / pages
    file_chunks: dict[str, list] = defaultdict(list)
    file_pages: dict[str, set] = defaultdict(set)
    file_domain: dict[str, str] = {}
    for v in vectors:
        loc = v.location
        f = (loc.file or "").replace("\\", "/")
        if not f:
            continue
        # normalize to corpus-relative
        if f.startswith("corpus/"):
            f = f[len("corpus/") :]
        file_chunks[f].append(v.id)
        file_domain[f] = loc.domain or f.split("/", 1)[0]
        if loc.page is not None:
            file_pages[f].add(loc.page)

    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()

    for file_rel, chunk_ids in sorted(file_chunks.items()):
        domain = file_domain.get(file_rel) or file_rel.split("/", 1)[0]
        if domain not in CORPUS_DOMAINS:
            # keep orphans but tag domain from path
            pass
        url = manifest.get(file_rel)
        doc_type = _doc_type(file_rel, url)
        base = Path(file_rel).name
        stem = Path(file_rel).stem.replace(".aspx", "")

        title = stem.replace("-", " ")
        blurb = ""
        page_path = corpus_dir / file_rel
        if page_path.exists() and file_rel.endswith(".txt"):
            raw = page_path.read_text(encoding="utf-8", errors="ignore")
            title = _title_from_page_text(raw, fallback=title)
            blurb = _blurb(raw, title)
        elif file_rel.endswith(".pdf") or "/files/" in file_rel:
            title = _title_from_pdf_name(base)

        aliases = []
        aliases.extend(_slug_aliases(stem))
        # title itself + stripped brand suffix
        aliases.append(title)
        for suffix in (" | הראל ביטוח ופיננסים", " - הראל", " | הראל"):
            if title.endswith(suffix):
                aliases.append(title[: -len(suffix)].strip())
        # de-dupe, drop empties / duplicates of title
        clean_aliases = []
        seen_a = {title.lower()}
        for a in aliases:
            a = _norm(a)
            if not a or a.lower() in seen_a:
                continue
            if len(a) < 3:
                continue
            seen_a.add(a.lower())
            clean_aliases.append(a)

        eid = _entry_id(domain, file_rel)
        if eid in seen_ids:
            eid = f"{eid}-{len(seen_ids)}"
        seen_ids.add(eid)

        pages_sorted = sorted(p for p in file_pages[file_rel] if p is not None)
        entries.append(
            CatalogEntry(
                id=eid,
                domain=domain,
                doc_type=doc_type,
                title=_norm(title),
                aliases=clean_aliases[:16],
                file=file_rel,
                url=url,
                n_chunks=len(chunk_ids),
                pages=pages_sorted[:50],
                blurb=blurb,
            )
        )

    _related_by_token(entries)

    domains = sorted({e.domain for e in entries})
    return Catalog(
        version=1,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_entries=len(entries),
        domains=domains,
        entries=entries,
    )


def save_catalog(catalog: Catalog, path: Path = DEFAULT_CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": catalog.version,
        "generated": catalog.generated,
        "n_entries": catalog.n_entries,
        "domains": catalog.domains,
        "entries": [asdict(e) for e in catalog.entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> Catalog:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [CatalogEntry(**e) for e in data["entries"]]
    return Catalog(
        version=int(data.get("version", 1)),
        generated=str(data.get("generated", "")),
        n_entries=len(entries),
        domains=list(data.get("domains") or []),
        entries=entries,
    )


def list_products(
    catalog: Catalog,
    *,
    domain: str | None = None,
    doc_types: tuple[str, ...] = ("product",),
) -> list[CatalogEntry]:
    """Agent-facing: products (and optionally hubs) in a domain."""
    out = []
    for e in catalog.entries:
        if domain and e.domain != domain:
            continue
        if e.doc_type not in doc_types:
            continue
        out.append(e)
    return out


# Boilerplate tokens that fire on almost every insurance page — ignore in scoring.
_STOP = frozenset(
    {
        "ביטוח",
        "ביטוחי",
        "פוליסה",
        "פוליסת",
        "הראל",
        "טופס",
        "בקשה",
        "מידע",
        "מהותי",
        "שאלות",
        "נפוצות",
        "כיסוי",
        "כיסויים",
        "תנאים",
        "תנאי",
        "הצעה",
        "לקוח",
        "שירות",
        "אתר",
        "עמוד",
        "של",
        "על",
        "עם",
        "את",
        "או",
        "גם",
        "כל",
        "לא",
        "זה",
        "זו",
        "הוא",
        "היא",
        "אני",
        "יש",
        "מה",
        "איך",
        "כדי",
        "בין",
        "לבין",
        "האם",
        "רוצה",
        "בכלל",
        "אצלכם",
        "insurance",
        "policy",
        "harel",
        "form",
        "the",
        "and",
        "for",
        "with",
    }
)

_HEB_PREFIXES = ("ה", "ו", "ל", "ב")  # proclitics only; do not strip מ/ש (breaks מקיף→קיף)


def _query_tokens(q: str) -> list[str]:
    """Distinctive query tokens; optionally strip one Hebrew proclitic."""
    raw = [t for t in re.split(r"[\s,.;:!?\"'()/\-]+", q) if len(t) >= 2]
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _STOP or t.isdigit():
            continue
        variants = [t]
        for p in _HEB_PREFIXES:
            if t.startswith(p) and len(t) - len(p) >= 3:
                rest = t[len(p) :]
                if rest not in _STOP:
                    variants.append(rest)
                break
        for cand in variants:
            if cand in _STOP or cand in seen or len(cand) < 3:
                continue
            seen.add(cand)
            out.append(cand)
    if not out:
        out = [t for t in raw if len(t) >= 4 and t not in _STOP]
    return out


def search_catalog(
    catalog: Catalog,
    query: str,
    *,
    domain: str | None = None,
    doc_types: tuple[str, ...] | None = None,
    limit: int = 20,
) -> list[tuple[CatalogEntry, float]]:
    """Lightweight lexical match over title/aliases/file/blurb.

    Prefers distinctive tokens (product names, specific nouns) over boilerplate
    like "ביטוח" / "פוליסה". Longer token hits weigh more.
    """
    q = _norm(query).lower()
    if not q:
        return []
    tokens = _query_tokens(q)
    scored: list[tuple[CatalogEntry, float]] = []
    for e in catalog.entries:
        if domain and e.domain != domain:
            continue
        if doc_types and e.doc_type not in doc_types:
            continue
        title_l = e.title.lower()
        alias_blob = " ".join(e.aliases).lower()
        hay = " ".join([title_l, e.file.lower(), e.blurb.lower(), alias_blob])
        score = 0.0
        # Full-query containment is rare but strong (short product-name queries).
        if len(q) >= 6 and q in hay:
            score += 6.0
        for tok in tokens:
            w = 1.0 + min(2.0, (len(tok) - 2) * 0.25)  # longer → heavier
            if tok in title_l:
                score += 2.5 * w
            elif tok in alias_blob:
                score += 2.0 * w
            elif tok in hay:
                score += 0.75 * w
        if score > 0:
            scored.append((e, score))
    scored.sort(key=lambda x: (-x[1], x[0].domain, x[0].title))
    return scored[:limit]


def catalog_files_for_query(
    catalog: Catalog,
    query: str,
    *,
    domain: str | None = None,
    domains: list[str] | None = None,
    min_score: float = 6.0,
    limit: int = 6,
    include_related: bool = True,
    score_margin: float = 2.5,
) -> tuple[list[str], list[CatalogEntry], list[tuple[str, float]]]:
    """Pick corpus files for filtered retrieval from a catalog search.

    When product pages clear `min_score`, take that product cluster (plus
    nearby branded PDFs/pages). Otherwise fall back to any strong top cluster.
    Empty files → caller keeps the unfiltered path.

    `domain` restricts to one domain; `domains` restricts to a set (e.g. router
    output). If both are set, `domain` wins.
    """
    allow = None
    if domain:
        allow = {domain}
    elif domains:
        allow = set(domains)

    if domain:
        scored = search_catalog(
            catalog, query, domain=domain, limit=max(limit, 24)
        )
    elif allow:
        merged: dict[str, tuple[CatalogEntry, float]] = {}
        for d in sorted(allow):
            for e, s in search_catalog(
                catalog, query, domain=d, limit=max(limit, 24)
            ):
                prev = merged.get(e.id)
                if prev is None or s > prev[1]:
                    merged[e.id] = (e, s)
        scored = sorted(merged.values(), key=lambda x: -x[1])[: max(limit, 24)]
    else:
        scored = search_catalog(
            catalog, query, domain=None, limit=max(limit, 24)
        )
    debug = [(e.file, s) for e, s in scored]
    strong = [(e, s) for e, s in scored if s >= min_score]
    if not strong:
        return [], [], debug

    q_tokens = set(_query_tokens(_norm(query).lower()))
    products = [(e, s) for e, s in strong if e.doc_type == "product"]

    def _title_shares_distinctive(e: CatalogEntry) -> bool:
        title_l = e.title.lower()
        return any(len(t) >= 3 and t in title_l for t in q_tokens)

    if products:
        best_p = products[0][1]
        chosen = [(e, s) for e, s in products if s >= best_p - score_margin]
        product_hay = " ".join(e.title.lower() for e, _ in chosen)
        product_toks = {t for t in q_tokens if len(t) >= 4 and t in product_hay}
        for e, s in strong:
            if e.doc_type == "product":
                continue
            if e.doc_type not in {
                "document",
                "policy",
                "page",
                "disclosure",
                "form",
            }:
                continue
            if s < max(min_score, best_p - 6.0):
                continue
            title_l = e.title.lower()
            # Extra docs must share a distinctive token with the product hit
            # (blocks generic "הראל/ביטוח" PDFs).
            if product_toks and not any(t in title_l for t in product_toks):
                continue
            if not product_toks and not _title_shares_distinctive(e):
                continue
            chosen.append((e, s))
        seen_ids: set[str] = set()
        uniq: list[tuple[CatalogEntry, float]] = []
        for e, s in sorted(chosen, key=lambda x: -x[1]):
            if e.id in seen_ids:
                continue
            seen_ids.add(e.id)
            uniq.append((e, s))
        chosen = uniq[:limit]
    else:
        best = strong[0][1]
        chosen = [(e, s) for e, s in strong if s >= best - score_margin][:limit]

    files: list[str] = []
    seen: set[str] = set()
    matched: list[CatalogEntry] = []
    for e, _ in chosen:
        matched.append(e)
        for f in [e.file, *(e.related_files if include_related else [])]:
            if f and f not in seen:
                seen.add(f)
                files.append(f)
    return files, matched, debug


def idxs_for_files(
    vectors,
    files: list[str] | set[str] | tuple[str, ...],
) -> list[int]:
    """Corpus indices whose location.file matches one of `files` (corpus-relative)."""
    want = set()
    for f in files:
        f = str(f).replace("\\", "/").strip()
        if f.startswith("corpus/"):
            f = f[len("corpus/") :]
        if f:
            want.add(f)
    if not want:
        return []
    out: list[int] = []
    for i, v in enumerate(vectors):
        path = (getattr(v.location, "file", None) or "").replace("\\", "/")
        if path.startswith("corpus/"):
            path = path[len("corpus/") :]
        if path in want:
            out.append(i)
    return out


def catalog_summary_for_prompt(catalog: Catalog, *, max_products_per_domain: int = 8) -> str:
    """Compact text an agent system prompt can include (not the full JSON)."""
    lines = ["Corpus catalog (products by domain):"]
    for domain in catalog.domains:
        products = list_products(catalog, domain=domain)
        if not products:
            continue
        names = [p.title for p in products[:max_products_per_domain]]
        extra = len(products) - len(names)
        tail = f" (+{extra} more)" if extra > 0 else ""
        lines.append(f"- {domain}: " + "; ".join(names) + tail)
    return "\n".join(lines)
