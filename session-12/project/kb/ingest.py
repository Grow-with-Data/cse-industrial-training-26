"""Build the ShopWise knowledge base: load → chunk → embed → write.

    uv run python kb/ingest.py                      # the policies (plain) + past tickets
    uv run python kb/ingest.py --reset              # drop them first
    uv run python kb/ingest.py --scheme contextual  # v9: chunks with situating context
    uv run python kb/ingest.py --scheme parent      # v9: small children, whole-section parents
    uv run python kb/ingest.py --tickets            # v9: past tickets only

Run it twice and nothing changes: every chunk gets an id derived from WHERE it
came from (source, section, offset), so a second run overwrites each point with
itself instead of adding a copy. That is the whole of idempotency here, and the
script prints the point count so you can watch it hold.

WHAT GOES IN, AND FROM WHICH FILE
    ShopWise's policies     the .md: headings become `section` metadata. Their
                            .pdf twins exist for the loader lesson and are NEVER
                            indexed too: both would return every clause twice.
    Regulator documents     the .pdf, the only form they come in. `section` is
                            the page ("p.4"), and `authority` says "regulator", so
                            an answer can never pass federal law off as a
                            ShopWise promise.
    v9: past tickets        kb/past_tickets.yml joined to data/tickets.yml, one
                            document per ticket, in its OWN collection. `from` is
                            dropped before anything is embedded.

Every policy chunk leaves here carrying the metadata a citation needs:
    source · section · page (PDFs) · format · authority · start_index

v9: FOUR COLLECTIONS, ONE PER SPACE AND SCHEME (ADR-0010)
    policies__<space>               plain chunks (session 8's)
    policies-contextual__<space>    each chunk prefixed with one model-written
                                    sentence that situates it in its document
    policies-parent__<space>        small children for matching; search returns
                                    the whole section they came from
    tickets__<space>                past tickets
    A chunking scheme is part of what a vector means, exactly like the embedding
    model: two schemes in one collection would return the same clause twice.

CONTEXTUAL CHUNKS COST ONE MODEL CALL EACH
    ~70 calls for this corpus, once. The sentences are saved in kb/contexts.json
    (committed), keyed by chunk id, so a rebuild, or a participant without a
    key, re-embeds without calling the model again. Delete an entry to have it
    rewritten.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

# `python kb/ingest.py` puts kb/ on sys.path, not the project folder above it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (MarkdownHeaderTextSplitter,  # noqa: E402
                                      RecursiveCharacterTextSplitter)
from pypdf import PdfReader  # noqa: E402

from core.config import settings  # noqa: E402
from kb import store  # noqa: E402

# Fixed forever: change it and every id changes, so every point is duplicated.
NAMESPACE = uuid.UUID("5e1f0a4e-0000-4000-8000-000000000008")

AUTHORITY = {"ftc-prompt-delivery": "regulator", "cfr-16-part-700": "regulator"}
PDF_ONLY = ["ftc-prompt-delivery", "cfr-16-part-700"]

KB_DIR = Path(__file__).resolve().parent
PAST_TICKETS_FILE = KB_DIR / "past_tickets.yml"
CONTEXTS_FILE = KB_DIR / "contexts.json"


# ── loading the policies (session 8, unchanged) ──────────────────────────────

def load_markdown(path: Path) -> list[Document]:
    """One Document per `## ` section, the heading kept in the text and recorded
    in metadata: the section a citation will name."""
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")],
                                          strip_headers=False)
    docs = splitter.split_text(path.read_text(encoding="utf-8"))
    for d in docs:
        d.metadata.update(source=path.stem, format="md",
                          authority=AUTHORITY.get(path.stem, "shop"))
        d.metadata.setdefault("section", "preamble")
        d.metadata.pop("title", None)
    return docs


def clean_pdf_text(text: str) -> str:
    """Re-join words a two-column PDF hyphenated across a line ("con -\\nsumer").

    Cleaning is part of loading. The text a PDF hands back is the layout's
    text, not the author's; this is the one repair worth making here."""
    return re.sub(r"(\w) ?-\s*\n\s*(\w)", r"\1\2", text)


def load_pdf(path: Path) -> list[Document]:
    """One Document per page. `section` is the page, 1-based, as a person counts.

    pypdf directly, as LangChain's own knowledge-base tutorial does: the
    PyPDFLoader wrapper lives in langchain-community, which is being sunset."""
    return [Document(page_content=clean_pdf_text(page.extract_text()),
                     metadata=dict(source=path.stem, section=f"p.{n}", page=n, format="pdf",
                                   authority=AUTHORITY.get(path.stem, "shop")))
            for n, page in enumerate(PdfReader(path).pages, 1)]


def load_corpus() -> list[Document]:
    """Every document the knowledge base indexes, never both twins. One Document
    per markdown section or PDF page: v9's parent scheme returns exactly these."""
    folder = settings.POLICIES_DIR
    docs: list[Document] = []
    for md in sorted(folder.glob("*.md")):
        docs += load_markdown(md)
    for stem in PDF_ONLY:
        docs += load_pdf(folder / f"{stem}.pdf")
    return docs


def chunk(docs: list[Document], size: int | None = None,
          overlap: int | None = None) -> list[Document]:
    """Split into chunks of about `size` characters. `start_index` records where
    in its section each chunk began: part of its stable id."""
    size = size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              add_start_index=True)
    return splitter.split_documents(docs)


def chunk_id(doc: Document) -> str:
    m = doc.metadata
    return str(uuid.uuid5(NAMESPACE, f"{m['source']}#{m['section']}#{m['start_index']}"))


def parent_id(doc: Document) -> str:
    """v9: the section a chunk came from, as an id: `warranty-policy#2. How long…`."""
    return f"{doc.metadata['source']}#{doc.metadata['section']}"


# ── v9: the contextual scheme ────────────────────────────────────────────────

def whole_documents(sections: list[Document]) -> dict[str, str]:
    """Each source file's full text, rebuilt from its sections: what the model
    reads to situate one chunk."""
    full: dict[str, list[str]] = {}
    for d in sections:
        full.setdefault(d.metadata["source"], []).append(d.page_content)
    return {source: "\n\n".join(parts) for source, parts in full.items()}


def contextualize(chunks: list[Document], sections: list[Document]) -> list[Document]:
    """Prefix every chunk with a sentence situating it in its document.

    The sentence comes from kb/contexts.json when it is there, and from one
    model call when it is not (then saved). Anthropic's contextual-retrieval
    recipe: the WHOLE document plus the chunk in, 50-100 tokens of context out,
    prepended before embedding. "The warranty period is 24 months" becomes
    findable as "ShopWise's warranty policy, the table of periods by product
    type: wearables..." even though the chunk never says "warranty policy"."""
    from models import build_model, stamp
    from prompts.loader import load_prompts

    prompt = load_prompts("rag")["context_prompt"]
    cache = json.loads(CONTEXTS_FILE.read_text(encoding="utf-8")) if CONTEXTS_FILE.exists() else {}
    full, model, written = whole_documents(sections), None, 0
    out = []
    for c in chunks:
        cid = chunk_id(c)
        if cid not in cache:
            model = model or build_model()
            reply = model.invoke(prompt.format(document=full[c.metadata["source"]],
                                               chunk=c.page_content),
                                 config={"run_name": "contextualize",
                                         **stamp("context_prompt", prompt)})
            cache[cid] = " ".join(reply.text.split())
            written += 1
            if written % 10 == 0:            # save as we go: a crash keeps what it paid for
                CONTEXTS_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        out.append(Document(page_content=f"{cache[cid]}\n\n{c.page_content}",
                            metadata={**c.metadata, "context": cache[cid]}))
    CONTEXTS_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    print(f"contexts        : {written} written by the model, {len(chunks) - written} from {CONTEXTS_FILE.name}")
    return out


# ── v9: the parent scheme ────────────────────────────────────────────────────

def children(sections: list[Document]) -> list[Document]:
    """Small chunks for precise matching, each knowing its parent section.
    kb/rag.small_to_big turns a hit back into the whole section."""
    kids = chunk(sections, size=settings.PARENT_CHILD_SIZE, overlap=50)
    for k in kids:
        k.metadata["parent_id"] = parent_id(k)
    return kids


# ── v9: past tickets ─────────────────────────────────────────────────────────

def load_past_tickets() -> list[Document]:
    """One Document per closed ticket: subject, the customer's words and the
    resolution, with category and product type as metadata.

    `from` never leaves this function. A past ticket is shown to OTHER customers
    as precedent; the person who wrote it is not part of the precedent."""
    inbox = {t["id"]: t for t in yaml.safe_load(
        (settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))}
    docs = []
    for row in yaml.safe_load(PAST_TICKETS_FILE.read_text(encoding="utf-8")):
        ticket = {**inbox.get(row["id"], {}), **row}
        ticket.pop("from", None)
        text = (f"Past ticket {ticket['id']} ({ticket['category']}): {ticket['subject']}\n"
                f"Customer wrote: {' '.join(ticket['body'].split())}\n"
                f"How it was resolved: {' '.join(ticket['resolution'].split())}")
        docs.append(Document(page_content=text, metadata=dict(
            source="past-tickets", section=ticket["id"], ticket_id=ticket["id"],
            category=ticket["category"], product_type=ticket["product_type"],
            closed=str(ticket["closed"]), authority="precedent", start_index=0)))
    return docs


# ── writing ──────────────────────────────────────────────────────────────────

def write(client, emb, name: str, docs: list[Document], reset: bool) -> int:
    """Create (or reset) one collection and upsert `docs` into it by stable id."""
    if reset and client.collection_exists(name):
        client.delete_collection(name)
        print(f"reset           : {name} dropped")
    store.ensure_collection(client, emb, name)
    store.vector_store(client, emb, mode="hybrid", name=name).add_documents(
        docs, ids=[chunk_id(d) for d in docs])
    total = client.count(name).count
    print(f"{name:<48} {len(docs):>4} documents · {total} points")
    if total > len(docs):
        # Ids are stable for fixed chunk settings. Change CHUNK_SIZE and the
        # old chunks' ids no longer come up, so they are never overwritten.
        print(f"WARNING         : {total - len(docs)} stale points from an earlier"
              f" chunking. Rebuild with --reset.")
    return total


def ingest(reset: bool = False, scheme: str | None = None, tickets: bool | None = None) -> None:
    """Build what was asked for. No arguments: the plain policies and the tickets."""
    print(f"embedding space : {settings.EMBED_MODEL}"
          + (f" · {settings.EMBED_DIM}" if settings.embed_provider == "google_genai" else ""))
    print(f"folder          : {settings.QDRANT_PATH}")
    schemes = [scheme] if scheme else ([] if tickets else ["plain"])
    tickets = (scheme is None) if tickets is None else tickets

    client = store.open_client()
    try:
        emb = store.embeddings()
        sections = load_corpus()
        for s in schemes:
            if s == "plain":
                docs = chunk(sections)
            elif s == "contextual":
                docs = contextualize(chunk(sections), sections)
            else:
                docs = children(sections)
            write(client, emb, settings.collection_for(s), docs, reset)
        if tickets:
            write(client, emb, settings.tickets_collection, load_past_tickets(), reset)

        if "plain" in schemes:
            hit = store.vector_store(client, emb).similarity_search(
                "opened headphones return window", k=1)[0]
            print(f"smoke query     : {hit.metadata['source']} · {hit.metadata['section']}")
    finally:
        client.close()   # release the folder lock for the next process


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the ShopWise knowledge base.")
    ap.add_argument("--reset", action="store_true", help="drop the collection(s) first")
    ap.add_argument("--scheme", choices=["plain", "contextual", "parent"],
                    help="build one policy collection (v9)")
    ap.add_argument("--tickets", action="store_true", help="build the past-tickets collection (v9)")
    args = ap.parse_args()
    if not settings.has_embed_key:
        sys.exit("No GEMINI_API_KEY in labs/.env. Add one, or set "
                 "SHOPWISE_EMBED_MODEL=fastembed:BAAI/bge-small-en-v1.5 for the free CPU model.")
    ingest(args.reset, args.scheme, True if args.tickets else None)
