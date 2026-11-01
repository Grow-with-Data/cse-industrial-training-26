"""Build the ShopWise knowledge base: load → chunk → embed → write.

    uv run python kb/ingest.py            # build or refresh it
    uv run python kb/ingest.py --reset    # drop the collection first

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

Every chunk leaves here carrying the metadata a citation needs:
    source · section · page (PDFs) · format · authority · start_index
"""
from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

# `python kb/ingest.py` puts kb/ on sys.path, not the project folder above it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (MarkdownHeaderTextSplitter,  # noqa: E402
                                      RecursiveCharacterTextSplitter)
from pypdf import PdfReader  # noqa: E402

from config import settings  # noqa: E402
from kb import store  # noqa: E402

# Fixed forever: change it and every id changes, so every point is duplicated.
NAMESPACE = uuid.UUID("5e1f0a4e-0000-4000-8000-000000000008")

AUTHORITY = {"ftc-prompt-delivery": "regulator", "cfr-16-part-700": "regulator"}
PDF_ONLY = ["ftc-prompt-delivery", "cfr-16-part-700"]


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
    """Every document the knowledge base indexes, never both twins."""
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


def ingest(reset: bool = False) -> None:
    print(f"embedding space : {settings.EMBED_MODEL}"
          + (f" · {settings.EMBED_DIM}" if settings.embed_provider == "google_genai" else ""))
    print(f"collection      : {settings.collection}")
    print(f"folder          : {settings.QDRANT_PATH}")

    client = store.open_client()
    try:
        if reset and client.collection_exists(settings.collection):
            client.delete_collection(settings.collection)
            print("reset           : collection dropped")
        emb = store.embeddings()
        store.ensure_collection(client, emb)

        chunks = chunk(load_corpus())
        ids = [chunk_id(c) for c in chunks]
        store.vector_store(client, emb, mode="hybrid").add_documents(chunks, ids=ids)
        print(f"chunks          : {len(chunks)}  (chunk_size {settings.CHUNK_SIZE},"
              f" overlap {settings.CHUNK_OVERLAP})")

        total = client.count(settings.collection).count
        print(f"points stored   : {total}")
        if total > len(chunks):
            # Ids are stable for fixed chunk settings. Change CHUNK_SIZE and the
            # old chunks' ids no longer come up, so they are never overwritten.
            print(f"WARNING         : {total - len(chunks)} stale points from an earlier"
                  f" chunking. Rebuild with --reset.")

        hit = store.vector_store(client, emb).similarity_search(
            "opened headphones return window", k=1)[0]
        print(f"smoke query     : {hit.metadata['source']} · {hit.metadata['section']}")
    finally:
        client.close()   # release the folder lock for the next process


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the ShopWise knowledge base.")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()
    if not settings.has_embed_key:
        sys.exit("No GEMINI_API_KEY in .env. Add one, or set "
                 "SHOPWISE_EMBED_MODEL=fastembed:BAAI/bge-small-en-v1.5 for the free CPU model.")
    ingest(args.reset)
