"""Ingestion: file -> text -> chunks -> embeddings -> vector database.

Re-ingesting the same file replaces its old chunks, so running it twice is safe.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config, llm, store, trace

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    """Read a file into Documents. PDFs give one Document per page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": n})
            for n, page in enumerate(reader.pages, start=1)
        ]
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": path.name, "page": 1})]
    raise ValueError(f"Unsupported file type {suffix}. Use one of {sorted(SUPPORTED)}")


def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,  # remember where each chunk started inside its page
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    return chunks


def chunk_id(chunk: Document, n: int) -> str:
    key = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.metadata.get('start_index')}|{n}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def ingest_file(path: Path) -> dict:
    trace.header(f"INGEST {path.name}")

    trace.step(1, "Load the file")
    pages = load_file(path)
    total_chars = sum(len(p.page_content) for p in pages)
    trace.info(f"{len(pages)} page(s), {total_chars} characters")
    if total_chars == 0:
        trace.warn("No text found (a scanned PDF needs OCR first). Nothing ingested.")
        return {"source": path.name, "chunks": 0}

    trace.step(2, f"Split into chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    chunks = split(pages)
    trace.info(f"{len(chunks)} chunk(s). First chunk:")
    trace.info(f"[dim]{trace.short(chunks[0].page_content, 200)}[/]")

    trace.step(3, f"Embed every chunk with {config.EMBEDDING_MODEL}")
    texts = [c.page_content for c in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):  # send at most 100 chunks per API call
        vectors.extend(llm.get_embeddings().embed_documents(texts[start : start + 100]))
    trace.vector("chunk 1 vector", vectors[0])

    trace.step(4, "Store id + vector + text + metadata in Chroma")
    db = store.get_store()
    # Remove this file's old chunks first, so re-ingesting does not create duplicates.
    db.delete(where={"source": path.name})
    ids = [chunk_id(c, n) for n, c in enumerate(chunks)]
    # We embedded ourselves (to show the vectors), so we write to the underlying
    # Chroma collection directly instead of letting LangChain embed a second time.
    db._collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    store.invalidate_keyword_index()  # BM25 index will be rebuilt on the next query
    trace.info(f"Stored {len(ids)} chunk(s). Collection now holds {db._collection.count()} chunk(s).")
    return {"source": path.name, "chunks": len(ids)}


def ingest_folder(folder: Path) -> list[dict]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        trace.warn(f"No .pdf/.txt/.md files found in {folder}")
    return [ingest_file(p) for p in files]
