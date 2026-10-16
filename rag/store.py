"""The vector database (Chroma, saved on disk) and a keyword index (BM25, in memory).

Chroma stores, for every chunk: an id, the embedding vector, the text, and metadata
(file name, page). It finds chunks whose vectors point in a similar direction to the
query vector ("similarity search").

BM25 is classic keyword search: it scores chunks by how often the query's words appear
in them, giving more weight to rare words. It catches exact terms (names, codes) that
embeddings sometimes miss. Combining both is "hybrid search".
"""

import re

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from rag import config, llm

_store: Chroma | None = None
_bm25: tuple[BM25Okapi, list[dict]] | None = None

_STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it of on or that the this "
    "to was what when where which who why will with you your do does did can".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]


def get_store() -> Chroma:
    global _store
    if _store is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=llm.get_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
            # Use cosine distance, so similarity = 1 - distance (range about -1..1).
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _store


def reset_cache() -> None:
    """Forget cached objects (used after ingestion and in tests)."""
    global _store, _bm25
    _store = None
    _bm25 = None


def invalidate_keyword_index() -> None:
    global _bm25
    _bm25 = None


def all_chunks() -> list[dict]:
    data = get_store().get(include=["documents", "metadatas"])
    return [
        {"id": i, "text": t, "metadata": m or {}}
        for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def keyword_index() -> tuple[BM25Okapi, list[dict]] | None:
    """Build the BM25 index from every chunk in Chroma (rebuilt after each ingestion)."""
    global _bm25
    if _bm25 is None:
        chunks = all_chunks()
        if not chunks:
            return None
        _bm25 = (BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks)
    return _bm25


def vectors_for(ids: list[str]) -> dict[str, list[float]]:
    """Fetch the stored embedding vectors for some chunk ids."""
    if not ids:
        return {}
    data = get_store().get(ids=ids, include=["embeddings"])
    return {i: list(v) for i, v in zip(data["ids"], data["embeddings"])}


def list_sources() -> list[dict]:
    counts: dict[str, int] = {}
    for c in all_chunks():
        name = c["metadata"].get("source", "?")
        counts[name] = counts.get(name, 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]
