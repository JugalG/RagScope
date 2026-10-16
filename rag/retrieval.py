"""Searching the documents: similarity search, keyword search, RRF fusion, relevance gate."""

import math

from rag import config, store, trace


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dense_search(query_vector: list[float], k: int) -> list[dict]:
    """Similarity search: nearest chunks to the query vector in Chroma."""
    db = store.get_store()
    results = db.similarity_search_by_vector_with_relevance_scores(query_vector, k=k)
    hits = []
    for rank, (doc, distance) in enumerate(results, start=1):
        hits.append(
            {
                "id": doc.id,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "dense_rank": rank,
                "similarity": 1 - distance,  # cosine distance -> cosine similarity
            }
        )
    trace.table(
        f"Similarity search (top {k}, cosine similarity: 1.0 = same direction)",
        ["rank", "similarity", "source", "text"],
        [[h["dense_rank"], f"{h['similarity']:.3f}", _where(h), trace.short(h["text"], 70)] for h in hits],
    )
    return hits


def keyword_search(query: str, k: int) -> list[dict]:
    """BM25 keyword search over every chunk."""
    index = store.keyword_index()
    if index is None:
        return []
    bm25, chunks = index
    terms = store.tokenize(query)
    trace.info(f"BM25 query terms: {trace.short(str(terms), 200)}")
    scores = bm25.get_scores(terms)
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order[:k]:
        if scores[i] <= 0:  # no query word appears in this chunk
            break
        c = chunks[i]
        hits.append({**c, "bm25_rank": len(hits) + 1, "bm25_score": float(scores[i])})
    trace.table(
        f"Keyword search / BM25 (top {k}, higher = more matching words)",
        ["rank", "bm25", "source", "text"],
        [[h["bm25_rank"], f"{h['bm25_score']:.2f}", _where(h), trace.short(h["text"], 70)] for h in hits],
    )
    return hits


def rrf_fuse(dense: list[dict], keyword: list[dict], k: int = config.RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).

    It only uses ranks, so it works even though cosine and BM25 scores are on
    different scales. A chunk found by both searches gets two contributions.
    """
    merged: dict[str, dict] = {}
    for hit in dense + keyword:
        merged.setdefault(hit["id"], {}).update(hit)
    for item in merged.values():
        d = 1 / (k + item["dense_rank"]) if "dense_rank" in item else 0.0
        b = 1 / (k + item["bm25_rank"]) if "bm25_rank" in item else 0.0
        item["rrf_dense"], item["rrf_bm25"], item["rrf"] = d, b, d + b
    fused = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
    trace.table(
        f"RRF fusion (k={k}): 1/(k+dense_rank) + 1/(k+bm25_rank)",
        ["#", "dense rank", "bm25 rank", "1/(k+d)", "1/(k+b)", "RRF", "text"],
        [
            [
                n,
                x.get("dense_rank", "-"),
                x.get("bm25_rank", "-"),
                f"{x['rrf_dense']:.4f}",
                f"{x['rrf_bm25']:.4f}",
                f"{x['rrf']:.4f}",
                trace.short(x["text"], 45),
            ]
            for n, x in enumerate(fused, start=1)
        ],
    )
    return fused


def relevance_gate(candidates: list[dict], query_vector: list[float]) -> list[dict]:
    """Keep only passages whose cosine similarity to the query is >= MIN_SIMILARITY.

    Keyword-only hits have no similarity score yet, so we fetch their stored vectors
    and compute it. If nothing passes, the app abstains instead of guessing.
    """
    missing = [c["id"] for c in candidates if "similarity" not in c]
    for cid, vec in store.vectors_for(missing).items():
        for c in candidates:
            if c["id"] == cid:
                c["similarity"] = cosine(query_vector, vec)
    kept = [c for c in candidates if c["similarity"] >= config.MIN_SIMILARITY]
    trace.table(
        f"Relevance gate (keep similarity >= {config.MIN_SIMILARITY})",
        ["similarity", "kept?", "text"],
        [
            [f"{c['similarity']:.3f}", "yes" if c in kept else "[red]no[/]", trace.short(c["text"], 60)]
            for c in candidates
        ],
    )
    return kept


def _where(hit: dict) -> str:
    m = hit["metadata"]
    return trace.short(f"{m.get('source', '?')} p{m.get('page', '?')}", 40)
