"""Reranking: re-order the candidates with a model that reads query and passage together.

Search ranks passages from separate vectors/word counts. A reranker (a "cross-encoder")
reads the query and each passage side by side, so it judges relevance more precisely.
It is slower, which is why we run it only on the few candidates the search returned.

Options (RERANKER in .env):
- "flashrank": small free model that runs on your CPU (downloaded once, about 30 MB).
- "llm":       asks the cheap OpenAI model to score each passage 0-10.
- "none":      keep the search order.
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag import config, llm, trace


@lru_cache
def _flashrank():
    from flashrank import Ranker  # imported here so the app still runs without it

    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(config.PROJECT_ROOT / "storage" / "flashrank"))


def _flashrank_scores(query: str, candidates: list[dict]) -> list[float]:
    from flashrank import RerankRequest

    passages = [{"id": n, "text": c["text"]} for n, c in enumerate(candidates)]
    results = _flashrank().rerank(RerankRequest(query=query, passages=passages))
    scores = [0.0] * len(candidates)
    for r in results:
        scores[r["id"]] = float(r["score"])
    return scores


class PassageScores(BaseModel):
    scores: list[int] = Field(description="One relevance score 0-10 per passage, in the given order")


def _llm_scores(query: str, candidates: list[dict]) -> list[float]:
    numbered = "\n\n".join(f"[{n}] {c['text'][:1200]}" for n, c in enumerate(candidates))
    result = llm.cheap_structured(PassageScores).invoke(
        [
            SystemMessage(
                "Score how useful each passage is for answering the query, from 0 (useless) "
                "to 10 (directly answers it). Return exactly one score per passage, in order."
            ),
            HumanMessage(f"Query: {query}\n\nPassages:\n{numbered}"),
        ]
    )
    scores = [float(s) for s in result.scores][: len(candidates)]
    return scores + [0.0] * (len(candidates) - len(scores))


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    method = config.RERANKER
    if method not in ("flashrank", "llm") or not candidates:
        trace.info("Reranker off: keeping search order.")
        return candidates[:top_k]
    try:
        scores = _flashrank_scores(query, candidates) if method == "flashrank" else _llm_scores(query, candidates)
    except Exception as exc:  # e.g. model download blocked; fall back instead of failing
        trace.warn(f"Reranker '{method}' failed ({type(exc).__name__}: {exc}); keeping search order.")
        return candidates[:top_k]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    before = {c["id"]: n for n, c in enumerate(candidates, start=1)}
    ordered = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    trace.table(
        f"Reranked with '{method}' (keeping top {top_k})",
        ["new #", "old #", "rerank score", "kept?", "text"],
        [
            [n, before[c["id"]], f"{c['rerank_score']:.3f}", "yes" if n <= top_k else "-", trace.short(c["text"], 55)]
            for n, c in enumerate(ordered, start=1)
        ],
    )
    return ordered[:top_k]
