"""The full question-answering pipeline, one step after another.

question
  -> 1 rewrite (cheap model, uses chat history)
  -> 2 embed the rewritten query
  -> 3 similarity search            \
  -> 4 keyword search (hybrid only)  > 5 RRF fusion
  -> 6 relevance gate (no relevant passage -> abstain, no LLM call)
  -> 7 rerank, keep TOP_K
  -> 8 generate answer with citations (+ citation validation)
  -> 9 grounding check (cheap model judge)
  -> 10 citations + final answer, save to chat history
"""

import time

from langchain_core.callbacks import get_usage_metadata_callback

from rag import config, generate, grounding, llm, rerank, retrieval, rewriter, trace
from rag.memory import Session

NO_ANSWER = "I couldn't find this in your documents, so I won't guess."


def answer_question(question: str, session: Session, mode: str | None = None) -> dict:
    mode = mode or config.SEARCH_MODE
    started = time.perf_counter()
    trace.header(f"QUESTION (session {session.id}, mode={mode})")
    with get_usage_metadata_callback() as usage:  # counts tokens of every model call inside
        result = _run(question, session, mode)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000)
    result["token_usage"] = {model: dict(u) for model, u in usage.usage_metadata.items()}
    result["session_id"] = session.id

    trace.step(10, "Final answer")
    trace.panel(result["answer"], f"Answer [{result['grounding']}]", "green" if not result["abstained"] else "red")
    if result["citations"]:
        trace.table(
            "Citations",
            ["label", "source", "page", "excerpt"],
            [[c["label"], trace.short(c["source"], 40), c["page"], trace.short(c["excerpt"], 60)] for c in result["citations"]],
        )
    for model, u in result["token_usage"].items():
        trace.info(f"tokens {model}: in={u.get('input_tokens')} out={u.get('output_tokens')}")
    trace.info(f"total time: {result['latency_ms']} ms")

    session.add_turn(question, result["answer"])
    return result


def _abstain(query: str, reason: str) -> dict:
    return {"answer": NO_ANSWER, "abstained": True, "abstain_reason": reason, "rewritten_query": query,
            "citations": [], "grounding": "not_needed", "grounding_report": []}


def _run(question: str, session: Session, mode: str) -> dict:
    recent = session.recent()

    trace.step(1, "Rewrite the question into a standalone search query")
    query = rewriter.rewrite(question, session.summary, recent)

    trace.step(2, f"Embed the query with {config.EMBEDDING_MODEL}")
    qvec = llm.get_embeddings().embed_query(query)
    trace.vector("query vector", qvec)

    trace.step(3, "Similarity search in Chroma")
    dense = retrieval.dense_search(qvec, config.CANDIDATES)

    if mode == "hybrid":
        trace.step(4, "Keyword search (BM25)")
        keyword = retrieval.keyword_search(query, config.CANDIDATES)
        trace.step(5, "Combine both result lists with Reciprocal Rank Fusion")
        candidates = retrieval.rrf_fuse(dense, keyword)
    else:
        trace.step("4-5", "Skipped keyword search and RRF (dense mode)")
        candidates = dense

    trace.step(6, "Relevance gate")
    candidates = retrieval.relevance_gate(candidates, qvec)
    if not candidates:
        trace.warn("Nothing relevant enough: abstaining WITHOUT calling the answer model.")
        return _abstain(query, "no_relevant_passages")

    trace.step(7, f"Rerank ({config.RERANKER})")
    passages = rerank.rerank(query, candidates, config.TOP_K)

    trace.step(8, "Generate the answer from the passages")
    labeled = generate.label_passages(passages)
    answer = generate.generate(question, labeled, session.summary, recent)
    if answer.abstained:
        return _abstain(query, "model_found_no_answer")

    trace.step(9, "Grounding check")
    answer, report, status = grounding.check(answer, labeled)
    if answer.abstained:
        result = _abstain(query, "claims_not_grounded")
        result["grounding_report"] = report
        result["grounding"] = status
        return result

    used = []
    for claim in answer.claims:
        used += [s for s in claim.sources if s not in used]
    text = " ".join(f"{c.text} [{', '.join(c.sources)}]" for c in answer.claims)
    citations = [
        {
            "label": label,
            "source": labeled[label]["metadata"].get("source"),
            "page": labeled[label]["metadata"].get("page"),
            "excerpt": labeled[label]["text"][:300],
        }
        for label in used
    ]
    return {"answer": text, "abstained": False, "abstain_reason": None, "rewritten_query": query,
            "citations": citations, "grounding": status, "grounding_report": report}
