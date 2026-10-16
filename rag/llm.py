"""Model factory: the single place where model clients are created.

This acts as a tiny "LLM gateway" inside the app:
- every call gets a timeout and automatic retries (max_retries),
- the answer model can fall back to a second model if it fails (.with_fallbacks),
- LLM_BASE_URL can route all traffic through an external gateway (LiteLLM, etc.)
  without changing any other code.

Tests replace these functions with fakes, so nothing else in the app needs to know.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag import config


def _chat(model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=config.LLM_BASE_URL,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.LLM_BASE_URL,
        max_retries=config.LLM_MAX_RETRIES,
    )


def cheap_structured(schema):
    """Cheap model that must reply in the shape of `schema` (a Pydantic class)."""
    return _chat(config.CHEAP_MODEL).with_structured_output(schema)


def cheap_text():
    """Cheap model that replies with plain text."""
    return _chat(config.CHEAP_MODEL)


def answer_structured(schema):
    """Answer model with structured output, plus an optional fallback model."""
    primary = _chat(config.ANSWER_MODEL).with_structured_output(schema)
    if config.FALLBACK_MODEL:
        backup = _chat(config.FALLBACK_MODEL).with_structured_output(schema)
        return primary.with_fallbacks([backup])
    return primary
