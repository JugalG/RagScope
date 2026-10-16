"""All settings in one place. Values come from the .env file (or the real environment).

Change a value in .env, then restart the program: .env is only read at start-up.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# override=True: values in this project's .env win over variables already set in Windows/the shell.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _get(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default  # blank values count as "not set"


def _float(name: str, default: float) -> float:
    return float(_get(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_get(name, str(default)))


# --- Models (all OpenAI by default) -------------------------------------------
# A cheap model does the small jobs (rewrite, rerank-by-LLM, grounding check, summaries).
# A stronger model writes the answer.
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = _get("ANSWER_MODEL", "gpt-4.1-mini")
CHEAP_MODEL = _get("CHEAP_MODEL", "gpt-4.1-nano")
FALLBACK_MODEL = _get("FALLBACK_MODEL", " ")  # optional: used if the answer model call fails

# --- "Gateway" settings ---------------------------------------------------------
# Default is OpenAI itself. Set LLM_BASE_URL to a gateway such as a
# LiteLLM proxy (e.g. http://localhost:4000) to route every call through it.
# Always set explicitly: otherwise a stray OPENAI_BASE_URL variable in Windows (for example
# from an Ollama setup) silently redirects chat calls but not embedding calls.
LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 30)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 2)

# --- Storage --------------------------------------------------------------------
CHROMA_DIR = PROJECT_ROOT / _get("CHROMA_DIR", "storage/chroma")
SESSIONS_FILE = PROJECT_ROOT / _get("SESSIONS_FILE", "storage/sessions.json")
UPLOAD_DIR = PROJECT_ROOT / _get("UPLOAD_DIR", "data/uploads")
COLLECTION_NAME = _get("COLLECTION_NAME", "docs")

# --- Chunking -------------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# --- Retrieval ------------------------------------------------------------------
SEARCH_MODE = _get("SEARCH_MODE", "hybrid")  # "dense" (similarity only) or "hybrid"
CANDIDATES = _int("CANDIDATES", 10)  # how many hits each search returns before fusing
RRF_K = _int("RRF_K", 60)  # standard constant for Reciprocal Rank Fusion
TOP_K = _int("TOP_K", 4)  # passages finally given to the answer model
# Passages with cosine similarity below this are treated as "not relevant".
# This is an UNTUNED starting point. Look at the scores printed in the terminal for
# your own documents and adjust it.
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.25)

# --- Reranking ------------------------------------------------------------------
RERANKER = _get("RERANKER", "flashrank")  # "flashrank" (small local model), "llm", or "none"

# --- Chat memory ----------------------------------------------------------------
HISTORY_MAX_TOKENS = _int("HISTORY_MAX_TOKENS", 1200)  # compress when history grows past this
HISTORY_KEEP_LAST = _int("HISTORY_KEEP_LAST", 4)  # messages always kept word-for-word

# --- Terminal output ------------------------------------------------------------
VERBOSE = _get("VERBOSE", "true").lower() in ("1", "true", "yes")


def require_api_key() -> None:
    """Stop early with a clear message instead of a confusing error deep inside a call."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and put your key in it.")
