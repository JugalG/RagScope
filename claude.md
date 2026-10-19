# CLAUDE.md: project context for Claude Code

## What this is
Simple RAG: a learning and portfolio project by a developer learning GenAI/RAG end to end.
It answers questions only from uploaded documents (PDF/TXT/MD), with citations, and shows every
pipeline step live (terminal + web "Pipeline trace" panel). An Evaluate page scores a test set under
two configurations and compares them side by side. Working name for the portfolio: **RAGScope**
("observable, grounded RAG"); the code still says "Simple RAG". Don't rename unless asked.

The owner is a beginner-to-intermediate learner. Keep code small, readable and commented in plain
English; prefer simple, explicit code over abstractions. Explain concepts when introducing them.

## How to run (Windows PowerShell; the owner develops on Windows)
```powershell
.venv\Scripts\activate
python -m pytest              # 23 tests, fake models, no API key needed
python cli.py ingest          # index data\ (3 fictional sample docs)
python server.py              # http://localhost:8000 (chat) and /eval (evaluation)
python cli.py ask "What does ERR-4012 mean?"
python cli.py compare --a "mode=dense,reranker=none" --b "mode=hybrid,reranker=flashrank"
```
README.md lists every command, setting, endpoint and troubleshooting step. Keep it up to date
with every feature change (the owner relies on it).

## Architecture (rag/)
Pipeline (`rag/pipeline.py`), 10 steps, each traced:
1 rewrite (cheap model, uses chat history) → 2 embed → 3 Chroma similarity search → 4 BM25
(hybrid only) → 5 RRF fusion → 6 relevance gate (MIN_SIMILARITY; abstain with no LLM call if empty)
→ 7 rerank (FlashRank local / LLM / none) → 8 answer as structured claims citing [S1].. (labels
validated in code, one retry, else abstain) → 9 grounding check per claim (cheap model judge;
unsupported claims removed) → 10 answer + citations, saved to session history.

- `config.py`: all settings from `.env` (load_dotenv with override=True, so .env beats Windows env vars).
  `config.overrides(...)` temporarily changes TUNABLE settings (used by evals).
- `llm.py`: the only place model clients are created (timeouts, retries, optional fallback,
  LLM_BASE_URL gateway). Tests monkeypatch these functions.
- `store.py` Chroma on disk (cosine) + in-memory BM25. `ingest.py` load/split/embed/upsert (idempotent per file).
- `memory.py`: sessions in storage/sessions.json; summary + last-N window; LangChain trim_messages.
- `trace.py`: every helper prints with rich AND emits an event dict to a contextvar sink (web SSE).
  `trace.silenced()` mutes both (evals). Escape rich markup for any user/document text (`trace.short`).
- `evals.py`: test sets `data/eval/*.jsonl` (id, question, expected_sources, must_include, kind);
  scoring, summary metrics, saved runs in storage/evals/, `compare()` and `diff_retrieval()`.
- `server.py`: FastAPI; streaming endpoints run work in a thread and stream SSE events
  (`_stream`); one request at a time via `_pipeline_lock`. Pages: `static/index.html`, `static/eval.html`
  (plain HTML/JS, no build step, light/dark via CSS variables).
- `cli.py`: ingest, ask, chat, sources, eval, compare, runs.

## Conventions
- Every new pipeline step must call trace helpers so it shows in the terminal and the web panel.
- Pipeline results carry `retrieval` (candidates with gate decisions, passages given to the model);
  evals depend on it.
- Tests use fakes (`tests/fakes.py`: bag-of-words embeddings, scripted models; shared fixture in
  `tests/conftest.py`). Add a test for each feature; check that a deliberate bug makes it fail.
- Never invent metrics or results; resume numbers must come from real eval runs.
- Never commit `.env` (gitignored). The owner's previous OpenAI key was invalidated.

## Models and environment
OpenAI: text-embedding-3-small, gpt-4.1-mini (answers), gpt-4.1-nano (cheap jobs).
Owner's Windows machine had leftover Ollama variables (OPENAI_API_KEY=ollama, OPENAI_BASE_URL);
the app pins LLM_BASE_URL=https://api.openai.com/v1 and .env overrides env vars because of this.

## Status
- Verified with the real OpenAI API: ingestion and the full chat pipeline on the sample docs
  (FlashRank downloaded and worked).
- Verified with fakes only: live trace UI, evaluation runner, scorecard, A/B comparison page.
- MIN_SIMILARITY=0.25 is untuned; real sample scores were ~0.44 (relevant) vs ~0.11-0.14 (unrelated).

## Next steps (agreed priority)
1. Run real evals; write 20-30 questions on the owner's own longer PDFs for resume numbers.
2. Live "what-if" settings panel on the chat page (rerun from search without re-embedding).
3. Clickable citations + chunk explorer.
4. Contextual retrieval (measure with evals); then feedback buttons, trace history, cost per step.
5. Observability export (LangSmith/Langfuse); later agentic RAG with LangGraph.