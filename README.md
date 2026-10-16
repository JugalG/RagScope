# Simple RAG

A small, readable retrieval-augmented generation (RAG) app for learning. You give it
PDF, TXT or Markdown files, ask questions, and it answers **only from your documents**,
with citations. Every step prints in the terminal so you can see what happens to a query.

About 1,000 lines of Python in `rag/`, one HTML file for the UI, no database server.

## What it does

```
question
  1  rewrite      cheap model turns a follow-up into a standalone search query
  2  embed        query -> vector (text-embedding-3-small)
  3  dense search Chroma similarity search (cosine)
  4  keyword      BM25 keyword search                       } hybrid mode only
  5  RRF          merge both lists by rank: 1/(60+rank)     }
  6  gate         drop passages below MIN_SIMILARITY; nothing left -> "I couldn't find this"
  7  rerank       FlashRank (local) or the cheap model re-orders; keep the best TOP_K
  8  generate     answer model writes claims, each citing [S1], [S2]...; labels are checked in code
  9  grounding    cheap model checks each claim against its cited passages; unsupported claims removed
 10  answer       final text + citations; saved to the chat history
```

Chat history: last 4 messages kept word-for-word; when history passes ~1200 tokens the
older messages are folded into a running summary by the cheap model (`rag/memory.py`).

## Files

| File | What it does |
|---|---|
| `rag/config.py` | every setting, read from `.env` |
| `rag/llm.py` | creates model clients: timeouts, retries, optional fallback model, optional gateway URL |
| `rag/ingest.py` | load file, split into chunks, embed, store in Chroma |
| `rag/store.py` | Chroma vector database (on disk) + BM25 keyword index |
| `rag/rewriter.py` | query rewriting with the cheap model |
| `rag/retrieval.py` | similarity search, keyword search, RRF, relevance gate |
| `rag/rerank.py` | FlashRank / LLM / no reranking |
| `rag/generate.py` | prompt, structured answer, citation validation + one retry |
| `rag/grounding.py` | claim-by-claim grounding check |
| `rag/memory.py` | sessions, summary compression, `trim_messages`, saved to JSON |
| `rag/pipeline.py` | runs the 10 steps in order |
| `rag/trace.py` | the coloured terminal output |
| `cli.py` | `ingest`, `ask`, `chat`, `sources` commands |
| `server.py` + `static/index.html` | web chat UI and JSON API |
| `tests/` | 10 tests using fake models (no API key needed) |

## Run it (Windows PowerShell; macOS/Linux in brackets)

```powershell
cd simple-rag
python -m venv .venv
.venv\Scripts\activate            # [source .venv/bin/activate]
pip install -r requirements.txt
copy .env.example .env            # [cp .env.example .env]
notepad .env                      # put your OpenAI key after OPENAI_API_KEY=
```

Then:

```powershell
python cli.py ingest              # indexes everything under data\ (3 sample files included)
python cli.py ask "What does ERR-4012 mean?"
python cli.py chat                # interactive chat with memory, full trace each turn
python server.py                  # web UI at http://localhost:8000 (trace prints in this terminal)
python -m pytest                  # tests, no key needed
```

Put your own files in `data\` and run `python cli.py ingest` again, or use **Upload file** in the UI.
Re-ingesting a file replaces its old chunks.

## Things to try (to learn what each part does)

- Ask `What does ERR-4012 mean?` with `--mode dense`, then `--mode hybrid`; compare the BM25 and RRF tables.
- Ask something unrelated ("Who won the 1998 World Cup?"): the gate should stop it before the answer model is called.
- In `chat`, ask "What is the refund policy for annual plans?" then "and for monthly ones?"; watch step 1 rewrite the follow-up.
- Set `HISTORY_MAX_TOKENS=200` in `.env` and chat a few turns to see the "History compressed" panel.
- Switch `RERANKER` between `flashrank`, `llm` and `none` and compare step 7.

## Honest limits

- `MIN_SIMILARITY=0.25` is a guess. Real `text-embedding-3-small` scores differ by document type; read the
  gate table for questions you know the answer to (and ones you know are not in the docs) and adjust.
- The grounding check is a model's judgement and can be wrong in both directions.
- The tests use fake models: they prove the plumbing works, not answer quality. Nothing here has been
  run against the real OpenAI API by the author yet.
- One request at a time, no login, sessions in a JSON file. Fine for learning, not for real users.
- If you change `EMBEDDING_MODEL`, delete `storage\chroma` and re-ingest (vector sizes differ).
- FlashRank downloads its model (about 30 MB) from Hugging Face on first use. If that fails
  (proxy, firewall), the app prints a warning and keeps the search order; set `RERANKER=llm` instead.

## Troubleshooting

- `OPENAI_API_KEY is not set`: create `.env` next to `cli.py` (not inside `rag\`).
- Connection errors to OpenAI: check for leftover proxy variables (`Get-ChildItem Env: | findstr /i proxy`)
  and test with `curl.exe -I https://api.openai.com/v1/models` (a 401 reply means the network path is fine).
- `.env` changes need a restart.

## Next phase

- **Observability:** uncomment the `LANGSMITH_*` lines in `.env`; every LangChain call (rewrite, answer,
  grounding, summary) then shows up as a trace with inputs, outputs, tokens and latency. The API response
  already includes `latency_ms` and per-model `token_usage`.
- **Evals:** write 20-30 questions about your own documents (include some that are *not* answerable),
  with the file each answer should come from; script `pipeline.answer_question` over them and measure
  hit rate (right file in citations), correct abstentions, and grounding status. Use that to tune
  `MIN_SIMILARITY`, `TOP_K` and dense vs hybrid.
