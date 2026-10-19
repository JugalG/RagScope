# Simple RAG

A small, readable retrieval-augmented generation (RAG) app for learning. You give it
PDF, TXT or Markdown files, ask questions, and it answers **only from your documents**,
with citations. Every step prints in the terminal **and streams live into a "Pipeline trace"
panel in the web UI**, so you can see what happens to a query. An **Evaluate** page runs a
test set under two configurations and compares them side by side.

About 1,500 lines of Python in `rag/`, two HTML pages, no database server.

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

- **Chat memory:** the last 4 messages are kept word-for-word; when history passes ~1200 tokens, older
  messages are folded into a running summary by the cheap model (`rag/memory.py`).
- **Model calls:** every call has a timeout and retries, can fall back to a second model, and can be routed
  through a gateway such as a LiteLLM proxy (`rag/llm.py`).
- **Evaluation:** a test set of questions with expected source files is scored under configuration A and B
  (`rag/evals.py`).

## Files

| File | What it does |
|---|---|
| `rag/config.py` | every setting, read from `.env`; temporary overrides used by evaluation runs |
| `rag/llm.py` | creates model clients: timeouts, retries, optional fallback model, optional gateway URL |
| `rag/ingest.py` | load file, split into chunks, embed, store in Chroma |
| `rag/store.py` | Chroma vector database (on disk) + BM25 keyword index |
| `rag/rewriter.py` | query rewriting with the cheap model |
| `rag/retrieval.py` | similarity search, keyword search, RRF, relevance gate |
| `rag/rerank.py` | FlashRank / LLM / no reranking |
| `rag/generate.py` | prompt, structured answer, citation validation + one retry |
| `rag/grounding.py` | claim-by-claim grounding check |
| `rag/memory.py` | sessions, summary compression, `trim_messages`, saved to JSON |
| `rag/pipeline.py` | runs the 10 steps in order; records what was retrieved |
| `rag/trace.py` | the step-by-step trace: coloured terminal output + events for the web UI |
| `rag/evals.py` | evaluation runs: scoring, scorecard, saving runs, A/B comparison |
| `cli.py` | terminal commands (see below) |
| `server.py` | web server and JSON API |
| `static/index.html` | chat page with the live Pipeline trace panel |
| `static/eval.html` | Evaluate page: scorecard and side-by-side comparison |
| `data/sample/` | 3 small fictional documents to try the app with |
| `data/eval/sample.jsonl` | 15 test questions for the sample documents |
| `tests/` | 23 tests using fake models (no API key needed) |

Created when you run it (not in the zip): `.env` (your key), `storage/chroma/` (vector database),
`storage/sessions.json` (chat history), `storage/evals/` (saved evaluation runs), `data/uploads/`.

## Setup (Windows PowerShell; macOS/Linux in brackets)

```powershell
cd simple-rag
python -m venv .venv
.venv\Scripts\activate            # [source .venv/bin/activate]
pip install -r requirements.txt
copy .env.example .env            # [cp .env.example .env]
notepad .env                      # put your OpenAI key after OPENAI_API_KEY=
python -m pytest                  # 23 passed (no key needed)
python cli.py ingest              # indexes everything under data\ (3 sample files included)
```

**Updating to a newer version of the zip:** copy the new files over your folder but keep `.env`,
`storage\` and `.venv\`. Run `pip install -r requirements.txt` only if `requirements.txt` changed.

## Daily use

```powershell
cd simple-rag
.venv\Scripts\activate
python server.py                  # then open http://localhost:8000   (Ctrl+C to stop)
```

| Page | What you see |
|---|---|
| http://localhost:8000 | chat on the left, live Pipeline trace on the right; **Upload file**; search mode; new chat |
| http://localhost:8000/eval | Evaluate page: run A vs B on a test set, scorecard, question-by-question comparison |

If a page looks out of date after an update, press **Ctrl+F5**.

## All terminal commands

| Command | What it does | API key? |
|---|---|---|
| `python cli.py ingest [path]` | index a folder or file (default `data\`); re-ingesting a file replaces its old chunks | yes |
| `python cli.py sources` | list files and chunk counts in the vector database | yes |
| `python cli.py ask "question" [--mode dense\|hybrid]` | one question with the full step-by-step trace | yes |
| `python cli.py chat [--mode ...] [--session ID]` | interactive chat with memory; `exit` to quit | yes |
| `python cli.py eval [--set NAME] [--label TEXT]` | run a test set with your `.env` settings and print a scorecard | yes |
| `python cli.py eval --mode dense --reranker none --top-k 3 --min-sim 0.3` | same, with settings changed for this run (also `--candidates`, `--rrf-k`) | yes |
| `python cli.py compare --a "mode=dense,reranker=none" --b "mode=hybrid,reranker=flashrank"` | run A and B, print both scorecards and what changed | yes |
| `python cli.py compare --runs RUN_ID_A RUN_ID_B` | compare two saved runs (no API calls) | no |
| `python cli.py runs` | list saved evaluation runs | no |
| `python server.py` | web app at http://localhost:8000 | yes |
| `python -m pytest` | run the tests with fake models | no |

Settings for `--a` / `--b`: `mode`, `reranker`, `top_k` (or `k`), `candidates`, `min_similarity`
(or `min_sim`), `rrf_k`. Anything you leave out comes from `.env`.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | (required) | your OpenAI key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | turns text into vectors |
| `ANSWER_MODEL` | `gpt-4.1-mini` | writes the answer |
| `CHEAP_MODEL` | `gpt-4.1-nano` | rewrite, LLM reranking, grounding check, history summaries |
| `FALLBACK_MODEL` | (empty) | tried automatically if the answer model call fails |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | change to route all calls through a gateway (e.g. LiteLLM) |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | 30, 2 | per call |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 800, 120 | characters; re-ingest after changing |
| `SEARCH_MODE` | `hybrid` | `dense` (similarity only) or `hybrid` (+ BM25 and RRF) |
| `CANDIDATES` | 10 | results taken from each search before fusing |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `MIN_SIMILARITY` | 0.25 | relevance gate; untuned starting point (see "Honest limits") |
| `TOP_K` | 4 | passages finally given to the answer model |
| `RERANKER` | `flashrank` | `flashrank`, `llm` or `none` |
| `HISTORY_MAX_TOKENS`, `HISTORY_KEEP_LAST` | 1200, 4 | chat memory compression |
| `VERBOSE` | `true` | print the trace in the terminal |

Values in `.env` win over variables already set in Windows. Restart after editing `.env`.

## How the live trace works

`rag/trace.py` is called at every step. Besides printing to the terminal, each call creates a small
event such as `{"type": "table", "title": "RRF fusion ...", "columns": [...], "rows": [...]}`.
The streaming endpoints run the work in a background thread and send each event to the browser as soon
as it happens (Server-Sent Events: one `data: {...}` line per event, then a final `result` or `error`
event). The chat page draws each step as a collapsible box with its tables, the query vector as a small
bar chart, and how long the step took. Uploads stream their ingestion steps the same way. During
evaluation runs the per-question traces are silenced and one progress line per question is sent instead.

The trace shows document excerpts and internal scores. That is the point for learning, but a public app
with many users would hide it or show it only to admins.

## Evaluation and side-by-side comparison

A test set is a JSON Lines file in `data/eval/` (one question per line). `data/eval/sample.jsonl` has
15 questions for the sample documents: 11 answerable and 4 the documents can't answer (two of them are
"near misses", such as `ERR-9999`, which shares words with real error codes).

```json
{"id": "a1", "question": "What does ERR-4012 mean?", "expected_sources": ["error_codes.md"], "must_include": ["five minutes"]}
{"id": "u1", "question": "What does ERR-9999 mean?", "expected_sources": []}
```

- `expected_sources`: the file name(s) the answer should come from. An empty list means the correct
  behaviour is to say "not found".
- `must_include` (optional): phrases the answer must contain (case-insensitive). Checked in code.
- `kind` (optional): a free-text tag shown in the tables, e.g. "paraphrase" or "near miss".

To make your own set, copy `sample.jsonl` to e.g. `data/eval/mydocs.jsonl` and edit it; it appears in
the Evaluate page's test set list and works with `--set mydocs`.

**How a question is scored**
- Answerable: **pass** if an expected file was among the passages given to the model, the answer cited
  an expected file, and every `must_include` phrase is in the answer.
- Unanswerable: **pass** if the app said "not found".

**Scorecard metrics**

| Metric | Meaning |
|---|---|
| Overall pass rate | questions handled correctly |
| Retrieval hit rate | answerable questions where an expected file reached the model |
| MRR | mean reciprocal rank of the first expected passage (1.0 = always first) |
| Answer rate | answerable questions that got an answer instead of "not found" |
| Citation accuracy | answers that cited an expected file |
| Key facts present | answers containing every `must_include` phrase |
| Fully grounded | answers where every claim passed the grounding check |
| Correct "not found" | unanswerable questions the app declined |
| Avg / p95 latency | time per question |
| Avg LLM tokens | chat-model tokens per question (embedding tokens not counted) |

**On the Evaluate page:** pick configurations A and B (presets: similarity only, hybrid,
hybrid + FlashRank, hybrid + LLM rerank, or set each value), then **Run A and B**. You get the scorecard
with B − A in green (better) or red (worse), a question-by-question table showing what B fixed or broke,
and, when you click a row, both answers side by side plus how each passage moved (▲ up, ▼ down, new,
dropped). **Compare one question** runs any single question under A and B. **Compare two saved runs**
re-opens old results without API calls. Runs are saved in `storage/evals/`.

**Read the numbers honestly.**
- The sample documents are tiny (one chunk each), so most settings score about the same on them.
  Differences between dense and hybrid search or between rerankers show up with longer documents.
  Write 20–30 questions about your own PDFs, including some that are not answerable.
- `must_include` is a strict word match: a correct answer that words a fact differently counts as a
  fail. Click failed rows to check.
- With 15 questions, one question is about 7 percentage points. Don't over-read small differences.

## API

| Method and path | Purpose |
|---|---|
| `POST /api/chat/stream` | ask a question; streams trace events, then the result |
| `POST /api/chat` | ask a question; returns only the final JSON (for scripts) |
| `GET /api/sessions/{id}` | chat history and summary of a session |
| `POST /api/upload/stream`, `POST /api/upload` | upload and ingest a file (.pdf/.txt/.md, max 10 MB) |
| `GET /api/sources` | files in the vector database |
| `GET /api/eval/sets`, `GET /api/eval/sets/{name}` | test sets and their questions |
| `GET /api/eval/settings` | current `.env` settings (pre-fills the A/B forms) |
| `POST /api/eval/run/stream` | run a test set under given settings; streams progress, saves the run |
| `GET /api/eval/runs`, `GET /api/eval/runs/{id}` | saved runs |
| `GET /api/eval/compare?a=ID&b=ID` | side-by-side comparison of two saved runs |
| `POST /api/eval/question/stream` | one question under settings A and B |

A chat result includes the answer, citations, grounding status and report, the rewritten query,
`retrieval` (candidates with gate decisions, and the passages given to the model), `latency_ms` and
per-model `token_usage`.

## Things to try (to learn what each part does)

- Ask `What does ERR-4012 mean?` with `--mode dense`, then `--mode hybrid`; compare the BM25 and RRF tables.
- Ask something unrelated ("Who won the 1998 World Cup?"): the gate stops it before the answer model is called.
- In `chat`, ask "What is the refund policy for annual plans?" then "and for monthly ones?"; watch step 1 rewrite the follow-up.
- Set `HISTORY_MAX_TOKENS=200` in `.env` and chat a few turns to see the "History compressed" panel.
- Switch `RERANKER` between `flashrank`, `llm` and `none` and compare step 7.
- On the Evaluate page, compare `min_similarity` 0.2 vs 0.4: a looser gate usually finds more answers but
  also answers questions it shouldn't. Find the value that balances hit rate and correct "not found".

## Honest limits

- `MIN_SIMILARITY=0.25` is a starting guess. Tune it with the Evaluate page on your own documents.
- The grounding check is a model's judgement and can be wrong in both directions.
- The tests use fake models: they prove the plumbing works, not answer quality. The chat pipeline has been
  run with the real OpenAI API on the sample documents; evaluation runs have only been tested with fakes so far.
- One request at a time, no login, sessions in a JSON file. Fine for learning, not for real users.
- If you change `EMBEDDING_MODEL`, delete `storage\chroma` and re-ingest (vector sizes differ).
- FlashRank downloads its model (about 30 MB) from Hugging Face on first use. If that fails (proxy,
  firewall), the app prints a warning and keeps the search order; set `RERANKER=llm` instead.

## Troubleshooting

| Message or symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | create `.env` next to `cli.py` (not inside `rag\`) |
| `Incorrect API key provided: ollama` | a leftover Windows variable from an Ollama setup. Remove it: `[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $null, "User")`, open a new terminal |
| `Rewrite failed (OpenAIConnectionError)` while embeddings work | a leftover `OPENAI_BASE_URL` pointing at Ollama. Keep `LLM_BASE_URL=https://api.openai.com/v1` in `.env`, and remove it: `[Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", $null, "User")` |
| `token_invalidated` / `Your API key has been invalidated` | the key was deleted or disabled (often because it leaked). Create a new key at platform.openai.com/api-keys; never commit `.env` |
| `insufficient_quota` | add credit or a payment method in your OpenAI billing settings |
| Connection or proxy errors | check `Get-ChildItem Env: \| findstr /i proxy` and test `curl.exe -I https://api.openai.com/v1/models` (a 401 reply means the network path is fine) |
| `Reranker 'flashrank' failed` | Hugging Face is blocked on your network; set `RERANKER=llm` |
| Every question says "couldn't find this" | read the gate table's similarity numbers and lower `MIN_SIMILARITY` |
| Activation fails: "running scripts is disabled" | run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Changed `.env` but nothing changed | restart the command or server |
| Page looks old after an update | Ctrl+F5 |

## Next steps

Ranked by value for this project (market gap plus portfolio value):

1. **Live "what-if" settings panel** on the chat page: change top-K, gate and reranker, then rerun from the
   search step without re-embedding.
2. **Clickable citations and a chunk explorer:** click [S1] to see the passage highlighted in its document.
3. **Contextual retrieval:** prefix each chunk with a summary of its document; measure the gain on the Evaluate page.
4. **Feedback buttons** that turn bad answers into new test questions; trace history; cost per step.
5. **Observability:** uncomment the `LANGSMITH_*` lines in `.env` to send every LangChain call to LangSmith,
   or add a Langfuse/OpenTelemetry exporter.
6. **Agentic RAG with LangGraph**, once the evaluation numbers can show whether it helps.
