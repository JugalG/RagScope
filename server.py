"""Web server: a chat UI with a live "Pipeline trace" panel, plus a JSON API.

Run:  python server.py     then open http://localhost:8000

Streaming endpoints (used by the UI) send Server-Sent Events: one `data: {...}` line per
trace event (step, info, table, vector, ...) while the pipeline runs, then a final
{"type": "result"} or {"type": "error"} event. The terminal still prints the same trace.
"""

import json
import queue
import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag import config, ingest, pipeline, store, trace
from rag.memory import SessionStore

app = FastAPI(title="Simple RAG")
sessions = SessionStore()
# One request at a time: keeps the trace readable and the app simple.
_pipeline_lock = threading.Lock()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    mode: str | None = Field(default=None, pattern="^(dense|hybrid)$")


def _run_chat(req: ChatRequest) -> dict:
    session = sessions.get(req.session_id)
    result = pipeline.answer_question(req.question.strip(), session, req.mode)
    sessions.save()
    return result


def _stream(job: Callable[[], dict], failure: str) -> StreamingResponse:
    """Run `job` in a background thread and stream its trace events as they happen."""
    events: queue.Queue = queue.Queue()

    def worker():
        with _pipeline_lock, trace.capture(events.put):
            try:
                events.put({"type": "result", "data": job()})
            except Exception as exc:
                trace.warn(f"{failure}: {type(exc).__name__}: {exc}")
                events.put({"type": "error", "message": f"{failure} ({type(exc).__name__}). See the trace."})
        events.put(None)  # tells the generator below to stop

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    return _stream(lambda: _run_chat(req), "The model provider call failed")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Same as /api/chat/stream but returns only the final JSON (handy for scripts)."""
    with _pipeline_lock:
        try:
            return _run_chat(req)
        except Exception as exc:
            trace.warn(f"Request failed: {type(exc).__name__}: {exc}")
            raise HTTPException(502, f"The model provider call failed ({type(exc).__name__}). See the server terminal.")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions.sessions:
        raise HTTPException(404, "Unknown session")
    s = sessions.sessions[session_id]
    return {
        "session_id": s.id,
        "summary": s.summary,
        "messages": [{"role": m.type, "content": m.content} for m in s.messages],
    }


@app.get("/api/sources")
def sources():
    return store.list_sources()


def _save_upload(file: UploadFile) -> Path:
    name = Path(file.filename or "").name  # drop any folder parts from the name
    if Path(name).suffix.lower() not in ingest.SUPPORTED:
        raise HTTPException(400, f"Only {sorted(ingest.SUPPORTED)} files are supported")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 10 MB")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    return path


@app.post("/api/upload/stream")
def upload_stream(file: UploadFile):
    path = _save_upload(file)
    return _stream(lambda: ingest.ingest_file(path), "Ingestion failed")


@app.post("/api/upload")
def upload(file: UploadFile):
    path = _save_upload(file)
    with _pipeline_lock:
        try:
            return ingest.ingest_file(path)
        except Exception as exc:
            trace.warn(f"Ingestion failed: {type(exc).__name__}: {exc}")
            raise HTTPException(502, f"Ingestion failed ({type(exc).__name__}). See the server terminal.")


if __name__ == "__main__":
    import uvicorn

    config.require_api_key()
    uvicorn.run(app, host="127.0.0.1", port=8000)
