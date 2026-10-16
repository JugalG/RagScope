"""Web server: a small chat UI plus a JSON API. The terminal shows each step.

Run:  python server.py     then open http://localhost:8000
"""

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import config, ingest, pipeline, store, trace
from rag.memory import SessionStore

app = FastAPI(title="Simple RAG")
sessions = SessionStore()
# One request at a time: keeps the terminal trace readable and the app simple.
_pipeline_lock = threading.Lock()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    mode: str | None = Field(default=None, pattern="^(dense|hybrid)$")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):  # a plain `def` runs in a worker thread, so blocking calls are fine
    with _pipeline_lock:
        session = sessions.get(req.session_id)
        try:
            result = pipeline.answer_question(req.question.strip(), session, req.mode)
        except Exception as exc:
            trace.warn(f"Request failed: {type(exc).__name__}: {exc}")
            raise HTTPException(502, f"The model provider call failed ({type(exc).__name__}). See the server terminal.")
        sessions.save()
    return result


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


@app.post("/api/upload")
def upload(file: UploadFile):
    name = Path(file.filename or "").name  # drop any folder parts from the name
    if Path(name).suffix.lower() not in ingest.SUPPORTED:
        raise HTTPException(400, f"Only {sorted(ingest.SUPPORTED)} files are supported")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 10 MB")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    with _pipeline_lock:
        try:
            result = ingest.ingest_file(path)
        except Exception as exc:
            trace.warn(f"Ingestion failed: {type(exc).__name__}: {exc}")
            raise HTTPException(502, f"Ingestion failed ({type(exc).__name__}). See the server terminal.")
    return result


if __name__ == "__main__":
    import uvicorn

    config.require_api_key()
    uvicorn.run(app, host="127.0.0.1", port=8000)
