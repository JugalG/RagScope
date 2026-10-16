from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag import config, ingest, pipeline, retrieval, store
from rag.generate import Answer, Claim
from rag.memory import Session, SessionStore
from tests import fakes

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def rec(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "RERANKER", "none")
    monkeypatch.setattr(config, "VERBOSE", False)
    monkeypatch.setattr(config, "MIN_SIMILARITY", 0.15)  # the fake embedder gives low scores
    recorder = fakes.Recorder()
    fakes.install(monkeypatch, recorder)
    store.reset_cache()
    yield recorder
    store.reset_cache()


def test_ingest_is_idempotent(rec):
    first = ingest.ingest_folder(SAMPLE)
    again = ingest.ingest_folder(SAMPLE)
    assert [r["chunks"] for r in first] == [r["chunks"] for r in again]
    total = sum(s["chunks"] for s in store.list_sources())
    assert total == sum(r["chunks"] for r in first)  # no duplicates after re-ingest


def test_hybrid_answer_with_citation(rec):
    ingest.ingest_folder(SAMPLE)
    result = pipeline.answer_question("What does ERR-4012 mean?", Session("s1"), "hybrid")
    assert not result["abstained"]
    assert result["grounding"] == "grounded"
    assert result["citations"][0]["source"] == "error_codes.md"


def test_irrelevant_question_abstains_without_answer_model(rec):
    ingest.ingest_folder(SAMPLE)
    result = pipeline.answer_question("quantum chromodynamics lattice gluon", Session("s1"), "dense")
    assert result["abstained"] and result["abstain_reason"] == "no_relevant_passages"
    assert "answer" not in rec.calls


def test_invalid_citation_retried_then_fixed(rec):
    ingest.ingest_folder(SAMPLE)
    rec.answers = [
        Answer(claims=[Claim(text="x", sources=["S99"])], abstained=False),
        Answer(claims=[Claim(text="Annual plans refund within 30 days.", sources=["S1"])], abstained=False),
    ]
    result = pipeline.answer_question("annual plan refund", Session("s1"))
    assert rec.calls["answer"] == 2 and not result["abstained"]


def test_invalid_citation_twice_abstains(rec):
    ingest.ingest_folder(SAMPLE)
    bad = Answer(claims=[Claim(text="x", sources=["S99"])], abstained=False)
    rec.answers = [bad, bad]
    result = pipeline.answer_question("annual plan refund", Session("s1"))
    assert result["abstained"]


def test_unsupported_claim_removed(rec):
    ingest.ingest_folder(SAMPLE)
    rec.answers = [Answer(claims=[Claim(text="good", sources=["S1"]), Claim(text="made up", sources=["S1"])], abstained=False)]
    rec.unsupported_claims = {2}
    result = pipeline.answer_question("annual plan refund", Session("s1"))
    assert result["grounding"] == "partially_grounded"
    assert "made up" not in result["answer"]


def test_rrf_scores():
    dense = [{"id": "a", "text": "a", "metadata": {}, "dense_rank": 1}, {"id": "b", "text": "b", "metadata": {}, "dense_rank": 2}]
    keyword = [{"id": "b", "text": "b", "metadata": {}, "bm25_rank": 1}]
    fused = retrieval.rrf_fuse(dense, keyword, k=60)
    assert fused[0]["id"] == "b"  # found by both lists
    assert fused[0]["rrf"] == pytest.approx(1 / 62 + 1 / 61)


def test_history_compression(rec, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_MAX_TOKENS", 60)
    monkeypatch.setattr(config, "HISTORY_KEEP_LAST", 2)
    s = Session("s1")
    for n in range(4):
        s.add_turn(f"question number {n} " * 5, f"answer number {n} " * 5)
    assert rec.calls.get("summary", 0) >= 1
    assert s.summary == "Fake summary of older turns."
    assert len(s.messages) <= 2 + 2


def test_sessions_persist(rec):
    st = SessionStore()
    s = st.get("abc")
    s.add_turn("hi", "hello")
    st.save()
    assert SessionStore().sessions["abc"].messages[0].content == "hi"


def test_api_upload_and_chat(rec):
    import server

    server.sessions = SessionStore()
    client = TestClient(server.app)
    up = client.post("/api/upload", files={"file": ("support.md", (SAMPLE / "support.md").read_bytes())})
    assert up.status_code == 200 and up.json()["chunks"] > 0
    bad = client.post("/api/upload", files={"file": ("x.exe", b"MZ")})
    assert bad.status_code == 400
    r = client.post("/api/chat", json={"question": "Business plan uptime target credit", "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and body["session_id"]
    hist = client.get(f"/api/sessions/{body['session_id']}").json()
    assert [m["role"] for m in hist["messages"]] == ["human", "ai"]
    assert client.get("/").status_code == 200
