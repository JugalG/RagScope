"""Fake models so the whole pipeline can be tested without an API key or network."""

import hashlib
import math

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from rag import store
from rag.generate import Answer, Claim
from rag.grounding import Verdict, Verdicts
from rag.rerank import PassageScores
from rag.rewriter import RewrittenQuery


class BagOfWordsEmbeddings(Embeddings):
    """Hashes words into a 256-dim vector: texts sharing words get similar vectors."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for w in store.tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class Recorder:
    """Records calls; answers are scripted per test."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.answers: list[Answer] = []  # consumed one per answer-model call
        self.unsupported_claims: set[int] = set()

    def count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1


class _Fake:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages):
        return self.fn(messages)


def install(monkeypatch, rec: Recorder):
    from rag import llm

    emb = BagOfWordsEmbeddings()
    monkeypatch.setattr(llm, "get_embeddings", lambda: emb)

    def cheap_structured(schema):
        def run(messages):
            rec.count(schema.__name__)
            text = messages[-1].content
            if schema is RewrittenQuery:
                return RewrittenQuery(search_query=text.rsplit("Latest user message:\n", 1)[-1])
            if schema is PassageScores:
                return PassageScores(scores=list(range(10, 0, -1)))
            if schema is Verdicts:
                n = text.count("CLAIM ")
                return Verdicts(verdicts=[
                    Verdict(claim_number=i, supported=i not in rec.unsupported_claims, reason="fake")
                    for i in range(1, n + 1)
                ])
            raise AssertionError(schema)
        return _Fake(run)

    def answer_structured(schema):
        def run(messages):
            rec.count("answer")
            if rec.answers:
                return rec.answers.pop(0)
            return Answer(claims=[Claim(text="Fake answer from S1.", sources=["S1"])], abstained=False)
        return _Fake(run)

    def cheap_text():
        return _Fake(lambda m: (rec.count("summary"), AIMessage("Fake summary of older turns."))[1])

    monkeypatch.setattr(llm, "cheap_structured", cheap_structured)
    monkeypatch.setattr(llm, "answer_structured", answer_structured)
    monkeypatch.setattr(llm, "cheap_text", cheap_text)
