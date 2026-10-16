"""Chat sessions: history, compression and context trimming.

Strategy (the common "summary + recent window" pattern):
1. Keep the last HISTORY_KEEP_LAST messages word-for-word.
2. When the whole history grows past HISTORY_MAX_TOKENS, the cheap model folds the
   older messages into a running summary, and those messages are dropped.
3. Before each model call, LangChain's trim_messages caps the recent window so a few
   very long messages can never blow up the prompt.

Sessions are saved to a JSON file so they survive a restart.
"""

import json
import threading
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from rag import config, llm, trace

_lock = threading.Lock()


class Session:
    def __init__(self, session_id: str, summary: str = "", messages: list[BaseMessage] | None = None):
        self.id = session_id
        self.summary = summary
        self.messages: list[BaseMessage] = messages or []

    def recent(self) -> list[BaseMessage]:
        """Recent messages for prompts, capped by tokens (oldest dropped first)."""
        return trim_messages(
            self.messages,
            max_tokens=config.HISTORY_MAX_TOKENS,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
        )

    def add_turn(self, question: str, answer: str) -> None:
        self.messages += [HumanMessage(question), AIMessage(answer)]
        self.compress_if_needed()

    def compress_if_needed(self) -> None:
        tokens = count_tokens_approximately(self.messages)
        trace.info(f"History: {len(self.messages)} message(s), ~{tokens} tokens (limit {config.HISTORY_MAX_TOKENS})")
        if tokens <= config.HISTORY_MAX_TOKENS or len(self.messages) <= config.HISTORY_KEEP_LAST:
            return
        old = self.messages[: -config.HISTORY_KEEP_LAST]
        keep = self.messages[-config.HISTORY_KEEP_LAST :]
        transcript = "\n".join(f"{m.type.upper()}: {m.content}" for m in old)
        try:
            reply = llm.cheap_text().invoke(
                [
                    SystemMessage(
                        "Update the running summary of a conversation. Keep names, numbers, the user's "
                        "goals and open questions. Max 120 words. Return only the summary."
                    ),
                    HumanMessage(f"Current summary:\n{self.summary or '(empty)'}\n\nNew messages:\n{transcript}"),
                ]
            )
        except Exception as exc:
            trace.warn(f"Summarising failed ({type(exc).__name__}); history left as is.")
            return
        self.summary = str(reply.content).strip()
        self.messages = keep
        trace.panel(
            f"Folded {len(old)} older message(s) into the summary; kept the last {len(keep)}.\n\n{self.summary}",
            "History compressed",
            "magenta",
        )

    def to_dict(self) -> dict:
        return {"summary": self.summary, "messages": messages_to_dict(self.messages)}


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if config.SESSIONS_FILE.exists():
            raw = json.loads(config.SESSIONS_FILE.read_text(encoding="utf-8"))
            for sid, data in raw.items():
                self.sessions[sid] = Session(sid, data["summary"], messages_from_dict(data["messages"]))

    def save(self) -> None:
        with _lock:
            config.SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {sid: s.to_dict() for sid, s in self.sessions.items()}
            config.SESSIONS_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def get(self, session_id: str | None) -> Session:
        if not session_id or session_id not in self.sessions:
            session_id = session_id or uuid.uuid4().hex[:12]
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]
