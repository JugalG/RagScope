"""Query rewriting with a cheap model.

Users ask follow-ups like "and what about the second one?". That text is a poor search
query on its own. The rewriter uses the chat history to turn it into a standalone,
search-friendly query, e.g. "refund policy for annual plans".
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag import config, llm, trace

REWRITE_INSTRUCTIONS = """You rewrite a user's latest message into ONE standalone search query
for a document search engine.
- Resolve pronouns and references ("it", "that one", "the second") using the conversation.
- Keep important names, numbers and technical terms exactly as written.
- Remove greetings and filler. Do not answer the question.
- If the message is already a clear standalone question, return it almost unchanged."""


class RewrittenQuery(BaseModel):
    search_query: str = Field(description="The standalone search query")


def rewrite(question: str, summary: str, recent: list[BaseMessage]) -> str:
    convo = "\n".join(f"{m.type.upper()}: {m.content}" for m in recent) or "(no earlier messages)"
    prompt = [
        SystemMessage(REWRITE_INSTRUCTIONS),
        HumanMessage(
            f"Summary of the older conversation:\n{summary or '(none)'}\n\n"
            f"Recent messages:\n{convo}\n\n"
            f"Latest user message:\n{question}"
        ),
    ]
    try:
        result = llm.cheap_structured(RewrittenQuery).invoke(prompt)
        query = result.search_query.strip() or question
    except Exception as exc:  # rewriting is an optimisation; never fail the request over it
        trace.warn(f"Rewrite failed ({type(exc).__name__}); using the original question.")
        query = question
    trace.info(f"original : {trace.short(question, 300)}")
    trace.info(f"rewritten: [bold]{trace.short(query, 300)}[/]  [dim](model: {config.CHEAP_MODEL})[/]")
    return query
