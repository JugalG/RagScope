"""Command line entry point.

  python cli.py ingest [folder_or_file]   # default folder: data/
  python cli.py ask "your question"        # one question, full step-by-step trace
  python cli.py chat                       # interactive chat with memory (type 'exit' to stop)
  python cli.py sources                    # what is in the vector database
"""

import argparse
import sys
from pathlib import Path

from rag import config, ingest, pipeline, store, trace
from rag.memory import SessionStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("path", nargs="?", default=str(config.PROJECT_ROOT / "data"))
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["dense", "hybrid"])
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--mode", choices=["dense", "hybrid"])
    p_chat.add_argument("--session", help="continue an earlier session id")
    sub.add_parser("sources")
    args = parser.parse_args()
    config.require_api_key()

    if args.command == "ingest":
        path = Path(args.path)
        results = ingest.ingest_folder(path) if path.is_dir() else [ingest.ingest_file(path)]
        trace.table("Ingested", ["file", "chunks"], [[r["source"], r["chunks"]] for r in results])
    elif args.command == "sources":
        trace.table("In the vector database", ["file", "chunks"], [[s["source"], s["chunks"]] for s in store.list_sources()])
    elif args.command == "ask":
        sessions = SessionStore()
        pipeline.answer_question(args.question, sessions.get(None), args.mode)
    elif args.command == "chat":
        sessions = SessionStore()
        session = sessions.get(args.session)
        trace.console.print(f"Session [bold]{session.id}[/]. Type 'exit' to quit.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("exit", "quit"):
                break
            if question:
                pipeline.answer_question(question, session, args.mode)
                sessions.save()


if __name__ == "__main__":
    sys.exit(main())
