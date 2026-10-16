"""Step-by-step trace of the pipeline: printed to the terminal AND sent to the web UI.

Every pipeline step calls one of these helpers, so you can watch what happens to a
query: rewrite -> embedding -> searches -> RRF -> gate -> rerank -> answer ->
grounding -> citations.

Two outputs:
- Terminal: pretty printing with the `rich` library (turn off with VERBOSE=false).
- Events: if a "sink" is active (see `capture`), each helper also sends a small dict
  such as {"type": "table", "title": ..., "columns": [...], "rows": [...]}. The web
  server streams these to the browser, which draws the live "Pipeline trace" panel.
"""

import contextvars
import math
import time
from contextlib import contextmanager
from typing import Callable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from rag import config

console = Console()

# The function that receives events for the current request (None = nobody listening).
_sink: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar("trace_sink", default=None)
_started: contextvars.ContextVar[float] = contextvars.ContextVar("trace_started", default=0.0)


@contextmanager
def capture(callback: Callable[[dict], None]):
    """Send every trace event produced inside this block to `callback`."""
    sink_token = _sink.set(callback)
    time_token = _started.set(time.perf_counter())
    try:
        yield
    finally:
        _sink.reset(sink_token)
        _started.reset(time_token)


def plain(text) -> str:
    """Remove rich styling tags, e.g. '[bold]hi[/]' -> 'hi', for the web UI."""
    try:
        return Text.from_markup(str(text)).plain
    except Exception:
        return str(text)


def emit(event: dict) -> None:
    sink = _sink.get()
    if sink is not None:
        event["t_ms"] = round((time.perf_counter() - _started.get()) * 1000)
        sink(event)


def _on() -> bool:
    return config.VERBOSE


def short(text: str, n: int = 90) -> str:
    """Shorten text for display and escape [brackets] so rich does not treat them as styling."""
    text = " ".join(str(text).split())
    return escape(text if len(text) <= n else text[: n - 3] + "...")


def header(title: str) -> None:
    emit({"type": "header", "title": plain(title)})
    if _on():
        console.print()
        console.print(Rule(f"[bold cyan]{title}"))


def step(number: int | str, title: str) -> None:
    emit({"type": "step", "number": str(number), "title": plain(title)})
    if _on():
        console.print(f"\n[bold yellow]STEP {number}[/] [bold]{title}[/]")


def info(text: str) -> None:
    emit({"type": "info", "text": plain(text)})
    if _on():
        console.print(f"  {text}")


def warn(text: str) -> None:
    emit({"type": "warn", "text": str(text)})
    if _on():
        console.print(f"  [bold red]![/] {escape(text)}")


def panel(text: str, title: str, style: str = "green") -> None:
    emit({"type": "panel", "title": str(title), "text": str(text), "style": style})
    if _on():
        console.print(Panel(escape(text), title=escape(title), border_style=style, expand=False))


def vector(name: str, values: list[float], show: int = 8) -> None:
    """Show an embedding: size, length (norm) and the first few numbers."""
    norm = math.sqrt(sum(v * v for v in values))
    emit({"type": "vector", "name": name, "dims": len(values), "norm": round(norm, 4),
          "values": [round(v, 4) for v in values[:48]]})
    if not _on():
        return
    first = ", ".join(f"{v:+.4f}" for v in values[:show])
    console.print(f"  {name}: {len(values)} numbers, length(norm)={norm:.3f}")
    console.print(f"  first {show}: [dim][{first}, ...][/]")


def table(title: str, columns: list[str], rows: list[list]) -> None:
    emit({"type": "table", "title": plain(title), "columns": [plain(c) for c in columns],
          "rows": [[plain(x) for x in r] for r in rows]})
    if not _on():
        return
    t = Table(title=escape(title), title_justify="left", show_lines=False)
    for c in columns:
        t.add_column(c, overflow="fold")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)
