"""Pretty step-by-step printing to the terminal (uses the `rich` library).

Every pipeline step calls one of these helpers, so you can watch what happens to a
query: rewrite -> embedding -> searches -> RRF -> gate -> rerank -> answer ->
grounding -> citations.
"""

import math

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from rag import config

console = Console()


def _on() -> bool:
    return config.VERBOSE


def short(text: str, n: int = 90) -> str:
    """Shorten text for display and escape [brackets] so rich does not treat them as styling."""
    text = " ".join(str(text).split())
    return escape(text if len(text) <= n else text[: n - 3] + "...")


def header(title: str) -> None:
    if _on():
        console.print()
        console.print(Rule(f"[bold cyan]{title}"))


def step(number: int | str, title: str) -> None:
    if _on():
        console.print(f"\n[bold yellow]STEP {number}[/] [bold]{title}[/]")


def info(text: str) -> None:
    if _on():
        console.print(f"  {text}")


def warn(text: str) -> None:
    if _on():
        console.print(f"  [bold red]![/] {escape(text)}")


def panel(text: str, title: str, style: str = "green") -> None:
    if _on():
        console.print(Panel(escape(text), title=escape(title), border_style=style, expand=False))


def vector(name: str, values: list[float], show: int = 8) -> None:
    """Print a summary of an embedding: size, length (norm) and the first few numbers."""
    if not _on():
        return
    norm = math.sqrt(sum(v * v for v in values))
    first = ", ".join(f"{v:+.4f}" for v in values[:show])
    console.print(f"  {name}: {len(values)} numbers, length(norm)={norm:.3f}")
    console.print(f"  first {show}: [dim][{first}, ...][/]")


def table(title: str, columns: list[str], rows: list[list]) -> None:
    if not _on():
        return
    t = Table(title=escape(title), title_justify="left", show_lines=False)
    for c in columns:
        t.add_column(c, overflow="fold")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)
