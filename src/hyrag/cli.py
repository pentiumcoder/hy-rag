"""`hyrag` command-line interface."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hyrag import __version__
from hyrag.chunking import get_chunker
from hyrag.config import get_settings
from hyrag.ingestion import DocumentStore, load_corpus
from hyrag.models import Chunk, ChunkStrategy, Document

app = typer.Typer(
    help="Hybrid-search RAG over internal documentation.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CorpusOption = Annotated[
    Path | None, typer.Option("--corpus", "-c", help="Corpus directory (default: settings).")
]


def _version(value: bool) -> None:
    if value:
        console.print(f"hyrag {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """hyrag: hybrid-search RAG pipeline."""


@app.command()
def load(corpus: CorpusOption = None) -> None:
    """Parse the corpus into data/processed, keeping originals in data/raw."""
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    report = store.ingest_directory(corpus or settings.corpus_dir)
    console.print(
        f"[green]added[/] {len(report.added)}  [yellow]updated[/] {len(report.updated)}  "
        f"[dim]unchanged[/] {len(report.unchanged)}  [red]skipped[/] {len(report.skipped)}"
    )
    _print_documents(store.load_documents())


def _print_documents(docs: list[Document]) -> None:
    table = Table(title=f"{len(docs)} documents")
    for col in ("source", "format", "title", "sections", "chars"):
        table.add_column(col, justify="right" if col in {"sections", "chars"} else "left")
    for d in docs:
        table.add_row(d.source, d.format, d.title, str(len(d.sections)), str(d.char_count))
    console.print(table)


def _stats_row(strategy: str, chunks: list[Chunk]) -> list[str]:
    sizes = [c.char_count for c in chunks] or [0]
    return [
        strategy,
        str(len(chunks)),
        f"{mean(sizes):.0f}",
        str(min(sizes)),
        str(max(sizes)),
    ]


@app.command()
def chunk(
    strategy: Annotated[
        ChunkStrategy | None, typer.Option("--strategy", "-s", help="Chunking strategy.")
    ] = None,
    corpus: CorpusOption = None,
    show: Annotated[int, typer.Option(help="Print this many sample chunks.")] = 0,
    compare: Annotated[bool, typer.Option(help="Compare all strategies.")] = False,
) -> None:
    """Chunk the corpus and print statistics."""
    settings = get_settings()
    docs = load_corpus(corpus or settings.corpus_dir)
    strategies = list(ChunkStrategy) if compare else [strategy or settings.chunk_strategy]

    table = Table(title=f"Chunking {len(docs)} documents")
    for col in ("strategy", "chunks", "avg chars", "min", "max"):
        table.add_column(col, justify="left" if col == "strategy" else "right")
    results: dict[ChunkStrategy, list[Chunk]] = {}
    for strat in strategies:
        results[strat] = get_chunker(strat, settings).chunk(docs)
        table.add_row(*_stats_row(strat.value, results[strat]))
    console.print(table)

    for strat, chunks in results.items():
        for c in chunks[:show]:
            console.rule(f"{strat.value} · {c.source} #{c.chunk_index} · {c.section_heading}")
            console.print(c.text, markup=False, highlight=False)


if __name__ == "__main__":  # pragma: no cover
    app()
