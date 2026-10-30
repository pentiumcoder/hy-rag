"""`hyrag` command-line interface."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hyrag import __version__
from hyrag.chunking import get_chunker
from hyrag.config import get_settings
from hyrag.embeddings import embedder_from_settings
from hyrag.evaluation import DEFAULT_GOLDEN, run_suite
from hyrag.generation import AnswerPipeline
from hyrag.index import Duplicate, IndexManager, IndexMismatchError
from hyrag.ingestion import DocumentStore, load_corpus
from hyrag.llm import LLMError
from hyrag.models import (
    AnswerResponse,
    AskOverrides,
    Chunk,
    ChunkStrategy,
    Document,
    RetrievalMode,
    ScoredChunk,
)
from hyrag.retrieval import HybridRetriever

app = typer.Typer(
    help="Hybrid-search RAG over internal documentation.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CorpusOption = Annotated[
    Path | None, typer.Option("--corpus", "-c", help="Corpus directory (default: settings).")
]
StrategyOption = Annotated[
    ChunkStrategy | None, typer.Option("--strategy", "-s", help="Chunking strategy.")
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
    strategy: StrategyOption = None,
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


@app.command()
def ingest(
    strategy: StrategyOption = None,
    all_strategies: Annotated[
        bool, typer.Option("--all", help="Index every chunking strategy.")
    ] = False,
    corpus: CorpusOption = None,
) -> None:
    """Load the corpus, then chunk, dedup, embed and index it (dense + BM25)."""
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    report = store.ingest_directory(corpus or settings.corpus_dir)
    docs = store.load_documents()
    console.print(
        f"documents: {len(docs)} ([green]added[/] {len(report.added)}, "
        f"[yellow]updated[/] {len(report.updated)}, [dim]unchanged[/] {len(report.unchanged)})"
    )
    embedder = embedder_from_settings(settings)
    strategies = list(ChunkStrategy) if all_strategies else [strategy or settings.chunk_strategy]

    table = Table(title=f"Index sync ({embedder.name})")
    for col in ("strategy", "added", "removed", "unchanged", "duplicates", "indexed"):
        table.add_column(col, justify="left" if col == "strategy" else "right")
    new_duplicates: list[Duplicate] = []
    for strat in strategies:
        result = IndexManager(settings, strat, embedder).sync(docs)
        note = " (rebuilt)" if result.rebuilt else " (repaired)" if result.repaired else ""
        table.add_row(
            strat.value + note,
            str(result.added),
            str(result.removed),
            str(result.unchanged),
            str(result.total_duplicates),
            str(result.total_chunks),
        )
        new_duplicates.extend(result.new_duplicates)
    console.print(table)
    for dup in new_duplicates[:10]:
        console.print(
            f"  [dim]skipped near-duplicate[/] {dup.source} ~ {dup.duplicate_of_source} "
            f"(cosine {dup.similarity:.3f})"
        )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="The search query.")],
    mode: Annotated[RetrievalMode, typer.Option("--mode", "-m")] = RetrievalMode.HYBRID,
    strategy: StrategyOption = None,
    k: Annotated[int | None, typer.Option("-k", help="Number of results.")] = None,
    rerank: Annotated[bool, typer.Option(help="Apply the cross-encoder reranker.")] = True,
    full: Annotated[bool, typer.Option(help="Print full chunk texts.")] = False,
) -> None:
    """Search the index and show every stage's score for each hit."""
    settings = get_settings()
    try:
        retriever = HybridRetriever.from_settings(settings, strategy)
    except IndexMismatchError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if len(retriever.index) == 0:
        console.print("[yellow]The index is empty. Run `hyrag ingest` first.[/]")
        raise typer.Exit(1)
    results = retriever.retrieve(query, mode, k, rerank=rerank)
    _print_results(results, f"{mode.value} · {retriever.index.strategy.value}", full)


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _print_results(results: list[ScoredChunk], title: str, full: bool) -> None:
    table = Table(title=title)
    for col in ("#", "source", "section", "dense", "bm25", "rrf", "rerank"):
        table.add_column(col, justify="left" if col in {"source", "section"} else "right")
    for r in results:
        c = r.chunk
        table.add_row(
            str(r.rank),
            c.source + (f" p.{c.page}" if c.page else ""),
            (c.section_heading or "").split(" > ")[-1],
            f"{_fmt(r.dense_score, '.3f')} #{r.dense_rank}" if r.dense_rank else "-",
            f"{_fmt(r.sparse_score, '.2f')} #{r.sparse_rank}" if r.sparse_rank else "-",
            _fmt(r.rrf_score, ".4f"),
            _fmt(r.rerank_score, ".2f"),
        )
    console.print(table)
    for r in results:
        text = r.chunk.text
        if not full and len(text) > 240:
            text = text[:240].replace("\n", " ") + " ..."
        console.rule(f"{r.rank}. {r.chunk.source}", align="left")
        console.print(text, markup=False, highlight=False)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    mode: Annotated[RetrievalMode, typer.Option("--mode", "-m")] = RetrievalMode.HYBRID,
    strategy: StrategyOption = None,
    k: Annotated[int | None, typer.Option("-k", help="Number of passages to use.")] = None,
    rerank: Annotated[bool, typer.Option(help="Apply the cross-encoder reranker.")] = True,
    verify: Annotated[
        bool | None, typer.Option("--verify/--no-verify", help="Check citations with the judge.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full response as JSON.")
    ] = False,
    context: Annotated[bool, typer.Option(help="Also print the retrieved passages.")] = False,
) -> None:
    """Answer a question with [n] citations, verified sources and a confidence score."""
    settings = get_settings()
    try:
        pipeline = AnswerPipeline.from_settings(settings, strategy)
    except IndexMismatchError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if len(pipeline.retriever_for(strategy or pipeline.default_strategy).index) == 0:
        console.print("[yellow]The index is empty. Run `hyrag ingest` first.[/]")
        raise typer.Exit(1)
    overrides = AskOverrides(mode=mode, strategy=strategy, top_k=k, rerank=rerank, verify=verify)
    try:
        with nullcontext() if as_json else console.status("Retrieving, answering, verifying..."):
            response = pipeline.ask(question, overrides)
    except LLMError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(response.model_dump_json(indent=2))
        return
    _print_answer(response)
    if context:
        _print_results(response.contexts, "retrieved passages", full=False)


_VERDICT_STYLE = {
    "SUPPORTED": "green",
    "PARTIAL": "yellow",
    "UNVERIFIED": "dim",
    "UNSUPPORTED": "bold red",
    "UNCITED": "bold red",
}


def _location(source: str, section: str | None, page: int | None = None) -> str:
    where = source + (f" p.{page}" if page else "")
    return where + (f" > {section.split(' > ')[-1]}" if section else "")


def _print_answer(r: AnswerResponse) -> None:
    if r.abstained and r.abstention:
        a = r.abstention
        body = Text(a.message)
        for label, items in (("Found", a.found), ("Missing", a.missing)):
            for item in items:
                body.append(f"\n{label}: ", style="bold").append(item)
        console.print(Panel(body, title="No confident answer", border_style="yellow"))
        if a.suggested_documents:
            console.print("Closest documents:")
            for doc in a.suggested_documents:
                console.print(Text(f"  - {_location(doc.source, doc.section_heading, doc.page)}"))
    else:
        text = Text(r.answer or "")
        text.highlight_regex(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]", "bold cyan")
        console.print(Panel(text, title="Answer", border_style="green"))
        table = Table(title="Claims", title_justify="left")
        for col in ("#", "claim", "cites", "verdict"):
            table.add_column(col, overflow="fold")
        for claim in r.claims:
            label = claim.label or "gap"
            table.add_row(
                str(claim.index),
                Text(claim.text),
                " ".join(f"[{n}]" for n in claim.citations) or "-",
                Text(label, style=_VERDICT_STYLE.get(label, "dim")),
            )
        console.print(table)
        for c in r.citations:
            line = Text(f"[{c.number}] {_location(c.source, c.section_heading, c.page)}")
            if c.flagged:
                line.append("  flagged", style="bold red")
            console.print(line)

    conf = r.confidence
    console.print(
        f"\n[bold]confidence {conf.score:.2f}[/] = 0.4 x retrieval {conf.retrieval:.2f} "
        f"({conf.retrieval_signal}) + 0.4 x citations {conf.citation_coverage:.2f} "
        f"+ 0.2 x completeness {conf.completeness:.2f}"
    )
    for warning in r.warnings:
        console.print(Text(f"! {warning}", style="yellow"))
    timings = "  ".join(f"{name} {ms / 1000:.1f}s" for name, ms in r.timings_ms.items())
    roles = (("generator", r.generator), ("judge", r.judge))
    models = "  ".join(f"{role} {name}" for role, name in roles if name)
    console.print(Text(f"{r.mode.value} / {r.strategy.value}  {models}  {timings}", style="dim"))


@app.command("eval")
def eval_command(
    offline: Annotated[
        bool,
        typer.Option(help="Fake embedder and stub LLM in a scratch directory: no models needed."),
    ] = False,
    retrieval_only: Annotated[
        bool, typer.Option("--retrieval-only", help="Skip generation (seconds instead of an hour).")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Default: HYRAG_EVAL_RESULTS_DIR.")
    ] = None,
    golden: Annotated[Path, typer.Option(help="Golden question set.")] = DEFAULT_GOLDEN,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    charts: Annotated[bool, typer.Option(help="Write PNG charts.")] = True,
    min_recall: Annotated[
        float | None, typer.Option(help="Exit 1 if the best recall@5 is below this.")
    ] = None,
) -> None:
    """Score retrieval on every configuration and answers on a selected few."""
    settings = get_settings()
    out_dir = output or settings.eval_results_dir
    with console.status("Evaluating...") as status:
        summary = run_suite(
            settings,
            out_dir,
            golden=golden,
            generation=not retrieval_only,
            offline=offline,
            limit=limit,
            charts=charts,
            progress=lambda message: status.update(f"Evaluating: {message}"),
        )

    table = Table(title=f"Retrieval ({summary['mode']}, answerable questions)")
    for col in ("config", "recall@1", "recall@5", "MRR"):
        table.add_column(col, justify="left" if col == "config" else "right")
    for row in summary["retrieval"]["configs"]:
        table.add_row(
            row["config"],
            f"{row['recall_at_1']:.1%}",
            f"{row['recall_at_5']:.1%}",
            f"{row['mrr']:.3f}",
        )
    console.print(table)

    generation = summary.get("generation")
    if generation:
        table = Table(title="Answers (tuned abstention policy)")
        for col in ("config", "answered", "correct", "faithful", "citations", "abstain P/R"):
            table.add_column(col, justify="left" if col == "config" else "right")
        for row in generation["configs"]:
            table.add_row(
                row["config"],
                f"{row['answered_rate']:.0%}",
                f"{row['correctness']:.0%}",
                f"{row['faithfulness']:.0%}",
                f"{row['citation_accuracy']:.0%}",
                f"{row['abstention_precision']:.0%}/{row['abstention_recall']:.0%}",
            )
        console.print(table)
        policy = generation["policy"]
        console.print(
            f"tuned policy: abstain_threshold={policy['abstain_threshold']} "
            f"answer_threshold={policy['answer_threshold']} judge_check={policy['judge_check']}"
        )
    console.print(f"[green]wrote[/] {out_dir / 'report.md'} and {out_dir / 'summary.json'}")

    best = (
        summary["retrieval"]["configs"][0]["recall_at_5"] if summary["retrieval"]["configs"] else 0
    )
    if min_recall is not None and best < min_recall:
        console.print(f"[red]best recall@5 {best:.1%} is below --min-recall {min_recall:.1%}[/]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (development).")] = False,
) -> None:
    """Run the HTTP API (OpenAPI docs at /docs)."""
    import uvicorn

    uvicorn.run("hyrag.api:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
