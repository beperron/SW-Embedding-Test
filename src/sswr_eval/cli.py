"""Typer CLI — one command per pipeline stage."""
from __future__ import annotations
import typer

app = typer.Typer(add_completion=False, help="SWRD retrieval evaluation pipeline.")


@app.command()
def export(expansion: bool = typer.Option(True, help="68,043 abstract-bearing corpus")):
    """Stage 1: pull corpus from Supabase to runs/corpus.parquet."""
    from . import corpus
    df = corpus.export(abstract_only=True)
    typer.echo(f"exported {len(df):,} docs -> {corpus.CORPUS_PARQUET}")


@app.command()
def profile():
    """Stage 0: profile the exported corpus -> reports/profile.md."""
    from . import corpus
    typer.echo(corpus.profile())


@app.command()
def embed(
    only: str = typer.Option("", help="comma-separated model keys to run (default: all enabled)"),
    limit: int = typer.Option(0, help="cap docs (smoke test)"),
):
    """Stage 2: embed corpus with each enabled dense model (local .npy storage)."""
    from . import embed as e
    keys = [k for k in only.split(",") if k] or None
    res = e.run(only=keys, limit=limit or None)
    for r in res:
        typer.echo(r)


@app.command()
def testset():
    """Stage 3: build queries + qrels (synthetic + known-item, local LLM)."""
    from . import testset as t
    t.run()


@app.command()
def retrieve():
    """Stage 4: dense + BM25 + hybrid first-stage retrieval."""
    from . import retrieve as r
    r.run()


@app.command()
def judge():
    """Stage 3b: pool first-stage candidates and LLM-judge graded relevance (run after retrieve)."""
    from . import testset as t
    t.judge()


@app.command()
def rerank():
    """Stage 5: apply rerankers to pooled candidates."""
    from . import rerank as r
    r.run()


@app.command()
def evaluate():
    """Stage 6: metrics, bootstrap CIs, significance tests."""
    from . import evaluate as ev
    ev.run()


@app.command()
def report():
    """Stage 7: leaderboard + cost frontier -> reports/report.md."""
    from . import report as rp
    rp.run()


if __name__ == "__main__":
    app()
