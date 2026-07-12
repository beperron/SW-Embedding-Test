"""Stage 0 (profile) and Stage 1 (export)."""
from __future__ import annotations
from collections import Counter
import pandas as pd
from tqdm import tqdm
from . import db, config

CORPUS_PARQUET = config.RUNS_DIR / "corpus.parquet"


def _doc_text(title: str | None, abstract: str | None) -> str:
    return f"{(title or '').strip()}\n\n{(abstract or '').strip()}".strip()


def export(abstract_only: bool = True) -> pd.DataFrame:
    """Pull the corpus from Supabase and write runs/corpus.parquet with a doc_text column."""
    total = db.count_papers(abstract_only=abstract_only)
    rows: list[dict] = []
    with tqdm(total=total, desc="export", unit="doc") as bar:
        for page in db.iter_papers(abstract_only=abstract_only):
            for r in page:
                r["doc_text"] = _doc_text(r.get("title"), r.get("abstract"))
                rows.append(r)
            bar.update(len(page))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df.to_parquet(CORPUS_PARQUET, index=False)
    return df


def load_corpus() -> pd.DataFrame:
    if not CORPUS_PARQUET.exists():
        raise FileNotFoundError("runs/corpus.parquet missing; run `export` first.")
    return pd.read_parquet(CORPUS_PARQUET)


def profile() -> str:
    df = load_corpus()
    n = len(df)
    yr = df["publication_year"].dropna()
    dt = Counter(df["document_type"].fillna("(null)"))
    ds = Counter(df["data_source"].fillna("(null)"))
    abs_len = df["doc_text"].str.len()
    lines = [
        "# SWRD corpus profile",
        "",
        f"- Exported documents (abstract-bearing): **{n:,}**",
        f"- publication_year range: {int(yr.min())}–{int(yr.max())}",
        f"- doc_text length: min {abs_len.min()}, median {int(abs_len.median())}, max {abs_len.max()}",
        f"- distinct document_type: {df['document_type'].nunique()}",
        f"- distinct data_source: {df['data_source'].nunique()}",
        "",
        "## Top document_type",
        *[f"- {k}: {v:,}" for k, v in dt.most_common(15)],
        "",
        "## Top data_source",
        *[f"- {k}: {v:,}" for k, v in ds.most_common(15)],
    ]
    out = "\n".join(lines)
    (config.REPORTS_DIR / "profile.md").write_text(out)
    return out
