"""Stage 7: leaderboard, quality-vs-cost frontier, written report."""
from __future__ import annotations
import json
import pandas as pd
from . import config
from .evaluate import METRICS_PARQUET


def _embed_meta() -> pd.DataFrame:
    rows = []
    for mp in config.EMB_DIR.glob("*.meta.json"):
        rows.append(json.loads(mp.read_text()))
    return pd.DataFrame(rows)


def run():
    if not METRICS_PARQUET.exists():
        raise FileNotFoundError("runs/metrics.parquet missing; run `evaluate` first.")
    agg = pd.read_parquet(METRICS_PARQUET)
    meta = _embed_meta()
    sig_path = config.RUNS_DIR / "significance.json"
    sig = json.loads(sig_path.read_text()) if sig_path.exists() else []

    pivot = agg.pivot_table(index="system", columns="metric", values="value", aggfunc="first")
    ndcg_col = "nDCG@10"
    if ndcg_col in pivot.columns:
        pivot = pivot.sort_values(ndcg_col, ascending=False)

    lines = ["# SWRD retrieval evaluation — results", ""]
    lines += ["## Leaderboard", "", pivot.round(4).to_markdown(), ""]

    # CI annotations for nDCG@10
    nd = agg[agg["metric"] == ndcg_col].sort_values("value", ascending=False)
    if len(nd):
        lines += ["## nDCG@10 with 95% bootstrap CI", ""]
        for _, r in nd.iterrows():
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if pd.notna(r["ci_low"]) else ""
            lines.append(f"- **{r['system']}**: {r['value']:.4f} {ci}  (n={int(r['n_queries'])})")
        lines.append("")

    # significance
    if sig:
        best = sig[0]["best"]
        lines += [f"## Significance vs best ({best}), Holm-corrected paired Wilcoxon on nDCG@10", ""]
        for s in sig:
            mark = "significantly worse" if s["sig_worse_than_best"] else "not sig. different"
            lines.append(f"- {s['system']}: p_holm={s['p_holm']:.4f} ({mark})")
        not_worse = [s["system"] for s in sig if not s["sig_worse_than_best"]]
        lines += ["", f"**Systems not significantly worse than {best}:** {', '.join(not_worse) or '(none)'}", ""]

    # cost frontier
    if len(meta):
        lines += ["## Quality vs cost (dense embedders, measured on M5 Max)", ""]
        cost = meta[["key", "dim", "n_docs", "seconds", "docs_per_sec", "peak_rss_mb", "tier"]].copy()
        cost = cost.sort_values("docs_per_sec", ascending=False)
        lines += [cost.round(2).to_markdown(index=False), ""]

    out = "\n".join(lines)
    (config.REPORTS_DIR / "report.md").write_text(out)
    print(out)
