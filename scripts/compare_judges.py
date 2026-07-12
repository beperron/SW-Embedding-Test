"""Compare the two committee judges (Qwen3.6-35B vs Gemma3-27B) over the shared pairs.
Splits labels into AGREEMENTS (consensus -> validated) and DISAGREEMENTS (-> tiebreaker).
Writes runs/judge_consensus.parquet, runs/judge_disagreements.parquet, runs/judge_agreement.json."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sswr_eval import config


def qwk(a, b, k=4):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (w * O).sum() / (w * e).sum()


def main():
    q = pd.read_parquet(config.RUNS_DIR / "eval_qrels.parquet")[["query_id", "paper_id", "relevance"]]
    g = pd.read_parquet(config.RUNS_DIR / "eval_qrels_gemma.parquet")[["query_id", "paper_id", "relevance"]]
    q = q.rename(columns={"relevance": "qwen"})
    g = g.rename(columns={"relevance": "gemma"})
    m = q.merge(g, on=["query_id", "paper_id"], how="inner")
    n = len(m)
    diff = (m["qwen"] - m["gemma"]).abs()
    exact = int((diff == 0).sum())
    within1 = int((diff <= 1).sum())
    k = qwk(m["qwen"], m["gemma"])

    res = {
        "n_pairs_compared": n,
        "judge_a": "qwen3.6:35b", "judge_b": "gemma3:27b",
        "exact_agreement_pct": round(100 * exact / n, 1),
        "within_one_pct": round(100 * within1 / n, 1),
        "quadratic_weighted_kappa": round(float(k), 3),
        "n_agree_exact": exact,
        "n_disagree": n - exact,
        "disagreement_by_magnitude": {int(d): int((diff == d).sum()) for d in [1, 2, 3]},
        "mean_abs_diff": round(float(diff.mean()), 3),
        "qwen_mean_grade": round(float(m["qwen"].mean()), 3),
        "gemma_mean_grade": round(float(m["gemma"].mean()), 3),
    }
    # 4x4 confusion (rows = qwen grade, cols = gemma grade)
    conf = pd.crosstab(m["qwen"], m["gemma"]).reindex(index=[0, 1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)
    res["confusion_qwen_rows_gemma_cols"] = conf.values.tolist()
    # where disagreements concentrate (which grade pairs)
    dd = m[diff > 0]
    res["top_disagreement_cells"] = (
        dd.groupby(["qwen", "gemma"]).size().sort_values(ascending=False).head(6)
        .reset_index().apply(lambda r: {"qwen": int(r["qwen"]), "gemma": int(r["gemma"]), "n": int(r[0])}, axis=1).tolist()
    )

    # consensus = exact-agreement pairs (validated); use the agreed grade
    consensus = m[diff == 0].copy()
    consensus["relevance"] = consensus["qwen"]
    consensus[["query_id", "paper_id", "relevance"]].to_parquet(config.RUNS_DIR / "judge_consensus.parquet", index=False)
    # disagreements (for tiebreaker)
    m[diff > 0][["query_id", "paper_id", "qwen", "gemma"]].assign(abs_diff=diff[diff > 0]).to_parquet(
        config.RUNS_DIR / "judge_disagreements.parquet", index=False)

    (config.RUNS_DIR / "judge_agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nConfusion matrix (rows = Qwen grade 0-3, cols = Gemma grade 0-3):")
    print(conf.to_string())


if __name__ == "__main__":
    main()
