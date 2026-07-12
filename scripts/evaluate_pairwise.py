"""Aggregate pairwise comparisons with Bradley-Terry -> per-(query,doc) relevance, then score every
system with continuous-gain nDCG@10. Also reports Kendall tau vs the BT gold ranking.

Output: runs/pairwise_qrels.parquet (query_id, paper_id, relevance[0..1], bt_theta)
        runs/metrics_pairwise.parquet (system, nDCG@10, kendall_tau, ...)
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from swrd_eval import config


def bradley_terry(docs, comps, iters=200, tol=1e-9):
    """MM algorithm (Hunter 2004). comps: list of (winner, loser). Returns dict doc->theta(log-strength)."""
    idx = {d: i for i, d in enumerate(docs)}
    n = len(docs)
    W = np.zeros(n)                  # total wins
    N = np.zeros((n, n))             # games between i,j
    for w, l in comps:
        wi, li = idx[w], idx[l]
        W[wi] += 1
        N[wi, li] += 1; N[li, wi] += 1
    p = np.ones(n)
    for _ in range(iters):
        p_new = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or N[i, j] == 0:
                    continue
                denom += N[i, j] / (p[i] + p[j])
            p_new[i] = (W[i] + 0.5) / (denom + 1e-12)   # +0.5 smoothing -> finite for winless docs
        p_new /= p_new.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new; break
        p = p_new
    theta = np.log(p + 1e-12)
    return {d: float(theta[idx[d]]) for d in docs}


def main():
    import os
    _IN=os.environ.get("PW_COMPARISONS","pairwise_comparisons.parquet")
    _QOUT=os.environ.get("PW_QRELS_OUT","pairwise_qrels.parquet")
    _MOUT=os.environ.get("PW_METRICS_OUT","metrics_pairwise.parquet")
    comps = pd.read_parquet(config.RUNS_DIR / _IN)
    qrel_rows = []
    gold = {}   # qid -> {doc: gain}
    for qid, grp in comps.groupby("query_id"):
        docs = sorted(set(grp["doc_a"]) | set(grp["doc_b"]))
        pairs = [(int(w), (int(a) if int(w) == int(b) else int(b)))
                 for a, b, w in zip(grp["doc_a"], grp["doc_b"], grp["winner"])]
        th = bradley_terry(docs, pairs)
        lo, hi = min(th.values()), max(th.values())
        rng = (hi - lo) or 1.0
        g = {d: (th[d] - lo) / rng for d in docs}     # min-max -> [0,1] gain
        gold[qid] = g
        for d in docs:
            qrel_rows.append({"query_id": qid, "paper_id": d, "relevance": g[d], "bt_theta": th[d]})
    pd.DataFrame(qrel_rows).to_parquet(config.RUNS_DIR / _QOUT, index=False)

    def ndcg_at(run, qid, k=10):
        g = gold.get(qid, {})
        ranked = run.sort_values("rank")["paper_id"].tolist()[:k]
        dcg = sum(g.get(int(p), 0.0) / np.log2(i + 2) for i, p in enumerate(ranked))
        ideal = sorted(g.values(), reverse=True)[:k]
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    files = sorted(glob.glob("runs/runs/dense.*.parquet")) + sorted(glob.glob("runs/runs/rerank.*.parquet")) \
        + sorted(glob.glob("runs/runs/hybrid.*.parquet")) + (["runs/runs/bm25.parquet"])
    rows = []
    qids = list(gold)
    for f in files:
        r = pd.read_parquet(f)
        sysname = f.split("/")[-1].replace(".parquet", "")
        nd = [ndcg_at(r[r.query_id == q], q) for q in qids if (r.query_id == q).any()]
        # Kendall tau vs BT gold order, over pooled docs the system ranks
        taus = []
        for q in qids:
            sub = r[r.query_id == q]
            common = [int(p) for p in sub.sort_values("rank")["paper_id"] if int(p) in gold[q]]
            if len(common) >= 3:
                sys_rank = list(range(len(common)))
                gold_rank = [-gold[q][d] for d in common]
                t, _ = kendalltau(sys_rank, gold_rank)
                if not np.isnan(t):
                    taus.append(t)
        rows.append({"system": sysname, "nDCG@10": float(np.mean(nd)),
                     "kendall_tau": float(np.mean(taus)) if taus else float("nan"),
                     "n_queries": len(nd)})
    lb = pd.DataFrame(rows).sort_values("nDCG@10", ascending=False)
    lb.to_parquet(config.RUNS_DIR / _MOUT, index=False)
    pd.set_option("display.max_rows", None)
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
