"""Stage 6: metrics, bootstrap CIs, significance tests."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import ir_measures
from ir_measures import nDCG, R, RR, AP, Success
from . import config, stats
from .retrieve import RUNS_SUBDIR
from .testset import QRELS_PARQUET, QUERIES_PARQUET

METRICS_PARQUET = config.RUNS_DIR / "metrics.parquet"
PERQ_PARQUET = config.RUNS_DIR / "per_query.parquet"


def _load_qrels() -> tuple[dict, set]:
    q = pd.read_parquet(QRELS_PARQUET)
    # max relevance per (query, doc) across judges
    q = q.groupby(["query_id", "paper_id"], as_index=False)["relevance"].max()
    qrels: dict[str, dict[str, int]] = {}
    for qid, pid, rel in zip(q["query_id"], q["paper_id"], q["relevance"]):
        qrels.setdefault(str(qid), {})[str(pid)] = int(rel)
    return qrels, set(q["query_id"])


def _load_run(path) -> dict:
    r = pd.read_parquet(path)
    run: dict[str, dict[str, float]] = {}
    for qid, pid, score in zip(r["query_id"], r["paper_id"], r["score"]):
        run.setdefault(str(qid), {})[str(pid)] = float(score)
    return run


def run():
    ev = config.eval_cfg()
    qrels, _ = _load_qrels()
    queries = pd.read_parquet(QUERIES_PARQUET)
    ki_qids = set(queries[queries["subset"] == "known_item"]["query_id"].astype(str))

    measures = [nDCG @ 10, R @ 10, R @ 100, RR @ 10, AP @ 100]
    run_files = (sorted(RUNS_SUBDIR.glob("dense.*.parquet"))
                 + ([RUNS_SUBDIR / "bm25.parquet"] if (RUNS_SUBDIR / "bm25.parquet").exists() else [])
                 + sorted(RUNS_SUBDIR.glob("hybrid.*.parquet"))
                 + sorted(RUNS_SUBDIR.glob("rerank.*.parquet")))

    perq_records = []   # system, query_id, metric, value
    agg_records = []    # system, metric, value, ci_low, ci_high
    primary_perq: dict[str, dict[str, float]] = {}  # system -> {qid: nDCG@10} for significance

    for rf in run_files:
        system = rf.stem
        run = _load_run(rf)
        # per-query for each measure
        per_metric: dict[str, dict[str, float]] = {str(m): {} for m in measures}
        for res in ir_measures.iter_calc(measures, qrels, run):
            per_metric[str(res.measure)][res.query_id] = res.value
            perq_records.append({"system": system, "query_id": res.query_id,
                                 "metric": str(res.measure), "value": res.value})
        for m in measures:
            vals = np.array(list(per_metric[str(m)].values()), dtype=float)
            mean, lo, hi = stats.bootstrap_ci(vals, ev.get("bootstrap_resamples", 10000), ev.get("seed", 0))
            agg_records.append({"system": system, "metric": str(m), "value": mean,
                                "ci_low": lo, "ci_high": hi, "n_queries": len(vals)})
        primary_perq[system] = per_metric[str(nDCG @ 10)]

        # known-item Success@1
        if ki_qids:
            ki_run = {q: d for q, d in run.items() if q in ki_qids}
            ki_qrels = {q: d for q, d in qrels.items() if q in ki_qids}
            if ki_run and ki_qrels:
                s1 = ir_measures.calc_aggregate([Success @ 1], ki_qrels, ki_run)
                agg_records.append({"system": system, "metric": "Success@1(known-item)",
                                    "value": s1[Success @ 1], "ci_low": None, "ci_high": None,
                                    "n_queries": len(ki_qrels)})

    agg = pd.DataFrame(agg_records)
    perq = pd.DataFrame(perq_records)
    agg.to_parquet(METRICS_PARQUET, index=False)
    perq.to_parquet(PERQ_PARQUET, index=False)

    # significance vs best system on nDCG@10 (Holm-corrected paired Wilcoxon)
    ndcg = agg[agg["metric"] == str(nDCG @ 10)].sort_values("value", ascending=False)
    sig = []
    if len(ndcg):
        best = ndcg.iloc[0]["system"]
        best_q = primary_perq[best]
        qids = sorted(best_q.keys())
        pvals, others = [], []
        for sysname in ndcg["system"]:
            if sysname == best:
                continue
            ov = primary_perq[sysname]
            a = np.array([best_q.get(q, 0.0) for q in qids])
            b = np.array([ov.get(q, 0.0) for q in qids])
            pvals.append(stats.paired_wilcoxon(a, b))
            others.append(sysname)
        adj = stats.holm_correction(pvals) if pvals else []
        for sysname, p, pa in zip(others, pvals, adj):
            sig.append({"best": best, "system": sysname, "p": p, "p_holm": pa,
                        "sig_worse_than_best": bool(pa < 0.05)})
    (config.RUNS_DIR / "significance.json").write_text(json.dumps(sig, indent=2))
    print(f"evaluated {len(run_files)} systems; wrote metrics.parquet, per_query.parquet, significance.json", flush=True)
    print(ndcg[["system", "value", "ci_low", "ci_high"]].to_string(index=False))
