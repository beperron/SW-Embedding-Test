"""Scoring-time de-duplication. Collapse (lowercased title + year) duplicate records to one canonical
id (the copy with the fullest abstract), then re-score the pairwise leaderboard with duplicates merged:
each ranked list is de-duplicated by canonical id and gold relevance is collapsed to canonical.
Compares the deduped leaderboard to the original. Also writes a clean deduplicated corpus.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sswr_eval import config

corp = pd.read_parquet(config.RUNS_DIR / "corpus.parquet")
corp["tl"] = corp["title"].astype(str).str.strip().str.lower()
corp["yr"] = corp["publication_year"].fillna(-1)
corp["_abslen"] = corp["abstract"].astype(str).str.len()

# canonical = fullest-abstract record per (title, year) group; map every id -> canonical id
canon = {}
n_groups = n_removed = 0
for (tl, yr), g in corp.groupby(["tl", "yr"]):
    ids = g["id"].tolist()
    keep = int(g.sort_values(["_abslen", "id"], ascending=[False, True])["id"].iloc[0])
    for i in ids:
        canon[int(i)] = keep
    if len(ids) > 1:
        n_groups += 1; n_removed += len(ids) - 1
print(f"dedup key = title+year | duplicate groups: {n_groups} | copies removed: {n_removed} | "
      f"corpus {len(corp)} -> {len(corp)-n_removed}", flush=True)

# deduplicated corpus for the manuscript / future runs
keep_ids = set(canon.values())
corp[corp["id"].isin(keep_ids)].drop(columns=["tl", "yr", "_abslen"]).to_parquet(
    config.RUNS_DIR / "corpus_dedup.parquet", index=False)

# gold relevance collapsed to canonical (max over duplicate members)
q = pd.read_parquet(config.RUNS_DIR / "pairwise_qrels.parquet")
gold = {}
for r in q.itertuples():
    c = canon.get(int(r.paper_id), int(r.paper_id))
    d = gold.setdefault(r.query_id, {})
    d[c] = max(d.get(c, 0.0), float(r.relevance))


def ndcg_dedup(run, qid, k=10):
    g = gold.get(qid, {})
    seen, ranked = set(), []
    for pid in run.sort_values("rank")["paper_id"]:
        c = canon.get(int(pid), int(pid))
        if c in seen:            # drop the 2nd copy of the same paper from the ranked list
            continue
        seen.add(c); ranked.append(c)
        if len(ranked) >= k:
            break
    dcg = sum(g.get(c, 0.0) / np.log2(i + 2) for i, c in enumerate(ranked))
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(sorted(g.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


files = sorted(glob.glob("runs/runs/dense.*.parquet")) + sorted(glob.glob("runs/runs/rerank.*.parquet")) \
    + sorted(glob.glob("runs/runs/hybrid.*.parquet")) + ["runs/runs/bm25.parquet"]
qids = list(gold)
rows = []
for f in files:
    r = pd.read_parquet(f); s = f.split("/")[-1].replace(".parquet", "")
    nd = [ndcg_dedup(r[r.query_id == qq], qq) for qq in qids if (r.query_id == qq).any()]
    rows.append({"system": s, "nDCG@10_dedup": float(np.mean(nd))})
dd = pd.DataFrame(rows)

orig = pd.read_parquet(config.RUNS_DIR / "metrics_pairwise.parquet")[["system", "nDCG@10"]]
m = orig.merge(dd, on="system").sort_values("nDCG@10", ascending=False)
m["delta"] = m["nDCG@10_dedup"] - m["nDCG@10"]
m.to_parquet(config.RUNS_DIR / "metrics_pairwise_dedup.parquet", index=False)

rho, _ = spearmanr(m["nDCG@10"], m["nDCG@10_dedup"])
dn = m[m.system.str.startswith("dense.")].copy(); dn["m"] = dn.system.str.replace("dense.", "")
print("\n=== DENSE leaderboard: original vs deduped pairwise nDCG@10 ===")
for _, r in dn.iterrows():
    print(f"  {r.m:18} {r['nDCG@10']:.4f} -> {r['nDCG@10_dedup']:.4f}  ({r.delta:+.4f})")
print(f"\nleaderboard rank correlation original vs deduped: Spearman rho = {rho:.4f}")
print("top-5 original :", list(orig.sort_values('nDCG@10',ascending=False).system.head(5)))
print("top-5 deduped  :", list(m.sort_values('nDCG@10_dedup',ascending=False).system.head(5)))
