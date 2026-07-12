"""Build the candidate pool for the Claude labeling swarm, chunk it, and ingest results.

Pool source for the first pass: BM25 (CPU-only; needs no embeddings). Dense/hybrid pools are
added later via `topup_pairs` once embeddings finish (incremental pooling).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config, corpus
from .testset import QUERIES_PARQUET, QRELS_PARQUET

LABEL_PAIRS = config.RUNS_DIR / "label_pairs.parquet"
CHUNK_DIR = config.RUNS_DIR / "label_chunks"
RESULTS_DIR = config.RUNS_DIR / "label_results"
ABS_TRUNC = 1500


def build_pairs(pool_depth: int | None = None) -> pd.DataFrame:
    import bm25s
    jc = config.judge_cfg()
    depth = pool_depth or jc.get("judge_pool_depth", 10)
    df = corpus.load_corpus()
    by_id = df.set_index("id")
    ids = df["id"].to_numpy()
    queries = pd.read_parquet(QUERIES_PARQUET)
    syn = queries[queries["subset"] == "synthetic"].reset_index(drop=True)

    # BM25 pool
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(df["doc_text"].tolist(), stopwords="en"))
    q_tokens = bm25s.tokenize(syn["query_text"].tolist(), stopwords="en")
    res_idx, _ = retriever.retrieve(q_tokens, k=depth)

    pairs: dict[tuple, dict] = {}
    for i, row in syn.iterrows():
        qid = row["query_id"]
        cand = {int(ids[res_idx[i, r]]) for r in range(res_idx.shape[1])}
        cand.add(int(row["seed_paper_id"]))  # always include the seed doc
        for pid in cand:
            if pid not in by_id.index:
                continue
            key = (qid, pid)
            if key in pairs:
                continue
            d = by_id.loc[pid]
            pairs[key] = {
                "pair_id": f"{qid}__{pid}",
                "query_id": qid,
                "query_text": row["query_text"],
                "paper_id": int(pid),
                "title": (d["title"] or "")[:300],
                "abstract": (d["abstract"] or "")[:ABS_TRUNC],
            }
    out = pd.DataFrame(list(pairs.values()))
    out.to_parquet(LABEL_PAIRS, index=False)
    return out


TOPUP_PAIRS = config.RUNS_DIR / "label_pairs_topup.parquet"
TOPUP_CHUNK_DIR = config.RUNS_DIR / "label_chunks_topup"
TOPUP_RESULTS = config.RUNS_DIR / "swarm_results_topup.json"


def build_topup_pairs(pool_depth: int = 10) -> pd.DataFrame:
    """Cross-system pool: contribute top-k from every retrieval run, keep only (query, paper)
    pairs not already judged. These are the new pairs the labeling swarm must grade."""
    from .retrieve import RUNS_SUBDIR
    df = corpus.load_corpus()
    by_id = df.set_index("id")
    queries = pd.read_parquet(QUERIES_PARQUET)
    qmap = dict(zip(queries["query_id"], queries["query_text"]))
    syn = set(queries[queries["subset"] == "synthetic"]["query_id"])
    existing = pd.read_parquet(QRELS_PARQUET) if QRELS_PARQUET.exists() else pd.DataFrame(columns=["query_id", "paper_id"])
    judged = set(zip(existing["query_id"], existing["paper_id"].astype(int)))

    pairs: dict[tuple, dict] = {}
    for rf in sorted(RUNS_SUBDIR.glob("*.parquet")):
        r = pd.read_parquet(rf)
        r = r[r["rank"] <= pool_depth]
        for qid, pid in zip(r["query_id"], r["paper_id"]):
            pid = int(pid)
            if qid not in syn or (qid, pid) in judged or (qid, pid) in pairs:
                continue
            if pid not in by_id.index:
                continue
            d = by_id.loc[pid]
            pairs[(qid, pid)] = {"pair_id": f"{qid}__{pid}", "query_id": qid,
                                 "query_text": qmap[qid], "paper_id": pid,
                                 "title": (d["title"] or "")[:300], "abstract": (d["abstract"] or "")[:ABS_TRUNC]}
    out = pd.DataFrame(list(pairs.values()))
    out.to_parquet(TOPUP_PAIRS, index=False)
    return out


def write_topup_chunks(chunk_size: int = 50) -> int:
    df = pd.read_parquet(TOPUP_PAIRS)
    TOPUP_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for f in TOPUP_CHUNK_DIR.glob("chunk_*.json"):
        f.unlink()
    for n, start in enumerate(range(0, len(df), chunk_size)):
        recs = df.iloc[start:start + chunk_size][["pair_id", "query_text", "title", "abstract"]].to_dict("records")
        (TOPUP_CHUNK_DIR / f"chunk_{n:04d}.json").write_text(json.dumps(recs, ensure_ascii=False))
    return (len(df) + chunk_size - 1) // chunk_size


def ingest_topup() -> int:
    """Append the top-up swarm grades (swarm_results_topup.json) to eval_qrels as claude-swarm."""
    pairs = pd.read_parquet(TOPUP_PAIRS).set_index("pair_id")
    grades = {r["pair_id"]: [int(g) for g in r["grades"]]
              for r in json.loads(TOPUP_RESULTS.read_text()) if r.get("grades")}
    rows, detail = [], []
    for pid, gs in grades.items():
        if pid not in pairs.index:
            continue
        med = int(np.round(np.median(gs)))
        p = pairs.loc[pid]
        rows.append({"query_id": p["query_id"], "paper_id": int(p["paper_id"]), "relevance": med,
                     "judge": "claude-swarm", "judge_model": "claude-swarm-5lens"})
        detail.append({"pair_id": pid, "grades": gs, "median": med,
                       "agreement": float(np.mean([1 if g == med else 0 for g in gs]))})
    new = pd.DataFrame(rows)
    existing = pd.read_parquet(QRELS_PARQUET)
    # drop any existing rows for the same (query,paper,judge) then append
    key = set(zip(new["query_id"], new["paper_id"]))
    mask = ~(existing.apply(lambda x: (x["query_id"], int(x["paper_id"])) in key and x["judge"] == "claude-swarm", axis=1))
    combined = pd.concat([existing[mask], new], ignore_index=True)
    combined.to_parquet(QRELS_PARQUET, index=False)
    dpath = config.RUNS_DIR / "swarm_label_detail_topup.parquet"
    pd.DataFrame(detail).to_parquet(dpath, index=False)
    return len(new)


def write_chunks(chunk_size: int = 40) -> int:
    df = pd.read_parquet(LABEL_PAIRS)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for f in CHUNK_DIR.glob("chunk_*.json"):
        f.unlink()
    n = 0
    for n, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start + chunk_size]
        recs = chunk[["pair_id", "query_text", "title", "abstract"]].to_dict("records")
        (CHUNK_DIR / f"chunk_{n:04d}.json").write_text(json.dumps(recs, ensure_ascii=False))
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    return n_chunks


SWARM_RESULTS = config.RUNS_DIR / "swarm_results.json"


def ingest_results() -> pd.DataFrame:
    """Aggregate the swarm's consolidated results (runs/swarm_results.json) into qrels.

    swarm_results.json: [{"pair_id": str, "grades": [int,...]}, ...].
    Final relevance per pair = median across the 5 lens grades. Agreement = fraction of
    lenses agreeing with the median.
    """
    pairs = pd.read_parquet(LABEL_PAIRS).set_index("pair_id")
    grades: dict[str, list[int]] = {}
    if SWARM_RESULTS.exists():
        for r in json.loads(SWARM_RESULTS.read_text()):
            if r.get("pair_id") is not None and r.get("grades"):
                grades[r["pair_id"]] = [int(g) for g in r["grades"]]
    else:  # fallback: per-lens files
        for f in sorted(RESULTS_DIR.glob("chunk_*.lens_*.json")):
            try:
                recs = json.loads(f.read_text())
            except Exception:
                continue
            for r in recs:
                if r.get("pair_id") is not None and r.get("grade") is not None:
                    grades.setdefault(r["pair_id"], []).append(int(r["grade"]))

    rows, detail = [], []
    for pair_id, gs in grades.items():
        if pair_id not in pairs.index:
            continue
        med = int(np.round(np.median(gs)))
        agree = float(np.mean([1 if g == med else 0 for g in gs]))
        p = pairs.loc[pair_id]
        rows.append({"query_id": p["query_id"], "paper_id": int(p["paper_id"]),
                     "relevance": med, "judge": "claude-swarm", "judge_model": "claude-swarm-5lens"})
        detail.append({"pair_id": pair_id, "grades": gs, "median": med, "agreement": agree})

    qrels_new = pd.DataFrame(rows)
    pd.DataFrame(detail).to_parquet(config.RUNS_DIR / "swarm_label_detail.parquet", index=False)

    # merge with any existing qrels (seed/known-item), dedup keeping max per (q,doc,judge)
    if QRELS_PARQUET.exists():
        existing = pd.read_parquet(QRELS_PARQUET)
        existing = existing[existing["judge"] != "claude-swarm"]
        combined = pd.concat([existing, qrels_new], ignore_index=True)
    else:
        combined = qrels_new
    combined.to_parquet(QRELS_PARQUET, index=False)
    return qrels_new
