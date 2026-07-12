"""Stage 4: first-stage retrieval — dense (exact cosine), BM25, hybrid (RRF).

Runs are written to runs/runs/{system}.parquet with columns:
  query_id, paper_id, rank, score
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config, corpus
from .embedders import build

RUNS_SUBDIR = config.RUNS_DIR / "runs"
RUNS_SUBDIR.mkdir(parents=True, exist_ok=True)


def _load_queries() -> pd.DataFrame:
    p = config.RUNS_DIR / "eval_queries.parquet"
    if not p.exists():
        raise FileNotFoundError("runs/eval_queries.parquet missing; run `testset` first.")
    return pd.read_parquet(p)


def _save_run(system: str, rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_parquet(RUNS_SUBDIR / f"{system}.parquet", index=False)


def _topk_dense(doc_mat: np.ndarray, q_mat: np.ndarray, ids: np.ndarray, k: int):
    """Cosine top-k. Vectors assumed L2-normalized -> dot product = cosine."""
    out = []
    for qi in range(q_mat.shape[0]):
        sims = doc_mat @ q_mat[qi]
        if k < len(sims):
            top = np.argpartition(-sims, k)[:k]
            top = top[np.argsort(-sims[top])]
        else:
            top = np.argsort(-sims)
        out.append([(int(ids[j]), float(sims[j])) for j in top])
    return out


def run_dense(depth: int = 100):
    df = corpus.load_corpus()
    ids = np.load(config.EMB_DIR / "ids.npy")
    assert np.array_equal(ids, df["id"].to_numpy()), "corpus/ids misaligned"
    queries = _load_queries()
    qtexts = queries["query_text"].tolist()
    qids = queries["query_id"].tolist()

    for m in config.enabled_dense():
        key = m["key"]
        npy = config.EMB_DIR / f"{key}.npy"
        if not npy.exists():
            print(f"skip dense {key}: no embeddings", flush=True)
            continue
        run_path = RUNS_SUBDIR / f"dense.{key}.parquet"
        if run_path.exists():
            print(f"skip dense {key}: run exists", flush=True)
            continue
        doc_mat = np.load(npy).astype(np.float32)
        # normalize defensively
        doc_mat /= (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12)
        emb = build(m)
        q_mat = emb.encode_queries(qtexts, batch_size=m.get("batch_size", 64)).astype(np.float32)
        q_mat /= (np.linalg.norm(q_mat, axis=1, keepdims=True) + 1e-12)
        results = _topk_dense(doc_mat, q_mat, ids, depth)
        rows = []
        for qid, res in zip(qids, results):
            for rank, (pid, score) in enumerate(res, 1):
                rows.append({"query_id": qid, "paper_id": pid, "rank": rank, "score": score})
        _save_run(f"dense.{key}", rows)
        print(f"dense {key}: {len(rows)} rows", flush=True)
        del doc_mat, emb, q_mat


def run_bm25(depth: int = 100):
    import bm25s
    df = corpus.load_corpus()
    ids = df["id"].to_numpy()
    queries = _load_queries()
    corpus_tokens = bm25s.tokenize(df["doc_text"].tolist(), stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    q_tokens = bm25s.tokenize(queries["query_text"].tolist(), stopwords="en")
    res_idx, res_scores = retriever.retrieve(q_tokens, k=depth)
    rows = []
    for i, qid in enumerate(queries["query_id"].tolist()):
        for rank in range(res_idx.shape[1]):
            j = res_idx[i, rank]
            rows.append({"query_id": qid, "paper_id": int(ids[j]), "rank": rank + 1,
                         "score": float(res_scores[i, rank])})
    _save_run("bm25", rows)
    print(f"bm25: {len(rows)} rows", flush=True)


def run_hybrid(dense_key: str | None = None, rrf_k: int = 60, depth: int = 100):
    """RRF fusion of the best dense run and BM25."""
    bm25_path = RUNS_SUBDIR / "bm25.parquet"
    if not bm25_path.exists():
        print("hybrid: bm25 run missing", flush=True)
        return
    # choose dense run: explicit key or first available
    dense_runs = sorted(RUNS_SUBDIR.glob("dense.*.parquet"))
    if dense_key:
        dense_path = RUNS_SUBDIR / f"dense.{dense_key}.parquet"
    elif dense_runs:
        dense_path = dense_runs[0]
    else:
        print("hybrid: no dense run", flush=True)
        return
    bm25 = pd.read_parquet(bm25_path)
    dense = pd.read_parquet(dense_path)
    rows = []
    for qid in set(bm25["query_id"]) | set(dense["query_id"]):
        scores: dict[int, float] = {}
        for run in (dense, bm25):
            sub = run[run["query_id"] == qid]
            for _, r in sub.iterrows():
                scores[int(r["paper_id"])] = scores.get(int(r["paper_id"]), 0.0) + 1.0 / (rrf_k + r["rank"])
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:depth]
        for rank, (pid, sc) in enumerate(ranked, 1):
            rows.append({"query_id": qid, "paper_id": pid, "rank": rank, "score": sc})
    tag = dense_path.stem.replace("dense.", "")
    _save_run(f"hybrid.{tag}+bm25", rows)
    print(f"hybrid {tag}+bm25: {len(rows)} rows", flush=True)


def run():
    ev = config.eval_cfg()
    depth = ev.get("retrieve_depth", 100)
    run_dense(depth)
    run_bm25(depth)
    # pre-registered dense model for the hybrid (chosen a priori, not by test-set performance)
    run_hybrid(dense_key=ev.get("hybrid_dense_key", "bge-m3"), rrf_k=ev.get("hybrid_rrf_k", 60), depth=depth)
    (RUNS_SUBDIR / "_done.json").write_text(json.dumps({"depth": depth}))
