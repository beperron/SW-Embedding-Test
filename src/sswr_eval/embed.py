"""Stage 2: embed the corpus with each enabled dense model. Local-first storage.

Per model we write:
  runs/embeddings/{key}.npy        float32 [N, dim], L2-normalized, aligned to corpus.parquet row order
  runs/embeddings/{key}.meta.json  shape, timing, throughput, peak RSS (for the cost analysis)
The id->row alignment is the corpus.parquet 'id' column order, saved once as ids.npy.
"""
from __future__ import annotations
import json
import time
import resource
import numpy as np
from . import config, corpus
from .embedders import build

IDS_NPY = config.EMB_DIR / "ids.npy"


def _ensure_ids(df) -> np.ndarray:
    ids = df["id"].to_numpy()
    if not IDS_NPY.exists():
        np.save(IDS_NPY, ids)
    else:
        saved = np.load(IDS_NPY)
        if not np.array_equal(saved, ids):
            raise RuntimeError("corpus id order changed; delete runs/embeddings and re-run.")
    return ids


def embed_model(model_cfg: dict, df, limit: int | None = None) -> dict:
    key = model_cfg["key"]
    out_npy = config.EMB_DIR / f"{key}.npy"
    meta_path = config.EMB_DIR / f"{key}.meta.json"
    texts = df["doc_text"].tolist()
    if limit:
        texts = texts[:limit]

    if out_npy.exists() and meta_path.exists():
        arr = np.load(out_npy, mmap_mode="r")
        if arr.shape[0] == len(texts):
            return {"key": key, "status": "skipped", "shape": list(arr.shape)}

    emb = build(model_cfg)
    t0 = time.time()
    vecs = emb.encode_docs(texts, batch_size=model_cfg.get("batch_size", 64))
    dt = time.time() - t0
    vecs = np.ascontiguousarray(vecs.astype(np.float32))
    np.save(out_npy, vecs)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes->MB on macOS
    meta = {
        "key": key,
        "hf_id": model_cfg["hf_id"],
        "backend": model_cfg["backend"],
        "dim": int(vecs.shape[1]),
        "n_docs": int(vecs.shape[0]),
        "seconds": round(dt, 2),
        "docs_per_sec": round(vecs.shape[0] / dt, 2),
        "peak_rss_mb": round(peak_mb, 1),
        "tier": model_cfg.get("tier"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"status": "embedded", **meta}


def run(only: list[str] | None = None, limit: int | None = None) -> list[dict]:
    df = corpus.load_corpus()
    _ensure_ids(df)
    results = []
    for m in config.enabled_dense():
        if only and m["key"] not in only:
            continue
        print(f"\n=== embedding {m['key']} ({m['hf_id']}) ===", flush=True)
        try:
            res = embed_model(m, df, limit=limit)
        except Exception as e:  # keep going; record failure
            res = {"key": m["key"], "status": "failed", "error": str(e)}
            print(f"!! {m['key']} failed: {e}", flush=True)
        print(res, flush=True)
        results.append(res)
    (config.EMB_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
    return results
