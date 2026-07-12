"""Pairwise-comparison relevance judging.

For each query, pool candidate docs (depth-10 union across systems, capped), then have an LLM judge
decide "which of two papers better answers this query?" over an adaptive-but-simple random-pairing
budget. Comparisons from one or more judges are pooled. Resume-safe + parallel.

Output: runs/pairwise_comparisons.parquet  (query_id, doc_a, doc_b, winner, judge_model)
Run:  JUDGE_MODEL=qwen3.6:35b M=10 PYTHONPATH=src python pairwise_judge.py [--limit N]
"""
from __future__ import annotations
import os, re, sys, glob, random
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ollama
from sswr_eval import config, corpus

JUDGE = os.environ.get("JUDGE_MODEL", "qwen3.6:35b")
M = int(os.environ.get("M", "10"))          # comparisons per doc (≈)
POOL_DEPTH = 10
POOL_CAP = 40                                # max candidates/query (most-pooled)
WORKERS = int(os.environ.get("JUDGE_WORKERS", "3"))
SEED = 20260630
OUT = config.RUNS_DIR / os.environ.get("JUDGE_OUT", "pairwise_comparisons.parquet")

# Judge client: local Ollama by default, or Ollama Cloud when JUDGE_HOST is set (frontier models).
_HOST = os.environ.get("JUDGE_HOST")
if _HOST:
    _key = os.environ.get("OLLAMA_API_KEY") or next(
        l.split("=", 1)[1].strip() for l in open(config.ROOT / ".env") if l.startswith("OLLAMA_API_KEY="))
    CLIENT = ollama.Client(host=_HOST, headers={"Authorization": "Bearer " + _key}, timeout=90)
else:
    CLIENT = ollama

import time, threading
_STATS = {"parse_fail": 0, "retries": 0, "errors": 0}   # parse_fail=billed-but-unusable (overspend signal)
_SLOCK = threading.Lock()

PROMPT = """You are an expert social work researcher. A user ran this search query:

QUERY: {query}

Two papers were retrieved. Decide which paper BETTER answers the user's information need.

PAPER A:
Title: {ta}
Abstract: {aa}

PAPER B:
Title: {tb}
Abstract: {ab}

Which paper better answers the query? Reply with ONLY the single letter A or B."""


def build_pools(limit=None):
    files = glob.glob("runs/runs/dense.*.parquet") + glob.glob("runs/runs/rerank.*.parquet") \
        + glob.glob("runs/runs/hybrid.*.parquet") + ["runs/runs/bm25.parquet"]
    cnt = {}
    for f in files:
        if not os.path.exists(f):
            continue
        r = pd.read_parquet(f)
        r = r[r["rank"] <= POOL_DEPTH]
        for qid, pid in zip(r["query_id"], r["paper_id"]):
            cnt.setdefault(qid, Counter())[int(pid)] += 1
    pools = {}
    for qid, c in cnt.items():
        pools[qid] = [pid for pid, _ in c.most_common(POOL_CAP)]
    qids = sorted(pools)
    if limit:
        qids = qids[:limit]
    return {q: pools[q] for q in qids}


def gen_pairs(docs, m, rng):
    """~m comparisons per doc via repeated random-permutation pairing (connected graph)."""
    pairs = set()
    n = len(docs)
    if n < 2:
        return []
    rounds = max(1, m // 2)
    for _ in range(rounds * 2):
        perm = docs[:]
        rng.shuffle(perm)
        for i in range(0, n - 1, 2):
            a, b = perm[i], perm[i + 1]
            pairs.add((min(a, b), max(a, b)))
    return list(pairs)


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    pools = build_pools(limit)
    queries = pd.read_parquet("runs/eval_queries.parquet")
    qmap = dict(zip(queries["query_id"], queries["query_text"]))
    df = corpus.load_corpus().set_index("id")
    rng = random.Random(SEED)

    # build all comparison tasks
    tasks = []
    for qid, docs in pools.items():
        for a, b in gen_pairs(docs, M, rng):
            # randomize display order to cancel position bias
            if rng.random() < 0.5:
                tasks.append((qid, a, b))
            else:
                tasks.append((qid, b, a))

    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")                       # preserve ALL judges' prior rows
        same = prev[prev["judge_model"] == JUDGE]            # dedup only against the CURRENT judge
        for q, da, db in zip(same["query_id"], same["doc_a"], same["doc_b"]):
            done.add((q, int(da), int(db)))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done]

    def judge_one(task):
        qid, da, db = task
        ra, rb = df.loc[da], df.loc[db]
        prompt = PROMPT.format(query=qmap[qid], ta=ra["title"], aa=(ra["abstract"] or "")[:1100],
                               tb=rb["title"], ab=(rb["abstract"] or "")[:1100])
        for attempt in range(6):
            try:
                r = CLIENT.chat(model=JUDGE, messages=[{"role": "user", "content": prompt}],
                                options={"temperature": 0.0}, think=False)
                t = re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.DOTALL | re.I)
                mch = re.search(r"\b([AB])\b", t.strip().upper())
                if not mch:
                    with _SLOCK: _STATS["parse_fail"] += 1     # billed but unparseable
                    return None
                winner = da if mch.group(1) == "A" else db
                return {"query_id": qid, "doc_a": int(da), "doc_b": int(db), "winner": int(winner), "judge_model": JUDGE}
            except Exception as e:
                msg = str(e).lower()
                transient = any(k in msg for k in ("429", "rate", "throttl", "timeout", "timed out",
                                                   "overload", "temporar", "connection", "500", "502", "503"))
                with _SLOCK:
                    _STATS["retries" if transient else "errors"] += 1
                if transient and attempt < 5:
                    time.sleep(min(30.0, 1.5 * (2 ** attempt)))   # exponential backoff on rate limits
                    continue
                return None
        return None

    print(f"pairwise judging: {len(tasks)} comparisons to do ({len(done)} cached) | judge={JUDGE} | {len(pools)} queries", flush=True)
    n_new = 0
    with tqdm(total=len(tasks), desc="pairwise", unit="cmp") as bar, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(judge_one, t) for t in tasks]):
            bar.update(1)
            res = fut.result()
            if res:
                rows.append(res); n_new += 1
                if n_new % 500 == 0:
                    pd.DataFrame(rows).to_parquet(OUT, index=False)
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print(f"done: {n_new} new comparisons; total {len(rows)} -> {OUT}", flush=True)
    print(f"drops: parse_fail={_STATS['parse_fail']} (billed-but-unusable), "
          f"transient_retries={_STATS['retries']}, hard_errors={_STATS['errors']}", flush=True)


if __name__ == "__main__":
    main()
