"""Stage 3: build the test collection.

generate(): synthetic NL queries (LLM) + known-item queries (judgment-free) -> eval_queries.parquet
            and the known-item / seed qrels.
judge():    pool candidates across first-stage runs, LLM-judge graded relevance -> eval_qrels.parquet
            (run AFTER retrieve).
"""
from __future__ import annotations
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import config, corpus

QUERIES_PARQUET = config.RUNS_DIR / "eval_queries.parquet"
QRELS_PARQUET = config.RUNS_DIR / "eval_qrels.parquet"


def _ollama_chat(model: str, prompt: str, temperature: float = 0.3, think=None) -> str:
    import ollama
    # think=False disables reasoning for models that support it (e.g. qwen3.6): the judge only
    # needs to emit a single digit, so skipping the hidden chain-of-thought is ~16x faster and
    # equivalent. Pass think=None to leave the model default untouched (e.g. for query generation).
    kw = {} if think is None else {"think": think}
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": temperature}, **kw)
    return resp["message"]["content"]


_QWORDS = ("how", "what", "why", "does", "do", "is", "are", "can", "which", "when", "should", "who", "where")


def _clean_queries(text: str, n: int) -> list[str]:
    """Keep short search-box PHRASES/fragments; reject full sentences and questions."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line)  # strip bullets/numbering
        line = line.strip().strip('"').strip()
        if not line:
            continue
        words = line.split()
        if not (2 <= len(words) <= 8):
            continue
        if "?" in line or line.endswith("."):          # no questions or full sentences
            continue
        if words[0].lower() in _QWORDS:                # no question/sentence stems
            continue
        if line.lower().startswith(("here", "query", "sure", "keyword", "phrase")):
            continue
        out.append(line)
    return out[:n]


def generate():
    jc = config.judge_cfg()
    df = corpus.load_corpus()
    df = df[df["abstract"].str.len() > 100].reset_index(drop=True)
    rng = np.random.default_rng(jc.get("sample_seed", 0))

    n_seed = jc.get("n_seed_docs", 250)
    per_seed = jc.get("queries_per_seed", 2)
    n_ki = jc.get("n_known_item", 250)
    gen_model = jc["generator_model"]

    seed_idx = rng.choice(len(df), size=min(n_seed, len(df)), replace=False)
    queries, qrels = [], []

    for idx in tqdm(seed_idx, desc="synthetic", unit="seed"):
        row = df.iloc[int(idx)]
        prompt = jc["generator_prompt"].format(n=per_seed, title=row["title"], abstract=row["abstract"][:2000])
        try:
            qs = _clean_queries(_ollama_chat(gen_model, prompt), per_seed)
        except Exception as e:
            print(f"gen fail {row['id']}: {e}", flush=True)
            continue
        for i, q in enumerate(qs):
            qid = f"syn-{int(row['id'])}-{i}"
            # queries are general topic searches; the seed is recorded for provenance only and is
            # rated by the judge like any other candidate (no auto-relevant seed qrel).
            queries.append({"query_id": qid, "subset": "synthetic", "query_text": q,
                            "seed_paper_id": int(row["id"]), "source": gen_model})

    # known-item queries (judgment-free; seed is the single relevant target)
    ki_idx = rng.choice(len(df), size=min(n_ki, len(df)), replace=False)
    ki_prompt = (
        "Write ONE specific search query (8-18 words) that uniquely identifies the paper below, "
        "as if searching for this exact paper. Return only the query.\n\nTitle: {title}\nAbstract: {abstract}"
    )
    for idx in tqdm(ki_idx, desc="known-item", unit="seed"):
        row = df.iloc[int(idx)]
        try:
            q = _clean_queries(_ollama_chat(gen_model, ki_prompt.format(title=row["title"], abstract=row["abstract"][:1500])), 1)
        except Exception as e:
            print(f"ki fail {row['id']}: {e}", flush=True)
            continue
        if not q:
            continue
        qid = f"ki-{int(row['id'])}"
        queries.append({"query_id": qid, "subset": "known_item", "query_text": q[0],
                        "seed_paper_id": int(row["id"]), "source": gen_model})
        qrels.append({"query_id": qid, "paper_id": int(row["id"]), "relevance": 3,
                      "judge": "known_item", "judge_model": gen_model})

    pd.DataFrame(queries).to_parquet(QUERIES_PARQUET, index=False)
    if qrels:  # only known-item qrels, if any; synthetic qrels come from judging
        pd.DataFrame(qrels).to_parquet(QRELS_PARQUET, index=False)
    print(f"generated {len(queries)} queries (phrase-style); qrels come from the judging stage", flush=True)


def _parse_grade(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"[0-3]", text)
    return int(m.group()) if m else None


def judge():
    """Pool first-stage candidates and LLM-judge graded relevance for synthetic queries."""
    from .retrieve import RUNS_SUBDIR
    jc = config.judge_cfg()
    pool_depth = jc.get("judge_pool_depth", 10)
    judge_model = jc["judge_model"]
    df = corpus.load_corpus().set_index("id")
    queries = pd.read_parquet(QUERIES_PARQUET)
    qmap = dict(zip(queries["query_id"], queries["query_text"]))
    syn_qids = set(queries[queries["subset"] == "synthetic"]["query_id"])

    # cross-system pool: every retrieval system contributes its top candidates
    run_files = list(RUNS_SUBDIR.glob("dense.*.parquet")) + [RUNS_SUBDIR / "bm25.parquet"]
    run_files += list(RUNS_SUBDIR.glob("hybrid.*.parquet"))
    run_files += list(RUNS_SUBDIR.glob("rerank.*.parquet"))  # include reranker-surfaced studies
    pools: dict[str, set] = {}
    for rf in run_files:
        if not rf.exists():
            continue
        r = pd.read_parquet(rf)
        r = r[r["rank"] <= pool_depth]
        for qid, pid in zip(r["query_id"], r["paper_id"]):
            if qid in syn_qids:
                pools.setdefault(qid, set()).add(int(pid))

    existing = pd.read_parquet(QRELS_PARQUET) if QRELS_PARQUET.exists() else pd.DataFrame()
    base_rows = existing.to_dict("records") if len(existing) else []
    judged_pairs = set(zip(existing["query_id"], existing["paper_id"])) if len(existing) else set()
    new_rows = []
    checkpoint = int(jc.get("judge_checkpoint", 250))
    # reasoning off by default (single-digit answer; ~16x faster and consistent). See _ollama_chat.
    think = jc.get("judge_think", False)
    # concurrent requests: match the Ollama server's OLLAMA_NUM_PARALLEL (extra workers just queue).
    workers = int(os.environ.get("JUDGE_WORKERS", 0) or jc.get("judge_workers", 0)
                  or os.environ.get("OLLAMA_NUM_PARALLEL", 3))
    workers = max(1, workers)

    def _flush():  # resume-safe: persist progress so an interrupted run loses nothing
        pd.DataFrame(base_rows + new_rows).to_parquet(QRELS_PARQUET, index=False)

    # build the work list once (skip already-judged pairs and docs missing from the corpus)
    tasks = [(qid, pid) for qid, pids in pools.items() for pid in pids
             if (qid, pid) not in judged_pairs and pid in df.index]

    def _judge_one(task):
        qid, pid = task
        row = df.loc[pid]
        prompt = jc["judge_prompt"].format(query=qmap[qid], title=row["title"],
                                           abstract=(row["abstract"] or "")[:2000])
        try:
            grade = _parse_grade(_ollama_chat(judge_model, prompt, temperature=0.0, think=think))
        except Exception:
            grade = None
        return qid, int(pid), grade

    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"judging {len(tasks)} pairs with {workers} concurrent workers (think={think})", flush=True)
    with tqdm(total=len(tasks), desc="judge", unit="pair") as bar, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_judge_one, t) for t in tasks]):
            bar.update(1)
            qid, pid, grade = fut.result()
            if grade is None:
                continue
            new_rows.append({"query_id": qid, "paper_id": pid, "relevance": grade,
                             "judge": "llm", "judge_model": judge_model})
            if len(new_rows) % checkpoint == 0:
                _flush()
    _flush()
    print(f"judged {len(new_rows)} new pairs; total qrels {len(base_rows)+len(new_rows)}", flush=True)


HUMAN_CSV = config.REPORTS_DIR / "human_validation.csv"


def export_human_validation(n: int = 150, seed: int = 20260627):
    """Export a stratified, blinded sample of model-rated pairs for a human expert to re-rate.
    The expert fills the empty `human_grade` column (0-3); model grades are NOT shown (blind)."""
    qrels = pd.read_parquet(QRELS_PARQUET)
    qrels = qrels[qrels["judge"] == "llm"]  # the single-model ratings
    queries = pd.read_parquet(QUERIES_PARQUET)
    qmap = dict(zip(queries["query_id"], queries["query_text"]))
    df = corpus.load_corpus().set_index("id")
    rng = np.random.default_rng(seed)
    per = max(1, n // 4)
    picks = []
    for g in [0, 1, 2, 3]:
        sub = qrels[qrels["relevance"] == g]
        take = min(per, len(sub))
        if take:
            picks.append(sub.sample(take, random_state=int(rng.integers(1e9))))
    sample = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    rows = []
    for _, r in sample.iterrows():
        pid = int(r["paper_id"])
        d = df.loc[pid] if pid in df.index else {"title": "", "abstract": ""}
        rows.append({"pair_id": f"{r['query_id']}__{pid}", "query_id": r["query_id"],
                     "query_text": qmap.get(r["query_id"], ""), "paper_id": pid,
                     "title": d["title"], "abstract": (d["abstract"] or "")[:2000], "human_grade": ""})
    out = pd.DataFrame(rows)
    out.to_csv(HUMAN_CSV, index=False)
    print(f"wrote {len(out)} pairs to {HUMAN_CSV} (blind; fill the human_grade column 0-3)")
    return out


def _quadratic_weighted_kappa(a, b, k=4):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    act_a = O.sum(1); act_b = O.sum(0); n = O.sum()
    E = np.outer(act_a, act_b) / n
    return 1 - (w * O).sum() / (w * E).sum()


def compute_human_agreement(completed_csv: str | None = None):
    """After the expert fills human_grade, compute model-vs-human agreement."""
    path = completed_csv or str(HUMAN_CSV)
    h = pd.read_csv(path)
    h = h[h["human_grade"].apply(lambda x: str(x).strip() not in ("", "nan"))].copy()
    h["human_grade"] = h["human_grade"].astype(int)
    qrels = pd.read_parquet(QRELS_PARQUET)
    qrels = qrels[qrels["judge"] == "llm"].copy()
    qrels["pair_id"] = qrels["query_id"].astype(str) + "__" + qrels["paper_id"].astype(str)
    mg = dict(zip(qrels["pair_id"], qrels["relevance"]))
    h["model_grade"] = h["pair_id"].map(mg)
    h = h.dropna(subset=["model_grade"]); h["model_grade"] = h["model_grade"].astype(int)
    exact = float((h["human_grade"] == h["model_grade"]).mean())
    within1 = float((abs(h["human_grade"] - h["model_grade"]) <= 1).mean())
    qwk = _quadratic_weighted_kappa(h["model_grade"], h["human_grade"])
    res = {"n": len(h), "percent_agreement": round(exact * 100, 1),
           "within_one_pct": round(within1 * 100, 1), "weighted_kappa": round(float(qwk), 3)}
    (config.RUNS_DIR / "human_agreement.json").write_text(json.dumps(res, indent=2))
    print(res)
    return res


def run():
    generate()
