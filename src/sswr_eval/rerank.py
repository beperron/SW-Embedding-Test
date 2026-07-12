"""Stage 5: rerank pooled first-stage candidates with cross-encoders.

For each enabled reranker and each first-stage dense run, re-score the top-N candidates
and write runs/runs/rerank.{reranker}.{base}.parquet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config, corpus
from .retrieve import RUNS_SUBDIR
from .testset import QUERIES_PARQUET


def _mxbai_prepare_inputs(self, queries, documents, *, instruction=None):
    """transformers-5-compatible replacement for MxbaiRerankV2.prepare_inputs.

    The upstream method calls tokenizer.prepare_for_model(), removed in transformers 5.x.
    This reproduces it (concatenate query + sep + doc ids, truncate the doc side to max_length,
    then wrap with the model's chat/task-prompt templates) using the current tokenizer API.
    """
    inputs = []
    instr = self.instruction_prompt.format(instruction=instruction) if instruction else None
    for query, document in zip(queries, documents):
        qp = self.query_prompt.format(query=query)
        if instr:
            qp = "".join([instr, self.sep, qp])
        qi = self.tokenizer(qp, return_tensors=None, add_special_tokens=False,
                            max_length=self.max_length * 3 // 4, truncation=True)["input_ids"]
        doc_maxlen = min(self.model_max_length - len(qi) - self.predefined_length, self.max_length)
        di = self.tokenizer(self.doc_prompt.format(document=document), return_tensors=None,
                            add_special_tokens=False, max_length=doc_maxlen, truncation=True)["input_ids"]
        ids1, ids2 = list(qi), list(self.sep_inputs) + list(di)
        combined = ids1 + ids2
        if len(combined) > self.max_length:                      # truncation="only_second"
            ids2 = ids2[:max(0, self.max_length - len(ids1))]
            combined = ids1 + ids2
        combined = self.concat_input_ids(combined)
        inputs.append({"input_ids": combined, "attention_mask": [1] * len(combined)})
    return self.tokenizer.pad(inputs, padding="longest", max_length=self.max_length_padding,
                              pad_to_multiple_of=8, return_tensors="pt")


class _CrossEncoder:
    def __init__(self, cfg: dict):
        self.backend = cfg["backend"]
        self.cfg = cfg
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        if cfg["backend"] == "flag":
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(cfg["hf_id"], use_fp16=True)
        elif cfg["backend"] == "mxbai":
            # mxbai-rerank-v2 is a generative (yes/no logit) reranker, not a plain cross-encoder;
            # it needs its own package + a transformers-5 patch (see _mxbai_prepare_inputs).
            import types
            from mxbai_rerank import MxbaiRerankV2
            self.model = MxbaiRerankV2(cfg["hf_id"], device=device)
            self.model.prepare_inputs = types.MethodType(_mxbai_prepare_inputs, self.model)
        else:  # st
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(cfg["hf_id"], device=device, trust_remote_code=True)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.backend == "flag":
            return list(self.model.compute_score(pairs, normalize=True))
        if self.backend == "mxbai":
            bs = self.cfg.get("batch_size", 32)
            qs = [p[0] for p in pairs]
            ds = [p[1] for p in pairs]
            out: list[float] = []
            for i in range(0, len(qs), bs):
                sc = self.model.predict(qs[i:i + bs], ds[i:i + bs])
                out.extend(float(x) for x in sc)
            return out
        return list(self.model.predict(pairs, batch_size=self.cfg.get("batch_size", 32)))


def run(top_n: int = 100):
    queries = pd.read_parquet(QUERIES_PARQUET)
    qmap = dict(zip(queries["query_id"], queries["query_text"]))
    df = corpus.load_corpus().set_index("id")
    dense_runs = sorted(RUNS_SUBDIR.glob("dense.*.parquet"))
    if not dense_runs:
        print("rerank: no dense runs", flush=True)
        return

    for rcfg in config.enabled_rerankers():
        rkey = rcfg["key"]
        try:
            ce = _CrossEncoder(rcfg)
        except Exception as e:
            print(f"rerank {rkey} load failed: {e}", flush=True)
            continue
        for base in dense_runs:
            base_tag = base.stem.replace("dense.", "")
            out_path = RUNS_SUBDIR / f"rerank.{rkey}.{base_tag}.parquet"
            if out_path.exists():
                continue
            run = pd.read_parquet(base)
            run = run[run["rank"] <= top_n]
            rows = []
            for qid, grp in run.groupby("query_id"):
                qtext = qmap.get(qid)
                if qtext is None:
                    continue
                pids = [int(p) for p in grp["paper_id"]]
                pairs = [(qtext, (df.loc[pid]["doc_text"] if pid in df.index else "")) for pid in pids]
                scores = ce.score(pairs)
                order = np.argsort(-np.asarray(scores))
                for rank, oi in enumerate(order, 1):
                    rows.append({"query_id": qid, "paper_id": pids[oi], "rank": rank,
                                 "score": float(scores[oi])})
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"rerank {rkey} on {base_tag}: {len(rows)} rows", flush=True)
        del ce
