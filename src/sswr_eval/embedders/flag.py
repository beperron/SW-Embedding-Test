"""FlagEmbedding backend (BGE-M3). Optional dependency."""
from __future__ import annotations
import numpy as np


class FlagEmbedder:
    def __init__(self, cfg: dict):
        from FlagEmbedding import BGEM3FlagModel

        self.cfg = cfg
        self.model = BGEM3FlagModel(cfg["hf_id"], use_fp16=True)
        self.max_len = cfg.get("max_seq", 1024)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = self.model.encode(texts, batch_size=batch_size, max_length=self.max_len)["dense_vecs"]
        return np.asarray(out, dtype=np.float32)

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size)
