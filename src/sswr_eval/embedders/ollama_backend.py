"""Ollama embedding backend (for locally-pulled models, e.g. Google EmbeddingGemma).

Avoids Hugging Face license gating by using the already-pulled Ollama model.
Task-specific prefixes (doc/query) are prepended to the text, matching the model card.
"""
from __future__ import annotations
import numpy as np
from tqdm import tqdm


class OllamaEmbedder:
    def __init__(self, cfg: dict):
        import ollama
        self.ollama = ollama
        self.cfg = cfg
        self.model = cfg["hf_id"]  # here hf_id holds the Ollama tag (e.g. "embeddinggemma:latest")
        self.normalize = cfg.get("normalize", True)
        self.doc_prefix = cfg.get("doc_prefix", "") or ""
        self.query_prefix = cfg.get("query_prefix", "") or ""

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"ollama:{self.model}", unit="batch"):
            chunk = [prefix + t for t in texts[i:i + batch_size]]
            resp = self.ollama.embed(model=self.model, input=chunk)
            vecs.extend(resp["embeddings"])
        arr = np.asarray(vecs, dtype=np.float32)
        if self.normalize:
            arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, self.doc_prefix, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, self.query_prefix, batch_size)
