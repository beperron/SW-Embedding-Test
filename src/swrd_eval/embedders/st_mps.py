"""sentence-transformers backend on Apple MPS."""
from __future__ import annotations
import numpy as np


class STEmbedder:
    def __init__(self, cfg: dict):
        from sentence_transformers import SentenceTransformer
        import torch

        self.cfg = cfg
        self.normalize = cfg.get("normalize", True)
        self.doc_prefix = cfg.get("doc_prefix", "") or ""
        self.query_prefix = cfg.get("query_prefix", "") or ""
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        model_kwargs = {}
        if cfg.get("tier", 1) >= 2:  # load multi-billion-param decoders in fp16 to fit/speed up
            model_kwargs["torch_dtype"] = torch.float16
        self.model = SentenceTransformer(
            cfg["hf_id"],
            device=device,
            trust_remote_code=cfg.get("trust_remote_code", False),
            model_kwargs=model_kwargs or None,
        )
        if cfg.get("max_seq"):
            self.model.max_seq_length = cfg["max_seq"]

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        inputs = [prefix + t for t in texts] if prefix else texts
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, self.doc_prefix, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, self.query_prefix, batch_size)
