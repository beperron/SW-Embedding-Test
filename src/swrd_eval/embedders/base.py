"""Embedder protocol and factory."""
from __future__ import annotations
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray: ...
    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray: ...


def build(model_cfg: dict) -> Embedder:
    backend = model_cfg.get("backend", "st")
    if backend == "st":
        from .st_mps import STEmbedder
        return STEmbedder(model_cfg)
    if backend == "flag":
        from .flag import FlagEmbedder
        return FlagEmbedder(model_cfg)
    if backend == "mlx":
        from .mlx_backend import MLXEmbedder
        return MLXEmbedder(model_cfg)
    if backend == "ollama":
        from .ollama_backend import OllamaEmbedder
        return OllamaEmbedder(model_cfg)
    raise ValueError(f"unknown backend {backend}")
