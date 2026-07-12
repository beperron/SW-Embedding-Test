"""Paths and config loading."""
from __future__ import annotations
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
EMB_DIR = RUNS_DIR / "embeddings"
LOG_DIR = RUNS_DIR / "logs"

for _d in (RUNS_DIR, REPORTS_DIR, EMB_DIR, LOG_DIR, REPORTS_DIR / "figures"):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def models_cfg() -> dict:
    return _load_yaml("models.yaml")


def eval_cfg() -> dict:
    return _load_yaml("eval.yaml")


def judge_cfg() -> dict:
    return _load_yaml("judge.yaml")


def enabled_dense() -> list[dict]:
    return [m for m in models_cfg()["dense"] if m.get("enabled")]


def enabled_rerankers() -> list[dict]:
    return [m for m in models_cfg()["rerankers"] if m.get("enabled")]


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
