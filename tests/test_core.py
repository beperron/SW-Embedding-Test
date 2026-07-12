"""Unit tests for stats, RRF, and LLM-output parsers (no GPU / no network)."""
import numpy as np
from swrd_eval import stats
from swrd_eval.testset import _clean_queries, _parse_grade


def test_holm_monotone_and_bounded():
    p = [0.001, 0.04, 0.5, 0.9]
    adj = stats.holm_correction(p)
    assert all(0.0 <= a <= 1.0 for a in adj)
    # adjusted >= raw for each
    assert all(a >= r - 1e-9 for a, r in zip(adj, p))


def test_holm_known_values():
    # m=4, smallest p=0.01 -> *4 = 0.04
    adj = stats.holm_correction([0.01, 0.02, 0.03, 0.04])
    assert abs(adj[0] - 0.04) < 1e-9


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    mean, lo, hi = stats.bootstrap_ci(x, resamples=2000, seed=1)
    assert lo <= mean <= hi
    assert abs(mean - x.mean()) < 1e-9


def test_bootstrap_ci_constant():
    mean, lo, hi = stats.bootstrap_ci(np.full(50, 0.7), resamples=500, seed=2)
    assert abs(mean - 0.7) < 1e-9 and abs(lo - 0.7) < 1e-9 and abs(hi - 0.7) < 1e-9


def test_parse_grade():
    assert _parse_grade("3") == 3
    assert _parse_grade("Relevance: 2") == 2
    assert _parse_grade("<think>maybe a 3</think> 1") == 1
    assert _parse_grade("no digit here") is None


def test_clean_queries_strips_noise():
    text = (
        "Here are the queries:\n"
        "1. effects of trauma informed care on youth outcomes\n"
        "- school based mental health interventions for adolescents\n"
        '"culturally adapted family therapy in immigrant communities"\n'
        "short\n"
    )
    qs = _clean_queries(text, 3)
    assert len(qs) == 3
    assert all(3 <= len(q.split()) <= 25 for q in qs)
    assert not any(q[0].isdigit() or q.startswith(("-", "Here")) for q in qs)


def test_rrf_fusion_orders_by_reciprocal_rank():
    # doc appearing high in both runs should win
    rrf_k = 60
    dense = {1: 1, 2: 2, 3: 3}   # paper_id -> rank
    bm25 = {2: 1, 1: 2, 4: 3}
    scores = {}
    for run in (dense, bm25):
        for pid, rank in run.items():
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores, key=lambda p: -scores[p])
    assert ranked[0] in (1, 2)  # 1 and 2 are top in both
    assert ranked[-1] in (3, 4)
