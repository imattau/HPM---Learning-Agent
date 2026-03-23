"""Tests for SP8: Cross-Task L4 Generalisation benchmark."""
from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.phyre_sim import SceneSnapshot
from benchmarks.phyre_cross_task_l4 import fit_global_l4, score_cross_task_l4, run_benchmark
from hpm.encoders.phyre_encoders import PhyreL2Encoder, PhyreL3Encoder


def _make_snapshot(
    ball_x=250.0, ball_y=400.0, ball_vx=0.0, ball_vy=0.0,
    mass=1.0, restitution=0.8, friction=0.3,
    goal_x=250.0, goal_y=100.0,
) -> SceneSnapshot:
    """Build a minimal 2-object SceneSnapshot (active ball + goal marker)."""
    return SceneSnapshot(
        positions=np.array([[ball_x, ball_y], [goal_x, goal_y]], dtype=np.float32),
        velocities=np.array([[ball_vx, ball_vy], [0.0, 0.0]], dtype=np.float32),
        masses=np.array([mass, 0.0], dtype=np.float32),
        restitutions=np.array([restitution, 0.0], dtype=np.float32),
        frictions=np.array([friction, 0.0], dtype=np.float32),
        goal_pos=np.array([goal_x, goal_y], dtype=np.float32),
        goal_radius=30.0,
        active_ball_idx=0,
    )


def _make_task(n_train: int = 3, correct_idx: int = 0) -> dict:
    """Build a synthetic task dict compatible with the benchmark."""
    init_snap = _make_snapshot(ball_x=250.0, ball_y=400.0)

    train_pairs = []
    for _ in range(n_train):
        final_snap = _make_snapshot(ball_x=250.0, ball_y=110.0, ball_vy=-20.0)
        train_pairs.append({"init": init_snap, "final": final_snap})

    candidates = []
    for i in range(5):
        # Candidate at correct_idx ends near goal (y~100), others are far
        final_y = 105.0 if i == correct_idx else 300.0 + i * 20
        final_snap = _make_snapshot(ball_x=250.0, ball_y=final_y)
        candidates.append({"action": {"family": "Bounce"}, "final": final_snap})

    return {
        "task_id": f"synthetic_{np.random.randint(10000):04d}",
        "family": "Bounce",
        "train": train_pairs,
        "test": {
            "init": init_snap,
            "candidates": candidates,
            "correct_idx": correct_idx,
        },
    }


def _make_task_list(n: int) -> list:
    rng = np.random.default_rng(99)
    return [_make_task(correct_idx=int(rng.integers(0, 5))) for i in range(n)]


class TestFitGlobalL4:
    def test_fit_returns_l4_head(self):
        from hpm.agents.l4_generative import L4GenerativeHead
        tasks = _make_task_list(10)
        l2_enc = PhyreL2Encoder()
        l3_enc = PhyreL3Encoder()
        head = fit_global_l4(tasks, l2_enc, l3_enc)
        assert isinstance(head, L4GenerativeHead)

    def test_fit_produces_valid_weights(self):
        tasks = _make_task_list(10)
        l2_enc = PhyreL2Encoder()
        l3_enc = PhyreL3Encoder()
        head = fit_global_l4(tasks, l2_enc, l3_enc)
        # After fitting on 10 tasks × 3 pairs = 30 pairs, W should not be None
        assert head._W is not None
        assert head._W.shape == (14, 12)

    def test_predict_not_none_after_fit(self):
        tasks = _make_task_list(10)
        l2_enc = PhyreL2Encoder()
        l3_enc = PhyreL3Encoder()
        head = fit_global_l4(tasks, l2_enc, l3_enc)
        dummy_l2 = np.zeros(14, dtype=np.float32)
        pred = head.predict(dummy_l2)
        assert pred is not None
        assert pred.shape == (12,)


class TestScoreCrossTaskL4:
    def test_returns_valid_index(self):
        tasks = _make_task_list(10)
        l2_enc = PhyreL2Encoder()
        l3_enc = PhyreL3Encoder()
        global_l4 = fit_global_l4(tasks[:8], l2_enc, l3_enc)
        for task in tasks[8:]:
            idx = score_cross_task_l4(task, global_l4, l2_enc, l3_enc)
            assert 0 <= idx < 5

    def test_fallback_when_head_unfitted(self):
        from hpm.agents.l4_generative import L4GenerativeHead
        task = _make_task()
        l2_enc = PhyreL2Encoder()
        l3_enc = PhyreL3Encoder()
        # Head with no data → _W is None → fallback to goal flag
        empty_head = L4GenerativeHead(feature_dim_in=14, feature_dim_out=12)
        idx = score_cross_task_l4(task, empty_head, l2_enc, l3_enc)
        assert 0 <= idx < 5


class TestRunBenchmark:
    def test_smoke_accuracy_in_range(self):
        tasks = _make_task_list(13)
        results = run_benchmark(tasks, seed=42, train_frac=0.8)
        assert set(results.keys()) == {"flat", "l2l3", "per_task_l4", "cross_task_l4"}
        for condition, acc in results.items():
            assert 0.0 <= acc <= 1.0, f"{condition} accuracy {acc} out of [0,1]"

    def test_all_conditions_return_float(self):
        tasks = _make_task_list(13)
        results = run_benchmark(tasks, seed=0, train_frac=0.7)
        for condition, acc in results.items():
            assert isinstance(acc, float), f"{condition} did not return float"

    def test_reproducible_with_same_seed(self):
        tasks = _make_task_list(20)
        r1 = run_benchmark(tasks, seed=7, train_frac=0.8)
        r2 = run_benchmark(tasks, seed=7, train_frac=0.8)
        # Deterministic conditions should match
        assert r1["l2l3"] == r2["l2l3"]
        assert r1["per_task_l4"] == r2["per_task_l4"]
        assert r1["cross_task_l4"] == r2["cross_task_l4"]

    def test_split_sizes(self):
        """run_benchmark completes without error for various train_frac values."""
        tasks = _make_task_list(20)
        results = run_benchmark(tasks, seed=42, train_frac=0.8)
        assert isinstance(results["cross_task_l4"], float)
