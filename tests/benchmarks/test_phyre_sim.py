import numpy as np
import pytest
import os
from benchmarks.phyre_sim import SceneSnapshot, simulate_scene, check_goal
from benchmarks.phyre_sim import generate_family_tasks, save_tasks, load_tasks


def _two_object_scene():
    """Minimal deterministic scene: one active ball above goal, no action object."""
    return SceneSnapshot(
        positions=np.array([[250.0, 400.0], [250.0, 50.0]]),
        velocities=np.array([[0.0, 0.0], [0.0, 0.0]]),
        masses=np.array([1.0, 0.0]),
        restitutions=np.array([0.3, 0.0]),
        frictions=np.array([0.5, 0.0]),
        goal_pos=np.array([250.0, 50.0]),
        goal_radius=30.0,
        active_ball_idx=0,
    )


def test_scene_snapshot_fields():
    snap = _two_object_scene()
    assert snap.positions.shape == (2, 2)
    assert snap.velocities.shape == (2, 2)
    assert snap.active_ball_idx == 0
    assert snap.goal_radius == 30.0


def test_simulate_scene_returns_snapshot():
    initial = _two_object_scene()
    final = simulate_scene(initial, action_obj=None, n_steps=120, dt=1/60)
    assert isinstance(final, SceneSnapshot)
    assert final.positions.shape == initial.positions.shape


def test_simulate_scene_ball_falls():
    """Ball with initial downward velocity should move closer to the floor."""
    initial = _two_object_scene()
    initial.velocities[0] = np.array([0.0, -100.0])
    final = simulate_scene(initial, action_obj=None, n_steps=120, dt=1/60)
    assert final.positions[0, 1] < initial.positions[0, 1]


def test_check_goal_true_when_ball_in_region():
    snap = _two_object_scene()
    snap.positions[0] = np.array([250.0, 55.0])
    assert check_goal(snap) is True


def test_check_goal_false_when_ball_outside():
    snap = _two_object_scene()
    snap.positions[0] = np.array([0.0, 400.0])
    assert check_goal(snap) is False


# ---------------------------------------------------------------------------
# Task 2: generation tests
# ---------------------------------------------------------------------------

def test_generate_family_tasks_schema():
    tasks = generate_family_tasks("Bounce", n_tasks=3, seed=42)
    assert len(tasks) == 3
    for t in tasks:
        assert "task_id" in t
        assert "family" in t and t["family"] == "Bounce"
        assert "train" in t and len(t["train"]) >= 1
        assert "test" in t
        test = t["test"]
        assert "init" in test
        assert "candidates" in test and len(test["candidates"]) == 5
        assert "correct_idx" in test
        assert isinstance(test["correct_idx"], int)
        assert 0 <= test["correct_idx"] < 5


def test_generate_family_tasks_correct_idx_works():
    """The candidate at correct_idx must achieve the goal."""
    tasks = generate_family_tasks("Projectile", n_tasks=3, seed=42)
    for t in tasks:
        test = t["test"]
        correct_candidate = test["candidates"][test["correct_idx"]]
        final = simulate_scene(test["init"], correct_candidate["action"])
        assert check_goal(final), f"Correct candidate failed goal check for {t['task_id']}"


def test_generate_all_families():
    for family in ["Projectile", "Bounce", "Slide", "Collision"]:
        tasks = generate_family_tasks(family, n_tasks=2, seed=0)
        assert len(tasks) == 2
        assert all(t["family"] == family for t in tasks)


def test_save_and_load_tasks(tmp_path):
    tasks = generate_family_tasks("Bounce", n_tasks=2, seed=42)
    path = str(tmp_path / "tasks.pkl")
    save_tasks(tasks, path)
    assert os.path.exists(path)
    loaded = load_tasks(path)
    assert len(loaded) == len(tasks)
    assert loaded[0]["task_id"] == tasks[0]["task_id"]


def test_all_candidates_share_same_test_initial():
    """All 5 candidates for a task must reference the same initial state."""
    tasks = generate_family_tasks("Collision", n_tasks=2, seed=42)
    for t in tasks:
        init = t["test"]["init"]
        for c in t["test"]["candidates"]:
            np.testing.assert_array_equal(
                c["test_initial"].positions, init.positions
            )


# ---------------------------------------------------------------------------
# Task 6: benchmark smoke tests
# ---------------------------------------------------------------------------

def test_run_benchmark_smoke():
    """Generate 3 tasks per family (12 total); run all 4 conditions."""
    from benchmarks.structured_phyre import run_benchmark
    tasks = []
    for family in ["Projectile", "Bounce", "Slide", "Collision"]:
        tasks.extend(generate_family_tasks(family, n_tasks=3, seed=99))
    assert len(tasks) == 12

    for condition in ["flat", "l2l3", "l4_only", "l4l5_full"]:
        score = run_benchmark(tasks, condition)
        assert 0.0 <= score <= 1.0, f"Score out of range for {condition}: {score}"


def test_load_tasks_from_file(tmp_path):
    from benchmarks.structured_phyre import load_tasks as st_load_tasks
    tasks = generate_family_tasks("Bounce", n_tasks=2, seed=42)
    path = str(tmp_path / "tasks.pkl")
    save_tasks(tasks, path)
    loaded = st_load_tasks(path)
    assert len(loaded) == 2
