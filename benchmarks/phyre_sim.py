"""PhyRE physics simulation helpers.

SceneSnapshot: frozen description of a 2D pymunk scene at one instant.
simulate_scene: runs pymunk forward dt*n_steps seconds; returns final SceneSnapshot.
check_goal: True if active ball centre is within goal_radius of goal_pos.

All pymunk calls are isolated here. Everything downstream (encoders, benchmark)
is pure numpy and never imports pymunk.
"""
from __future__ import annotations
import dataclasses
import pickle
import random
import numpy as np

try:
    import pymunk
    _PYMUNK_AVAILABLE = True
except ImportError:
    _PYMUNK_AVAILABLE = False


@dataclasses.dataclass
class SceneSnapshot:
    """One-instant description of a PhyRE scene.

    positions:    (N, 2) float32  — x, y in [0, 500]
    velocities:   (N, 2) float32  — vx, vy in physics units
    masses:       (N,)   float32  — 0 = static body
    restitutions: (N,)   float32  — [0, 1]
    frictions:    (N,)   float32  — [0, 1]
    goal_pos:     (2,)   float32  — centre of circular goal region
    goal_radius:  float           — radius of goal region (default 30)
    active_ball_idx: int          — index into positions of the ball that must reach goal
    """
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    restitutions: np.ndarray
    frictions: np.ndarray
    goal_pos: np.ndarray
    goal_radius: float
    active_ball_idx: int


def check_goal(snapshot: SceneSnapshot) -> bool:
    """Return True if active ball centre is within goal_radius of goal_pos."""
    ball_pos = snapshot.positions[snapshot.active_ball_idx]
    dist = float(np.linalg.norm(ball_pos - snapshot.goal_pos))
    return dist <= snapshot.goal_radius


def simulate_scene(
    initial: SceneSnapshot,
    action_obj: dict | None,
    n_steps: int = 120,
    dt: float = 1 / 60,
) -> SceneSnapshot:
    """Run pymunk simulation; return final SceneSnapshot.

    action_obj: dict with keys {type, position, angle, mass, restitution, friction}
                or None (no action object placed).
    type: one of 'ramp'|'bumper'|'surface'|'ball'
    """
    if not _PYMUNK_AVAILABLE:
        raise RuntimeError("pymunk is required for simulate_scene. pip install pymunk")

    space = pymunk.Space()
    space.gravity = (0, -500)

    walls = [
        pymunk.Segment(space.static_body, (0, 0), (500, 0), 2),
        pymunk.Segment(space.static_body, (0, 500), (500, 500), 2),
        pymunk.Segment(space.static_body, (0, 0), (0, 500), 2),
        pymunk.Segment(space.static_body, (500, 0), (500, 500), 2),
    ]
    for w in walls:
        w.elasticity = 0.4
        w.friction = 0.5
        space.add(w)

    bodies = []
    for i in range(len(initial.positions)):
        mass = float(initial.masses[i])
        if mass <= 0:
            continue
        moment = pymunk.moment_for_circle(mass, 0, 15)
        body = pymunk.Body(mass, moment)
        body.position = tuple(initial.positions[i])
        body.velocity = tuple(initial.velocities[i])
        shape = pymunk.Circle(body, 15)
        shape.elasticity = float(initial.restitutions[i])
        shape.friction = float(initial.frictions[i])
        space.add(body, shape)
        bodies.append((i, body))

    if action_obj is not None:
        _add_action_object(space, action_obj)

    for _ in range(n_steps):
        space.step(dt)

    final_positions = initial.positions.copy()
    final_velocities = initial.velocities.copy()
    for idx, body in bodies:
        final_positions[idx] = np.array(body.position, dtype=np.float32)
        final_velocities[idx] = np.array(body.velocity, dtype=np.float32)

    return SceneSnapshot(
        positions=final_positions,
        velocities=final_velocities,
        masses=initial.masses.copy(),
        restitutions=initial.restitutions.copy(),
        frictions=initial.frictions.copy(),
        goal_pos=initial.goal_pos.copy(),
        goal_radius=initial.goal_radius,
        active_ball_idx=initial.active_ball_idx,
    )


def _add_action_object(space, action_obj: dict) -> None:
    obj_type = action_obj["type"]
    pos = action_obj["position"]
    angle = action_obj.get("angle", 0.0)
    mass = action_obj.get("mass", 1.0)
    restitution = action_obj.get("restitution", 0.5)
    friction = action_obj.get("friction", 0.5)

    if obj_type in ("ramp", "surface"):
        import math
        length = action_obj.get("length", 80)
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        seg = pymunk.Segment(
            space.static_body,
            (pos[0] - dx, pos[1] - dy),
            (pos[0] + dx, pos[1] + dy),
            3,
        )
        seg.elasticity = restitution
        seg.friction = friction
        space.add(seg)
    elif obj_type == "bumper":
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        static_body.position = tuple(pos)
        shape = pymunk.Circle(static_body, 20)
        shape.elasticity = restitution
        shape.friction = friction
        space.add(static_body, shape)
    elif obj_type == "ball":
        moment = pymunk.moment_for_circle(mass, 0, 15)
        body = pymunk.Body(mass, moment)
        body.position = tuple(pos)
        if "velocity" in action_obj:
            body.velocity = tuple(action_obj["velocity"])
        shape = pymunk.Circle(body, 15)
        shape.elasticity = restitution
        shape.friction = friction
        space.add(body, shape)


# ---------------------------------------------------------------------------
# Family scene builders
# Strategy: run a calibration simulation with the correct action, then place
# goal_pos at the ball's final position so check_goal always succeeds.
# ---------------------------------------------------------------------------

def _calibrate_goal(snap_no_goal, action_obj):
    """Simulate and return (snap_with_goal, action_obj) where goal = ball's final pos."""
    # Temporary snapshot with a dummy goal_pos for simulation
    tmp = SceneSnapshot(
        positions=snap_no_goal.positions.copy(),
        velocities=snap_no_goal.velocities.copy(),
        masses=snap_no_goal.masses.copy(),
        restitutions=snap_no_goal.restitutions.copy(),
        frictions=snap_no_goal.frictions.copy(),
        goal_pos=np.array([250.0, 250.0], dtype=np.float32),
        goal_radius=30.0,
        active_ball_idx=snap_no_goal.active_ball_idx,
    )
    final = simulate_scene(tmp, action_obj)
    idx = tmp.active_ball_idx
    goal_pos = final.positions[idx].copy()
    snap_with_goal = SceneSnapshot(
        positions=snap_no_goal.positions.copy(),
        velocities=snap_no_goal.velocities.copy(),
        masses=snap_no_goal.masses.copy(),
        restitutions=snap_no_goal.restitutions.copy(),
        frictions=snap_no_goal.frictions.copy(),
        goal_pos=goal_pos,
        goal_radius=30.0,
        active_ball_idx=snap_no_goal.active_ball_idx,
    )
    return snap_with_goal, action_obj


def _make_projectile_scene(rng):
    bx = rng.uniform(60, 150)
    by = rng.uniform(350, 450)
    positions = np.array([[bx, by]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    masses = np.array([1.0], dtype=np.float32)
    restitutions = np.array([0.3], dtype=np.float32)
    frictions = np.array([0.5], dtype=np.float32)
    # ramp placed to redirect ball rightward and down
    ramp_x = rng.uniform(200, 350)
    ramp_y = rng.uniform(150, 300)
    action_obj = {"type": "ramp", "position": [ramp_x, ramp_y],
                  "angle": -0.5, "mass": 0, "restitution": 0.6, "friction": 0.2, "length": 120}
    snap_proto = SceneSnapshot(positions, velocities, masses, restitutions, frictions,
                               np.zeros(2, dtype=np.float32), 30.0, active_ball_idx=0)
    return _calibrate_goal(snap_proto, action_obj)


def _make_bounce_scene(rng):
    bx = rng.uniform(50, 180)
    by = rng.uniform(200, 380)
    vx = rng.uniform(100, 200)
    positions = np.array([[bx, by]], dtype=np.float32)
    velocities = np.array([[vx, 0.0]], dtype=np.float32)
    masses = np.array([1.0], dtype=np.float32)
    restitutions = np.array([0.8], dtype=np.float32)
    frictions = np.array([0.1], dtype=np.float32)
    bumper_x = bx + rng.uniform(80, 160)
    bumper_y = by + rng.uniform(-30, 30)
    action_obj = {"type": "bumper", "position": [bumper_x, bumper_y],
                  "mass": 0, "restitution": 0.95, "friction": 0.05}
    snap_proto = SceneSnapshot(positions, velocities, masses, restitutions, frictions,
                               np.zeros(2, dtype=np.float32), 30.0, active_ball_idx=0)
    return _calibrate_goal(snap_proto, action_obj)


def _make_slide_scene(rng):
    bx = rng.uniform(50, 150)
    by = rng.uniform(300, 440)
    vx = rng.uniform(60, 120)
    positions = np.array([[bx, by]], dtype=np.float32)
    velocities = np.array([[vx, 0.0]], dtype=np.float32)
    masses = np.array([1.0], dtype=np.float32)
    restitutions = np.array([0.1], dtype=np.float32)
    frictions = np.array([0.9], dtype=np.float32)
    surf_y = by - rng.uniform(10, 30)
    action_obj = {"type": "surface", "position": [250, surf_y],
                  "angle": 0.0, "length": 350, "mass": 0, "restitution": 0.05, "friction": 0.02}
    snap_proto = SceneSnapshot(positions, velocities, masses, restitutions, frictions,
                               np.zeros(2, dtype=np.float32), 30.0, active_ball_idx=0)
    return _calibrate_goal(snap_proto, action_obj)


def _make_collision_scene(rng):
    bx = rng.uniform(300, 420)
    by = rng.uniform(200, 350)
    positions = np.array([[bx, by]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    masses = np.array([1.0], dtype=np.float32)
    restitutions = np.array([0.7], dtype=np.float32)
    frictions = np.array([0.3], dtype=np.float32)
    ball_x = bx + rng.uniform(40, 80)
    ball_vx = rng.uniform(-250, -150)
    action_obj = {"type": "ball", "position": [ball_x, by],
                  "mass": 2.0, "restitution": 0.7, "friction": 0.3,
                  "velocity": [ball_vx, 0.0]}
    snap_proto = SceneSnapshot(positions, velocities, masses, restitutions, frictions,
                               np.zeros(2, dtype=np.float32), 30.0, active_ball_idx=0)
    return _calibrate_goal(snap_proto, action_obj)


_FAMILY_BUILDERS = {
    "Projectile": _make_projectile_scene,
    "Bounce": _make_bounce_scene,
    "Slide": _make_slide_scene,
    "Collision": _make_collision_scene,
}

_DISTRACTOR_TYPES = {
    "Projectile": ["bumper", "surface", "ball", "ramp"],
    "Bounce":     ["ramp", "surface", "ball", "bumper"],
    "Slide":      ["ramp", "bumper", "ball", "surface"],
    "Collision":  ["ramp", "bumper", "surface", "ball"],
}


def _make_distractor(family: str, distractor_idx: int, rng) -> dict:
    wrong_type = _DISTRACTOR_TYPES[family][distractor_idx]
    return {
        "type": wrong_type,
        "position": [rng.uniform(100, 400), rng.uniform(100, 400)],
        "angle": rng.uniform(-1.0, 1.0),
        "mass": rng.uniform(0.5, 3.0),
        "restitution": rng.uniform(0.1, 0.9),
        "friction": rng.uniform(0.1, 0.9),
        "length": 80,
    }


def generate_family_tasks(family: str, n_tasks: int = 60, seed: int = 42,
                          n_train_pairs: int = 3) -> list:
    """Generate n_tasks for the given physics family.

    Task schema:
        task_id:  str "{family_lower}_{index:03d}"
        family:   str
        train:    list of {"init": SceneSnapshot, "final": SceneSnapshot}
        test:     {"init": SceneSnapshot,
                   "candidates": [{"action": dict, "final": SceneSnapshot,
                                   "test_initial": SceneSnapshot}],
                   "correct_idx": int}

    Only tasks where the correct candidate achieves the goal are kept.
    Attempts up to 3×n_tasks scenes before returning what's available.
    All 5 candidates share the same test_initial (= test["init"]).

    n_train_pairs: number of training pairs to generate per task (default 3).
    """
    builder = _FAMILY_BUILDERS[family]
    rng = random.Random(seed)
    tasks = []
    attempts = 0
    max_attempts = n_tasks * 3

    while len(tasks) < n_tasks and attempts < max_attempts:
        attempts += 1
        try:
            init_snap, correct_action = builder(rng)
            final_correct = simulate_scene(init_snap, correct_action)
        except Exception:
            continue

        if not check_goal(final_correct):
            continue

        train_pairs = []
        for _ in range(n_train_pairs):
            try:
                tr_init, tr_action = builder(rng)
                tr_final = simulate_scene(tr_init, tr_action)
                if check_goal(tr_final):
                    train_pairs.append({"init": tr_init, "final": tr_final})
            except Exception:
                pass

        if not train_pairs:
            continue

        correct_idx = rng.randint(0, 4)
        distractors = [_make_distractor(family, i, rng) for i in range(4)]
        candidates = []
        distractor_cursor = 0
        for ci in range(5):
            if ci == correct_idx:
                action, final = correct_action, final_correct
            else:
                action = distractors[distractor_cursor]
                distractor_cursor += 1
                try:
                    final = simulate_scene(init_snap, action)
                except Exception:
                    final = init_snap
            candidates.append({"action": action, "final": final,
                                "test_initial": init_snap})

        tasks.append({
            "task_id": f"{family.lower()}_{len(tasks):03d}",
            "family": family,
            "train": train_pairs,
            "test": {"init": init_snap, "candidates": candidates,
                     "correct_idx": correct_idx},
        })

    return tasks


def save_tasks(tasks: list, path: str) -> None:
    """Serialise task list to path using pickle (trusted internal data only)."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tasks, f)


def load_tasks(path: str) -> list:
    """Deserialise task list from pickle file produced by save_tasks."""
    with open(path, "rb") as f:
        return pickle.load(f)
