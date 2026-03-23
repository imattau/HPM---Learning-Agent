"""PhyRE-specific LevelEncoder implementations.

All three encoders accept observation = (initial: SceneSnapshot, final: SceneSnapshot[, extras])
and return list[np.ndarray] of length 1.

PhyreL1Encoder (feature_dim=16): kinematic change — delta position and velocity per object.
PhyreL2Encoder (feature_dim=14): material properties — mass, restitution, friction.
PhyreL3Encoder (feature_dim=12): aggregate physics — momentum, energy, collision count.
"""
from __future__ import annotations
import numpy as np

MAX_OBJECTS = 3
SPACE_NORM = 500.0
VEL_NORM = 200.0


class PhyreL1Encoder:
    """Level-1: kinematic change features.

    feature_dim = 16:
        [0:12]  delta_x, delta_y, delta_vx, delta_vy per object (padded to MAX_OBJECTS=3)
        [12]    active ball distance to goal at t=0 / SPACE_NORM
        [13]    active ball distance to goal at t=T / SPACE_NORM
        [14:16] action object position (zeros — passed via observation extension)
    """
    feature_dim: int = 16
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic) -> list:
        initial, final = observation[0], observation[1]
        n = min(len(initial.positions), MAX_OBJECTS)
        kinematic = np.zeros(MAX_OBJECTS * 4, dtype=np.float32)
        for i in range(n):
            delta_pos = (final.positions[i] - initial.positions[i]) / SPACE_NORM
            delta_vel = (final.velocities[i] - initial.velocities[i]) / VEL_NORM
            kinematic[i * 4: i * 4 + 2] = delta_pos
            kinematic[i * 4 + 2: i * 4 + 4] = delta_vel

        idx = initial.active_ball_idx
        dist_init = float(np.linalg.norm(initial.positions[idx] - initial.goal_pos)) / SPACE_NORM
        dist_final = float(np.linalg.norm(final.positions[idx] - final.goal_pos)) / SPACE_NORM

        action_pos = np.zeros(2, dtype=np.float32)
        vec = np.concatenate([
            kinematic,
            np.array([dist_init, dist_final], dtype=np.float32),
            action_pos,
        ])
        return [vec.astype(np.float32)]


class PhyreL2Encoder:
    """Level-2: material property features.

    feature_dim = 14:
        [0:9]   mass/10, restitution, friction per object (padded to MAX_OBJECTS=3)
        [9:12]  action object: mass/10, restitution, friction (zeros if absent)
        [12]    l1_weight from epistemic (0.0 if None)
        [13]    l1_loss   from epistemic (0.0 if None)
    """
    feature_dim: int = 14
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic) -> list:
        initial = observation[0]
        n = min(len(initial.positions), MAX_OBJECTS)
        material = np.zeros(MAX_OBJECTS * 3, dtype=np.float32)
        for i in range(n):
            material[i * 3] = float(initial.masses[i]) / 10.0
            material[i * 3 + 1] = float(initial.restitutions[i])
            material[i * 3 + 2] = float(initial.frictions[i])

        action_material = np.zeros(3, dtype=np.float32)
        epi_weight = epi_loss = 0.0
        if epistemic is not None:
            epi_weight, epi_loss = float(epistemic[0]), float(epistemic[1])

        vec = np.concatenate([
            material,
            action_material,
            np.array([epi_weight, epi_loss], dtype=np.float32),
        ])
        return [vec.astype(np.float32)]


class PhyreL3Encoder:
    """Level-3: aggregate physics features.

    Observation = (initial: SceneSnapshot, final: SceneSnapshot, n_collisions: int)

    feature_dim = 12:
        [0]   momentum change magnitude / (mass * VEL_NORM)
        [1]   energy dissipation ratio: 1 - KE_final / max(KE_initial, 1e-8)
        [2]   net displacement magnitude / SPACE_NORM
        [3]   net displacement direction / pi
        [4]   goal_achieved flag (0.0 or 1.0)
        [5]   max velocity across all objects / VEL_NORM
        [6]   collision count / 10
        [7]   action mass/restitution ratio (0.0 — populated when action info available)
        [8]   active ball final speed / VEL_NORM
        [9]   action object displacement magnitude / SPACE_NORM (0.0 if absent)
        [10]  l2_weight from epistemic (0.0 if None)
        [11]  l2_loss   from epistemic (0.0 if None)
    """
    feature_dim: int = 12
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic) -> list:
        from benchmarks.phyre_sim import check_goal
        initial, final, n_collisions = observation[0], observation[1], observation[2]
        idx = initial.active_ball_idx
        mass = float(initial.masses[idx]) if idx < len(initial.masses) else 1.0

        p_init = mass * initial.velocities[idx]
        p_final = mass * final.velocities[idx]
        momentum_change = float(np.linalg.norm(p_final - p_init)) / max(mass * VEL_NORM, 1e-8)

        def ke(snap):
            total = 0.0
            for i in range(len(snap.velocities)):
                m = float(snap.masses[i])
                if m > 0:
                    total += 0.5 * m * float(np.dot(snap.velocities[i], snap.velocities[i]))
            return total

        ke_init = ke(initial)
        ke_final = ke(final)
        energy_dissipation = float(np.clip(1.0 - ke_final / max(ke_init, 1e-8), 0.0, 1.0))

        displacement = final.positions[idx] - initial.positions[idx]
        disp_magnitude = float(np.linalg.norm(displacement)) / SPACE_NORM
        disp_angle = float(np.arctan2(displacement[1], displacement[0])) / np.pi
        goal_achieved = 1.0 if check_goal(final) else 0.0
        max_vel = max(float(np.linalg.norm(final.velocities[i]))
                      for i in range(len(final.velocities))) / VEL_NORM
        collision_norm = float(n_collisions) / 10.0
        active_speed = float(np.linalg.norm(final.velocities[idx])) / VEL_NORM

        epi_weight = epi_loss = 0.0
        if epistemic is not None:
            epi_weight, epi_loss = float(epistemic[0]), float(epistemic[1])

        vec = np.array([
            momentum_change, energy_dissipation, disp_magnitude, disp_angle,
            goal_achieved, max_vel, collision_norm,
            0.0,            # action_ratio
            active_speed,
            0.0,            # action_disp
            epi_weight, epi_loss,
        ], dtype=np.float32)
        return [vec]
