import numpy as np
import pytest
from benchmarks.phyre_sim import SceneSnapshot
from hpm.encoders.phyre_encoders import PhyreL1Encoder, PhyreL2Encoder, PhyreL3Encoder


def _make_snap(n_objects: int = 3) -> SceneSnapshot:
    positions = np.zeros((n_objects, 2), dtype=np.float32)
    velocities = np.zeros((n_objects, 2), dtype=np.float32)
    masses = np.ones(n_objects, dtype=np.float32)
    restitutions = np.full(n_objects, 0.5, dtype=np.float32)
    frictions = np.full(n_objects, 0.5, dtype=np.float32)
    goal_pos = np.array([250.0, 250.0], dtype=np.float32)
    return SceneSnapshot(positions, velocities, masses, restitutions, frictions,
                         goal_pos, 30.0, active_ball_idx=0)


# ---------------------------------------------------------------------------
# PhyreL1Encoder tests (Task 3)
# ---------------------------------------------------------------------------

def test_l1_encoder_feature_dim():
    enc = PhyreL1Encoder()
    assert enc.feature_dim == 16
    assert enc.max_steps_per_obs == 1


def test_l1_encoder_output_shape():
    enc = PhyreL1Encoder()
    init = _make_snap(3)
    final = _make_snap(3)
    final.positions[0] = np.array([100.0, 200.0])
    final.velocities[0] = np.array([50.0, -30.0])
    result = enc.encode((init, final), epistemic=None)
    assert len(result) == 1
    assert result[0].shape == (16,)
    assert result[0].dtype == np.float32


def test_l1_encoder_normalisation():
    enc = PhyreL1Encoder()
    init = _make_snap(3)
    final = _make_snap(3)
    final.positions[0] = np.array([500.0, 0.0])   # delta_x = 500 -> normalised 1.0
    result = enc.encode((init, final), epistemic=None)
    assert abs(result[0][0] - 1.0) < 1e-5


def test_l1_encoder_padding_for_fewer_objects():
    enc = PhyreL1Encoder()
    init = _make_snap(1)
    final = _make_snap(1)
    result = enc.encode((init, final), epistemic=None)
    assert result[0].shape == (16,)
    assert np.all(result[0][4:12] == 0.0)   # objects 1 and 2 zero-padded


def test_l1_encoder_epistemic_ignored():
    enc = PhyreL1Encoder()
    init = _make_snap(2)
    final = _make_snap(2)
    res_none = enc.encode((init, final), epistemic=None)
    res_epi = enc.encode((init, final), epistemic=(0.5, 0.3))
    np.testing.assert_array_equal(res_none[0], res_epi[0])


def test_l1_encoder_goal_distance_features():
    enc = PhyreL1Encoder()
    init = _make_snap(1)    # ball at (0,0), goal at (250,250)
    final = _make_snap(1)
    result = enc.encode((init, final), epistemic=None)
    expected_dist_init = np.linalg.norm(np.array([0.0, 0.0]) - np.array([250.0, 250.0])) / 500.0
    assert abs(result[0][12] - expected_dist_init) < 1e-4


# ---------------------------------------------------------------------------
# PhyreL2Encoder tests (Task 4)
# ---------------------------------------------------------------------------

def test_l2_encoder_feature_dim():
    enc = PhyreL2Encoder()
    assert enc.feature_dim == 14
    assert enc.max_steps_per_obs == 1


def test_l2_encoder_output_shape():
    result = PhyreL2Encoder().encode((_make_snap(3), _make_snap(3)), epistemic=None)
    assert len(result) == 1
    assert result[0].shape == (14,)
    assert result[0].dtype == np.float32


def test_l2_encoder_epistemic_threading():
    enc = PhyreL2Encoder()
    init, final = _make_snap(2), _make_snap(2)
    res_no = enc.encode((init, final), epistemic=None)
    res_epi = enc.encode((init, final), epistemic=(0.7, 0.15))
    assert res_no[0][-2] == 0.0 and res_no[0][-1] == 0.0
    assert abs(res_epi[0][-2] - 0.7) < 1e-5
    assert abs(res_epi[0][-1] - 0.15) < 1e-5


def test_l2_encoder_material_values():
    enc = PhyreL2Encoder()
    init = _make_snap(1)
    init.masses[0] = 5.0
    init.restitutions[0] = 0.8
    init.frictions[0] = 0.3
    result = enc.encode((init, _make_snap(1)), epistemic=None)
    assert abs(result[0][0] - 0.5) < 1e-5   # mass/10
    assert abs(result[0][1] - 0.8) < 1e-5   # restitution
    assert abs(result[0][2] - 0.3) < 1e-5   # friction


def test_l2_encoder_padding_for_fewer_objects():
    enc = PhyreL2Encoder()
    result = enc.encode((_make_snap(1), _make_snap(1)), epistemic=None)
    assert np.all(result[0][3:9] == 0.0)    # objects 1 and 2 zero-padded


# ---------------------------------------------------------------------------
# PhyreL3Encoder tests (Task 5)
# ---------------------------------------------------------------------------

def _make_snap_with_motion(ball_pos_init=(250.0, 400.0), ball_pos_final=(250.0, 50.0),
                           ball_vel_init=(0.0, 0.0), ball_vel_final=(0.0, -150.0),
                           goal_pos=(250.0, 50.0), n_collisions=2):
    init = _make_snap(1)
    final = _make_snap(1)
    init.positions[0] = np.array(ball_pos_init, dtype=np.float32)
    final.positions[0] = np.array(ball_pos_final, dtype=np.float32)
    init.velocities[0] = np.array(ball_vel_init, dtype=np.float32)
    final.velocities[0] = np.array(ball_vel_final, dtype=np.float32)
    init.goal_pos = np.array(goal_pos, dtype=np.float32)
    final.goal_pos = np.array(goal_pos, dtype=np.float32)
    return init, final, n_collisions


def test_l3_encoder_feature_dim():
    enc = PhyreL3Encoder()
    assert enc.feature_dim == 12
    assert enc.max_steps_per_obs == 1


def test_l3_encoder_output_shape():
    enc = PhyreL3Encoder()
    init, final, n_col = _make_snap_with_motion()
    result = enc.encode((init, final, n_col), epistemic=None)
    assert len(result) == 1
    assert result[0].shape == (12,)
    assert result[0].dtype == np.float32


def test_l3_encoder_goal_achieved_flag():
    enc = PhyreL3Encoder()
    init, final, n_col = _make_snap_with_motion(
        ball_pos_final=(250.0, 250.0), goal_pos=(250.0, 250.0))
    result = enc.encode((init, final, n_col), epistemic=None)
    assert result[0][4] == 1.0


def test_l3_encoder_goal_not_achieved_flag():
    enc = PhyreL3Encoder()
    init, final, n_col = _make_snap_with_motion(
        ball_pos_final=(0.0, 0.0), goal_pos=(250.0, 250.0))
    result = enc.encode((init, final, n_col), epistemic=None)
    assert result[0][4] == 0.0


def test_l3_encoder_collision_count_normalised():
    enc = PhyreL3Encoder()
    init, final, _ = _make_snap_with_motion()
    result = enc.encode((init, final, 5), epistemic=None)
    assert abs(result[0][6] - 0.5) < 1e-5   # 5/10 = 0.5


def test_l3_encoder_energy_dissipation():
    enc = PhyreL3Encoder()
    init = _make_snap(1)
    final = _make_snap(1)
    init.velocities[0] = np.array([200.0, 0.0])   # KE_init = 0.5*1*40000 = 20000
    final.velocities[0] = np.array([0.0, 0.0])    # KE_final = 0
    result = enc.encode((init, final, 0), epistemic=None)
    assert abs(result[0][1] - 1.0) < 1e-5          # 1 - 0/20000 = 1.0


def test_l3_encoder_epistemic_threading():
    enc = PhyreL3Encoder()
    init, final, n_col = _make_snap_with_motion()
    result = enc.encode((init, final, n_col), epistemic=(0.6, 0.25))
    assert abs(result[0][10] - 0.6) < 1e-5
    assert abs(result[0][11] - 0.25) < 1e-5
