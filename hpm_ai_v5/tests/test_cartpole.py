import pytest
import numpy as np
from hpm_ai_v5.adapter.physics import CartpoleStateAdapter, RunningNormaliserAdapter, RewardToGoalAdapter
from hpm_ai_v5.postprocessors.numeric import MultiNumericPostprocessor
from hpm_ai_v5.postprocessors.physics import CartpoleForecastPostprocessor
from hpm_ai_v5.adapter.packet import AdapterPacket
from hpm_ai_v5.core import State, Action, Pattern
from hpm_ai_v5.planning.cartpole import CartpoleBenchmark, CartpoleEnvConfig, DEFAULT_CARTPOLE_VARIANTS
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.core import PatternEngine
from hpm_ai_v5.core.evaluator import PolygraphScore
from hpm_ai_v5.polygraphs.base import PolygraphGenerator, PolygraphView


class _TestPolygraphGenerator(PolygraphGenerator):
    def generate(self, raw, *, context=None):
        context = context or {}
        return [PolygraphView(name="test_view", state=State(value=(1.0, -1.0, 0.5), context=context))]

def test_cartpole_state_adapter():
    adapter = CartpoleStateAdapter(action_history_len=3)
    obs = {"position": 0.1, "velocity": 0.2, "angle": 0.1, "angular_velocity": -0.1}
    # Provide last action to context
    packet = AdapterPacket(raw=obs, context={"last_action": 0.5})
    packet = adapter.run(packet)
    
    assert len(packet.states) == 1
    val = packet.states[0].value
    assert isinstance(val, tuple)
    # 3 (actions) + 2 (pos, vel) + 2 (sin, cos) + 1 (ang_vel) + 1 (error) = 9
    assert len(val) == 9
    assert val[0] == 0.0 # t-2
    assert val[1] == 0.0 # t-1
    assert val[2] == 0.5 # t (current last action)
    assert val[3] == 0.1 # pos
    assert val[4] == 0.2 # vel
    assert np.isclose(val[5], np.sin(0.1)) # sin
    assert np.isclose(val[6], np.cos(0.1)) # cos
    assert val[7] == -0.1 # ang_vel
    # error = 0.1^2 + 0.1 * (-0.1)^2 = 0.01 + 0.001 = 0.011
    assert np.isclose(val[8], 0.011)

def test_running_normaliser_adapter():
    adapter = RunningNormaliserAdapter()
    
    # First state
    packet = AdapterPacket(raw=None)
    packet.states.append(State(value=(1.0, 2.0)))
    packet = adapter.run(packet)
    
    assert len(packet.states) == 2
    # First transform with one point usually results in 0s
    assert packet.states[-1].value == (0.0, 0.0)
    
    # Second state
    packet.states.append(State(value=(3.0, 4.0)))
    packet = adapter.run(packet)
    assert len(packet.states) == 4
    # Mean (2, 3), std (1, 1). (3, 4) -> (1, 1)
    val = packet.states[-1].value
    assert np.isclose(val[0], 1.0)
    assert np.isclose(val[1], 1.0)

def test_reward_to_goal_adapter():
    adapter = RewardToGoalAdapter(decay=0.9)
    packet = AdapterPacket(raw=None, context={"reward": 1.0})
    packet = adapter.run(packet)
    assert packet.goal["utility"] == 1.0
    
    packet = AdapterPacket(raw=None, context={"reward": 1.0})
    packet = adapter.run(packet)
    assert np.isclose(packet.goal["utility"], 1.9)

def test_multi_numeric_postprocessor():
    post = MultiNumericPostprocessor()
    action = Action(action_type="apply_delta", value=5.0, confidence=1.0)
    packet = AdapterPacket(raw=None, core_action=action)
    packet = post.run(packet)
    assert packet.validated_output == 1.0 # clamped to default max
    
    action = Action(action_type="apply_delta", value=-10.0, confidence=1.0)
    packet = AdapterPacket(raw=None, core_action=action)
    packet = post.run(packet)
    assert packet.validated_output == -1.0 # clamped to default min

@pytest.mark.parametrize("episodes", [2])
def test_cartpole_benchmark_smoke(episodes):
    benchmark = CartpoleBenchmark()
    # Run a very short version for smoke test
    result = benchmark.run(episodes=episodes, max_steps=10)
    assert result.total_episodes == episodes
    assert len(result.episode_lengths) == episodes
    assert result.average_length >= 0
    assert result.evaluation_average >= 0

def test_cartpole_variant_config_is_applied():
    benchmark = CartpoleBenchmark(env_config=DEFAULT_CARTPOLE_VARIANTS["cartpole_heavy"])
    assert benchmark.env.mass_pole == 0.5
    assert benchmark.env.config.name == "cartpole_heavy"

def test_cartpole_transfer_state_round_trip():
    source = CartpoleBenchmark(env_config=CartpoleEnvConfig(name="source"))
    source_result = source.run(episodes=2, max_steps=5, total_episodes=2)
    snapshot = source.export_transfer_state()

    target = CartpoleBenchmark(env_config=DEFAULT_CARTPOLE_VARIANTS["cartpole_light"])
    target.import_transfer_state(snapshot)

    assert target.env_config.name == "cartpole_light"
    assert target.postprocessor.q_table == source.postprocessor.q_table
    assert target.pattern_manager.archive.keys() == source.pattern_manager.archive.keys()
    assert source_result.average_length >= 0

def test_cartpole_evaluate_restores_mutable_learning_state():
    benchmark = CartpoleBenchmark()
    benchmark.run(episodes=2, max_steps=5, total_episodes=2)
    before = benchmark.export_transfer_state()

    eval_result = benchmark.run(episodes=2, max_steps=5, evaluate=True)
    after = benchmark.export_transfer_state()

    assert eval_result.average_length >= 0
    assert before.postprocessor_state["q_table"] == after.postprocessor_state["q_table"]
    assert before.pattern_manager.archive.keys() == after.pattern_manager.archive.keys()

def test_cartpole_q_policy_prefers_pattern_key_when_available():
    post = CartpoleForecastPostprocessor()
    pattern = Pattern(name="pattern_7", template=(0.12, -0.33, 0.0), support=3, density=1.8, utility=2.2)
    action = Action(action_type="apply_delta", value=None, confidence=0.9, selected_pattern=pattern)
    key = post._policy_state_key(action, {"angle": -0.2, "angular_velocity": 0.3, "position": 0.0, "velocity": 0.0})
    assert key == ("cartpole_q", -2.0, 1.0, 0.0, 0.0)

def test_cartpole_q_policy_falls_back_without_pattern():
    post = CartpoleForecastPostprocessor()
    action = Action(action_type="apply_delta", value=None, confidence=0.9, selected_pattern=None)
    key = post._policy_state_key(action, {"angle": -0.2, "angular_velocity": 0.3, "position": 0.1, "velocity": -0.2})
    assert key == ("cartpole_q", -2.0, 1.0, 0.0, 0.0)

def test_cartpole_policy_key_uses_magnitude_buckets():
    post = CartpoleForecastPostprocessor()
    action = Action(action_type="apply_delta", value=None, confidence=0.9, selected_pattern=None)
    key = post._policy_state_key(action, {"angle": 0.01, "angular_velocity": 1.0, "position": 2.0, "velocity": -1.0})
    assert key == ("cartpole_q", 0.0, 2.0, 2.0, -2.0)

def test_pipeline_keeps_primary_forecast_when_polygraph_selects_view():
    post = CartpoleForecastPostprocessor()
    primary_engine = PatternEngine()
    pipeline = HPMPipeline(
        preprocessor=CartpoleStateAdapter(),
        engine=primary_engine,
        postprocessor=post,
        polygraph_generator=_TestPolygraphGenerator(),
        polygraph_every_n_steps=1,
        polygraph_min_patterns=0,
        polygraph_confidence_skip=0.99,
    )
    pipeline.register_preprocessor(RunningNormaliserAdapter())
    pipeline.register_preprocessor(RewardToGoalAdapter())
    post.pipeline = pipeline

    primary_pattern = Pattern(name="primary", template=(0.1, 0.2, 0.3))
    primary_forecast = State(value=(0.0, 0.0, 1.0, 0.1, 0.2), context={})
    primary_action = Action(
        action_type="apply_delta",
        value=primary_forecast.value,
        confidence=0.4,
        selected_pattern=primary_pattern,
        forecast=primary_forecast,
        trace={},
    )
    primary_engine.act = lambda goal=None, horizon=1, top_k=3: primary_action

    class _StubViewEngine:
        def observe(self, state):
            return None
        def act(self, goal=None, horizon=1, top_k=3):
            return Action(
                action_type="apply_delta",
                value=(1.0, -1.0, 0.5),
                confidence=0.7,
                selected_pattern=Pattern(name="view", template=(1.0,)),
                forecast=State(value=(1.0, -1.0, 0.5), context={}),
                trace={},
            )

    pipeline.view_engines["test_view"] = _StubViewEngine()
    pipeline.polygraph_evaluator.score_engine = lambda engine: PolygraphScore(
        concentration=1.0,
        average_density=1.0,
        fragmentation=0.0,
        score=0.8,
    )
    pipeline.polygraph_evaluator.select_view = lambda scores: "test_view"

    result = pipeline.step(
        {"position": 0.0, "velocity": 0.0, "angle": 0.1, "angular_velocity": 0.0},
        goal={},
        context={"last_action": 0.0, "reward": 0.0},
    )

    assert result.action.selected_view == "test_view"
    assert result.action.forecast == primary_forecast
    assert result.action.selected_pattern == primary_pattern
