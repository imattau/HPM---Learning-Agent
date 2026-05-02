from hpm_ai_v5 import AgentInput, BaseAgent, HPMPipeline
from hpm_ai_v5.adapter import AdapterPacket, AdapterRegistry
from hpm_ai_v5.core import CoreConfig, Delta, Pattern, PatternEngine, PatternSequence, PatternStore, State
from hpm_ai_v5.core.action import Action
from hpm_ai_v5.core.evaluator import PolygraphEvaluator
from hpm_ai_v5.postprocessors.numeric import NumericPostprocessor
from hpm_ai_v5.polygraphs.numeric import NumericPolygraphGenerator
from hpm_ai_v5.preprocessors.numeric import NumericPreprocessor


def test_delta_between_numeric_states() -> None:
    delta = Delta.between(1, 4)

    assert delta.value == 3.0
    assert delta.magnitude == 3.0
    assert delta.level == "state"


def test_delta_can_carry_level_information() -> None:
    delta = Delta.between((1.0, 2.0), (2.0, 3.0), level="pattern")

    assert delta.level == "pattern"


def test_store_matches_exact_and_near() -> None:
    store = PatternStore(exact_threshold=0.0, near_threshold=0.5)
    store.add(Pattern(name="rise", template=(1.0, 2.0)))

    exact = store.match((1.0, 2.0))
    near = store.match((1.0, 2.25))
    novel = store.match((4.0, 9.0))

    assert exact.status == "exact"
    assert near.status == "near"
    assert novel.status == "novel"


def test_canonical_rotation_matches_reversed_order() -> None:
    pattern = Pattern(name="cycle", template=(1.0, 2.0))

    assert pattern.distance((2.0, 1.0)) == 0.0


def test_strict_canonicalization_preserves_order() -> None:
    pattern = Pattern(name="cycle", template=(1.0, 2.0))

    assert pattern.distance((2.0, 1.0), canonicalization_mode="strict") > 0.0


def test_distance_scale_changes_matching_range() -> None:
    store = PatternStore(config=CoreConfig(distance_scale=10.0, near_threshold=0.5))
    store.add(Pattern(name="rise", template=(1.0, 2.0)))

    near = store.match((1.0, 6.0))

    assert near.status == "near"


def test_repeated_sequence_learns_compact_template() -> None:
    store = PatternStore()

    pattern = store.learn((1.0, 2.0, 1.0, 2.0, 1.0, 2.0))

    assert pattern.template == (1.0, 2.0)


def test_pattern_simulate_cycles_through_template() -> None:
    pattern = Pattern(name="cycle", template=(1.0, 2.0))

    path = pattern.simulate(State(value=0.0), horizon=3)

    assert [state.value for state in path] == [1.0, 3.0, 4.0]


def test_sequence_simulate_uses_phase_offset() -> None:
    engine = PatternEngine()
    first = Pattern(name="first", template=(1.0,))
    second = Pattern(name="second", template=(2.0,))
    third = Pattern(name="third", template=(3.0,))
    engine.store.add(first)
    engine.store.add(second)
    engine.store.add(third)
    sequence = PatternSequence(pattern_names=("first", "second", "third"))

    state = State(value=0.0, context={"phase": 1, "position": 0})
    path = sequence.simulate(
        state,
        horizon=3,
        resolver=engine.store.get,
        start_offset=engine._sequence_offset(state, 3),
    )

    assert [state.value for state in path] == [2.0, 5.0, 6.0]


def test_engine_learns_and_reuses_pattern() -> None:
    engine = PatternEngine()

    assert engine.observe(State(value=0.0)) is None

    first = engine.observe(State(value=3.0))
    second = engine.observe(State(value=6.0))

    assert first is not None
    assert first.status == "novel"
    assert first.pattern is not None
    assert len(engine.store.patterns) == 1

    assert second is not None
    assert second.status == "exact"
    assert second.pattern is first.pattern
    assert second.pattern.support == 2

    action = engine.act(goal={"utility": 1.0}, horizon=1)
    assert action.action_type == "apply_delta"
    assert action.selected_pattern is not None
    assert action.forecast.value == 9.0
    assert action.reasoning_trace is not None
    assert action.reasoning_trace.selected_action["action_type"] == "apply_delta"
    assert action.reasoning_trace.validation["status"] == "accepted"
    assert action.reasoning_trace.candidate_patterns


def test_engine_history_is_bounded() -> None:
    engine = PatternEngine(config=CoreConfig(history_limit=3))

    for value in (0.0, 1.0, 2.0, 3.0, 4.0):
        engine.observe(State(value=value))

    assert len(engine.history) == 3
    assert [state.value for state in engine.history] == [2.0, 3.0, 4.0]


def test_pattern_decay_caps_context_memory() -> None:
    pattern = Pattern(
        name="aging",
        template=(1.0,),
        density=8.0,
        utility=4.0,
        context_memory={f"ctx{i}": float(i + 1) for i in range(12)},
    )

    pattern.decay(density_decay=0.5, utility_decay=0.5, context_decay=0.5, context_memory_limit=4)

    assert pattern.density == 4.0
    assert pattern.utility == 2.0
    assert len(pattern.context_memory) == 4
    assert list(pattern.context_memory.values()) == sorted(pattern.context_memory.values(), reverse=True)


def test_fresh_pattern_can_overtake_stale_high_density_pattern() -> None:
    engine = PatternEngine(config=CoreConfig(density_decay=0.25, utility_decay=0.25))
    stale = Pattern(name="stale", template=(1.0,), density=12.0, utility=12.0)
    fresh = Pattern(name="fresh", template=(1.0,), density=1.0, utility=1.0)

    engine.store.add(stale)
    engine.store.add(fresh)
    engine.current_state = State(value=0.0)
    engine.history = [State(value=0.0)]

    assert engine.select(goal={"beta": 1.0, "delta": 1.0}) is stale

    for _ in range(10):
        stale.decay(density_decay=0.25, utility_decay=0.25)
        fresh.reinforce(density_boost=0.75)
        fresh.reward(utility_boost=0.75)

    assert engine.select(goal={"beta": 1.0, "delta": 1.0}) is fresh


def test_context_memory_can_override_density() -> None:
    engine = PatternEngine()
    broad = Pattern(name="broad", template=(1.0,), density=10.0)
    local = Pattern(name="local", template=(1.0,), density=1.0)
    context = {"mode": "fast"}
    local.context_memory[engine.store.context_signature(context)] = 5.0

    engine.store.add(broad)
    engine.store.add(local)
    engine.current_state = State(value=0.0, context=context)
    engine.history = [State(value=0.0, context=context)]

    selected = engine.select(goal={"utility": 0.0})

    assert selected is local


def test_prune_removes_weak_patterns() -> None:
    store = PatternStore(max_patterns=2)
    strong = Pattern(name="strong", template=(1.0,), support=3, density=3.0)
    medium = Pattern(name="medium", template=(2.0,), support=2, density=2.0)
    weak = Pattern(name="weak", template=(3.0,), support=0, density=0.0)

    store.add(strong)
    store.add(medium)
    store.add(weak)

    removed = store.prune()

    assert removed == 1
    assert weak not in store.patterns


def test_repeating_residual_creates_higher_order_pattern() -> None:
    engine = PatternEngine()
    engine.store.add(Pattern(name="base", template=(1.0, 2.0), support=1, density=1.0))
    engine.current_state = State(value=(0.0, 0.0))
    engine.history = [State(value=(0.0, 0.0))]

    first = engine.observe(State(value=(1.0, 2.4)))
    second = engine.observe(State(value=(2.0, 4.8)))

    assert first is not None
    assert first.status == "near"
    assert second is not None
    assert len(engine.store.patterns) == 2
    assert any(pattern.name.startswith("residual_") for pattern in engine.store.patterns)


def test_pipeline_preprocesses_acts_and_postprocesses() -> None:
    pipeline = HPMPipeline(
        preprocessor=NumericPreprocessor(),
        engine=PatternEngine(),
        postprocessor=NumericPostprocessor(),
    )

    first = pipeline.step(1.0, context={"minimum": 0.0, "maximum": 10.0})
    second = pipeline.step(4.0, context={"minimum": 0.0, "maximum": 10.0})
    third = pipeline.step(7.0, context={"minimum": 0.0, "maximum": 10.0})

    assert first.input.state.value == 1.0
    assert first.input.context["domain"] == "numeric"
    assert second.action.action_type == "apply_delta"
    assert third.output == 10.0


def test_polygraph_pipeline_prefers_stable_view() -> None:
    pipeline = HPMPipeline(
        preprocessor=NumericPreprocessor(),
        engine=PatternEngine(),
        postprocessor=NumericPostprocessor(),
        polygraph_generator=NumericPolygraphGenerator(),
    )

    last_result = None
    for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        last_result = pipeline.step(value, context={"minimum": -10.0, "maximum": 20.0})

    assert last_result is not None
    assert last_result.polygraph_scores is not None
    assert "noisy_delta" in last_result.polygraph_scores
    assert last_result.action.selected_view in {"value_delta", "trend"}
    assert last_result.polygraph_scores["noisy_delta"].fragmentation >= 2.0


def test_polygraph_agreement_prefers_consensus() -> None:
    evaluator = PolygraphEvaluator()
    actions = {
        "value_delta": Action(action_type="apply_delta", value=2.0, confidence=0.9, selected_pattern=Pattern(name="stable"), trace={}, forecast=None),
        "trend": Action(action_type="apply_delta", value=2.0, confidence=0.8, selected_pattern=Pattern(name="stable"), trace={}, forecast=None),
        "noisy_delta": Action(action_type="apply_delta", value=5.0, confidence=0.2, selected_pattern=Pattern(name="noisy"), trace={}, forecast=None),
    }
    scores = {
        "value_delta": evaluator.score_engine(PatternEngine()),
        "trend": evaluator.score_engine(PatternEngine()),
        "noisy_delta": evaluator.score_engine(PatternEngine()),
    }

    agreement = evaluator.agreement(actions, scores)

    assert agreement.selected_key == "pattern:stable"
    assert agreement.support >= 2


def test_long_horizon_pipeline_uses_polygraph_agreement() -> None:
    pipeline = HPMPipeline(
        preprocessor=NumericPreprocessor(),
        engine=PatternEngine(),
        postprocessor=NumericPostprocessor(),
        polygraph_generator=NumericPolygraphGenerator(),
    )

    last_result = None
    for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        last_result = pipeline.step(
            value,
            goal={"plan_horizon": 2.0},
            context={"minimum": -10.0, "maximum": 20.0},
        )

    assert last_result is not None
    assert last_result.polygraph_agreement is not None
    assert last_result.action.trace["selection_mode"] == "agreement"
    assert "polygraph_agreement" in last_result.action.trace


def test_adapter_registry_resolves_dependencies_in_order() -> None:
    class CleanAdapter:
        name = "clean"
        requires = []
        provides = ["clean"]

        def run(self, packet: AdapterPacket) -> AdapterPacket:
            packet.clean = f"clean:{packet.raw}"
            return packet

    class TokenAdapter:
        name = "tokens"
        requires = ["clean"]
        provides = ["tokens"]

        def run(self, packet: AdapterPacket) -> AdapterPacket:
            packet.tokens = packet.clean.split(":")
            return packet

    registry = AdapterRegistry()
    registry.register(CleanAdapter())
    registry.register(TokenAdapter())

    packet = registry.run(AdapterPacket(raw="abc"), target_outputs=["tokens"])

    assert packet.clean == "clean:abc"
    assert packet.tokens == ["clean", "abc"]
    assert [entry["adapter"] for entry in packet.trace] == ["clean", "tokens"]


def test_base_agent_steps_through_pipeline() -> None:
    agent = BaseAgent(
        name="NumericAgent",
        core=PatternEngine(),
        preprocessors=[NumericPreprocessor()],
        postprocessors=[NumericPostprocessor()],
    )

    output = agent.step(AgentInput(raw=1.0, context={"minimum": 0.0, "maximum": 10.0}))

    assert output.action_type in {"unknown", "defer", "apply_delta"}
    assert output.valid in {True, False}
    assert agent.state["turn_history"]


def test_base_agent_preserves_adapter_trace() -> None:
    agent = BaseAgent(
        name="NumericAgent",
        core=PatternEngine(),
        preprocessors=[NumericPreprocessor()],
        postprocessors=[NumericPostprocessor()],
    )

    agent.step(AgentInput(raw=1.0, context={"minimum": 0.0, "maximum": 10.0}))
    result = agent.step(AgentInput(raw=2.0, context={"minimum": 0.0, "maximum": 10.0}))

    assert result.trace["agent"] == "NumericAgent"
    assert result.trace["adapter_trace"]
    assert "core_decision" in result.trace


def test_repeating_pattern_order_promotes_sequence() -> None:
    engine = PatternEngine()
    context = {"mode": "repeat"}

    engine.pattern_trace = ["p1", "p2", "p1", "p2"]
    engine._maybe_promote_sequence(engine.store.context_signature(context))

    assert len(engine.sequences) == 1
    assert engine.sequences[0].pattern_names == ("p1", "p2")


def test_engine_uses_sequence_forecast_when_sequence_scores_higher() -> None:
    engine = PatternEngine()
    first = Pattern(name="p1", template=(1.0,), support=1, density=0.0)
    second = Pattern(name="p2", template=(5.0,), support=1, density=0.0)
    engine.store.add(first)
    engine.store.add(second)
    promoted = PatternSequence(pattern_names=("p1", "p2"), support=3, density=10.0)
    engine.sequences = [promoted]
    engine.current_state = State(value=0.0)
    engine.history = [State(value=0.0)]

    action = engine.act(goal={"utility": 0.0}, horizon=2)

    assert action.trace["forecast_source"] == "sequence"
    assert action.selected_sequence is promoted
    assert action.forecast.value == 6.0


def test_engine_prefers_better_trajectory_over_better_next_step() -> None:
    engine = PatternEngine()
    greedy = Pattern(name="greedy", template=(1.0,), support=1, density=6.0)
    step_a = Pattern(name="step_a", template=(1.0,), support=1, density=0.0)
    step_b = Pattern(name="step_b", template=(4.0,), support=1, density=0.0)
    engine.store.add(greedy)
    engine.store.add(step_a)
    engine.store.add(step_b)
    engine.sequences = [PatternSequence(pattern_names=("step_a", "step_b"), support=2, density=0.0)]
    engine.current_state = State(value=0.0)
    engine.history = [State(value=0.0)]

    action = engine.act(goal={"target": 5.0, "utility": 0.0}, horizon=2)

    assert action.reasoning_trace is not None
    assert action.reasoning_trace.selected_action["trajectory_mode"] == "full"
    assert action.reasoning_trace.score_trace["sequence_trajectory_score"] >= action.reasoning_trace.score_trace["pattern_trajectory_score"]
    assert action.selected_sequence is not None
    assert action.selected_sequence.pattern_names == ("step_a", "step_b")
    assert action.forecast.value == 5.0


def test_engine_changes_selection_with_context_and_goal() -> None:
    engine = PatternEngine()
    context_sensitive = Pattern(name="contextual", template=(4.0,), support=1, density=1.0, utility=0.5)
    utility_sensitive = Pattern(name="goal", template=(9.0,), support=1, density=0.5, utility=4.0)
    dense_but_generic = Pattern(name="dense", template=(1.0,), support=1, density=6.0, utility=0.0)

    context_signature = engine.store.context_signature({"mode": "fast"})
    context_sensitive.context_memory[context_signature] = 5.0

    engine.store.add(context_sensitive)
    engine.store.add(utility_sensitive)
    engine.store.add(dense_but_generic)

    engine.current_state = State(value=0.0, context={"mode": "fast"})
    engine.history = [State(value=0.0, context={"mode": "fast"})]

    contextual_action = engine.act(goal={"beta": 0.5, "gamma": 3.0, "delta": 0.1}, horizon=1)
    assert contextual_action.selected_pattern is context_sensitive
    assert contextual_action.forecast.value == 4.0

    engine.current_state = State(value=0.0, context={"mode": "slow"})
    engine.history = [State(value=0.0, context={"mode": "slow"})]

    goal_action = engine.act(goal={"beta": 0.1, "gamma": 0.0, "delta": 3.0}, horizon=1)
    assert goal_action.selected_pattern is utility_sensitive
    assert goal_action.forecast.value == 9.0


def test_reasoning_trace_exposes_observations_and_rejections() -> None:
    engine = PatternEngine()
    engine.store.add(Pattern(name="short", template=(1.0,), support=1, density=0.5))
    engine.store.add(Pattern(name="long", template=(2.0,), support=1, density=0.25))
    engine.current_state = State(value=1.0, context={"mode": "trace"})
    engine.history = [State(value=0.0, context={"mode": "trace"}), State(value=1.0, context={"mode": "trace"})]

    action = engine.act(goal={"utility": 0.0}, horizon=1)

    assert action.reasoning_trace is not None
    assert action.reasoning_trace.observations == [0.0, 1.0]
    assert any(key.startswith("pattern:") for key in action.reasoning_trace.rejected_candidates)
    assert action.reasoning_trace.forecast is not None
    assert action.reasoning_trace.score_trace["confidence"] == action.confidence
