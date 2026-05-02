from hpm_ai_v5 import AgentInput, BaseAgent, HPMPipeline
from hpm_ai_v5.adapter import AdapterPacket, AdapterRegistry
from hpm_ai_v5.core import Delta, Pattern, PatternEngine, PatternStore, State
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


def test_repeated_sequence_learns_compact_template() -> None:
    store = PatternStore()

    pattern = store.learn((1.0, 2.0, 1.0, 2.0, 1.0, 2.0))

    assert pattern.template == (1.0, 2.0)


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
