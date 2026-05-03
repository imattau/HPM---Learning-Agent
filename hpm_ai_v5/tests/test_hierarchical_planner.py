import pytest
from typing import Any
from hpm_ai_v5.agents.hierarchical_planner import HierarchicalPlanningAgent
from hpm_ai_v5.agents.base import AgentInput
from hpm_ai_v5.core import PatternEngine, State, Action
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.preprocessors.base import Preprocessor, PreprocessedInput
from hpm_ai_v5.postprocessors.base import Postprocessor
from hpm_ai_v5.adapter import AdapterPacket

class SimplePreprocessor(Preprocessor):
    name = "simple_pre"
    requires: list[str] = []
    provides: list[str] = ["state"]
    def run(self, packet: AdapterPacket) -> AdapterPacket:
        packet.states.append(State(value=packet.raw, context=packet.context))
        return packet
    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = self.run(AdapterPacket(raw=raw, context=context or {}))
        return PreprocessedInput(state=packet.states[-1], context=packet.context, raw=raw, packet=packet)

class SimplePostprocessor(Postprocessor):
    name = "simple_post"
    requires: list[str] = []
    provides: list[str] = ["output"]
    def run(self, packet: AdapterPacket) -> AdapterPacket:
        packet.validated_output = packet.core_action.value
        return packet
    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> Any:
        return action.value

@pytest.fixture
def agent():
    engine = PatternEngine()
    pre = SimplePreprocessor()
    post = SimplePostprocessor()
    pipeline = HPMPipeline(preprocessor=pre, engine=engine, postprocessor=post)
    
    # Prerequisite map: "open_chest" -> ["find_key", "pull_lever"]
    prerequisite_map = {
        "open_chest": ["find_key", "pull_lever"]
    }
    
    def completion_checker(goal, state):
        # State is just the raw input in this simple test
        if goal == "find_key" and "has_key" in state:
            return True
        if goal == "pull_lever" and "lever_pulled" in state:
            return True
        if goal == "open_chest" and "chest_open" in state:
            return True
        return False

    return HierarchicalPlanningAgent(
        name="test_hierarchical",
        core=engine,
        pipeline=pipeline,
        prerequisite_map=prerequisite_map,
        completion_checker=completion_checker
    )

def test_goal_stack_decomposition(agent):
    agent.push_goal("open_chest")
    # Stack should be ["open_chest", "pull_lever", "find_key"] (reversed order for LIFO)
    assert agent.goal_stack == ["open_chest", "pull_lever", "find_key"]

def test_goal_stack_execution(agent):
    agent.push_goal("open_chest")
    
    # Step 1: find_key
    out = agent.step(AgentInput(raw="initial_state"))
    assert agent.goal_stack[-1] == "find_key"
    
    # Step 2: fulfill find_key
    out = agent.step(AgentInput(raw="has_key"))
    # find_key should be popped, next is pull_lever
    assert "find_key" not in agent.goal_stack
    assert agent.goal_stack[-1] == "pull_lever"
    
    # Step 3: fulfill pull_lever
    out = agent.step(AgentInput(raw="lever_pulled"))
    assert "pull_lever" not in agent.goal_stack
    assert agent.goal_stack[-1] == "open_chest"
    
    # Step 4: fulfill open_chest
    out = agent.step(AgentInput(raw="chest_open"))
    assert len(agent.goal_stack) == 0
    assert out.trace.get("status") == "completed"

def test_nested_decomposition(agent):
    # Add another level: "find_key" -> ["unlock_gate"]
    agent.prerequisite_map["find_key"] = ["unlock_gate"]
    agent.push_goal("open_chest")
    
    # Expected stack: ["open_chest", "pull_lever", "find_key", "unlock_gate"]
    assert agent.goal_stack == ["open_chest", "pull_lever", "find_key", "unlock_gate"]

def test_engine_swapping(agent):
    special_engine = PatternEngine()
    agent.engines["find_key"] = special_engine
    
    # 1. Goal with special engine
    agent.push_goal("find_key")
    agent.step(AgentInput(raw="initial"))
    assert agent.pipeline.engine == special_engine
    
    # 2. Fulfill that goal, move to a goal WITHOUT special engine
    agent.prerequisite_map["open_chest"] = ["find_key", "pull_lever"]
    agent.goal_stack = [] # reset
    agent.push_goal("open_chest")
    
    # top is find_key
    agent.step(AgentInput(raw="initial"))
    assert agent.pipeline.engine == special_engine
    
    # fulfill find_key
    agent.step(AgentInput(raw="has_key"))
    # now top is pull_lever, which has no special engine
    assert agent.goal_stack[-1] == "pull_lever"
    assert agent.pipeline.engine == agent.core
