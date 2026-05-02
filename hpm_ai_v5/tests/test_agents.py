from hpm_ai_v5.agents import AgentPacket, AgentPipeline, BaseAgent, AgentInput
from hpm_ai_v5.core import PatternEngine
from hpm_ai_v5.postprocessors.numeric import NumericPostprocessor
from hpm_ai_v5.preprocessors.numeric import NumericPreprocessor


def test_agent_pipeline_runs_agents_in_order() -> None:
    class RouterAgent:
        name = "router"

        def step(self, packet: AgentPacket) -> AgentPacket:
            packet.agent_trace.append(self.name)
            packet.context["route"] = "numeric"
            return packet

    class OutputAgent:
        name = "output"

        def step(self, packet: AgentPacket) -> AgentPacket:
            packet.agent_trace.append(self.name)
            packet.final_output = {"route": packet.context.get("route"), "content": packet.raw_input}
            return packet

    pipeline = AgentPipeline(agents=[RouterAgent(), OutputAgent()])
    packet = pipeline.run(AgentPacket(raw_input=5.0))

    assert packet.final_output == {"route": "numeric", "content": 5.0}
    assert packet.agent_trace == ["router", "output"]
    assert [entry["agent"] for entry in packet.trace] == ["router", "output"]


def test_base_agent_step_packet_composes_with_agent_pipeline_shape() -> None:
    agent = BaseAgent(
        name="NumericAgent",
        core=PatternEngine(),
        preprocessors=[NumericPreprocessor()],
        postprocessors=[NumericPostprocessor()],
    )
    packet = AgentPacket(raw_input=1.0, context={"minimum": 0.0, "maximum": 10.0})

    updated = agent.step_packet(packet)

    assert updated.candidate_outputs
    assert updated.core_actions
    assert updated.agent_trace == ["NumericAgent"]
    assert updated.trace and updated.trace[-1]["agent"] == "NumericAgent"


def test_agent_pipeline_dispatches_base_agent_step_packet() -> None:
    agent = BaseAgent(
        name="NumericAgent",
        core=PatternEngine(),
        preprocessors=[NumericPreprocessor()],
        postprocessors=[NumericPostprocessor()],
    )
    pipeline = AgentPipeline(agents=[agent])
    packet = AgentPacket(raw_input=1.0, context={"minimum": 0.0, "maximum": 10.0})

    updated = pipeline.run(packet)

    assert updated.agent_trace == ["NumericAgent"]
    assert updated.candidate_outputs
    assert updated.core_actions
    assert updated.trace and updated.trace[-1]["role"] == "agent"
