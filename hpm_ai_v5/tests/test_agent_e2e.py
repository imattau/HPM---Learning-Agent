from hpm_ai_v5.agents import AgentPacket, AgentPipeline, BaseAgent
from hpm_ai_v5.core import PatternEngine
from hpm_ai_v5.postprocessors.numeric import NumericPostprocessor
from hpm_ai_v5.preprocessors.numeric import NumericPreprocessor


def test_numeric_agent_end_to_end_preprocess_core_postprocess() -> None:
    agent = BaseAgent(
        name="NumericAgent",
        core=PatternEngine(),
        preprocessors=[NumericPreprocessor()],
        postprocessors=[NumericPostprocessor()],
    )
    pipeline = AgentPipeline(agents=[agent])
    packet = AgentPacket(raw_input=1.0, context={"minimum": 0.0, "maximum": 10.0})

    first = pipeline.run(packet)
    packet.raw_input = 4.0
    second = pipeline.run(packet)

    assert first is second
    assert second.agent_trace == ["NumericAgent", "NumericAgent"]
    assert second.candidate_outputs
    assert second.core_actions
    assert second.final_output == 7.0
    assert second.state["agent_outputs"]
    assert agent.state["last_packet"] is not None
    assert any(entry["role"] == "adapter" for entry in agent.state["last_packet"].trace)
    assert any(entry["role"] == "agent" for entry in second.trace)
