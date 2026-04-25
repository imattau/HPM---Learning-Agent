from hpm_ai_v3.augmented_agent import AugmentedHPMAgent

def create_numeric_agent():
    return AugmentedHPMAgent(
        tool_names=["numeric_eval", "unit_converter"],
        base_patterns=[], # Pure tool specialist
        agent_id="numeric_agent"
    )
