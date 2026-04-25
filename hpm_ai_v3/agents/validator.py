from hpm_ai_v3.augmented_agent import AugmentedHPMAgent

def create_validator_agent():
    return AugmentedHPMAgent(
        tool_names=["dimensional_analysis"],
        base_patterns=[], # Pure tool specialist
        agent_id="validator_agent"
    )
