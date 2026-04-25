from hpm_ai_v3.augmented_agent import AugmentedHPMAgent

def create_algebra_agent():
    return AugmentedHPMAgent(
        tool_names=["sympy_solve"],
        base_patterns=[], # Pure tool specialist
        agent_id="algebra_agent"
    )
