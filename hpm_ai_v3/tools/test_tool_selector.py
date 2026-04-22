# hpm_ai_v3/tools/test_tool_selector.py
import numpy as np
import pytest

def test_cosine_similar_vectors():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert ts._cosine(a, b) == pytest.approx(1.0)

def test_cosine_orthogonal_vectors():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert ts._cosine(a, b) == pytest.approx(0.0)

def test_cosine_zero_vector():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    assert ts._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_apply_returns_same_length_as_weights():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    patterns = [
        ActionPattern("python_call", module="str", function="split"),
        ActionPattern("python_call", module="textblob", function="TextBlob"),
        ActionPattern("python_call", module="math", function="sqrt"),
    ]
    weights = np.array([1.0, 1.0, 1.0])
    adjusted = ts.apply("Count words in: hello world", weights, patterns)
    assert len(adjusted) == 3
    assert all(w >= 0 for w in adjusted)

def test_apply_boosts_relevant_pattern():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm, alpha=1.0)
    split_pat = ActionPattern("python_call", module="builtins", function="str.split")
    sqrt_pat = ActionPattern("python_call", module="math", function="sqrt")
    patterns = [split_pat, sqrt_pat]
    weights = np.array([1.0, 1.0])
    adjusted = ts.apply("split the words in this text", weights, patterns)
    # str.split should be boosted relative to sqrt for a word-splitting task
    assert adjusted[0] >= adjusted[1]

def test_action_pattern_tool_description_module_function():
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    p = ActionPattern("python_call", module="math", function="sqrt")
    assert "math" in p.tool_description
    assert "sqrt" in p.tool_description

def test_action_pattern_tool_description_action_only():
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    p = ActionPattern("list_modules")
    assert p.tool_description == "list_modules"

def test_act_with_tool_selector_does_not_crash():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    agent = UnifiedDiscoveryAgent(context_dim=64)
    lm = LanguageModelPattern()
    agent.tool_selector = ToolSelector(lm, alpha=0.5)
    task = {"text": "Count words in: hello world", "type": "pretraining", "answer": 2.0}
    agent.current_task = task
    result = agent.act(step_idx=0, step_population=False)
    assert "action" in result or "status" in result

def test_act_without_tool_selector_unchanged():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    agent = UnifiedDiscoveryAgent(context_dim=64)
    assert agent.tool_selector is None
    task = {"text": "10 + 5", "type": "pretraining", "answer": 15.0}
    agent.current_task = task
    result = agent.act(step_idx=0, step_population=False)
    assert "action" in result or "status" in result

def test_unified_agent_has_tool_selector_when_lm_provided():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
    assert agent.tool_selector is not None

def test_unified_agent_no_tool_selector_without_lm():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    agent = UnifiedDiscoveryAgent(context_dim=64)
    assert agent.tool_selector is None
