import pytest
import numpy as np
from hpm_ai_v3.tools.perception import summarize_task, summarize_pool
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.innate import register_innate_tools
from hpm_ai_v3.agents.base_discovery import ActionPattern
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent

def test_summarize_task_grid():
    task = {
        "text": "Solve this ARC task",
        "grid": [[1, 2], [3, 4]],
        "meta": {"id": "arc_001"},
        "answer": [[2, 4], [6, 8]]
    }
    summary = summarize_task(task)
    print(f"\nTask Summary:\n{summary}")
    assert "Task has 3 visible keys" in summary
    assert "2x2 grid of ints" in summary
    assert "'meta': dict (keys=['id'])" in summary
    assert "answer" not in summary # Should not reveal answer

def test_summarize_pool_complex():
    pool = [42, "hello", [1, 2, 3], np.zeros((3, 3))]
    summary = summarize_pool(pool)
    print(f"\nPool Summary:\n{summary}")
    assert "Pool has 4 items" in summary
    assert "int (value=42)" in summary
    assert "string (len=5" in summary
    assert "list (len=3" in summary
    assert "numpy array (shape=(3, 3)" in summary

def test_agent_perception_integration():
    """Verify agent act() correctly passes full objects to perception tools."""
    from hpm_ai_v3.tools.memory import register_memory_tools
    ToolRegistry.clear()
    register_memory_tools()
    register_innate_tools()
    
    # UnifiedDiscoveryAgent implements all abstract methods
    agent = UnifiedDiscoveryAgent()
    task = {"text": "What is in my task?", "data": [1, 2, 3], "answer": "nothing"}
    agent.current_task = task
    
    # Use tool_name property which combines module and function
    target_tool = "hpm_ai_v3.tools.perception.summarize_task"
    
    # Find or Inject the summarize_task pattern in the population
    perception_pattern = None
    for p in agent.population.patterns:
        if hasattr(p, 'tool_name') and p.tool_name == target_tool:
            perception_pattern = p
            break
            
    if perception_pattern is None:
        # Register it first so creation works
        ToolRegistry.register("summarize_task", summarize_task, ["task"], "result", 0.005,
                              module="hpm_ai_v3.tools.perception", function="summarize_task")
        perception_pattern = ActionPattern("summarize_task", module="hpm_ai_v3.tools.perception", function="summarize_task")
        agent.population.patterns.append(perception_pattern)
    
    assert perception_pattern.tool_name == target_tool
    
    # Mock act to pick this pattern
    import numpy as np
    from unittest.mock import patch
    
    # We need to ensure weights give 100% to perception_pattern
    # But act() does np.random.choice based on p=probs
    # We can mock the chosen index
    pattern_idx = agent.population.patterns.index(perception_pattern)
    
    with patch('numpy.random.choice', return_value=pattern_idx):
        result = agent.act(step_idx=0)
    
    print(f"\nAgent Perception Result:\n{result['result']}")
    assert "Task has 2 visible keys" in result["result"]
    assert "data" in result["result"]
    assert "list (len=3" in result["result"]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
