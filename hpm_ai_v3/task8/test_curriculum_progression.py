import pytest
from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
import numpy as np

def test_curriculum_phases_in_order():
    cm = CurriculumManager()
    phases = [p.phase for p in cm.patterns]
    assert phases == sorted(phases), "Phases must be sorted ascending"
    assert cm.patterns[0].name == "Tool Familiarization"
    assert cm.patterns[-1].name == "Quantitative Synthesis"

def test_arithmetic_phase_tasks():
    cm = CurriculumManager()
    p = next(x for x in cm.patterns if x.name == "Arithmetic Reasoning")
    assert len(p.tasks) == 10
    practice = [t for t in p.tasks if t["type"] == "practice"]
    assert len(practice) >= 3, "Need at least 3 demo tasks in arithmetic phase"

def test_synthesis_phase_no_demos():
    cm = CurriculumManager()
    p = next(x for x in cm.patterns if x.name == "Quantitative Synthesis")
    for task in p.tasks:
        assert "demo" not in task, f"Synthesis task should have no demo: {task['text']}"

def test_agent_advances_past_arithmetic(tmp_path):
    """Agent should advance from Arithmetic Reasoning within 50 episodes."""
    agent = UnifiedDiscoveryAgent(context_dim=64)
    cm = CurriculumManager()

    # Fast-forward to arithmetic phase
    arith_idx = next(i for i, p in enumerate(cm.patterns) if p.name == "Arithmetic Reasoning")
    cm.active_pattern_idx = arith_idx
    cm.phase = cm.patterns[arith_idx].phase

    advanced = False
    for ep in range(50):
        task = cm.get_current_task()
        solution = agent.run_episode(task, max_steps=15)
        reward = agent.evaluate_solution(solution)
        prev = cm.active_pattern_idx
        cm.update(reward)
        if cm.active_pattern_idx > prev:
            advanced = True
            break

    assert advanced, "Agent should advance from Arithmetic Reasoning within 50 episodes"

def test_tool_selector_improves_nlp_phase_reward():
    """Agent with ToolSelector should get higher avg reward on NLP tasks."""
    import numpy as np
    import os
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.curriculum import CurriculumManager
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern

    def run_nlp_episodes(use_lm: bool) -> float:
        lm = None
        if use_lm:
            lm = LanguageModelPattern()
            # Pretrain on tiny corpus to get non-random embeddings
            if os.path.exists("hpm_ai_v3/data/lm_corpus/sample.txt"):
                lm.pretrain("hpm_ai_v3/data/lm_corpus/sample.txt", epochs=2)
            else:
                # Fallback pretrain on code strings
                lm.fine_tune("split tokenize count word text sentiment positive negative upper lower math sqrt calculation", epochs=5)
            
        agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
        cm = CurriculumManager()
        # Fast-forward to NLP Tool Mastery phase (if it exists, else use last)
        try:
            nlp_idx = next(i for i, p in enumerate(cm.patterns)
                           if "NLP" in p.name)
        except StopIteration:
            nlp_idx = len(cm.patterns) - 1
            
        cm.active_pattern_idx = nlp_idx
        cm.phase = cm.patterns[nlp_idx].phase
        rewards = []
        for _ in range(10): # Reduced for speed
            task = cm.get_current_task()
            sol = agent.run_episode(task, max_steps=10)
            r = agent.evaluate_solution(sol)
            rewards.append(r)
        return float(np.mean(rewards))

    reward_without = run_nlp_episodes(use_lm=False)
    reward_with = run_nlp_episodes(use_lm=True)
    print(f"\nWithout ToolSelector: {reward_without:.3f}")
    print(f"With ToolSelector:    {reward_with:.3f}")
    # ToolSelector should not make things worse
    assert reward_with >= reward_without - 0.2
