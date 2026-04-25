"""
MetaTrainingLoop — wraps an existing agent training loop with meta-cognitive oversight.
Calls the agent's run_episode() N times, then triggers a meta-step.
"""
from typing import Any, List, Optional
import numpy as np

from ..meta_cognitive_pattern import MetaCognitivePattern


class MetaTrainingLoop:
    def __init__(self, agent: Any, curriculum: Any,
                 meta_pattern: MetaCognitivePattern,
                 N: int = 10, max_meta_steps: int = 100):
        self.agent = agent
        self.curriculum = curriculum
        self.meta = meta_pattern
        self.N = N
        self.max_meta_steps = max_meta_steps

        # Initialise tracking attributes on agent if not present
        if not hasattr(agent, "_meta_success_history"):
            agent._meta_success_history = []
        if not hasattr(agent, "_steps_since_advance"):
            agent._steps_since_advance = 0

    def run(self):
        """Run max_meta_steps meta-steps."""
        for meta_step in range(self.max_meta_steps):
            print(f"\n=== Meta-Step {meta_step + 1}/{self.max_meta_steps} ===")
            self.run_meta_step()

    def run_meta_step(self):
        """Run N base episodes, then one meta update."""
        episode_rewards: List[float] = []

        for _ in range(self.N):
            reward = self._run_single_episode()
            episode_rewards.append(reward)
            self.agent._meta_success_history.append(reward)
            if len(self.agent._meta_success_history) > 50:
                self.agent._meta_success_history.pop(0)
            self.agent._steps_since_advance += 1

        # Meta observation and directive
        features = self.meta.observe(self.agent, self.curriculum)
        directive = self.meta.observe_and_act(self.agent, self.curriculum)

        # Meta reward
        meta_reward = self.meta.compute_meta_reward(self.agent, self.curriculum)
        self.meta.record_transition(features, directive, meta_reward)
        self.meta.update_policy()

        self._reset_use_counts()

        print(f"  [MetaLoop] N={self.N} episodes, mean_reward={np.mean(episode_rewards):.3f}, "
              f"meta_reward={meta_reward:.3f}, directive={directive.name}")

    def _run_single_episode(self) -> float:
        """Run one episode on the agent. Returns episode reward."""
        task = self.curriculum.get_current_task()
        if hasattr(self.agent, "run_episode"):
            solution = self.agent.run_episode(task)
            reward = self.agent.evaluate_solution(solution)
        else:
            # Fallback: single act() call
            self.agent.current_task = task
            result = self.agent.act()
            reward = 1.0 if result.get("status") == "success" else -0.5
        
        self.curriculum.update(reward)
        return reward

    def _reset_use_counts(self):
        """Reset recent_use_count on all ActionPatterns after each meta-step."""
        for p in self.agent.population.patterns:
            if hasattr(p, "recent_use_count"):
                p.recent_use_count = 0
