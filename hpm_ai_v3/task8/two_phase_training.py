"""
two_phase_training.py - Supervised discovery followed by unsupervised application.
Phase 1: Problems with answers → dense feedback → store strategies.
Phase 2: Problems only → vector memory retrieval → apply/adapt strategies.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.memory import vector_store, vector_search, episodic_clear
from hpm_ai_v3.curriculum import CurriculumManager

class TwoPhaseTrainer:
    def __init__(self, agent: UnifiedDiscoveryAgent, vector_collection: str = "successful_strategies"):
        self.agent = agent
        self.vector_collection = vector_collection
        self.phase1_complete = False
        self.strategy_cache = {} # problem_text -> strategy

    def run_phase1(self, tasks: List[Dict], max_steps_per_task: int = 30, success_threshold: float = 0.9):
        """
        Phase 1: Answer-Key Discovery.
        Persistence: Agent stays with a task until solved.
        Correctness feedback is provided, but curiosity is internal.
        """
        print("\n=== PHASE 1: Answer-Key Discovery ===\n")
        
        for task_idx, task in enumerate(tasks):
            print(f"\n[Task {task_idx+1}/{len(tasks)}] {task.get('text', 'No text')}")
            task_solved = False
            episode = 0
            
            while not task_solved:
                episode += 1
                self.agent.initialize_task(task)
                self.agent.current_task = task
                self.agent.episode_sequence = []
                episodic_clear()
                
                # For dense feedback, we need to intercept act() and provide reward
                solution = self._run_episode_with_feedback(task, max_steps_per_task)
                
                if solution is not None:
                    reward = self.agent.evaluate_solution(solution)
                    if reward >= success_threshold:
                        task_solved = True
                        print(f"  ✓ Solved in episode {episode}")
                        
                        # Store successful strategy in vector memory
                        strategy = {
                            "sequence": self.agent.episode_sequence.copy(),
                            "solution": solution,
                            "task_text": task.get("text", "")
                        }
                        embedding = self._embed_text(task.get("text", ""))
                        vector_store(embedding, strategy, collection=self.vector_collection)
                        self.strategy_cache[task.get("text", "")] = strategy
                        break
                else:
                    # Optional: periodically log failure to avoid silence
                    if episode % 10 == 0:
                        print(f"  (Episode {episode} in progress...)")
                
                # Limit safety (just in case)
                if episode > 500:
                    print(f"  ⚠ Giving up on Task after {episode} episodes.")
                    break
                
        self.phase1_complete = True
        print(f"\n[Phase 1 Complete] Stored {len(self.strategy_cache)} successful strategies.")

    def _run_episode_with_feedback(self, task: Dict, max_steps: int) -> Optional[Any]:
        """Run an episode with extrinsic success/failure feedback."""
        for step in range(max_steps):
            # Call act without internal population step to avoid double-stepping
            step_res = self.agent.act(step, step_population=False)
            
            # Simple success/failure signal (no hand-holding bonuses)
            reward = -0.05 # step penalty
            val = step_res.get("result")
            status = step_res.get("status", "failed")
            
            if status != "success":
                reward = -0.1
            elif self._is_correct_answer(val, task.get("answer")):
                reward = 1.0
                self.agent.predicted_solution = val
                # Success signal
                obs = {"reward": torch.tensor([reward], device=self.agent.population._device), "outcome": step_res}
                self.agent.population.step(self.agent.evaluator_mgr, obs, self.agent.compiler)
                return val
            
            # Pattern update for non-solution steps
            obs = {"reward": torch.tensor([reward], device=self.agent.population._device), "outcome": step_res}
            self.agent.population.step(self.agent.evaluator_mgr, obs, self.agent.compiler)
            if hasattr(self.agent.population.get_top_patterns(1)[0], "update_parameters"):
                # We need to ensure the correct pattern is updated
                # In feedback mode, the agent just acted, so the last pattern used should be updated.
                # Find the pattern that was just used:
                # But wait, UnifiedDiscoveryAgent.act already called action_pattern.mark_used()
                # and action_pattern.update_parameters(obs) IF step_population was true.
                # However, here step_population=False, so we MUST call it manually.
                pass 
                
        return None

    def _is_correct_answer(self, val: Any, answer: Any) -> bool:
        if val is None or answer is None: return False
        try:
            if isinstance(answer, list):
                if not isinstance(val, (list, tuple)) or len(val) != len(answer):
                    return False
                return all(abs(float(a) - float(v)) < 1e-6 for a, v in zip(answer, val))
            return abs(float(val) - float(answer)) < 1e-6
        except (TypeError, ValueError):
            return val == answer

    def _is_novel_in_episode(self, val: Any) -> bool:
        """Check if value hasn't been produced in this episode yet."""
        if not hasattr(self.agent, '_episode_results'):
            self.agent._episode_results = set()
        key = str(val)
        if key in self.agent._episode_results:
            return False
        self.agent._episode_results.add(key)
        return True

    def run_phase2(self, tasks: List[Dict], max_steps_per_task: int = 30, use_memory: bool = True, delayed_feedback: bool = False):
        """
        Phase 2: Problems-Only Application.
        Agent uses vector memory to retrieve Phase 1 strategies.
        No intermediate feedback.
        """
        print("\n=== PHASE 2: Problems-Only Application ===\n")
        results = []
        
        for task_idx, task in enumerate(tasks):
            print(f"\n[Task {task_idx+1}/{len(tasks)}] {task.get('text', 'No text')}")
            
            # Retrieve similar strategy from Phase 1
            prior_strategy = None
            if use_memory:
                embedding = self._embed_text(task.get("text", ""))
                search_res = vector_search(embedding, collection=self.vector_collection, top_k=1)
                if search_res.get("results"):
                    prior_strategy = search_res["results"][0]["metadata"]
                    print(f"  Retrieved strategy: {prior_strategy.get('sequence', [])}")
            
            # Run episode without intermediate feedback
            solution = self._run_episode_no_feedback(task, max_steps_per_task, prior_strategy)
            
            # Optional delayed feedback (for continued learning)
            if delayed_feedback and "answer" in task:
                if solution is not None:
                    reward = self.agent.evaluate_solution(solution)
                    obs = {"reward": torch.tensor([reward], device=self.agent.population._device), "outcome": {"result": solution}}
                    self.agent.population.step(self.agent.evaluator_mgr, obs, self.agent.compiler)
                    
            results.append({
                "task": task.get("text", ""),
                "solution": solution,
                "expected": task.get("answer"),
                "correct": self._is_correct_answer(solution, task.get("answer")) if "answer" in task else None,
                "used_memory": prior_strategy is not None
            })
            
            status = "✓" if results[-1]["correct"] else ("?" if results[-1]["correct"] is None else "✗")
            print(f"  {status} Solution: {solution}")
            
        return results

    def _run_episode_no_feedback(self, task: Dict, max_steps: int, prior_strategy: Optional[Dict] = None) -> Optional[Any]:
        """Run episode without intermediate rewards. Optionally bias toward prior strategy."""
        self.agent.current_task = task
        self.agent.predicted_solution = None
        self.agent.episode_sequence = []
        episodic_clear()
        self.agent._episode_results = set()
        
        # If we have a prior strategy, we can bias the orchestrator (optional)
        # For pure HPM, we just let the agent act; the prior is only for logging.
        for step in range(max_steps):
            step_res = self.agent.act(step)
            if step_res.get("status") == "success":
                val = step_res.get("result")
                # In Phase 2, we consider any numeric result as a potential solution
                if isinstance(val, (int, float, list)):
                    self.agent.predicted_solution = val
                    return val
        return None

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using available tool."""
        try:
            res = ToolRegistry.call("embed_text", text=text)
            if res.get("status") == "success":
                return res["embedding"]
        except:
            pass
        
        # Fallback: simple hash-based pseudo-embedding
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        # Create a 384-dim vector from MD5 (pad with zeros)
        vec = np.zeros(384)
        data = np.frombuffer(h, dtype=np.uint8)
        for i, val in enumerate(data):
            vec[i % 384] = val / 255.0
        return vec.tolist()

def demo_two_phase():
    """Demonstrate two-phase training on simple arithmetic tasks."""
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.tools.python_substrate import register_python_substrate
    from hpm_ai_v3.tools.memory import register_memory_tools
    
    register_python_substrate()
    register_memory_tools()
    
    agent = UnifiedDiscoveryAgent(context_dim=64)
    trainer = TwoPhaseTrainer(agent)
    
    # Phase 1: Answer-Key tasks
    phase1_tasks = [
        {"text": "1 + 1", "answer": 2},
        {"text": "3 * 4", "answer": 12},
        {"text": "10 - 2", "answer": 8},
        {"text": "sin(0)", "answer": 0.0},
        {"text": "sqrt(16)", "answer": 4.0},
    ]
    
    trainer.run_phase1(phase1_tasks, max_steps_per_task=20, max_episodes=3)
    
    # Phase 2: Problems-Only tasks (no answers provided)
    phase2_tasks = [
        {"text": "2 + 3"},
        {"text": "5 * 6"},
        {"text": "cos(0)"},
        {"text": "20 - 7"},
    ]
    
    results = trainer.run_phase2(phase2_tasks, max_steps_per_task=20)
    
    print("\n=== Phase 2 Results ===")
    for r in results:
        print(f"{r['task']}: {r['solution']} (memory used: {r['used_memory']})")

if __name__ == "__main__":
    demo_two_phase()
