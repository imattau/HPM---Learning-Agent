"""
curriculum.py - Agnostic JSON Importer for HPM agents.
Loads curriculums from JSON files into the agent's population as generative patterns.
"""

import os
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from hpm_ai_v3.agents.base_discovery import ActionPattern


class CurriculumPattern(ActionPattern):
    """
    A specific curriculum phase (e.g., Phase -1: Sensory Priming)
    treated as a generative pattern in the population.
    """
    def __init__(self, name: str, phase: int, tasks: List[Dict]):
        # We treat this as an 'orchestrator' that generates tasks
        super().__init__("curriculum_gen")
        self.name = name
        self.phase = phase
        self.tasks = tasks
        
        self.weight = 1.0 # Base weight for competition
        self.accuracy = 0.0 # Will track the agent's success on this curriculum
        self.affective_score = 3.0 # High curiosity initially

    def sample_task(self, difficulty: float = 0.0) -> Dict[str, Any]:
        """Sample a task from this curriculum's manifold, proportional to difficulty."""
        if not self.tasks:
            return {"type": "pretraining", "text": "Empty curriculum", "answer": 0.0}
        
        # Difficulty scales the window of tasks the agent sees
        num_tasks = len(self.tasks)
        max_idx = max(1, int(difficulty * num_tasks))
        # Add some randomness to see tasks slightly above current difficulty
        max_idx = min(num_tasks, max_idx + 2) 
        
        subset = self.tasks[:max_idx]
        return random.choice(subset)


class CurriculumImporter:
    """
    Loads curriculums from data/curriculums/*.json and converts them into
    Generative HPM Patterns.
    """
    @staticmethod
    def load_all(curriculum_dir: str = "hpm_ai_v3/data/curriculums") -> List[CurriculumPattern]:
        patterns = []
        if not os.path.exists(curriculum_dir):
            print(f"[Warning] Curriculum directory not found: {curriculum_dir}")
            return []
            
        for filename in sorted(os.listdir(curriculum_dir)):
            if filename.endswith(".json"):
                path = os.path.join(curriculum_dir, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        name = data.get("name", filename)
                        phase = data.get("phase", 0)
                        tasks = data.get("tasks", [])
                        
                        pattern = CurriculumPattern(name, phase, tasks)
                        patterns.append(pattern)
                        print(f"  [Importer] Loaded Curriculum: {name} (Phase {phase}, {len(tasks)} tasks)")
                except Exception as e:
                    print(f"  [Error] Failed to load {filename}: {e}")
                    
        return patterns


class CurriculumManager:
    """
    Legacy wrapper for training scripts, now uses the Importer.
    """
    def __init__(self):
        self.importer = CurriculumImporter()
        self.patterns = self.importer.load_all()
        # Sort patterns by phase
        self.patterns = sorted(self.patterns, key=lambda x: x.phase)
        
        # Training script expects these
        self.active_pattern_idx = 0
        self.phase = self.patterns[0].phase if self.patterns else 0
        self.difficulty = 0.0
        
        self.window_size = 10
        self.recent_rewards = []

    def get_current_task(self) -> Dict[str, Any]:
        """In a pure HPM model, the agent would choose the pattern. 
        For now, we help it by picking the 'active' phase."""
        if not self.patterns:
            return {"type": "pretraining", "text": "Empty curriculum", "answer": 0.0}
        
        p = self.patterns[self.active_pattern_idx]
        return p.sample_task(difficulty=self.difficulty)

    def update(self, reward: float):
        """Update the active curriculum phase and difficulty based on performance."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        self.difficulty = min(1.0, avg_reward) if avg_reward > 0 else 0.0
        
        # If mastery reached (> 0.8), advance the "focus" phase
        if avg_reward >= 0.8 and len(self.recent_rewards) >= 5:
            if self.active_pattern_idx < len(self.patterns) - 1:
                self.active_pattern_idx += 1
                self.phase = self.patterns[self.active_pattern_idx].phase
                self.recent_rewards = []
                print(f"--- CURRICULUM: Advancing to {self.patterns[self.active_pattern_idx].name} (Phase {self.phase}) ---")

    def advance_phase(self):
        """Directly advance curriculum phase (called by meta-directive)."""
        if self.active_pattern_idx < len(self.patterns) - 1:
            self.active_pattern_idx += 1
            self.phase = self.patterns[self.active_pattern_idx].phase
            self.recent_rewards = []
            print(f"--- META: Advancing to {self.patterns[self.active_pattern_idx].name} (Phase {self.phase}) ---")

    def set_difficulty(self, delta: float):
        """Adjust difficulty by delta, clamped to [0.0, 1.0]."""
        self.difficulty = max(0.0, min(1.0, self.difficulty + delta))

    def is_complete(self) -> bool:
        # Complete if mastery reached on the final phase
        if not self.patterns: return True
        if self.active_pattern_idx < len(self.patterns) - 1: return False
        return np.mean(self.recent_rewards) > 0.8 if self.recent_rewards else False
