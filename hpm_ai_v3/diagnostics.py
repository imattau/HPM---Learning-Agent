"""
diagnostics.py - Performance logging and diagnostic monitoring for HPM agents.
Tracks LM loss, tool selection distribution, and population stratification.
"""

import json
import csv
import os
import time
from typing import Dict, Any, List


class HPMLogger:
    """
    Records per-episode and per-epoch metrics to verify learning dynamics.
    """
    def __init__(self, log_dir: str = "./logs/diagnostics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_log = os.path.join(log_dir, "episodes.csv")
        self.lm_log = os.path.join(log_dir, "lm_stats.csv")
        
        # Initialize CSV files
        if not os.path.exists(self.episode_log):
            with open(self.episode_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "phase", "step0_tool", "reward", "success", "top_patterns"])
        
        if not os.path.exists(self.lm_log):
            with open(self.lm_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "episode", "loss", "internal_accuracy"])

    def log_episode(self, episode: int, phase: str, step0_tool: str, reward: float, population: Any):
        """Record episode outcome and population state."""
        success = 1 if reward > 0.9 else 0
        
        # Extract top 3 patterns by weight
        patterns = sorted(population.patterns, key=lambda p: p.weight, reverse=True)
        top_p = [f"{getattr(p, 'tool_name', p.id)}:{p.weight:.3f}" for p in patterns[:3]]
        
        with open(self.episode_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, phase, step0_tool, reward, success, "|".join(top_p)])

    def log_lm_stats(self, episode: int, loss: float, accuracy: float = 0.0):
        """Record LM training progress."""
        with open(self.lm_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), episode, loss, accuracy])

    def get_summary(self) -> Dict[str, Any]:
        """Compute high-level success metrics from logs."""
        try:
            with open(self.episode_log, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows: return {}
                
                recent = rows[-100:]
                avg_reward = sum(float(r['reward']) for r in recent) / len(recent)
                success_rate = sum(int(r['success']) for r in recent) / len(recent)
                
                return {
                    "total_episodes": len(rows),
                    "avg_reward_recent": avg_reward,
                    "success_rate_recent": success_rate,
                    "last_phase": rows[-1]['phase']
                }
        except Exception as e:
            return {"error": str(e)}
