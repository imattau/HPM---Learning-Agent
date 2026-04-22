"""
train_cold_start.py - Main entry point for training HPM agents from scratch.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Any, List

from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.agents.agent_persistence import save_full_checkpoint, load_full_checkpoint
from hpm_ai_v3.tools.memory import register_memory_tools, vector_store

try:
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    _LM_AVAILABLE = True
except ImportError:
    _LM_AVAILABLE = False


def make_text_embedding(text: str, lm_pattern=None) -> list:
    """
    Return a fixed-length float vector for text.
    Uses LM embedding if lm_pattern is provided and pretrained,
    otherwise falls back to bag-of-chars (first 384 ASCII/extended chars).
    """
    if lm_pattern is not None and getattr(lm_pattern, "_pretrained", False):
        return lm_pattern.sample({"action": "embed", "text": text})
    
    # Bag-of-chars fallback: count of characters, normalized
    emb = np.zeros(384)
    for char in str(text).lower():
        idx = ord(char) % 384
        emb[idx] += 1.0
    if np.sum(emb) > 0:
        emb /= np.sum(emb)
    return emb.tolist()


def train_cold_start(max_episodes: int = 2000, checkpoint_dir: str = "./checkpoints/cold_start"):
    print(f"=== HPM Cold Start Training Started ===")
    print(f"Checkpoint Dir: {checkpoint_dir}\n")
    
    # Initialize
    agent = UnifiedDiscoveryAgent(context_dim=64)
    curriculum = CurriculumManager()
    register_memory_tools(persistence_path=os.path.join(checkpoint_dir, "kv_store.json"))
    
    # Optional LM substrate
    lm_pattern = LanguageModelPattern() if _LM_AVAILABLE else None
    
    # Try to resume
    if os.path.exists(checkpoint_dir):
        print("[Persistence] Attempting to load latest checkpoint...")
        load_full_checkpoint(agent, checkpoint_dir, vector_collections=["successes"])
    
    start_time = time.time()
    
    for ep in range(max_episodes):
        task = curriculum.get_current_task()
        print(f"\n[Episode {ep}] Phase {curriculum.phase} (Diff: {curriculum.difficulty:.2f})")
        print(f"Task: {task['text']}")
        
        # Run agent on task
        solution = agent.run_episode(task, max_steps=50)
        reward = agent.evaluate_solution(solution)
        
        print(f"Result: {solution} | Reward: {reward:.4f}")
        
        # Memory Accumulation: Store successes
        if reward > 0.9:
            emb = make_text_embedding(task['text'], lm_pattern=lm_pattern)
            vector_store(emb, {"task_text": task['text'], "solution": solution}, collection="successes")
        
        # Update curriculum
        curriculum.update(reward)
        
        # Periodic Checkpointing
        if ep % 100 == 0 and ep > 0:
            save_full_checkpoint(agent, checkpoint_dir, vector_collections=["successes"])
            
        if curriculum.is_complete():
            print("\n--- CURRICULUM MASTERED! ---")
            break
            
    total_time = time.time() - start_time
    print(f"\n=== Training Complete in {total_time:.1f}s ===")
    
    # Save final state
    save_full_checkpoint(agent, checkpoint_dir, vector_collections=["successes"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    
    train_cold_start(max_episodes=args.episodes)
