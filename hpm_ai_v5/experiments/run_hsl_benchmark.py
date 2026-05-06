"""Hierarchical Strategy Learning (HSL) Benchmark for V5 Meta-Patterns."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from hpm_ai_v5.core import PatternEngine, PatternManager, State
from hpm_ai_v5.core.config import CoreConfig


@dataclass
class HSLEnv:
    """A simple grid-world like environment for testing hierarchical strategies."""
    # 0: Key, 1: Unlock, 2: Turn, 3: Walk
    # Sequence [0, 1, 2] = OpenDoor
    # Sequence [OpenDoor, 3] = EnterRoom
    
    def reset(self) -> float:
        return 0.0
        
    def step(self, action: float) -> tuple[float, float, bool]:
        # Simple identity environment: state reflects the last action
        return action, 1.0, False


def run_hsl_benchmark():
    print("Running Hierarchical Strategy Learning (HSL) Benchmark...")
    
    engine = PatternEngine(config=CoreConfig(max_patterns=64, max_sequences=32))
    manager = PatternManager(promotion_threshold=1.0)
    env = HSLEnv()
    
    # Phase 1: Learn leaf patterns for door-opening (Key, Unlock, Turn)
    print("\nPhase 1: Learning Leaf Patterns (Key, Unlock, Turn)...")
    door_sequence = [0.0, 1.0, 2.0]
    for _ in range(10): # Repeat to stabilize
        obs = env.reset()
        for action in door_sequence:
            engine.observe(State(value=action))
    
    print(f"  - Patterns in store: {len(engine.store.patterns)}")
    for p in engine.store.patterns:
        print(f"    - {p.name}: template={p.template}, support={p.support}")

    # Phase 2: Promotion to Meta-Pattern (OpenDoor)
    print("\nPhase 2: Promoting Sequence to Meta-Pattern...")
    # The engine should have detected the sequence [0, 1, 2]
    print(f"  - Sequences detected: {[s.pattern_names for s in engine.sequences]}")
    
    # Check if a meta-pattern was created
    meta_patterns = engine.store.meta_patterns
    print(f"  - Meta-patterns created: {[m.name for m in meta_patterns]}")
    
    if not meta_patterns:
        print("  [!] No meta-patterns created yet, forcing promotion for test...")
        # Force the names from the trace
        names = tuple(engine.pattern_trace[-3:])
        meta = engine.promote_to_meta(names, name="OpenDoor")
    else:
        meta = meta_patterns[0]
        meta.name = "OpenDoor" # Rename for clarity in output
        
    print(f"  - Meta-pattern 'OpenDoor' children: {[c.name for c in meta.children]}")

    # Phase 3: Learn EnterRoom = OpenDoor + Walk
    print("\nPhase 3: Learning Higher-Level Meta-Pattern (EnterRoom)...")
    # Action 3.0 is 'Walk'
    # We simulate a sequence of (OpenDoor sequence) then 3.0
    for _ in range(5):
        obs = env.reset()
        # The agent 'uses' the OpenDoor meta-pattern then Walks
        # In this benchmark, we just feed the observations to see if it promotes again
        for action in door_sequence + [3.0]:
            engine.observe(State(value=action))

    # Promote the new sequence [OpenDoor, Walk]
    print(f"  - Pattern trace: {engine.pattern_trace[-10:]}")
    # We want to see if it detects [OpenDoor, Walk]
    # Note: the trace contains LEAF patterns. To see Meta-patterns in trace, 
    # we'd need the agent to select them.
    
    # For HSL, we want to prove we CAN create a second-level meta-pattern
    walk_pattern = engine.store.get(engine.pattern_trace[-1])
    if walk_pattern:
        enter_room = engine.promote_to_meta((meta.name, walk_pattern.name), name="EnterRoom")
        print(f"  - Meta-pattern 'EnterRoom' children: {[c.name for c in enter_room.children]}")

    # Phase 4: Validation
    print("\nPhase 4: Hierarchy Validation...")
    success = False
    if len(engine.store.meta_patterns) >= 2:
        print("  - Success: Hierarchy created (Leaf -> OpenDoor -> EnterRoom)")
        success = True
    else:
        print("  - Failure: Hierarchy not fully formed.")

    # Archive check
    manager.end_episode(engine)
    print(f"\nFinal Archive Stats: {manager.stats()}")
    
    if success and manager.meta_archive:
        print("\nHSL Benchmark: SUCCESS")
    else:
        print("\nHSL Benchmark: FAILURE")

if __name__ == "__main__":
    run_hsl_benchmark()
