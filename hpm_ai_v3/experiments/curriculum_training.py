"""
curriculum_training.py - Full curriculum training run using LM (L4) + MetaCognitive (L5).

Runs a single agent through the full curriculum, reporting phase advancement,
success rates, and meta-directive activity at regular intervals.

Usage:
    PYTHONPATH=. python3 hpm_ai_v3/experiments/curriculum_training.py
    PYTHONPATH=. python3 hpm_ai_v3/experiments/curriculum_training.py --episodes 2000
    PYTHONPATH=. python3 hpm_ai_v3/experiments/curriculum_training.py --no-lm
    PYTHONPATH=. python3 hpm_ai_v3/experiments/curriculum_training.py --no-l5
"""

import os
import time
import argparse
import numpy as np
from collections import defaultdict

from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern
from hpm_ai_v3.agents.meta_training import MetaTrainingLoop
from hpm_ai_v3.diagnostics import HPMLogger


def run(episodes: int = 1000, use_lm: bool = True, use_l5: bool = True,
        report_interval: int = 50, corpus_path: str = "hpm_ai_v3/data/lm_corpus/rich_corpus.txt"):

    print("=" * 60)
    print(f"HPM Curriculum Training  |  episodes={episodes}  LM={use_lm}  L5={use_l5}")
    print("=" * 60)

    # ── LM (L4) ──────────────────────────────────────────────────────────────
    lm = None
    if use_lm:
        lm = LanguageModelPattern()
        if os.path.exists(corpus_path):
            print(f"[LM] Pretraining on {corpus_path} ...")
            lm.pretrain(corpus_path, epochs=10)
        else:
            print(f"[LM] Corpus not found at {corpus_path}, skipping pretraining.")

    # ── Agent + Curriculum ────────────────────────────────────────────────────
    agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
    curriculum = CurriculumManager()
    logger = HPMLogger(log_dir="./logs/curriculum_training")

    # ── MetaCognitive (L5) ───────────────────────────────────────────────────
    meta = None
    meta_loop = None
    if use_l5:
        meta = MetaCognitivePattern()
        if lm and hasattr(agent, 'tool_selector'):
            meta._tool_selector = agent.tool_selector
            lm._tool_selector = agent.tool_selector
        meta_loop = MetaTrainingLoop(agent, curriculum, meta, N=10)

    # ── Training loop ─────────────────────────────────────────────────────────
    start = time.time()
    phase_history = []          # (episode, phase_name)
    directive_counts = defaultdict(int)
    ep = 0

    while ep < episodes:
        phase_name = curriculum.patterns[curriculum.active_pattern_idx].name

        if use_l5:
            meta_loop.run_meta_step()
            recent = agent._meta_success_history[-10:] if agent._meta_success_history else [0.0] * 10
            for i, r in enumerate(recent):
                logger.log_episode(ep + i, phase_name, "meta_guided", r, agent.population)

            # Track which directive fired
            if hasattr(meta, '_last_directive') and meta._last_directive:
                directive_counts[meta._last_directive] += 1

            ep += 10
        else:
            task = curriculum.get_current_task()
            solution = agent.run_episode(task, max_steps=10)
            reward = agent.evaluate_solution(solution)
            logger.log_episode(ep, phase_name, "standard", reward, agent.population)
            curriculum.update(reward)
            ep += 1

        # Track phase changes
        current_phase = curriculum.patterns[curriculum.active_pattern_idx].name
        if not phase_history or phase_history[-1][1] != current_phase:
            phase_history.append((ep, current_phase))
            print(f"  [Ep {ep:>5}] ► Phase advanced → {current_phase}")

        if ep % report_interval == 0:
            recent_rewards = (
                agent._meta_success_history[-report_interval:]
                if use_l5 and agent._meta_success_history
                else []
            )
            success_rate = np.mean([r > 0.8 for r in recent_rewards]) if recent_rewards else 0.0
            elapsed = time.time() - start
            print(f"  [Ep {ep:>5}] phase={current_phase:<30} success={success_rate:.0%}  t={elapsed:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    summary = logger.get_summary()
    final_phase = curriculum.patterns[curriculum.active_pattern_idx].name

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Episodes     : {episodes}")
    print(f"  Time         : {elapsed:.1f}s")
    print(f"  Final phase  : {final_phase}")
    print(f"  Phases seen  : {len(phase_history)}")
    print()
    print("Phase progression:")
    for ep_num, name in phase_history:
        print(f"    ep {ep_num:>5} → {name}")

    if directive_counts:
        print()
        print("Meta-directive activity:")
        for directive, count in sorted(directive_counts.items(), key=lambda x: -x[1]):
            print(f"    {directive:<30} {count:>4}x")

    print()
    print(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--no-lm", action="store_true")
    parser.add_argument("--no-l5", action="store_true")
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--corpus", default="hpm_ai_v3/data/lm_corpus/rich_corpus.txt")
    args = parser.parse_args()

    run(
        episodes=args.episodes,
        use_lm=not args.no_lm,
        use_l5=not args.no_l5,
        report_interval=args.report_interval,
        corpus_path=args.corpus,
    )
