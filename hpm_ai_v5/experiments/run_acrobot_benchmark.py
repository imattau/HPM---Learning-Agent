"""Acrobot Transfer (ACT) benchmark script."""

from __future__ import annotations

import time
from pathlib import Path

from hpm_ai_v5.planning.cartpole import CartpoleBenchmark, DEFAULT_CARTPOLE_VARIANTS
from hpm_ai_v5.planning.acrobot import AcrobotBenchmark, AcrobotEnvConfig


def run_act_benchmark(
    cartpole_eps: int = 50,
    acrobot_scratch_eps: int = 200,
    acrobot_fine_tune_eps: int = 200,
    max_steps: int = 500,
):
    print(f"=== Phase 1: Training Source (Cartpole) - {cartpole_eps} eps ===")
    cartpole_source = CartpoleBenchmark(env_config=DEFAULT_CARTPOLE_VARIANTS["cartpole"])
    start_time = time.time()
    cartpole_res = cartpole_source.run(episodes=cartpole_eps, total_episodes=cartpole_eps, max_steps=max_steps)
    print(f"Cartpole source trained. Result: {cartpole_res.result}, Avg len: {cartpole_res.average_length:.1f}")
    source_state = cartpole_source.export_transfer_state()
    
    print(f"\n=== Phase 2: Training Acrobot from Scratch - {acrobot_scratch_eps} eps ===")
    acrobot_scratch = AcrobotBenchmark()
    scratch_res = acrobot_scratch.run(episodes=acrobot_scratch_eps, total_episodes=acrobot_scratch_eps, max_steps=max_steps)
    print(f"Acrobot scratch trained. Result: {scratch_res.result}, Avg len: {scratch_res.average_length:.1f}")
    print(f"Goal reached in {sum(1 for l in scratch_res.episode_lengths if l < max_steps)}/{acrobot_scratch_eps} episodes")

    print(f"\n=== Phase 3: Transfer from Cartpole to Acrobot - {acrobot_fine_tune_eps} eps ===")
    acrobot_transfer = AcrobotBenchmark()
    acrobot_transfer.import_transfer_state(source_state)
    
    print("Running Zero-Shot Evaluation...")
    zero_shot_res = acrobot_transfer.run(episodes=10, evaluate=True, max_steps=max_steps)
    print(f"Zero-shot avg len: {zero_shot_res.average_length:.1f}")
    
    print(f"Fine-tuning on Acrobot - {acrobot_fine_tune_eps} eps...")
    fine_tune_res = acrobot_transfer.run(episodes=acrobot_fine_tune_eps, total_episodes=acrobot_fine_tune_eps, max_steps=max_steps)
    print(f"Fine-tuned avg len: {fine_tune_res.average_length:.1f}")
    print(f"Goal reached in {sum(1 for l in fine_tune_res.episode_lengths if l < max_steps)}/{acrobot_fine_tune_eps} episodes")

    print("\n=== Phase 4: Catastrophic Forgetting Evaluation ===")
    # Test the acrobot-trained agent back on Cartpole
    acrobot_state = acrobot_transfer.export_transfer_state()
    cartpole_eval = CartpoleBenchmark()
    cartpole_eval.import_transfer_state(acrobot_state)
    
    forgetting_res = cartpole_eval.run(episodes=20, evaluate=True, max_steps=max_steps)
    print(f"Cartpole performance after Acrobot training: {forgetting_res.average_length:.1f}")
    eval_avg = cartpole_res.evaluation_average if cartpole_res.evaluation_average > 0 else 1.0
    ratio = forgetting_res.average_length / eval_avg
    print(f"Forgetting Ratio: {ratio:.2%}")

    print(f"\nBenchmark completed in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        run_act_benchmark(cartpole_eps=5, acrobot_scratch_eps=5, acrobot_fine_tune_eps=5, max_steps=20)
    else:
        run_act_benchmark()
