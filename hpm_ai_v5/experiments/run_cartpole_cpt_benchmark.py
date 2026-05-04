"""Run the cross-physics transfer benchmark for CartPole variants."""

from __future__ import annotations

from hpm_ai_v5.planning.cartpole import run_cross_physics_transfer


def main() -> None:
    result = run_cross_physics_transfer()
    print(f"Source CartPole average: {result.source_average:.2f}")
    for target in result.targets:
        print(
            f"{target.variant}: zero-shot={target.zero_shot_average:.2f}, "
            f"fine-tune={target.fine_tune_average:.2f}, scratch={target.scratch_average:.2f}, "
            f"forgetting={target.forgetting_average:.2f}"
        )


if __name__ == "__main__":
    main()
