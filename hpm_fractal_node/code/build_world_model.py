"""
Build and serialize the code world model to disk.

Run once (or after vocab/stdlib changes):
    PYTHONPATH=. python3 hpm_fractal_node/code/build_world_model.py

Writes:
    data/code_world_model.npz
    data/code_world_model.json

Subsequent experiment runs load in <0.1s via load_world_model().
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpm_fractal_node.code.code_world_model import build_code_world_model, save_world_model

OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "code_world_model"


def main() -> None:
    print("Building code world model ...")
    print("  (First run scans stdlib — subsequent runs use disk cache)")
    t0 = time.time()

    forest, prior_nodes = build_code_world_model()
    prior_ids = {n.id for n in prior_nodes}

    elapsed = time.time() - t0
    print(f"  Built {len(prior_nodes)} prior nodes in {elapsed:.1f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_world_model(forest, prior_ids, OUTPUT_PATH)
    print(f"  Saved to {OUTPUT_PATH}.npz / .json")
    print("Done. Experiments will load this model instantly.")


if __name__ == "__main__":
    main()
