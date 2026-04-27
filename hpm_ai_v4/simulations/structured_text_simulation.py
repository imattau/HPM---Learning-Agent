"""Structured text simulation for record-like JSON output."""
import argparse
import json
import os
from typing import Any, Dict, List, Optional

from hpm_ai_v4.io.adapters import StructuredTextAdapter
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.library_registry import LibraryRegistry


def _corrupt_structured_text(text: str) -> str:
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch in {"{", "}", ":", ","} and idx % 7 == 0:
            continue
        if ch == '"' and idx % 5 == 0:
            continue
        if ch.isdigit() and idx % 11 == 0:
            continue
        out.append(ch)
    return "".join(out)


def _structured_text_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No metrics recorded.")
        return
    final = history[-1]
    print("\n" + "=" * 60)
    print("STRUCTURED TEXT REPORT")
    print("=" * 60)
    print(f"repair_agreement={final.get('repair_agreement', 0.0):.3f} "
          f"corruption_agreement={final.get('corruption_agreement', 0.0):.3f} "
          f"repair_improvement={final.get('repair_improvement', 0.0):.3f} "
          f"parseable={bool(final.get('parseable', False))}")


def run_structured_text_simulation(
    total_steps: int = 24,
    chunk_size: int = 1,
    warmup_records: int = 4,
    num_workers: int = 1,
    use_dict: bool = False,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a small structured-text repair benchmark."""
    adapter = StructuredTextAdapter()
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    records = [
        {"kind": "bundle", "phase": "train", "level": 1, "count": 3, "value": 0.5},
        {"kind": "bundle", "phase": "train", "level": 2, "count": 5, "value": 0.75},
        {"kind": "bundle", "phase": "validation", "level": 3, "count": 7, "value": 1.0},
        {"kind": "bundle", "phase": "validation", "level": 4, "count": 9, "value": 1.25},
    ]

    for rec in records[:warmup_records]:
        layered.observe_text(adapter.to_text(rec), feedback_mode="target")

    history: List[Dict[str, Any]] = []
    print(
        f"Starting structured text simulation: records={len(records)} "
        f"warmup={warmup_records} workers={num_workers} dict={use_dict}"
    )

    for step in range(total_steps):
        target = records[step % len(records)]
        target_text = adapter.to_text(target)
        corrupted_text = _corrupt_structured_text(target_text)
        repaired_text = layered.repair_text(
            corrupted_text=corrupted_text,
            target_text=target_text,
            use_constraints=False,
            mode="target",
            update_policy=True,
        )
        corruption_stats = layered.evaluate_generated_text(corrupted_text, target_text)
        repair_stats = layered.evaluate_generated_text(repaired_text, target_text)
        parsed = adapter.from_text(repaired_text)
        repair_stats["repair_improvement"] = repair_stats["token_agreement"] - corruption_stats["token_agreement"]
        repair_stats["corruption_agreement"] = corruption_stats["token_agreement"]
        repair_stats["repair_agreement"] = repair_stats["token_agreement"]
        repair_stats["parseable"] = isinstance(parsed, dict)
        repair_stats["record_kind"] = target["kind"]
        layered.observe_text(target_text, feedback_mode="target")
        history.append(repair_stats)

        if step % max(1, chunk_size) == 0:
            print(
                f"[step {step:4d}] repair_agree={repair_stats['token_agreement']:.3f} "
                f"improve={repair_stats['repair_improvement']:.3f} "
                f"parseable={repair_stats['parseable']}"
            )
            print(f"  target:   {target_text}")
            print(f"  corrupt:  {corrupted_text}")
            print(f"  repaired: {repaired_text}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "structured_text_library")
    layered.save_bundle(base)
    if registry_path:
        registry = LibraryRegistry(registry_path)
        densities = [p.compression() for p in layered.l1.patterns]
        registry.upsert(
            name="structured_text_seed",
            path=base + ".l1.pkl",
            domain="structured_text",
            status="seed",
            bundle_kind="stacked",
            level_contract="l1-l5",
            obs_dims=[5, 10, 10, 32, 64],
            source="canonical-json",
            density_mean=float(sum(densities) / len(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=len(layered.l1.patterns),
            notes="auto-registered from structured text simulation",
        )
    print(f"Final structured text library saved: {base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    _structured_text_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Structured text simulation")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--chunk-size", type=int, default=1)
    p.add_argument("--warmup-records", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    p.add_argument("--registry", default=None, help="Optional registry path to auto-populate")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_structured_text_simulation(
        total_steps=args.steps,
        chunk_size=args.chunk_size,
        warmup_records=args.warmup_records,
        num_workers=args.workers,
        use_dict=args.dict,
        checkpoint_dir=args.checkpoint_dir,
        registry_path=args.registry,
    )
