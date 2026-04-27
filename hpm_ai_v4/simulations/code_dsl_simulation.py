"""Code/DSL simulation for tiny stack-machine programs."""
import argparse
import os
from typing import Any, Dict, List, Optional

from hpm_ai_v4.io.adapters import CodeDSLAdapter
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


def _corrupt_program(program: str) -> str:
    out: List[str] = []
    for idx, ch in enumerate(program):
        if ch == " " and idx % 3 == 0:
            continue
        if ch == "\n" and idx % 5 == 0:
            continue
        if ch.isalpha() and idx % 11 == 0:
            out.append(ch.lower())
            continue
        out.append(ch)
    return "".join(out)


def _code_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No metrics recorded.")
        return
    final = history[-1]
    print("\n" + "=" * 60)
    print("CODE/DSL REPORT")
    print("=" * 60)
    print(f"repair_agreement={final.get('repair_agreement', 0.0):.3f} "
          f"corruption_agreement={final.get('corruption_agreement', 0.0):.3f} "
          f"repair_improvement={final.get('repair_improvement', 0.0):.3f} "
          f"parseable={bool(final.get('parseable', False))} "
          f"execution_match={bool(final.get('execution_match', False))}")


def run_code_dsl_simulation(
    total_steps: int = 16,
    warmup_programs: int = 2,
    num_workers: int = 1,
    use_dict: bool = False,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a small code/DSL repair benchmark over stack-machine programs."""
    adapter = CodeDSLAdapter()
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    text_signals = TextSignalExtractor()

    programs = [
        "push 2\npush 3\nadd\npush 4\nmul\nreturn",
        "push 10\npush 7\nsub\npush 6\nadd\nreturn",
        "push 8\ndup\nmul\nreturn",
        "push 9\npush 3\ndiv\npush 2\nsub\nreturn",
    ]

    for program in programs[:warmup_programs]:
        layered.observe_code_dsl(adapter.to_text(program), feedback_mode="target")

    history: List[Dict[str, Any]] = []
    print(
        f"Starting code/DSL simulation: programs={len(programs)} "
        f"warmup={warmup_programs} workers={num_workers} dict={use_dict}"
    )

    for step in range(total_steps):
        target_program = programs[step % len(programs)]
        target_text = adapter.to_text(target_program)
        corrupted_text = _corrupt_program(target_text)
        repaired_text = layered.repair_text(
            corrupted_text=corrupted_text,
            target_text=target_text,
            use_constraints=False,
            mode="target",
            update_policy=True,
        )

        corruption_stats = layered.evaluate_generated_text(corrupted_text, target_text)
        repair_stats = layered.evaluate_generated_text(repaired_text, target_text)
        repair_stats["repair_improvement"] = repair_stats["token_agreement"] - corruption_stats["token_agreement"]
        repair_stats["corruption_agreement"] = corruption_stats["token_agreement"]
        repair_stats["repair_agreement"] = repair_stats["token_agreement"]
        repair_stats["parseable"] = bool(adapter.from_text(repaired_text))
        target_value = adapter.execute(target_text)
        repaired_value = adapter.execute(repaired_text)
        repair_stats["target_value"] = target_value
        repair_stats["repaired_value"] = repaired_value
        repair_stats["execution_match"] = repaired_value == target_value and repaired_value is not None
        signal_pack = text_signals.analyze(
            repaired_text,
            context_texts=[corrupted_text, target_text],
            target_text=target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        repair_stats.update(signal_pack.to_dict())
        repair_stats["text_signal_score"] = signal_pack.combined_score()
        layered.observe_code_dsl(
            target_text,
            generated_program=repaired_text,
            feedback_mode="hybrid",
            self_feedback_weight=0.02,
            feedback_signal=signal_pack.to_dict(),
        )
        history.append(repair_stats)

        if step % 4 == 0:
            print(
                f"[step {step:4d}] repair_agree={repair_stats['repair_agreement']:.3f} "
                f"improve={repair_stats['repair_improvement']:.3f} "
                f"parseable={repair_stats['parseable']} exec={repair_stats['execution_match']}"
            )
            print(f"  target:   {target_text!r}")
            print(f"  corrupt:  {corrupted_text!r}")
            print(f"  repaired: {repaired_text!r}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "code_dsl_library")
    layered.save_bundle(base)
    if registry_path:
        registry = LibraryRegistry(registry_path)
        densities = [p.compression() for p in layered.l1.patterns]
        registry.upsert(
            name="code_dsl_seed",
            path=base + ".l1.pkl",
            domain="code",
            status="seed",
            bundle_kind="stacked",
            level_contract="l1-l5",
            obs_dims=[5, 10, 10, 32, 64],
            source="stack-machine-dsl",
            density_mean=float(sum(densities) / len(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=len(layered.l1.patterns),
            notes="auto-registered from code/DSL simulation",
        )
    print(f"Final code/DSL library saved: {base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    _code_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Code/DSL simulation")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--warmup-programs", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    p.add_argument("--registry", default=None, help="Optional registry path to auto-populate")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_code_dsl_simulation(
        total_steps=args.steps,
        warmup_programs=args.warmup_programs,
        num_workers=args.workers,
        use_dict=args.dict,
        checkpoint_dir=args.checkpoint_dir,
        registry_path=args.registry,
    )
