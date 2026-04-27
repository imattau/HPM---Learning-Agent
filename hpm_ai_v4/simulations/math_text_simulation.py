"""Mixed math/text simulation with SymPy-backed symbolic feedback."""
import argparse
import os
from typing import Any, Dict, List, Optional

from hpm_ai_v4.io.adapters import MathTextAdapter, SympyMathAdapter
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


def _corrupt_math_text(text: str) -> str:
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch.isspace() and idx % 7 == 0:
            continue
        if ch in "=+*/^" and idx % 5 == 0:
            continue
        if ch.isalpha() and idx % 13 == 0:
            out.append(ch.lower())
            continue
        out.append(ch)
    return "".join(out)


def _first_math_span(adapter: MathTextAdapter, text: str) -> str:
    spans = adapter.extract_math_spans(text)
    if spans:
        return spans[0]
    return text


def _math_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No metrics recorded.")
        return
    final = history[-1]
    print("\n" + "=" * 60)
    print("MATH/TEXT REPORT")
    print("=" * 60)
    print(
        f"repair_agreement={final.get('repair_agreement', 0.0):.3f} "
        f"equivalent={bool(final.get('equivalent', False))} "
        f"parseable={bool(final.get('parseable', False))} "
        f"symbolic_score={final.get('symbolic_score', 0.0):.3f} "
        f"mixed_domain_score={final.get('mixed_domain_score', 0.0):.3f}"
    )


def run_math_text_simulation(
    total_steps: int = 12,
    warmup_cases: int = 2,
    num_workers: int = 1,
    use_dict: bool = True,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a mixed math/text repair benchmark with symbolic feedback."""
    adapter = MathTextAdapter()
    sympy_adapter = SympyMathAdapter()
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    text_signals = TextSignalExtractor()

    targets = [
        "The area is A = pi * r^2.",
        "If x + 2 = 5 then x = 3.",
        "The force is F = m * a.",
        "Energy is E = m * c^2.",
    ]

    for target in targets[:warmup_cases]:
        layered.observe_text(target, feedback_mode="target")

    history: List[Dict[str, Any]] = []
    print(
        f"Starting math/text simulation: cases={len(targets)} "
        f"warmup={warmup_cases} workers={num_workers} dict={use_dict}"
    )

    for step in range(total_steps):
        target_text = targets[step % len(targets)]
        corrupted_text = _corrupt_math_text(target_text)
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
        repair_stats["corrupted_text"] = corrupted_text

        math_signal = text_signals.analyze(
            repaired_text,
            context_texts=[corrupted_text, target_text],
            target_text=target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        repair_stats.update(math_signal.to_dict())
        repair_stats["text_signal_score"] = math_signal.combined_score()

        target_span = _first_math_span(adapter, target_text)
        repaired_span = _first_math_span(adapter, repaired_text)
        symbolic_feedback = sympy_adapter.feedback(
            repaired_span,
            target_text=target_span,
            substitutions={"pi": 3.141592653589793, "r": 2, "x": 3, "m": 2, "a": 4, "c": 3},
        )
        repair_stats.update(symbolic_feedback)
        repair_stats["symbolic_feedback_score"] = float(symbolic_feedback.get("symbolic_score", 0.0))
        repair_stats["mixed_domain_score"] = float(math_signal.mixed_domain_score)
        repair_stats["math_spans"] = adapter.extract_math_spans(repaired_text)

        layered.observe_text(
            target_text,
            feedback_mode="hybrid",
            generated_text=repaired_text,
            self_feedback_weight=0.02,
            feedback_signal={
                **math_signal.to_dict(),
                **symbolic_feedback,
                "symbolic_target_span": target_span,
                "symbolic_generated_span": repaired_span,
                "kind": "math_text",
                "task_family": "math_text",
            },
        )

        repair_stats["parseable"] = bool(symbolic_feedback.get("parseable", False))
        repair_stats["equivalent"] = bool(symbolic_feedback.get("equivalent", False))
        history.append(repair_stats)

        if step % 4 == 0:
            print(
                f"[step {step:4d}] repair_agree={repair_stats['repair_agreement']:.3f} "
                f"improve={repair_stats['repair_improvement']:.3f} "
                f"sym={repair_stats['symbolic_score']:.3f} "
                f"mixed={repair_stats['mixed_domain_score']:.3f} "
                f"parseable={repair_stats['parseable']} eq={repair_stats['equivalent']}"
            )
            print(f"  target:   {target_text!r}")
            print(f"  corrupt:  {corrupted_text!r}")
            print(f"  repaired: {repaired_text!r}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "math_text_library")
    layered.save_bundle(base)
    if registry_path:
        registry = LibraryRegistry(registry_path)
        densities = [p.compression() for p in layered.l1.patterns]
        registry.upsert(
            name="math_text_seed",
            path=base + ".l1.pkl",
            domain="math",
            status="seed",
            bundle_kind="stacked",
            level_contract="l1-l5",
            obs_dims=[5, 10, 10, 32, 64],
            source="mixed-math-text",
            density_mean=float(sum(densities) / len(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=len(layered.l1.patterns),
            notes="auto-registered from math/text simulation",
        )
    print(f"Final math/text library saved: {base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    _math_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Mixed math/text simulation")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup-cases", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    p.add_argument("--registry", default=None, help="Optional registry path to auto-populate")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_math_text_simulation(
        total_steps=args.steps,
        warmup_cases=args.warmup_cases,
        num_workers=args.workers,
        use_dict=args.dict,
        checkpoint_dir=args.checkpoint_dir,
        registry_path=args.registry,
    )
