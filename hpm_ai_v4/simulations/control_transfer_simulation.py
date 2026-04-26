"""Control-transfer benchmark for the learned l4/l5 policy stack."""
import argparse
import os
from statistics import mean
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.text_full_simulation import _ids_to_text, _text_metrics_snapshot
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary


def _mean_phase(history: List[Dict[str, Any]], phase: str) -> Dict[str, float]:
    snaps = [snap for snap in history if snap.get("phase") == phase]
    if not snaps:
        return {}
    keys = {
        "target_agreement",
        "plausibility",
        "compression_mi",
        "l2_compression_mi",
        "l3_compression_mi",
        "transfer_gain",
        "planned_agreement",
        "planned_plausibility",
    }
    summary: Dict[str, float] = {}
    for key in keys:
        vals = [float(s.get(key, 0.0)) for s in snaps if key in s]
        if vals:
            summary[key] = float(mean(vals))
    return summary


def _print_phase(label: str, metrics: Dict[str, float]) -> None:
    if not metrics:
        print(f"{label.upper()}: no metrics")
        return
    print("\n" + "=" * 60)
    print(f"{label.upper()} REPORT")
    print("=" * 60)
    print(
        f"target_agreement={metrics.get('target_agreement', 0.0):.3f} "
        f"plausibility={metrics.get('plausibility', 0.0):.3f} "
        f"L1_mi={metrics.get('compression_mi', 0.0):.3f} "
        f"L2_mi={metrics.get('l2_compression_mi', 0.0):.3f} "
        f"L3_mi={metrics.get('l3_compression_mi', 0.0):.3f} "
        f"transfer_gain={metrics.get('transfer_gain', 0.0):.3f}"
    )


def _validate_snapshot(
    layered: LayeredAgent,
    target_text: str,
    generated_text: str,
    planned_text: str,
    step: int,
    recent_chars: List[int],
    baseline_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    planned_stats = layered.evaluate_generated_text(planned_text, target_text) if planned_text else {
        "token_agreement": 0.0,
        "plausibility": 0.0,
    }
    target_stats = layered.evaluate_generated_text(generated_text, target_text)
    target_stats["planned_agreement"] = planned_stats["token_agreement"]
    target_stats["planned_plausibility"] = planned_stats["plausibility"]
    if baseline_stats:
        target_stats["baseline_agreement"] = baseline_stats.get("token_agreement", 0.0)
        target_stats["baseline_plausibility"] = baseline_stats.get("plausibility", 0.0)
        target_stats["transfer_gain"] = target_stats["token_agreement"] - baseline_stats.get("token_agreement", 0.0)
    else:
        target_stats["transfer_gain"] = 0.0
        target_stats["baseline_agreement"] = 0.0
        target_stats["baseline_plausibility"] = 0.0

    snap = _text_metrics_snapshot(
        layered=layered,
        recent_chars=recent_chars,
        step=step,
        generated_text=generated_text,
        target_text=target_text,
        target_stats=target_stats,
    )
    snap["phase"] = "validation"
    snap["planned_agreement"] = planned_stats["token_agreement"]
    snap["planned_plausibility"] = planned_stats["plausibility"]
    snap["baseline_agreement"] = target_stats["baseline_agreement"]
    snap["baseline_plausibility"] = target_stats["baseline_plausibility"]
    snap["transfer_gain"] = target_stats["transfer_gain"]
    snap["decoder_choice"] = layered._last_decoder_choice
    snap["policy_age"] = layered.decoder_policy._age
    snap["l4_compression_mi"] = layered.l4_metrics()["mi"]
    snap["l5_compression_mi"] = layered.l5_metrics()["mi"]
    return snap


def run_control_transfer_simulation(
    corpus_path: str,
    train_steps: int = 20_000,
    validation_steps: int = 4_000,
    log_every: int = 1_000,
    chunk_size: int = 120,
    prompt_size: int = 200,
    warmup_chars: int = 500,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Train the decoder-policy stack, reload it, and validate transfer on held-out text."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    train_agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    if library_path and os.path.exists(library_path + ".l1.pkl"):
        train_agent.load_bundle(library_path)
        print(f"Loaded library from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    needed = warmup_chars + train_steps + validation_steps + chunk_size + prompt_size
    raw_ids = [next(stream_iter) for _ in range(needed)]
    corpus_text = _ids_to_text(raw_ids)

    train_text = corpus_text[: warmup_chars + train_steps]
    validation_text = corpus_text[warmup_chars + train_steps : warmup_chars + train_steps + validation_steps + chunk_size]

    warmup_text = train_text[:warmup_chars]
    if warmup_text:
        train_agent.observe_text(warmup_text, feedback_mode="target")

    history: List[Dict[str, Any]] = []
    print(
        f"Starting control-transfer simulation: train_steps={train_steps} validation_steps={validation_steps} "
        f"chunk={chunk_size} warmup={warmup_chars} prompt={prompt_size} workers={num_workers} dict={use_dict}"
    )

    train_cursor = 0
    while train_cursor < train_steps:
        target_text = train_text[warmup_chars + train_cursor : warmup_chars + train_cursor + chunk_size]
        if not target_text:
            break
        seed_start = max(0, warmup_chars + train_cursor - prompt_size)
        seed_text = train_text[seed_start:warmup_chars + train_cursor]
        if use_dict:
            generated_text = train_agent.generate_constrained_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                allowed_words=set(train_agent.dictionary.words) if train_agent.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
            )
        else:
            generated_text = train_agent.generate_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
            )
        train_agent.observe_text(
            target_text,
            feedback_mode="hybrid",
            generated_text=generated_text,
            self_feedback_weight=0.02,
        )

        snap = _text_metrics_snapshot(
            layered=train_agent,
            recent_chars=raw_ids[max(0, warmup_chars + train_cursor - prompt_size): warmup_chars + train_cursor + len(target_text)],
            step=warmup_chars + train_cursor,
            generated_text=generated_text,
            target_text=target_text,
            target_stats=train_agent.evaluate_generated_text(generated_text, target_text),
        )
        snap["phase"] = "train"
        snap["l4_compression_mi"] = train_agent.l4_metrics()["mi"]
        snap["l5_compression_mi"] = train_agent.l5_metrics()["mi"]
        snap["decoder_choice"] = train_agent._last_decoder_choice
        history.append(snap)

        if train_cursor % log_every == 0:
            print(
                f"[train {train_cursor:6d}] gen_agree={snap['target_agreement']:.3f} "
                f"plaus={snap['plausibility']:.3f} L1 mi={snap['compression_mi']:.3f} "
                f"L2 mi={snap['l2_compression_mi']:.3f} L3 mi={snap['l3_compression_mi']:.3f} "
                f"decoder={snap['decoder_choice']}"
            )
        train_cursor += chunk_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_control_transfer_library")
    train_agent.save_bundle(final_base)

    loaded = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    loaded.load_bundle(final_base)
    baseline = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    validation_warmup = validation_text[:warmup_chars]
    if validation_warmup:
        loaded.observe_text(validation_warmup, feedback_mode="target")
        baseline.observe_text(validation_warmup, feedback_mode="target")

    val_cursor = 0
    while val_cursor < validation_steps:
        target_text = validation_text[warmup_chars + val_cursor:warmup_chars + val_cursor + chunk_size]
        if not target_text:
            break
        seed_start = max(0, warmup_chars + val_cursor - prompt_size)
        seed_text = validation_text[seed_start:warmup_chars + val_cursor]
        horizon = max(8, min(80, len(target_text) // 2))
        if use_dict:
            loaded_text = loaded.generate_constrained_text(
                steps=horizon,
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                allowed_words=set(loaded.dictionary.words) if loaded.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
                update_policy=False,
            )
            baseline_text = baseline.generate_constrained_text(
                steps=horizon,
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                allowed_words=set(baseline.dictionary.words) if baseline.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
                update_policy=False,
            )
        else:
            loaded_text = loaded.generate_text(
                steps=horizon,
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                update_policy=False,
            )
            baseline_text = baseline.generate_text(
                steps=horizon,
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                update_policy=False,
            )
        loaded_planned = loaded.plan_text_continuation(
            target_text=target_text,
            seed_text=seed_text[-200:] if seed_text else None,
            horizon=horizon,
            strategy="beam",
            lookback=240,
        )
        baseline_stats = baseline.evaluate_generated_text(baseline_text, target_text)
        snap = _validate_snapshot(
            layered=loaded,
            target_text=target_text,
            generated_text=loaded_text,
            planned_text=loaded_planned,
            step=warmup_chars + train_steps + val_cursor,
            recent_chars=raw_ids[max(0, warmup_chars + train_steps + val_cursor - prompt_size): warmup_chars + train_steps + val_cursor + len(target_text)],
            baseline_stats=baseline_stats,
        )
        history.append(snap)

        if val_cursor % log_every == 0:
            print(
                f"[val   {val_cursor:6d}] loaded_agree={snap['target_agreement']:.3f} "
                f"baseline={snap['baseline_agreement']:.3f} gain={snap['transfer_gain']:.3f} "
                f"plaus={snap['plausibility']:.3f} decoder={snap['decoder_choice']}"
            )
            print(f"  target:   {target_text[:120]!r}")
            print(f"  planned:  {loaded_planned[:120]!r}")
            print(f"  loaded:   {loaded_text[:120]!r}")
            print(f"  baseline: {baseline_text[:120]!r}")

        val_cursor += chunk_size

    train_agent.save_bundle(final_base)
    print(f"Final control-transfer library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    print("\n" + "=" * 60)
    print("CONTROL TRANSFER REPORT")
    print("=" * 60)
    _print_phase("train", _mean_phase(history, "train"))
    _print_phase("validation", _mean_phase(history, "validation"))
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Control-transfer benchmark for the stacked HPM text system")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--train-steps", type=int, default=20_000)
    p.add_argument("--validation-steps", type=int, default=4_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--prompt-size", type=int, default=200)
    p.add_argument("--warmup-chars", type=int, default=500)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_control_transfer_simulation(
        corpus_path=args.corpus,
        train_steps=args.train_steps,
        validation_steps=args.validation_steps,
        log_every=args.log_every,
        chunk_size=args.chunk_size,
        prompt_size=args.prompt_size,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
