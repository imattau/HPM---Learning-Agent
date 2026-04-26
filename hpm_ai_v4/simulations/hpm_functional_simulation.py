"""Functional end-to-end benchmark for the stacked HPM system."""
import argparse
import os
from statistics import mean
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.control_transfer_simulation import run_control_transfer_simulation
from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.text_full_simulation import _ids_to_text, _text_metrics_snapshot
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary


def _mean_metric(history: List[Dict[str, Any]], phase: str, key: str) -> float:
    vals = [float(s.get(key, 0.0)) for s in history if s.get("phase") == phase and key in s]
    return float(mean(vals)) if vals else 0.0


def _report(history: List[Dict[str, Any]]) -> None:
    train_agree = _mean_metric(history, "train", "target_agreement")
    val_agree = _mean_metric(history, "validation", "target_agreement")
    transfer_gain = _mean_metric(history, "validation", "transfer_gain")
    print("\n" + "=" * 60)
    print("HPM FUNCTIONAL REPORT")
    print("=" * 60)
    print(f"train_target_agreement={train_agree:.3f}")
    print(f"validation_target_agreement={val_agree:.3f}")
    print(f"transfer_gain={transfer_gain:.3f}")


def run_hpm_functional_simulation(
    corpus_path: str,
    train_steps: int = 2_000,
    validation_steps: int = 400,
    log_every: int = 200,
    chunk_size: int = 80,
    prompt_size: int = 120,
    warmup_chars: int = 200,
    num_workers: int = 1,
    use_dict: bool = True,
    checkpoint_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Train the stack, save/reload it, and validate on held-out text."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None

    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    needed = warmup_chars + train_steps + validation_steps + chunk_size + prompt_size
    raw_ids = [next(stream_iter) for _ in range(needed)]
    corpus_text = _ids_to_text(raw_ids)
    train_text = corpus_text[: warmup_chars + train_steps]
    validation_text = corpus_text[warmup_chars + train_steps : warmup_chars + train_steps + validation_steps + chunk_size]

    if train_text[:warmup_chars]:
        layered.observe_text(train_text[:warmup_chars], feedback_mode="target")
    for ch in train_text[warmup_chars:]:
        layered.perceive(94 if ch == "\n" else ord(ch) - 32)

    history: List[Dict[str, Any]] = []
    train_snapshot = _text_metrics_snapshot(
        layered=layered,
        recent_chars=raw_ids[: len(train_text)],
        step=len(train_text),
        generated_text="",
        target_text="",
        target_stats={"token_agreement": 0.0, "plausibility": 0.0, "target_chars": len(train_text), "self_chars": 0},
    )
    train_snapshot["phase"] = "train"
    train_snapshot["transfer_gain"] = 0.0
    train_snapshot["l4_compression_mi"] = layered.l4_metrics()["mi"]
    train_snapshot["l5_compression_mi"] = layered.l5_metrics()["mi"]
    train_snapshot["decoder_choice"] = layered._last_decoder_choice
    history.append(train_snapshot)

    cursor = 0
    while cursor < train_steps:
        target_text = train_text[warmup_chars + cursor : warmup_chars + cursor + chunk_size]
        if not target_text:
            break
        seed_start = max(0, warmup_chars + cursor - prompt_size)
        seed_text = train_text[seed_start:warmup_chars + cursor]
        generated_text = layered.generate_constrained_text(
            steps=max(8, min(80, len(target_text) // 2)),
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
            allowed_words=set(layered.dictionary.words) if layered.dictionary else None,
            strict_dictionary=bool(layered.dictionary),
            strict_grammar=bool(layered.grammar),
        ) if use_dict else layered.generate_text(
            steps=max(8, min(80, len(target_text) // 2)),
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
        )
        layered.observe_text(target_text, feedback_mode="hybrid", generated_text=generated_text, self_feedback_weight=0.02)
        snap = _text_metrics_snapshot(
            layered=layered,
            recent_chars=raw_ids[max(0, warmup_chars + cursor - prompt_size): warmup_chars + cursor + len(target_text)],
            step=warmup_chars + cursor,
            generated_text=generated_text,
            target_text=target_text,
            target_stats=layered.evaluate_generated_text(generated_text, target_text),
        )
        snap["phase"] = "train"
        snap["transfer_gain"] = 0.0
        history.append(snap)
        snap["decoder_choice"] = layered._last_decoder_choice
        snap["l4_compression_mi"] = layered.l4_metrics()["mi"]
        snap["l5_compression_mi"] = layered.l5_metrics()["mi"]
        if cursor % log_every == 0:
            print(
                f"[train {cursor:5d}] agree={snap['target_agreement']:.3f} plaus={snap['plausibility']:.3f} "
                f"L1={snap['compression_mi']:.3f} L2={snap['l2_compression_mi']:.3f} L3={snap['l3_compression_mi']:.3f}"
            )
        cursor += chunk_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "hpm_functional_library")
    layered.save_bundle(base)

    loaded = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    loaded.load_bundle(base)
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
        generated_text = loaded.generate_constrained_text(
            steps=horizon,
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
            allowed_words=set(loaded.dictionary.words) if loaded.dictionary else None,
            strict_dictionary=bool(loaded.dictionary),
            strict_grammar=bool(loaded.grammar),
            update_policy=False,
        ) if use_dict else loaded.generate_text(
            steps=horizon,
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
            update_policy=False,
        )
        baseline_text = baseline.generate_constrained_text(
            steps=horizon,
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
            allowed_words=set(baseline.dictionary.words) if baseline.dictionary else None,
            strict_dictionary=bool(baseline.dictionary),
            strict_grammar=bool(baseline.grammar),
            update_policy=False,
        ) if use_dict else baseline.generate_text(
            steps=horizon,
            seed_text=seed_text[-200:] if seed_text else None,
            target_text=target_text,
            mode="target",
            include_seed=False,
            update_policy=False,
        )
        baseline_stats = baseline.evaluate_generated_text(baseline_text, target_text)
        snap = _text_metrics_snapshot(
            layered=loaded,
            recent_chars=raw_ids[max(0, warmup_chars + train_steps + val_cursor - prompt_size): warmup_chars + train_steps + val_cursor + len(target_text)],
            step=warmup_chars + train_steps + val_cursor,
            generated_text=generated_text,
            target_text=target_text,
            target_stats={
                **loaded.evaluate_generated_text(generated_text, target_text),
                "baseline_agreement": baseline_stats.get("token_agreement", 0.0),
                "baseline_plausibility": baseline_stats.get("plausibility", 0.0),
                "transfer_gain": loaded.evaluate_generated_text(generated_text, target_text)["token_agreement"] - baseline_stats.get("token_agreement", 0.0),
            },
        )
        snap["phase"] = "validation"
        snap["transfer_gain"] = snap["target_agreement"] - baseline_stats.get("token_agreement", 0.0)
        snap["decoder_choice"] = loaded._last_decoder_choice
        snap["policy_age"] = loaded.decoder_policy._age
        history.append(snap)
        if val_cursor % log_every == 0:
            print(
                f"[val   {val_cursor:5d}] agree={snap['target_agreement']:.3f} baseline={baseline_stats.get('token_agreement', 0.0):.3f} "
                f"gain={snap['transfer_gain']:.3f}"
            )
        val_cursor += chunk_size

    loaded.save_bundle(base)
    _report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Functional end-to-end benchmark for the stacked HPM system")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--train-steps", type=int, default=2_000)
    p.add_argument("--validation-steps", type=int, default=400)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--chunk-size", type=int, default=80)
    p.add_argument("--prompt-size", type=int, default=120)
    p.add_argument("--warmup-chars", type=int, default=200)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_hpm_functional_simulation(
        corpus_path=args.corpus,
        train_steps=args.train_steps,
        validation_steps=args.validation_steps,
        log_every=args.log_every,
        chunk_size=args.chunk_size,
        prompt_size=args.prompt_size,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        checkpoint_dir=args.checkpoint_dir,
    )
