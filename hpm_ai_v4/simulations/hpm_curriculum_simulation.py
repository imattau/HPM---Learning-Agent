"""Self-curriculum benchmark for the HPM stack.

The simulation selects which environment family to practice next using the
reasoner's episodic polygraph. This keeps curriculum choice learned and soft:
high-resonance families are revisited more often, but underused families still
receive exploration pressure so the policy does not collapse.
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CurriculumAdapter, EpisodeBundleAdapter
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.simulations.hpm_environment_simulation import (
    ToyStateWorld,
    _choose_action,
    _episode_reward,
    _recent_context,
)
from hpm_ai_v4.tools.serializer import PatternSerializer


def _family_policy_name(family: int) -> str:
    return f"family_{int(family)}"


def _family_context(agent: HPMAgent, adapter: CurriculumAdapter, family: int, phase: str, obs: int) -> List[int]:
    context = list(agent.obs_buffer[-12:]) if agent.obs_buffer else []
    context.extend(
        adapter.to_observations(
            {"family": _family_policy_name(family), "phase": phase, "obs": obs},
            max_length=3,
        )
    )
    return context


def _choose_family(agent: HPMAgent, families: Sequence[int], context: Sequence[int]) -> int:
    control = agent.reasoner.control_context(context)
    prior = control.get("family_prior", {})
    scores: Dict[int, float] = {}
    for family in families:
        policy = _family_policy_name(family)
        score = float(prior.get(policy, 0.0))
        seen = sum(
            1
            for ep in agent.reasoner.memory
            if ep.policy == policy and not ep.is_summary
        )
        score += 0.04 / ((seen + 1) ** 0.5)
        score += 0.01 * float(control.get("community_strength", 0.0))
        scores[int(family)] = score
    if not any(scores.values()):
        return int(families[0])
    return max(scores.items(), key=lambda item: item[1])[0]


def _save_bundle(agent: HPMAgent, base_path: str, registry: Optional[LibraryRegistry] = None) -> None:
    PatternSerializer.save(agent.patterns, base_path + ".pkl")
    with open(base_path + ".reasoner.json", "w", encoding="utf-8") as f:
        json.dump(agent.reasoner.state_dict(), f)
    bundle_adapter = EpisodeBundleAdapter()
    descriptor = {"kind": "bundle", "phase": "save", "level": getattr(agent.development, "level", "unknown"), "count": agent.reasoner.memory_size}
    tokens = bundle_adapter.encode_bundle(descriptor)
    with open(base_path + ".bundle.json", "w", encoding="utf-8") as f:
        json.dump({"descriptor": descriptor, "tokens": tokens, "decoded": bundle_adapter.decode_bundle(tokens)}, f)
    if registry is not None:
        densities = [p.compression() for p in agent.patterns]
        registry.upsert(
            name=os.path.basename(base_path),
            path=base_path + ".pkl",
            domain="curriculum",
            status="seed",
            source="curriculum-policy",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(np.min(densities)) if densities else 0.0,
            density_max=float(np.max(densities)) if densities else 0.0,
            pattern_count=len(agent.patterns),
            notes="auto-registered from curriculum simulation",
        )


def _load_bundle(agent: HPMAgent, base_path: str) -> int:
    loaded = 0
    if os.path.exists(base_path + ".pkl"):
        agent.patterns = PatternSerializer.load(base_path + ".pkl")
        loaded += 1
    reasoner_path = base_path + ".reasoner.json"
    if os.path.exists(reasoner_path):
        with open(reasoner_path, "r", encoding="utf-8") as f:
            agent.reasoner.load_state_dict(json.load(f))
        loaded += 1
    bundle_path = base_path + ".bundle.json"
    if os.path.exists(bundle_path):
        bundle_adapter = EpisodeBundleAdapter()
        with open(bundle_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        bundle_adapter.decode_bundle(payload.get("tokens", []))
        loaded += 1
    return loaded


def _episode_snapshot(
    agent: HPMAgent,
    phase: str,
    step: int,
    family: int,
    reward: float,
    exact_match: int,
    baseline_match: Optional[int] = None,
) -> Dict[str, Any]:
    top3 = sorted(agent.patterns, key=lambda p: -p.weight)[:3]
    mi = float(mean([p.compression() for p in top3])) if top3 else 0.0
    control = agent.reasoner.control_context()
    snap = {
        "phase": phase,
        "step": step,
        "family": int(family),
        "reward": float(reward),
        "exact_match": int(exact_match),
        "compression_mi": mi,
        "memory_size": agent.reasoner.memory_size,
        "community_strength": float(control.get("community_strength", 0.0)),
        "retrieved_count": int(control.get("retrieved_count", 0)),
        "summary_count": int(control.get("summary_count", 0)),
        "dev_stage": getattr(agent.development, "level", "unknown"),
    }
    if baseline_match is not None:
        snap["baseline_exact_match"] = int(baseline_match)
        snap["transfer_gain"] = float(exact_match - baseline_match)
    else:
        snap["baseline_exact_match"] = 0
        snap["transfer_gain"] = 0.0
    return snap


def _mean_metric(history: List[Dict[str, Any]], phase: str, key: str) -> float:
    vals = [float(s.get(key, 0.0)) for s in history if s.get("phase") == phase and key in s]
    return float(mean(vals)) if vals else 0.0


def _report(history: List[Dict[str, Any]]) -> None:
    train_exact = _mean_metric(history, "train", "exact_match")
    val_exact = _mean_metric(history, "validation", "exact_match")
    val_gain = _mean_metric(history, "validation", "transfer_gain")
    family_counts: Dict[int, int] = {}
    for snap in history:
        if snap.get("phase") == "train":
            family_counts[int(snap.get("family", 0))] = family_counts.get(int(snap.get("family", 0)), 0) + 1
    print("\n" + "=" * 60)
    print("HPM CURRICULUM REPORT")
    print("=" * 60)
    print(f"train_exact_match={train_exact:.3f}")
    print(f"validation_exact_match={val_exact:.3f}")
    print(f"validation_transfer_gain={val_gain:.3f}")
    print(f"train_family_diversity={len([v for v in family_counts.values() if v > 0])}")


def run_hpm_curriculum_simulation(
    train_episodes: int = 240,
    validation_episodes: int = 80,
    warmup_episodes: int = 24,
    episode_length: int = 4,
    log_every: int = 40,
    train_families: Sequence[int] = (0, 1, 2),
    validation_families: Sequence[int] = (3,),
    num_workers: int = 1,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Train a curriculum policy and validate on held-out environment families."""
    agent = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)
    baseline = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)
    family_adapter = CurriculumAdapter([_family_policy_name(f) for f in sorted(set(train_families) | set(validation_families))])

    history: List[Dict[str, Any]] = []

    # Warm up on the training family set so the polygraph has episodes to query.
    for idx in range(warmup_episodes):
        family = int(train_families[idx % len(train_families)])
        world = ToyStateWorld(family=family)
        obs = world.reset()
        agent.perceive_and_learn(obs)
        context = _family_context(agent, family_adapter, family, "warmup", obs)
        action = _choose_action(agent, context)
        next_obs, reward, _, info = world.step(action)
        agent.perceive_and_learn(next_obs)
        agent.reasoner.record_episode(
            context,
            action=action,
            reward=reward,
            tag="curriculum",
            metadata={
                **_episode_reward(agent, reward, obs, action, context),
                "phase": "warmup",
                "family": family,
            },
            stage=getattr(agent.development, "level", "unknown"),
            policy=_family_policy_name(family),
        )

    for step in range(train_episodes):
        family = _choose_family(agent, train_families, agent.obs_buffer)
        world = ToyStateWorld(family=family)
        obs = world.reset()
        total_reward = 0.0
        correct_actions = 0
        context = _family_context(agent, family_adapter, family, "train", obs)
        for _ in range(episode_length):
            action = _choose_action(agent, context)
            next_obs, reward, done, info = world.step(action)
            agent.perceive_and_learn(next_obs)
            correct_actions += int(action == info["correct_action"])
            total_reward += reward
            context = _family_context(agent, family_adapter, family, "train", next_obs)
            if done:
                break

        exact_match = int(correct_actions == episode_length)
        agent.reasoner.record_episode(
            context,
            action=int(world.executed[-1] if world.executed else 0),
            reward=float(total_reward / max(1, episode_length)),
            tag="curriculum",
            metadata={
                "phase": "train",
                "family": family,
                "episode_length": episode_length,
                "correct_actions": correct_actions,
                "reward": float(total_reward / max(1, episode_length)),
            },
            stage=getattr(agent.development, "level", "unknown"),
            policy=_family_policy_name(family),
        )
        snap = _episode_snapshot(agent, "train", step, family, total_reward / max(1, episode_length), exact_match)
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[train {step:5d}] family={family} exact={snap['exact_match']} "
                f"reward={snap['reward']:.3f} mem={snap['memory_size']} stage={snap['dev_stage']}"
            )

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "hpm_curriculum_library")
    registry = LibraryRegistry(registry_path) if registry_path else None
    _save_bundle(agent, base, registry=registry)

    loaded = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)
    _load_bundle(loaded, base)
    baseline = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)

    for idx in range(warmup_episodes):
        family = int(validation_families[idx % len(validation_families)])
        world = ToyStateWorld(family=family)
        obs = world.reset()
        loaded.perceive_and_learn(obs)
        baseline.perceive_and_learn(obs)

    for step in range(validation_episodes):
        family = int(validation_families[step % len(validation_families)])
        loaded_world = ToyStateWorld(family=family)
        baseline_world = ToyStateWorld(family=family)

        obs_loaded = loaded_world.reset()
        obs_baseline = baseline_world.reset()
        loaded.perceive_and_learn(obs_loaded)
        baseline.perceive_and_learn(obs_baseline)

        loaded_context = _family_context(loaded, family_adapter, family, "validation", obs_loaded)
        baseline_context = _family_context(baseline, family_adapter, family, "validation", obs_baseline)

        loaded_reward = 0.0
        baseline_reward = 0.0
        loaded_correct = 0
        baseline_correct = 0

        for _ in range(episode_length):
            loaded_action = _choose_action(loaded, loaded_context)
            baseline_action = _choose_action(baseline, baseline_context)

            next_loaded_obs, reward_loaded, done_loaded, info_loaded = loaded_world.step(loaded_action)
            next_baseline_obs, reward_baseline, done_baseline, info_baseline = baseline_world.step(baseline_action)

            loaded.perceive_and_learn(next_loaded_obs)
            baseline.perceive_and_learn(next_baseline_obs)

            loaded_correct += int(loaded_action == info_loaded["correct_action"])
            baseline_correct += int(baseline_action == info_baseline["correct_action"])
            loaded_reward += reward_loaded
            baseline_reward += reward_baseline

            loaded_context = _family_context(loaded, family_adapter, family, "validation", next_loaded_obs)
            baseline_context = _family_context(baseline, family_adapter, family, "validation", next_baseline_obs)

            if done_loaded and done_baseline:
                break

        exact_match = int(loaded_correct == episode_length)
        baseline_match = int(baseline_correct == episode_length)
        loaded.reasoner.record_episode(
            loaded_context,
            action=int(loaded_world.executed[-1] if loaded_world.executed else 0),
            reward=float(loaded_reward / max(1, episode_length)),
            tag="curriculum",
            metadata={
                "phase": "validation",
                "family": family,
                "episode_length": episode_length,
                "correct_actions": loaded_correct,
                "reward": float(loaded_reward / max(1, episode_length)),
            },
            stage=getattr(loaded.development, "level", "unknown"),
            policy=_family_policy_name(family),
        )
        baseline.reasoner.record_episode(
            baseline_context,
            action=int(baseline_world.executed[-1] if baseline_world.executed else 0),
            reward=float(baseline_reward / max(1, episode_length)),
            tag="curriculum",
            metadata={
                "phase": "validation",
                "family": family,
                "episode_length": episode_length,
                "correct_actions": baseline_correct,
                "reward": float(baseline_reward / max(1, episode_length)),
            },
            stage=getattr(baseline.development, "level", "unknown"),
            policy=_family_policy_name(family),
        )

        snap = _episode_snapshot(
            loaded,
            "validation",
            step,
            family,
            loaded_reward / max(1, episode_length),
            exact_match,
            baseline_match=baseline_match,
        )
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[val   {step:5d}] family={family} exact={snap['exact_match']} "
                f"baseline={baseline_match} gain={snap['transfer_gain']:.3f} "
                f"mem={snap['memory_size']} stage={snap['dev_stage']}"
            )

    _save_bundle(loaded, base, registry=registry)
    _report(history)
    return history


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-curriculum benchmark for the HPM stack")
    p.add_argument("--train-episodes", type=int, default=240)
    p.add_argument("--validation-episodes", type=int, default=80)
    p.add_argument("--warmup-episodes", type=int, default=24)
    p.add_argument("--episode-length", type=int, default=4)
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--checkpoint-dir", default=".")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_hpm_curriculum_simulation(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        warmup_episodes=args.warmup_episodes,
        episode_length=args.episode_length,
        log_every=args.log_every,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )
