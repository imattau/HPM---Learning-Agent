"""Environment-state benchmark for the HPM stack.

This simulation is intentionally narrow: a small discrete environment with
state/action/outcome episodes recorded through the existing reasoner polygraph.
The point is to ground the stack in non-text interaction without introducing a
separate controller or memory engine.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import EnvironmentStateAdapter, EpisodeBundleAdapter
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass
class ToyStateWorld:
    """Small discrete environment with a latent transition family."""

    family: int = 0
    obs_dim: int = 2
    state_dim: int = 6
    start_state: int = 0

    def __post_init__(self) -> None:
        self.state = self.start_state % self.state_dim
        self.step_count = 0

    def reset(self, start_state: Optional[int] = None) -> int:
        if start_state is not None:
            self.start_state = int(start_state)
        self.state = self.start_state % self.state_dim
        self.step_count = 0
        self.executed: List[int] = []
        return self.observe()

    def observe(self) -> int:
        return int(self.state % self.obs_dim)

    def _correct_action(self, state: Optional[int] = None) -> int:
        s = self.state if state is None else int(state)
        parity = s % 2
        if self.family % 2 == 0:
            return parity
        return 1 - parity

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        action = int(action) % self.obs_dim
        self.executed.append(action)
        correct = self._correct_action()
        reward = 1.0 if action == correct else 0.0
        delta = 1 + (1 if action == correct else 2) + self.family
        self.state = (self.state + delta) % self.state_dim
        self.step_count += 1
        obs = self.observe()
        info = {
            "state": self.state,
            "correct_action": correct,
            "family": self.family,
        }
        return obs, reward, False, info


def _recent_context(
    agent: HPMAgent,
    env: ToyStateWorld,
    obs: int,
    state_adapter: EnvironmentStateAdapter,
    action: Optional[int] = None,
) -> List[int]:
    context = list(agent.obs_buffer[-16:]) if agent.obs_buffer else []
    structured = {
        "state": env.state,
        "family": env.family,
        "obs": obs,
        "action": action if action is not None else 0,
    }
    context.extend(state_adapter.to_observations(structured, max_length=8))
    return context


def _choose_action(agent: HPMAgent, context: Sequence[int]) -> int:
    model_action = agent.act()
    hits = agent.reasoner.retrieve_memory(context, top_k=6)
    scores = {0: 0.0, 1: 0.0}
    scores[int(model_action) % 2] += 0.35
    for ep in hits:
        weight = max(0.05, float(ep.reward))
        if ep.metadata:
            weight += 0.15 * float(ep.metadata.get("reward", 0.0))
            weight += 0.10 * float(ep.metadata.get("plausibility", 0.0))
        scores[int(ep.action) % 2] += weight
    return max(scores.items(), key=lambda item: item[1])[0]


def _episode_reward(agent: HPMAgent, reward: float, obs: int, action: int, context: Sequence[int]) -> Dict[str, Any]:
    control = agent.reasoner.control_context(context)
    return {
        "reward": reward,
        "obs": obs,
        "action": action,
        "community_strength": control.get("community_strength", 0.0),
        "retrieved_count": control.get("retrieved_count", 0),
        "summary_count": control.get("summary_count", 0),
    }


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
            domain="environment",
            status="seed",
            bundle_kind="flat",
            level_contract="agent",
            obs_dims=[agent.obs_dim],
            source=f"family:{getattr(agent, 'family', 'unknown')}",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(np.min(densities)) if densities else 0.0,
            density_max=float(np.max(densities)) if densities else 0.0,
            pattern_count=len(agent.patterns),
            notes="auto-registered from environment simulation",
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


def _snapshot(
    agent: HPMAgent,
    phase: str,
    step: int,
    action_correct: int,
    reward: float,
    baseline_correct: Optional[int] = None,
    baseline_reward: Optional[float] = None,
) -> Dict[str, Any]:
    top3 = sorted(agent.patterns, key=lambda p: -p.weight)[:3]
    mi = float(np.mean([p.compression() for p in top3])) if top3 else 0.0
    control = agent.reasoner.control_context()
    snap = {
        "phase": phase,
        "step": step,
        "action_accuracy": float(action_correct),
        "reward": float(reward),
        "compression_mi": mi,
        "pop_size": len(agent.patterns),
        "memory_size": agent.reasoner.memory_size,
        "dev_stage": getattr(agent.development, "level", "unknown"),
        "community_strength": float(control.get("community_strength", 0.0)),
        "retrieved_count": int(control.get("retrieved_count", 0)),
        "summary_count": int(control.get("summary_count", 0)),
    }
    if baseline_correct is not None:
        snap["baseline_action_accuracy"] = float(baseline_correct)
        snap["baseline_reward"] = float(baseline_reward if baseline_reward is not None else 0.0)
        snap["transfer_gain"] = float(action_correct) - float(baseline_correct)
    else:
        snap["baseline_action_accuracy"] = 0.0
        snap["baseline_reward"] = 0.0
        snap["transfer_gain"] = 0.0
    return snap


def _mean_metric(history: List[Dict[str, Any]], phase: str, key: str) -> float:
    vals = [float(s.get(key, 0.0)) for s in history if s.get("phase") == phase and key in s]
    return float(mean(vals)) if vals else 0.0


def _report(history: List[Dict[str, Any]]) -> None:
    train_acc = _mean_metric(history, "train", "action_accuracy")
    val_acc = _mean_metric(history, "validation", "action_accuracy")
    val_gain = _mean_metric(history, "validation", "transfer_gain")
    print("\n" + "=" * 60)
    print("HPM ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"train_action_accuracy={train_acc:.3f}")
    print(f"validation_action_accuracy={val_acc:.3f}")
    print(f"validation_transfer_gain={val_gain:.3f}")


def run_hpm_environment_simulation(
    train_steps: int = 2_000,
    validation_steps: int = 500,
    warmup_steps: int = 120,
    log_every: int = 200,
    state_dim: int = 6,
    train_family: int = 0,
    validation_family: int = 1,
    num_workers: int = 1,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Train on one environment family and validate on a held-out family."""
    train_agent = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)
    state_adapter = EnvironmentStateAdapter(obs_dim=8)
    train_world = ToyStateWorld(family=train_family, state_dim=state_dim)
    validation_world = ToyStateWorld(family=validation_family, state_dim=state_dim)

    history: List[Dict[str, Any]] = []
    obs = train_world.reset()
    for _ in range(warmup_steps):
        train_agent.perceive_and_learn(obs)
        obs, _, _, _ = train_world.step(obs)

    train_correct = 0
    train_reward = 0.0
    for step in range(train_steps):
        train_agent.perceive_and_learn(obs)
        context = _recent_context(train_agent, train_world, obs, state_adapter)
        action = _choose_action(train_agent, context)
        next_obs, reward, _, info = train_world.step(action)
        correct = int(action == info["correct_action"])
        train_correct += correct
        train_reward += reward
        metadata = _episode_reward(train_agent, reward, obs, action, context)
        metadata.update(info)
        metadata["phase"] = "train"
        train_agent.reasoner.record_episode(
            context,
            action=action,
            reward=reward,
            tag="environment",
            metadata=metadata,
            stage=getattr(train_agent.development, "level", "unknown"),
            policy="state-policy",
        )
        snap = _snapshot(train_agent, "train", step, correct, reward)
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[train {step:5d}] action_acc={snap['action_accuracy']:.3f} reward={snap['reward']:.3f} "
                f"stage={snap['dev_stage']} memory={snap['memory_size']}"
            )
        obs = next_obs

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "hpm_environment_library")
    registry = LibraryRegistry(registry_path) if registry_path else None
    _save_bundle(train_agent, base, registry=registry)

    loaded = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)
    _load_bundle(loaded, base)
    baseline = HPMAgent(obs_dim=2, num_initial_patterns=4, num_workers=num_workers)

    val_world_loaded = ToyStateWorld(family=validation_family, state_dim=state_dim)
    val_world_baseline = ToyStateWorld(family=validation_family, state_dim=state_dim)
    val_obs = val_world_loaded.reset(start_state=validation_family % state_dim)
    baseline_obs = val_world_baseline.reset(start_state=validation_family % state_dim)
    for _ in range(warmup_steps):
        loaded.perceive_and_learn(val_obs)
        baseline.perceive_and_learn(baseline_obs)
        val_obs, _, _, _ = val_world_loaded.step(val_obs)
        baseline_obs, _, _, _ = val_world_baseline.step(baseline_obs)

    val_correct = 0
    val_reward = 0.0
    baseline_correct = 0
    baseline_reward = 0.0
    for step in range(validation_steps):
        loaded.perceive_and_learn(val_obs)
        baseline.perceive_and_learn(baseline_obs)

        loaded_ctx = _recent_context(loaded, val_world_loaded, val_obs, state_adapter)
        baseline_ctx = _recent_context(baseline, val_world_baseline, baseline_obs, state_adapter)

        loaded_action = _choose_action(loaded, loaded_ctx)
        baseline_action = _choose_action(baseline, baseline_ctx)

        next_loaded_obs, loaded_r, _, loaded_info = val_world_loaded.step(loaded_action)
        next_baseline_obs, baseline_r, _, baseline_info = val_world_baseline.step(baseline_action)

        loaded_correct = int(loaded_action == loaded_info["correct_action"])
        baseline_correct_step = int(baseline_action == baseline_info["correct_action"])
        val_correct += loaded_correct
        val_reward += loaded_r
        baseline_correct += baseline_correct_step
        baseline_reward += baseline_r

        loaded_meta = _episode_reward(loaded, loaded_r, val_obs, loaded_action, loaded_ctx)
        loaded_meta.update(loaded_info)
        loaded_meta["phase"] = "validation"
        loaded.reasoner.record_episode(
            loaded_ctx,
            action=loaded_action,
            reward=loaded_r,
            tag="environment",
            metadata=loaded_meta,
            stage=getattr(loaded.development, "level", "unknown"),
            policy="state-policy",
        )
        baseline_meta = _episode_reward(baseline, baseline_r, val_obs, baseline_action, baseline_ctx)
        baseline_meta.update(baseline_info)
        baseline_meta["phase"] = "validation"
        baseline.reasoner.record_episode(
            baseline_ctx,
            action=baseline_action,
            reward=baseline_r,
            tag="environment",
            metadata=baseline_meta,
            stage=getattr(baseline.development, "level", "unknown"),
            policy="state-policy",
        )

        snap = _snapshot(
            loaded,
            "validation",
            step,
            loaded_correct,
            loaded_r,
            baseline_correct=baseline_correct_step,
            baseline_reward=baseline_r,
        )
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[val   {step:5d}] action_acc={snap['action_accuracy']:.3f} baseline={baseline_correct_step:.3f} "
                f"gain={snap['transfer_gain']:.3f} memory={snap['memory_size']}"
            )
        val_obs = next_loaded_obs
        _ = next_baseline_obs  # keep the baseline trajectory moving independently
    _save_bundle(loaded, base, registry=registry)
    _report(history)
    return history


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Environment-state benchmark for the HPM stack")
    p.add_argument("--train-steps", type=int, default=2_000)
    p.add_argument("--validation-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=120)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--state-dim", type=int, default=6)
    p.add_argument("--train-family", type=int, default=0)
    p.add_argument("--validation-family", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--checkpoint-dir", default=".")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_hpm_environment_simulation(
        train_steps=args.train_steps,
        validation_steps=args.validation_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        state_dim=args.state_dim,
        train_family=args.train_family,
        validation_family=args.validation_family,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )
