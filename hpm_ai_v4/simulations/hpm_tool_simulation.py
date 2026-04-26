"""Tool/action benchmark for the HPM stack.

This simulation exercises multi-step tool use with learned template selection
from the reasoner polygraph. The goal is to keep the HPM control loop intact:
episodes are stored, repeated tool families consolidate, and decoder/planning
choices are biased by the learned episodic ecology rather than a hard controller.
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
from hpm_ai_v4.io.adapters import EpisodeBundleAdapter, ToolActionAdapter
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


TOOL_NAMES = {
    0: "inspect",
    1: "shift",
    2: "flip",
    3: "commit",
}


@dataclass(frozen=True)
class ToolTemplate:
    name: str
    sequence: Tuple[int, ...]


TEMPLATES: List[ToolTemplate] = [
    ToolTemplate("inspect_shift_commit", (0, 1, 3)),
    ToolTemplate("inspect_flip_commit", (0, 2, 3)),
    ToolTemplate("inspect_shift_shift_commit", (0, 1, 1, 3)),
    ToolTemplate("inspect_flip_shift_commit", (0, 2, 1, 3)),
]


class ToolChainWorld:
    """Small environment where action sequences must match a hidden tool family."""

    def __init__(self, family: int = 0, obs_dim: int = 4, state_dim: int = 6):
        self.family = int(family) % len(TEMPLATES)
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.reset()

    @property
    def target_template(self) -> ToolTemplate:
        return TEMPLATES[self.family]

    def reset(self) -> int:
        self.state = self.family % self.state_dim
        self.step_idx = 0
        self.executed: List[int] = []
        return self._observation()

    def _observation(self) -> int:
        # Compact observable state that still lets the stack learn family-specific cues.
        return int((self.state + self.step_idx + self.family) % self.obs_dim)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        action = int(action) % self.obs_dim
        self.executed.append(action)

        if action == 0:  # inspect
            self.state = (self.state + self.family + 1) % self.state_dim
        elif action == 1:  # shift
            self.state = (self.state + 1) % self.state_dim
        elif action == 2:  # flip
            self.state = (self.state * 2 + 1) % self.state_dim
        elif action == 3:  # commit
            pass

        self.step_idx += 1
        done = action == 3 or self.step_idx >= max(len(t.sequence) for t in TEMPLATES)
        reward = 0.0
        if action == 3:
            reward = self._commit_reward()

        info = {
            "family": self.family,
            "target_template": self.target_template.name,
            "executed_template": self._executed_template_name(),
            "step_idx": self.step_idx,
            "state": self.state,
            "target_sequence": list(self.target_template.sequence),
            "executed_sequence": list(self.executed),
        }
        return self._observation(), reward, done, info

    def _commit_reward(self) -> float:
        target = list(self.target_template.sequence)
        executed = list(self.executed)
        exact = executed == target
        prefix_match = sum(1 for a, b in zip(executed, target) if a == b)
        prefix_score = prefix_match / max(1, len(target))
        state_bonus = 1.0 if self.state == (self.family + len(executed)) % self.state_dim else 0.0
        return 0.7 * float(exact) + 0.2 * prefix_score + 0.1 * state_bonus

    def _executed_template_name(self) -> str:
        executed = tuple(self.executed)
        for template in TEMPLATES:
            if executed == template.sequence:
                return template.name
        return "partial"


def _template_scores(agent: HPMAgent, context: Sequence[int]) -> Dict[str, float]:
    hits = agent.reasoner.retrieve_memory(context, top_k=8)
    control = agent.reasoner.control_context(context)
    family_prior = control.get("family_prior", {})
    scores: Dict[str, float] = {}

    for template in TEMPLATES:
        score = 0.0
        score += 0.25 * float(family_prior.get(template.name, 0.0))
        for ep in hits:
            if ep.policy == template.name:
                base = max(0.05, float(ep.reward))
                if ep.metadata:
                    base += 0.15 * float(ep.metadata.get("reward", 0.0))
                    base += 0.10 * float(ep.metadata.get("token_agreement", 0.0))
                score += base
            elif ep.community != "unknown" and ep.metadata and ep.metadata.get("template") == template.name:
                score += 0.25 * max(0.0, float(ep.reward))
        if not hits and template == TEMPLATES[0]:
            score += 0.05
        scores[template.name] = score

    if not any(scores.values()):
        scores[TEMPLATES[0].name] = 1.0
    return scores


def _choose_template(agent: HPMAgent, context: Sequence[int]) -> ToolTemplate:
    scores = _template_scores(agent, context)
    best_name = max(scores.items(), key=lambda item: item[1])[0]
    for template in TEMPLATES:
        if template.name == best_name:
            return template
    return TEMPLATES[0]


def _plan_template(agent: HPMAgent, template: ToolTemplate) -> List[int]:
    plan = agent.reasoner.plan_sequence(
        list(template.sequence),
        horizon=len(template.sequence),
        strategy="beam",
        require_valid_words=False,
        require_grammatical=False,
        target_weight=2.0,
    )
    return plan or list(template.sequence)


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
            domain="tool",
            status="seed",
            source="tool-action",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(np.min(densities)) if densities else 0.0,
            density_max=float(np.max(densities)) if densities else 0.0,
            pattern_count=len(agent.patterns),
            notes="auto-registered from tool simulation",
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


def _context_from_world(agent: HPMAgent, world: ToolChainWorld, obs: int, action_adapter: ToolActionAdapter) -> List[int]:
    context = list(agent.obs_buffer[-12:]) if agent.obs_buffer else []
    context.extend([world.family, world.state, world.step_idx, obs])
    last_action = world.executed[-1] if world.executed else 0
    context.extend(action_adapter.to_observations({"action": TOOL_NAMES.get(last_action, "inspect")}, max_length=1))
    return context


def _record_tool_episode(
    agent: HPMAgent,
    context: Sequence[int],
    action: int,
    reward: float,
    world: ToolChainWorld,
    action_adapter: ToolActionAdapter,
    phase: str,
    template_name: str,
    planned: Sequence[int],
) -> None:
    metadata = {
        "phase": phase,
        "family": world.family,
        "template": template_name,
        "planned_sequence": list(planned),
        "target_sequence": list(world.target_template.sequence),
        "executed_sequence": list(world.executed),
        "reward": reward,
        "state": world.state,
        "action_name": action_adapter.act(action),
        "token_agreement": 1.0 if list(world.executed) == list(world.target_template.sequence) else 0.0,
        "plausibility": 1.0 if action in TOOL_NAMES else 0.0,
    }
    agent.reasoner.record_episode(
        context,
        action=action,
        reward=reward,
        tag="tool",
        metadata=metadata,
        stage=getattr(agent.development, "level", "unknown"),
        policy=template_name,
    )


def _episode_snapshot(
    agent: HPMAgent,
    phase: str,
    step: int,
    reward: float,
    exact_match: int,
    planned: Sequence[int],
    baseline_match: Optional[int] = None,
) -> Dict[str, Any]:
    top3 = sorted(agent.patterns, key=lambda p: -p.weight)[:3]
    mi = float(mean([p.compression() for p in top3])) if top3 else 0.0
    control = agent.reasoner.control_context()
    snap = {
        "phase": phase,
        "step": step,
        "reward": float(reward),
        "exact_match": int(exact_match),
        "planned_length": len(list(planned)),
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
    print("\n" + "=" * 60)
    print("HPM TOOL REPORT")
    print("=" * 60)
    print(f"train_exact_match={train_exact:.3f}")
    print(f"validation_exact_match={val_exact:.3f}")
    print(f"validation_transfer_gain={val_gain:.3f}")


def run_hpm_tool_simulation(
    train_episodes: int = 200,
    validation_episodes: int = 60,
    warmup_episodes: int = 20,
    log_every: int = 40,
    train_families: Sequence[int] = (0, 1, 2),
    validation_families: Sequence[int] = (3,),
    num_workers: int = 1,
    checkpoint_dir: str = ".",
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Train on tool families, then validate on held-out tool families."""
    agent = HPMAgent(obs_dim=4, num_initial_patterns=4, num_workers=num_workers)
    baseline = HPMAgent(obs_dim=4, num_initial_patterns=4, num_workers=num_workers)
    action_adapter = ToolActionAdapter()

    history: List[Dict[str, Any]] = []

    # Warmup across training families to seed the polygraph.
    for idx in range(warmup_episodes):
        family = int(train_families[idx % len(train_families)])
        world = ToolChainWorld(family=family)
        obs = world.reset()
        agent.perceive_and_learn(obs)
        context = _context_from_world(agent, world, obs, action_adapter)
        template = _choose_template(agent, context)
        planned = _plan_template(agent, template)
        for action in planned:
            next_obs, reward, done, info = world.step(action)
            agent.perceive_and_learn(next_obs)
            _record_tool_episode(agent, context, action, reward, world, action_adapter, "train", template.name, planned)
            if done:
                break
            context = _context_from_world(agent, world, next_obs, action_adapter)
        snap = _episode_snapshot(agent, "train", idx, reward, int(planned == list(world.target_template.sequence)), planned)
        history.append(snap)

    for step in range(train_episodes):
        family = int(train_families[step % len(train_families)])
        world = ToolChainWorld(family=family)
        obs = world.reset()
        agent.perceive_and_learn(obs)
        context = _context_from_world(agent, world, obs, action_adapter)
        template = _choose_template(agent, context)
        planned = _plan_template(agent, template)
        reward = 0.0
        exact_match = 0
        for action in planned:
            next_obs, reward, done, info = world.step(action)
            agent.perceive_and_learn(next_obs)
            _record_tool_episode(agent, context, action, reward, world, action_adapter, "train", template.name, planned)
            if done:
                exact_match = int(list(world.executed) == list(world.target_template.sequence))
                break
            context = _context_from_world(agent, world, next_obs, action_adapter)

        snap = _episode_snapshot(agent, "train", step, reward, exact_match, planned)
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[train {step:5d}] exact={snap['exact_match']} reward={snap['reward']:.3f} "
                f"stage={snap['dev_stage']} mem={snap['memory_size']} template={template.name}"
            )

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "hpm_tool_library")
    registry = LibraryRegistry(registry_path) if registry_path else None
    _save_bundle(agent, base, registry=registry)

    loaded = HPMAgent(obs_dim=4, num_initial_patterns=4, num_workers=num_workers)
    _load_bundle(loaded, base)
    if not baseline.patterns:
        baseline = HPMAgent(obs_dim=4, num_initial_patterns=4, num_workers=num_workers)

    # Validation uses the same family set but new episodes / reload state.
    for idx in range(warmup_episodes):
        family = int(validation_families[idx % len(validation_families)])
        world = ToolChainWorld(family=family)
        obs = world.reset()
        loaded.perceive_and_learn(obs)
        baseline.perceive_and_learn(obs)

    for step in range(validation_episodes):
        family = int(validation_families[step % len(validation_families)])
        loaded_world = ToolChainWorld(family=family)
        baseline_world = ToolChainWorld(family=family)

        obs_loaded = loaded_world.reset()
        obs_baseline = baseline_world.reset()
        loaded.perceive_and_learn(obs_loaded)
        baseline.perceive_and_learn(obs_baseline)

        loaded_ctx = _context_from_world(loaded, loaded_world, obs_loaded, action_adapter)
        baseline_ctx = _context_from_world(baseline, baseline_world, obs_baseline, action_adapter)

        loaded_template = _choose_template(loaded, loaded_ctx)
        baseline_template = _choose_template(baseline, baseline_ctx)
        loaded_plan = _plan_template(loaded, loaded_template)
        baseline_plan = _plan_template(baseline, baseline_template)

        loaded_reward = 0.0
        baseline_reward = 0.0
        loaded_match = 0
        baseline_match = 0

        for action in loaded_plan:
            next_obs, loaded_reward, done, info = loaded_world.step(action)
            loaded.perceive_and_learn(next_obs)
            _record_tool_episode(loaded, loaded_ctx, action, loaded_reward, loaded_world, action_adapter, "validation", loaded_template.name, loaded_plan)
            if done:
                loaded_match = int(list(loaded_world.executed) == list(loaded_world.target_template.sequence))
                break
            loaded_ctx = _context_from_world(loaded, loaded_world, next_obs, action_adapter)

        for action in baseline_plan:
            next_obs, baseline_reward, done, info = baseline_world.step(action)
            baseline.perceive_and_learn(next_obs)
            _record_tool_episode(baseline, baseline_ctx, action, baseline_reward, baseline_world, action_adapter, "validation", baseline_template.name, baseline_plan)
            if done:
                baseline_match = int(list(baseline_world.executed) == list(baseline_world.target_template.sequence))
                break
            baseline_ctx = _context_from_world(baseline, baseline_world, next_obs, action_adapter)

        snap = _episode_snapshot(
            loaded,
            "validation",
            step,
            loaded_reward,
            loaded_match,
            loaded_plan,
            baseline_match=baseline_match,
        )
        history.append(snap)
        if step % log_every == 0:
            print(
                f"[val   {step:5d}] exact={snap['exact_match']} baseline={baseline_match} "
                f"gain={snap['transfer_gain']:.3f} mem={snap['memory_size']} template={loaded_template.name}"
            )

    _save_bundle(loaded, base)
    _report(history)
    return history


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tool/action benchmark for the HPM stack")
    p.add_argument("--train-episodes", type=int, default=200)
    p.add_argument("--validation-episodes", type=int, default=60)
    p.add_argument("--warmup-episodes", type=int, default=20)
    p.add_argument("--log-every", type=int, default=40)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--checkpoint-dir", default=".")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_hpm_tool_simulation(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        warmup_episodes=args.warmup_episodes,
        log_every=args.log_every,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
    )
