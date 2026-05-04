"""Cartpole physics benchmark for v5 continuous control."""

from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..adapter.physics import CartpoleStateAdapter, RewardToGoalAdapter, RunningNormaliserAdapter, TDErrorAdapter
from ..adapter.trajectory_buffer import TrajectoryBufferAdapter
from ..agents import ScoringWeightAdaptationAgent
from ..core import PatternEngine, PatternManager
from ..pipeline import HPMPipeline
from ..polygraphs.action_policy import ActionPolygraphGenerator
from ..postprocessors.physics import CartpoleForecastPostprocessor


@dataclass(frozen=True, slots=True)
class CartpoleEnvConfig:
    name: str = "cartpole"
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02


DEFAULT_CARTPOLE_VARIANTS: dict[str, CartpoleEnvConfig] = {
    "cartpole": CartpoleEnvConfig(name="cartpole"),
    "cartpole_heavy": CartpoleEnvConfig(name="cartpole_heavy", mass_pole=0.5),
    "cartpole_light": CartpoleEnvConfig(name="cartpole_light", mass_pole=0.02),
    "cartpole_short": CartpoleEnvConfig(name="cartpole_short", length=0.25),
    "cartpole_long": CartpoleEnvConfig(name="cartpole_long", length=1.0),
}


class CartpoleEnv:
    """Lightweight cartpole simulation (Euler integration)."""

    def __init__(self, config: CartpoleEnvConfig | None = None) -> None:
        self.config = config or CartpoleEnvConfig()
        self.gravity = self.config.gravity
        self.mass_cart = self.config.mass_cart
        self.mass_pole = self.config.mass_pole
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = self.config.length  # half length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = self.config.force_mag
        self.tau = self.config.tau  # seconds between updates
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12 degrees approx 0.21 rad
        self.x_threshold = 2.4
        self.reset()

    def reset(self) -> dict[str, float]:
        # Low noise initial state
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._obs()

    def step(self, action: float) -> tuple[dict[str, float], float, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = action * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0
        return self._obs(), reward, done

    def _obs(self) -> dict[str, float]:
        x, x_dot, theta, theta_dot = self.state
        return {
            "position": float(x),
            "velocity": float(x_dot),
            "angle": float(theta),
            "angular_velocity": float(theta_dot),
        }


@dataclass(frozen=True, slots=True)
class CartpoleResult:
    result: str
    reason: str
    average_length: float
    episode_lengths: list[int]
    total_episodes: int
    evaluation_average: float
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CartpoleTransferState:
    env_name: str
    env_config: CartpoleEnvConfig
    engine: PatternEngine
    pattern_manager: PatternManager
    normaliser: RunningNormaliserAdapter
    postprocessor_state: dict[str, Any]
    swa: ScoringWeightAdaptationAgent


@dataclass(frozen=True, slots=True)
class TransferTargetResult:
    variant: str
    zero_shot_average: float
    fine_tune_average: float
    scratch_average: float
    fine_tune_passed: bool
    zero_shot_passed: bool
    forgetting_average: float
    forgetting_ratio: float
    forgetting_passed: bool
    fine_tune_episodes_to_threshold: int | None
    scratch_episodes_to_threshold: int | None


@dataclass(frozen=True, slots=True)
class CrossPhysicsTransferResult:
    source_average: float
    targets: list[TransferTargetResult]


class CartpoleBenchmark:
    """Benchmark for continuous control via HPM core."""

    def __init__(
        self,
        engine: PatternEngine | None = None,
        *,
        env_config: CartpoleEnvConfig | None = None,
        pattern_manager: PatternManager | None = None,
        use_pattern_manager: bool = True,
    ) -> None:
        from ..core.config import CoreConfig

        self.engine = engine or PatternEngine(config=CoreConfig(max_patterns=64, density_decay=0.01, utility_decay=0.005, max_sequences=32))
        self.env_config = env_config or CartpoleEnvConfig()
        self.env = CartpoleEnv(config=self.env_config)
        self.use_pattern_manager = use_pattern_manager
        self.pattern_manager = pattern_manager or PatternManager(
            promotion_threshold=0.5,
            min_support=2,
            archive_decay_rate=0.0,
        )
        self.swa = ScoringWeightAdaptationAgent(learning_rate=0.15, exploration_rate=0.2)

        # Pipeline setup
        self.state_adapter = CartpoleStateAdapter()
        self.normaliser = RunningNormaliserAdapter()
        self.td_error = TDErrorAdapter(error_alpha=0.1)
        self.reward_adapter = RewardToGoalAdapter(decay=0.95)
        self.trajectory_buffer = TrajectoryBufferAdapter(window=5)

        self.postprocessor = CartpoleForecastPostprocessor(
            action_index=2,
            confidence_threshold=0.6,
            max_blend_alpha=0.4,
        )

        self.pipeline = HPMPipeline(
            preprocessor=self.state_adapter,
            engine=self.engine,
            postprocessor=self.postprocessor,
            polygraph_generator=ActionPolygraphGenerator(),
            polygraph_every_n_steps=1,
            polygraph_min_patterns=0,
            polygraph_confidence_skip=0.99,
        )
        self.pipeline.register_preprocessor(self.normaliser)
        self.pipeline.register_preprocessor(self.reward_adapter)
        self.pipeline.register_preprocessor(self.td_error)
        self.pipeline.register_preprocessor(self.trajectory_buffer)

    def export_transfer_state(self) -> CartpoleTransferState:
        return CartpoleTransferState(
            env_name=self.env_config.name,
            env_config=self.env_config,
            engine=copy.deepcopy(self.engine),
            pattern_manager=copy.deepcopy(self.pattern_manager),
            normaliser=copy.deepcopy(self.normaliser),
            postprocessor_state=copy.deepcopy(self.postprocessor.export_state()),
            swa=copy.deepcopy(self.swa),
        )

    def import_transfer_state(self, state: CartpoleTransferState) -> None:
        self.engine = copy.deepcopy(state.engine)
        self.pattern_manager = copy.deepcopy(state.pattern_manager)
        self.normaliser = copy.deepcopy(state.normaliser)
        self.swa = copy.deepcopy(state.swa)
        self.pipeline.engine = self.engine
        self.postprocessor.import_state(copy.deepcopy(state.postprocessor_state))

    def save_transfer_state(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self.export_transfer_state(), handle)

    def load_transfer_state(self, path: str | Path) -> None:
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        self.import_transfer_state(state)

    @staticmethod
    def _episodes_to_threshold(lengths: list[int], threshold: int) -> int | None:
        for index, length in enumerate(lengths, start=1):
            if length >= threshold:
                return index
        return None

    def run(
        self,
        episodes: int = 50,
        max_steps: int = 1000,
        global_episode_start: int = 0,
        total_episodes: int = 100,
        *,
        evaluate: bool = False,
    ) -> CartpoleResult:
        episode_lengths = []
        epsilon_start = 0.3
        epsilon_min = 0.01
        restore_state = self.export_transfer_state() if evaluate else None
        original_learning_enabled = self.postprocessor.learning_enabled
        self.postprocessor.learning_enabled = not evaluate

        for ep in range(episodes):
            obs = self.env.reset()
            self.reward_adapter.reset()
            self.state_adapter.reset()
            if self.use_pattern_manager:
                self.pattern_manager.start_episode(self.engine, context={"task": self.env_config.name})

            # Decay epsilon based on global progress
            global_ep = global_episode_start + ep
            if evaluate:
                self.postprocessor.q_epsilon = 0.0
            else:
                self.postprocessor.q_epsilon = max(epsilon_min, epsilon_start * (1 - global_ep / total_episodes))

            # Reset engine history for new episode
            self.engine.history = []

            steps = 0
            done = False
            last_action = 0.0
            prev_forecast = None
            # Carry terminal reward=0 from previous episode so the Q-table
            # penalises the (state, action) that caused failure. Without this,
            # reward is always 1.0 during an episode and all Q-values saturate
            # at ~10 with no discrimination between correct and incorrect actions.
            carry: dict = {"reward": 0.0}

            while not done and steps < max_steps:
                # 1. Get adaptive weights from SWA
                swa_weights = self.swa.current_weights("cartpole")

                # Step the pipeline — reward in carry is from the previous step
                # (or 0.0 at episode start for terminal penalty from last episode)
                context = {
                    "last_action": last_action,
                    "prev_forecast": prev_forecast,
                    "action_index": 5,  # sin(theta) index in (a0,a1,a2,pos,vel,sin,cos,ang_vel,err)
                    **carry,
                }
                
                # Goal with learned adaptive weights and sequence execution enabled
                goal = {
                    **swa_weights,
                    "delta": swa_weights.get("delta", 1.0) * 5.0, # Amplify utility importance for physics
                    "sequence_execution": True,
                    "plan_horizon": 3,
                }
                
                result = self.pipeline.step(obs, goal=goal, context=context)

                # Capture pattern matched at this state — used for retroactive reward below
                last_match = self.engine.last_match

                # Action comes from CartpoleForecastPostprocessor — heuristic
                # baseline with confidence-gated engine blending.
                action = float(result.output) if result.output is not None else (
                    1.0 if obs.get("angle", 0.0) > 0 else -1.0
                )

                # Store normalized forecast for TD error in next step
                if result.action and result.action.forecast:
                    prev_forecast = result.action.forecast.value
                else:
                    prev_forecast = None

                obs, reward, done = self.env.step(action)

                # Retroactive pattern reinforcement: reward the pattern that was
                # active when this action was taken with the environment's feedback.
                # This ties pattern utility to actual survival, not just observation.
                if not evaluate and last_match is not None and last_match.pattern is not None:
                    last_match.pattern.reward(reward)

                # 2. Update SWA agent with performance feedback
                if not evaluate:
                    running_utility = self.reward_adapter._running_utility
                    self.swa.observe_reward(
                        self.env_config.name,
                        result.action.selected_pattern,
                        result.input.state,
                        running_utility,
                    )

                # Thread actual env reward into carry so Q-update next step
                # uses the real outcome, including reward=0 on terminal steps.
                carry = {**result.carry_context, "reward": reward}
                last_action = action
                steps += 1

            episode_lengths.append(steps)

            if self.use_pattern_manager and not evaluate:
                self.pattern_manager.end_episode(self.engine, context={"task": self.env_config.name})

        avg_len = sum(episode_lengths) / len(episode_lengths)
        # Evaluate on learned policy (last half) — early exploration inflates failure rate
        eval_window = episode_lengths[len(episode_lengths) // 2:]
        eval_avg = sum(eval_window) / len(eval_window)
        passed = eval_avg > 150

        self.postprocessor.learning_enabled = original_learning_enabled
        if restore_state is not None:
            self.import_transfer_state(restore_state)

        return CartpoleResult(
            result="success" if passed else "failure",
            reason=f"Eval avg {eval_avg:.1f} steps (last {len(eval_window)} eps), overall {avg_len:.1f}" + ("" if passed else " (required > 150)"),
            average_length=avg_len,
            episode_lengths=episode_lengths,
            total_episodes=episodes,
            evaluation_average=eval_avg,
            trace={"config": {"episodes": episodes, "max_steps": max_steps, "env": self.env_config.name, "evaluate": evaluate}},
        )


def run_cross_physics_transfer(
    *,
    source_episodes: int = 100,
    evaluation_episodes: int = 20,
    fine_tune_episodes: int = 50,
    max_steps: int = 1000,
    variants: tuple[str, ...] = ("cartpole_heavy", "cartpole_light", "cartpole_short", "cartpole_long"),
    success_threshold: int = 500,
) -> CrossPhysicsTransferResult:
    source = CartpoleBenchmark(env_config=DEFAULT_CARTPOLE_VARIANTS["cartpole"])
    source_result = source.run(episodes=source_episodes, max_steps=max_steps, total_episodes=source_episodes)
    source_state = source.export_transfer_state()

    targets: list[TransferTargetResult] = []
    for variant_name in variants:
        variant_config = DEFAULT_CARTPOLE_VARIANTS[variant_name]

        zero_shot = CartpoleBenchmark(env_config=variant_config)
        zero_shot.import_transfer_state(source_state)
        zero_shot_result = zero_shot.run(episodes=evaluation_episodes, max_steps=max_steps, evaluate=True)

        fine_tune = CartpoleBenchmark(env_config=variant_config)
        fine_tune.import_transfer_state(source_state)
        fine_tune_result = fine_tune.run(
            episodes=fine_tune_episodes,
            max_steps=max_steps,
            total_episodes=fine_tune_episodes,
        )
        forgetting = fine_tune.run(episodes=evaluation_episodes, max_steps=max_steps, evaluate=True)
        forgetting_ratio = 0.0 if source_result.evaluation_average <= 0 else forgetting.evaluation_average / source_result.evaluation_average

        scratch = CartpoleBenchmark(env_config=variant_config)
        scratch_result = scratch.run(
            episodes=fine_tune_episodes,
            max_steps=max_steps,
            total_episodes=fine_tune_episodes,
        )

        targets.append(
            TransferTargetResult(
                variant=variant_name,
                zero_shot_average=zero_shot_result.average_length,
                fine_tune_average=fine_tune_result.average_length,
                scratch_average=scratch_result.average_length,
                fine_tune_passed=(fine_tune._episodes_to_threshold(fine_tune_result.episode_lengths, success_threshold) or (fine_tune_episodes + 1)) <= 30,
                zero_shot_passed=zero_shot_result.average_length > 100.0,
                forgetting_average=forgetting.average_length,
                forgetting_ratio=forgetting_ratio,
                forgetting_passed=forgetting_ratio >= 0.8,
                fine_tune_episodes_to_threshold=fine_tune._episodes_to_threshold(fine_tune_result.episode_lengths, success_threshold),
                scratch_episodes_to_threshold=scratch._episodes_to_threshold(scratch_result.episode_lengths, success_threshold),
            )
        )

    return CrossPhysicsTransferResult(source_average=source_result.average_length, targets=targets)
