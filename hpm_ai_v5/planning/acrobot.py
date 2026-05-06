"""Acrobot physics benchmark for v5 continuous control."""

from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..adapter.physics import AcrobotStateAdapter, RewardToGoalAdapter, RunningNormaliserAdapter, TDErrorAdapter
from ..adapter.trajectory_buffer import TrajectoryBufferAdapter
from ..agents import ScoringWeightAdaptationAgent
from ..core import PatternEngine, PatternManager
from ..pipeline import HPMPipeline
from ..polygraphs.physics import AcrobotPolygraphGenerator
from ..postprocessors.physics import AcrobotForecastPostprocessor


@dataclass(frozen=True, slots=True)
class AcrobotEnvConfig:
    name: str = "acrobot"
    dt: float = 0.2
    link_length_1: float = 1.0
    link_length_2: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * math.pi
    max_vel_2: float = 9 * math.pi


class AcrobotEnv:
    """Lightweight Acrobot simulation (Runge-Kutta integration)."""

    def __init__(self, config: AcrobotEnvConfig | None = None) -> None:
        self.config = config or AcrobotEnvConfig()
        self.reset()

    def reset(self) -> dict[str, float]:
        # Initial state: hanging down with small noise
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._obs()

    def _dsdt(self, s_augmented: np.ndarray) -> np.ndarray:
        m1 = self.config.link_mass_1
        m2 = self.config.link_mass_2
        l1 = self.config.link_length_1
        lc1 = self.config.link_com_pos_1
        lc2 = self.config.link_com_pos_2
        I1 = self.config.link_moi
        I2 = self.config.link_moi
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * math.cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * math.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * math.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * math.cos(theta1 - math.pi / 2.0)
            + phi2
        )
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * math.sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])

    def step(self, action: float) -> tuple[dict[str, float], float, bool]:
        # action is torque in {-1, 0, 1}
        torque = float(np.clip(action, -1.0, 1.0))
        
        # Runge-Kutta 4 integration
        s_augmented = np.append(self.state, torque)
        k1 = self._dsdt(s_augmented)
        k2 = self._dsdt(s_augmented + self.config.dt / 2.0 * k1)
        k3 = self._dsdt(s_augmented + self.config.dt / 2.0 * k2)
        k4 = self._dsdt(s_augmented + self.config.dt * k3)
        
        self.state = self.state + self.config.dt / 6.0 * (k1[:-1] + 2 * k2[:-1] + 2 * k3[:-1] + k4[:-1])
        
        # Clip velocities
        self.state[2] = np.clip(self.state[2], -self.config.max_vel_1, self.config.max_vel_1)
        self.state[3] = np.clip(self.state[3], -self.config.max_vel_2, self.config.max_vel_2)
        
        # Goal: tip height > 1.0
        # Height: -cos(theta1) - cos(theta1 + theta2)
        # Initially both links down: theta1=0, theta2=0 -> height = -1 - 1 = -2
        # Goal: θ1 near π, θ2 near 0 -> height = -(-1) - (-1) = 2
        theta1, theta2, dtheta1, dtheta2 = self.state
        height = -math.cos(theta1) - math.cos(theta1 + theta2)
        done = bool(height >= 1.0)
        
        # Reward: -1 per step until goal
        reward = 0.0 if done else -1.0
        return self._obs(), reward, done

    def _obs(self) -> dict[str, float]:
        theta1, theta2, dtheta1, dtheta2 = self.state
        return {
            "theta1": float(theta1),
            "theta2": float(theta2),
            "theta1_dot": float(dtheta1),
            "theta2_dot": float(dtheta2),
        }


@dataclass(frozen=True, slots=True)
class AcrobotResult:
    result: str
    reason: str
    average_length: float
    episode_lengths: list[int]
    total_episodes: int
    evaluation_average: float
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AcrobotTransferState:
    env_name: str
    engine: PatternEngine
    pattern_manager: PatternManager
    normaliser: RunningNormaliserAdapter
    postprocessor_state: dict[str, Any]
    swa: ScoringWeightAdaptationAgent


class AcrobotBenchmark:
    """Benchmark for Acrobot swing-up via HPM core."""

    def __init__(
        self,
        engine: PatternEngine | None = None,
        *,
        env_config: AcrobotEnvConfig | None = None,
        pattern_manager: PatternManager | None = None,
        use_pattern_manager: bool = True,
    ) -> None:
        from ..core.config import CoreConfig

        self.engine = engine or PatternEngine(config=CoreConfig(max_patterns=128, density_decay=0.01, utility_decay=0.005, max_sequences=64))
        self.env_config = env_config or AcrobotEnvConfig()
        self.env = AcrobotEnv(config=self.env_config)
        self.use_pattern_manager = use_pattern_manager
        self.pattern_manager = pattern_manager or PatternManager(
            promotion_threshold=0.5,
            min_support=2,
            archive_decay_rate=0.0,
        )
        self.swa = ScoringWeightAdaptationAgent(learning_rate=0.1, exploration_rate=0.2)

        # Pipeline setup
        self.state_adapter = AcrobotStateAdapter()
        self.normaliser = RunningNormaliserAdapter()
        self.td_error = TDErrorAdapter(error_alpha=0.1)
        self.reward_adapter = RewardToGoalAdapter(decay=0.99)
        self.trajectory_buffer = TrajectoryBufferAdapter(window=5)

        self.postprocessor = AcrobotForecastPostprocessor(
            action_index=2, # index in action history buffer
            confidence_threshold=0.6,
            max_blend_alpha=0.4,
        )

        self.pipeline = HPMPipeline(
            preprocessor=self.state_adapter,
            engine=self.engine,
            postprocessor=self.postprocessor,
            polygraph_generator=AcrobotPolygraphGenerator(),
            polygraph_every_n_steps=1,
            polygraph_min_patterns=0,
            polygraph_confidence_skip=0.99,
        )
        self.pipeline.register_preprocessor(self.normaliser)
        self.pipeline.register_preprocessor(self.reward_adapter)
        self.pipeline.register_preprocessor(self.td_error)
        self.pipeline.register_preprocessor(self.trajectory_buffer)

    def export_transfer_state(self) -> AcrobotTransferState:
        return AcrobotTransferState(
            env_name=self.env_config.name,
            engine=copy.deepcopy(self.engine),
            pattern_manager=copy.deepcopy(self.pattern_manager),
            normaliser=copy.deepcopy(self.normaliser),
            postprocessor_state=copy.deepcopy(self.postprocessor.export_state()),
            swa=copy.deepcopy(self.swa),
        )

    def import_transfer_state(self, state: Any) -> None:
        # Support importing from CartpoleTransferState too if fields match
        self.engine = copy.deepcopy(state.engine)
        self.pattern_manager = copy.deepcopy(state.pattern_manager)
        self.normaliser = copy.deepcopy(state.normaliser)
        self.swa = copy.deepcopy(state.swa)
        self.pipeline.engine = self.engine
        
        # Try to import postprocessor state if it's compatible
        if hasattr(state, "postprocessor_state"):
             self.postprocessor.import_state(copy.deepcopy(state.postprocessor_state))

    def run(
        self,
        episodes: int = 200,
        max_steps: int = 500,
        global_episode_start: int = 0,
        total_episodes: int = 200,
        *,
        evaluate: bool = False,
    ) -> AcrobotResult:
        episode_lengths = []
        epsilon_start = 0.2 # Reduced from 0.3
        epsilon_min = 0.05
        restore_state = self.export_transfer_state() if evaluate else None
        original_learning_enabled = self.postprocessor.learning_enabled
        self.postprocessor.learning_enabled = not evaluate

        for ep in range(episodes):
            print(".", end="", flush=True)
            if (ep + 1) % 50 == 0:
                 print(f" {ep + 1}/{episodes}")
            obs = self.env.reset()
            self.reward_adapter.reset()
            self.state_adapter.reset()
            if self.use_pattern_manager:
                self.pattern_manager.start_episode(self.engine, context={"task": self.env_config.name})

            global_ep = global_episode_start + ep
            if evaluate:
                self.postprocessor.q_epsilon = 0.0
            else:
                self.postprocessor.q_epsilon = max(epsilon_min, epsilon_start * (1 - global_ep / total_episodes))

            self.engine.history = []

            steps = 0
            done = False
            last_action = 0.0
            prev_forecast = None
            carry: dict = {"reward": -1.0}

            while not done and steps < max_steps:
                swa_weights = self.swa.current_weights("acrobot")

                context = {
                    "last_action": last_action,
                    "prev_forecast": prev_forecast,
                    **carry,
                }
                
                goal = {
                    **swa_weights,
                    "delta": swa_weights.get("delta", 1.0) * 10.0, 
                    "sequence_execution": True,
                    "sequence_atomic_threshold": 0.1, # Eagerly use sequences
                    "plan_horizon": 4,
                }
                
                result = self.pipeline.step(obs, goal=goal, context=context)
                last_match = self.engine.last_match

                if not evaluate and result.action:
                    # Use shaped reward for dense reinforcement
                    shaped_reward = float(result.carry_context.get("carry_shaped_reward", 0.0))
                    
                    # Reward selected pattern
                    if last_match and last_match.pattern:
                        last_match.pattern.reward(max(0.0, shaped_reward))
                    
                    # Reward selected sequence if it was used
                    if result.action.selected_sequence:
                        result.action.selected_sequence.reward(max(0.0, shaped_reward))
                        if shaped_reward > 0.5:
                             print(f"  [Sequence] Rewarded {result.action.selected_sequence.pattern_names} with {shaped_reward:.2f}")

                action = float(result.output) if result.output is not None else 0.0

                if result.action and result.action.forecast:
                    prev_forecast = result.action.forecast.value
                else:
                    prev_forecast = None

                obs, reward, done = self.env.step(action)

                if done:
                    print(f"  [Acrobot] Episode {ep} reached goal in {steps} steps!")

                if not evaluate:
                    running_utility = self.reward_adapter._running_utility
                    self.swa.observe_reward(
                        self.env_config.name,
                        result.action.selected_pattern,
                        result.input.state,
                        running_utility,
                    )

                carry = {**result.carry_context, "reward": reward}
                last_action = action
                steps += 1

            episode_lengths.append(steps)

            if self.use_pattern_manager and not evaluate:
                self.pattern_manager.end_episode(self.engine, context={"task": self.env_config.name})

        avg_len = sum(episode_lengths) / len(episode_lengths)
        eval_window = episode_lengths[max(0, len(episode_lengths)-20):]
        eval_avg = sum(eval_window) / len(eval_window)
        # Success if we hit the goal occasionally (Acrobot is hard)
        passed = any(l < max_steps for l in eval_window)

        self.postprocessor.learning_enabled = original_learning_enabled
        if restore_state is not None:
            self.import_transfer_state(restore_state)

        return AcrobotResult(
            result="success" if passed else "failure",
            reason=f"Eval avg {eval_avg:.1f} steps, goal reached in {sum(1 for l in eval_window if l < max_steps)}/20 eps",
            average_length=avg_len,
            episode_lengths=episode_lengths,
            total_episodes=episodes,
            evaluation_average=eval_avg,
            trace={"config": {"episodes": episodes, "max_steps": max_steps, "env": self.env_config.name, "evaluate": evaluate}},
        )
