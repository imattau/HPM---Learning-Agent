"""Cartpole physics benchmark for v5 continuous control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..adapter import AdapterPacket, AdapterRegistry
from ..adapter.physics import CartpoleStateAdapter, RewardToGoalAdapter, RunningNormaliserAdapter, TDErrorAdapter
from ..adapter.recent_buffer import RecentBufferAdapter
from ..adapter.trajectory_buffer import TrajectoryBufferAdapter
from ..agents import ScoringWeightAdaptationAgent
from ..core import PatternEngine
from ..pipeline import HPMPipeline
from ..polygraphs.action_policy import ActionPolygraphGenerator
from ..postprocessors.physics import CartpoleForecastPostprocessor


class CartpoleEnv:
    """Lightweight cartpole simulation (Euler integration)."""

    def __init__(self) -> None:
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # half length
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between updates
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
    trace: dict[str, Any]


class CartpoleBenchmark:
    """Benchmark for continuous control via HPM core."""

    def __init__(self, engine: PatternEngine | None = None) -> None:
        from ..core.config import CoreConfig
        self.engine = engine or PatternEngine(config=CoreConfig(max_patterns=64, density_decay=0.01, utility_decay=0.005, max_sequences=32))
        self.env = CartpoleEnv()
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
        self.postprocessor.pipeline = self.pipeline  # give postprocessor access to view engine scores

    def run(self, episodes: int = 50, max_steps: int = 1000, global_episode_start: int = 0, total_episodes: int = 100) -> CartpoleResult:
        episode_lengths = []
        epsilon_start = 0.3
        epsilon_min = 0.01
        
        for ep in range(episodes):
            obs = self.env.reset()
            self.reward_adapter.reset()
            # self.normaliser.reset() # Keep normaliser stats across episodes
            self.state_adapter.reset()
            
            # Decay epsilon based on global progress
            global_ep = global_episode_start + ep
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
                if last_match is not None and last_match.pattern is not None:
                    last_match.pattern.reward(reward)
                
                # 2. Update SWA agent with performance feedback
                # Use the running utility from RewardToGoalAdapter for denser feedback
                running_utility = self.reward_adapter._running_utility
                self.swa.observe_reward(
                    "cartpole", 
                    result.action.selected_pattern, 
                    result.input.state, 
                    running_utility
                )
                
                # Thread actual env reward into carry so Q-update next step
                # uses the real outcome, including reward=0 on terminal steps.
                carry = {**result.carry_context, "reward": reward}
                last_action = action
                steps += 1
                
            episode_lengths.append(steps)
            
        avg_len = sum(episode_lengths) / len(episode_lengths)
        # Evaluate on learned policy (last half) — early exploration inflates failure rate
        eval_window = episode_lengths[len(episode_lengths) // 2:]
        eval_avg = sum(eval_window) / len(eval_window)
        passed = eval_avg > 150

        return CartpoleResult(
            result="success" if passed else "failure",
            reason=f"Eval avg {eval_avg:.1f} steps (last {len(eval_window)} eps), overall {avg_len:.1f}" + ("" if passed else " (required > 150)"),
            average_length=avg_len,
            episode_lengths=episode_lengths,
            total_episodes=episodes,
            trace={"config": {"episodes": episodes, "max_steps": max_steps}}
        )
