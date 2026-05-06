"""Regime Shift Adaptation (RSA) Benchmark for HPM v5."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .cartpole import CartpoleEnv, CartpoleEnvConfig, CartpoleStateAdapter, TDErrorAdapter, TrajectoryBufferAdapter
from ..adapter import AdapterPacket
from ..adapter.changepoint import ChangepointAdapter
from ..adapter.physics import RewardToGoalAdapter, RunningNormaliserAdapter
from ..adapter.validation_only import ValidationOnlyAdapter
from ..agents import ScoringWeightAdaptationAgent, SelfAdaptiveAgent
from ..core import PatternEngine, PatternManager
from ..core.config import CoreConfig
from ..pipeline import HPMPipeline
from ..polygraphs.action_policy import ActionPolygraphGenerator


@dataclass(frozen=True, slots=True)
class RSAResult:
    episodes: list[int]
    scores: list[int]
    detection_points: list[int]
    regime_shifts: list[int]
    status: str


class RSABenchmark:
    """Benchmark for Cartpole with sudden regime shifts."""

    def __init__(
        self,
        engine: PatternEngine | None = None,
        pattern_manager: PatternManager | None = None,
    ) -> None:
        self.engine = engine or PatternEngine(config=CoreConfig(
            max_patterns=128, 
            density_decay=0.01, 
            utility_decay=0.005,
            max_sequences=64,
            history_limit=10
        ))
        self.pattern_manager = pattern_manager or PatternManager(
            promotion_threshold=0.5,
            min_support=2,
            archive_decay_rate=0.01
        )
        
        # Adapters
        self.normaliser = RunningNormaliserAdapter()
        self.state_adapter = CartpoleStateAdapter()
        self.td_error = TDErrorAdapter(error_alpha=0.1)
        self.reward_adapter = RewardToGoalAdapter(decay=0.95)
        self.trajectory_buffer = TrajectoryBufferAdapter(window=5)
        
        # Changepoint: Monitoring Polygraph Agreement
        self.changepoint = ChangepointAdapter(window_size=32, threshold=1.5, cooldown=20)
        
        # SWA Agent
        self.swa = ScoringWeightAdaptationAgent(
            learning_rate=0.2,
            exploration_rate=0.15,
        )
        
        # Pipeline
        self.pipeline = HPMPipeline(
            preprocessor=self.state_adapter,
            engine=self.engine,
            postprocessor=ValidationOnlyAdapter(), 
            polygraph_generator=ActionPolygraphGenerator(),
        )
        self.pipeline.register_preprocessor(self.normaliser)
        self.pipeline.register_preprocessor(self.td_error)
        self.pipeline.register_preprocessor(self.reward_adapter)
        self.pipeline.register_preprocessor(self.trajectory_buffer)
        
        # Self-Adaptive Agent Wrapper
        self.agent = SelfAdaptiveAgent(
            name="rsa_agent",
            core=self.engine,
            pipeline=self.pipeline,
            changepoint=self.changepoint,
            swa=self.swa
        )

    def run(
        self,
        phases: list[dict[str, Any]],
        max_steps: int = 1000,
    ) -> RSAResult:
        episode_lengths = []
        detection_points = []
        regime_shifts = []
        
        global_ep = 0
        
        for phase in phases:
            name = phase["name"]
            count = phase["count"]
            config = phase["config"]
            
            print(f"\nPhase: {name} ({count} episodes, mass={config.mass_cart})")
            regime_shifts.append(global_ep)
            
            env = CartpoleEnv(config=config)
            
            for ep in range(count):
                obs = env.reset()
                total_reward = 0
                env_name = f"cartpole_{config.mass_cart}"
                
                for step in range(max_steps):
                    # Use the consolidated Agent step
                    res = self.agent.step_adaptive(obs, env_name=env_name)
                    
                    # Action Selection (HPM + Epsilon)
                    if self.agent.swa.rng.random() < self.agent.adaptation_epsilon:
                        action = self.agent.swa.rng.randint(0, 1)
                    else:
                        action = int(res.action.value[0]) if res.action.value else 0
                    
                    # Environment Step
                    next_obs, reward, done = env.step(float(action))
                    total_reward += 1
                    
                    # Feedback to SWA layer
                    self.agent.observe_reward(env_name, res, float(reward))
                    
                    # Check for detection
                    if res.carry_context.get("regime_changed"):
                        print(f"\n[!] Regime Shift Detected at Episode {global_ep}, Step {step}")
                        detection_points.append(global_ep)
                    
                    obs = next_obs
                    if done:
                        break
                
                self.pattern_manager.end_episode(self.engine)
                episode_lengths.append(total_reward)
                global_ep += 1
                print(".", end="", flush=True)
                
        return RSAResult(
            episodes=list(range(global_ep)),
            scores=episode_lengths,
            detection_points=detection_points,
            regime_shifts=regime_shifts,
            status="complete"
        )
