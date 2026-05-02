"""Multi-Episode Lifecycle Benchmark (MELB).

Tests that full memory lifecycle (working memory → promotion → context-sensitive
seeding) produces genuine cross-episode transfer: patterns learned in episode 1
accelerate learning in episode 3 when structural context matches, but NOT when
the context is different.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapter import AdapterPacket, AdapterRegistry, PatternStoreSizeAdapter
from ..adapter.feature_adapters import (
    AutocorrelationAdapter,
    EntropyAdapter,
    NormalisationAdapter,
    PrefixBufferAdapter,
)
from ..adapter.reward import RewardAdapter
from ..core import PatternEngine
from ..core.pattern_manager import PatternManager, build_context_signature
from ..core.state import State as _State


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MELBEpisodeResult:
    episode: int
    name: str
    stream_family: str
    patterns_seeded: int
    patterns_promoted: int
    archive_size_after: int
    prediction_accuracy: float
    final_store_size: int
    context_signature: str


@dataclass(frozen=True, slots=True)
class MELBResult:
    result: str  # "success" or "failure"
    reason: str
    episodes: tuple[MELBEpisodeResult, ...]
    transfer_accuracy_gain: float  # ep3 accuracy - ep1 accuracy (should be >= 0)
    context_isolation_verified: bool  # ep3 seeded > 0 AND ep2 seeded == 0 from period-2 archive
    trace: dict[str, Any]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class MultiEpisodeLifecycleBenchmark:
    """Run four structured episodes through a shared PatternManager and check
    that cross-episode transfer is context-selective."""

    def __init__(self, manager: PatternManager | None = None) -> None:
        self.manager = manager or PatternManager(
            promotion_threshold=0.0,
            min_support=3,
            max_archive_size=256,
            retrieval_top_k=8,
            archive_decay_rate=0.02,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_registry(self) -> tuple[AdapterRegistry, PrefixBufferAdapter, RewardAdapter]:
        registry = AdapterRegistry()
        norm = NormalisationAdapter(mode="zscore", window_size=8)
        acorr = AutocorrelationAdapter(window_size=12, max_lag=4)
        entropy = EntropyAdapter(window_size=12)
        store_size = PatternStoreSizeAdapter(base_patterns=16, min_patterns=8, max_patterns=64)
        prefix = PrefixBufferAdapter(buffer_size=2, value_mode="tuple", include_history_context=True)
        reward = RewardAdapter(correct_reward=1.0, incorrect_penalty=0.0, decay=0.9)
        registry.register(norm)
        registry.register(acorr)
        registry.register(entropy)
        registry.register(store_size)
        registry.register(prefix)
        registry.register(reward)
        return registry, prefix, reward

    def _run_episode(
        self,
        episode_num: int,
        name: str,
        stream_family: str,
        stream: tuple[float, ...],
        seed_context: dict,
    ) -> MELBEpisodeResult:
        """Run one episode and return its result. Mutates self.manager."""
        engine = PatternEngine()

        # Seed from archive before the episode starts.
        # Use fallback_global=False so period-4 episodes don't receive period-2
        # patterns (or vice-versa) — context isolation is a core MELB requirement.
        patterns_seeded = self.manager.seed_engine(engine, context=seed_context, fallback_global=False)

        # Build fresh adapter registry per episode
        registry, prefix_adapter, reward_adapter = self._build_registry()

        correct = 0
        total = 0
        previous_prediction: Any = None
        previous_history_window: Any = None
        last_packet: AdapterPacket | None = None

        for index, raw in enumerate(stream):
            ctx: dict = {"step": index, "episode": episode_num}
            # Communicate previous prediction so RewardAdapter can evaluate it
            if previous_prediction is not None:
                ctx["predicted_next"] = previous_prediction
            packet = AdapterPacket(raw=raw, context=ctx)
            # Run all adapters (order matters — registry preserves insertion order)
            for adapter in registry.adapters.values():
                packet = adapter.run(packet)

            last_packet = packet
            # The last state has goal injected by RewardAdapter; use the state
            # before the reward state (second-to-last) as the observation state.
            # RewardAdapter appends an extra running_reward State, so states[-2]
            # is the prefix/feature state and states[-1] is the reward state.
            obs_state = packet.states[-2] if len(packet.states) >= 2 else packet.states[-1]
            state = obs_state
            history_window = (packet.context or {}).get("history_window")

            # Update engine store capacity from store_size adapter
            recommended = packet.context.get("recommended_max_patterns")
            if recommended is not None:
                engine.store.max_patterns = int(recommended)

            # Score previous prediction (reward_correct from RewardAdapter)
            if previous_prediction is not None:
                actual = state.value
                if previous_prediction == actual:
                    correct += 1
                total += 1

            # Record transition for prefix adapter
            if previous_history_window is not None and hasattr(prefix_adapter, "record_transition"):
                prefix_adapter.record_transition(previous_history_window, state.value)

            # Observe in engine — use earned utility from RewardAdapter via state.goal.
            # The obs_state has goal injected by RewardAdapter; fall back to a small
            # positive utility so new patterns can survive archive decay.
            earned_goal = state.goal if state.goal is not None else {"utility": 0.1}
            engine.observe(_State(value=state.value, step=index, goal=earned_goal, context=state.context))

            # Predict next
            predicted = None
            if hasattr(prefix_adapter, "predict_transition") and history_window is not None:
                predicted = prefix_adapter.predict_transition(history_window)
            if predicted is None:
                action = engine.act(
                    goal={"utility": 0.0, "min_confidence": 0.3},
                    horizon=1,
                )
                if action.forecast is not None:
                    predicted = action.forecast.value

            previous_prediction = predicted
            previous_history_window = history_window

        prediction_accuracy = correct / total if total > 0 else 0.0

        # Extract context from last packet for end_episode
        # Use only dominant_period for the archive signature so that seeding
        # context (which also uses only dominant_period) matches exactly.
        episode_context: dict = {}
        if last_packet is not None:
            dp = last_packet.context.get("dominant_period")
            if dp is not None:
                episode_context["dominant_period"] = dp

        end_info = self.manager.end_episode(engine, context=episode_context)
        context_sig = build_context_signature(episode_context)

        return MELBEpisodeResult(
            episode=episode_num,
            name=name,
            stream_family=stream_family,
            patterns_seeded=patterns_seeded,
            patterns_promoted=end_info["promoted"],
            archive_size_after=end_info["archive_size"],
            prediction_accuracy=prediction_accuracy,
            final_store_size=len(engine.store.patterns),
            context_signature=context_sig,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> MELBResult:
        # Episode streams
        period2_stream: tuple[float, ...] = (1.0, 2.0) * 8         # 16 values, period-2
        period4_stream: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0) * 4  # 16 values, period-4
        period2_transfer: tuple[float, ...] = (5.0, 9.0) * 8       # same structure, different values
        period4_transfer: tuple[float, ...] = (3.0, 1.0, 4.0, 1.0) * 4  # period-4, different values

        # Context hints per episode (used for seeding).
        # Use only dominant_period so the signature is stable across episodes
        # regardless of the actual entropy value computed by the adapter pipeline.
        period2_ctx = {"dominant_period": 2}
        period4_ctx = {"dominant_period": 4}

        trace: dict[str, Any] = {}

        # Episode 1: period-2 training
        ep1 = self._run_episode(1, "period2_train", "period2", period2_stream, seed_context=period2_ctx)
        trace["ep1"] = {"archive_stats": self.manager.archive_stats_by_signature()}

        # Episode 2: period-4 training (should NOT receive period-2 patterns)
        ep2 = self._run_episode(2, "period4_train", "period4", period4_stream, seed_context=period4_ctx)
        trace["ep2"] = {"archive_stats": self.manager.archive_stats_by_signature()}

        # Episode 3: period-2 transfer (same structure as ep1, different values)
        ep3 = self._run_episode(3, "period2_transfer", "period2", period2_transfer, seed_context=period2_ctx)
        trace["ep3"] = {"archive_stats": self.manager.archive_stats_by_signature()}

        # Episode 4: period-4 cross-structure control
        ep4 = self._run_episode(4, "period4_transfer", "period4", period4_transfer, seed_context=period4_ctx)
        trace["ep4"] = {"archive_stats": self.manager.archive_stats_by_signature()}

        episodes = (ep1, ep2, ep3, ep4)

        # Evaluate success criteria
        transfer_accuracy_gain = ep3.prediction_accuracy - ep1.prediction_accuracy
        context_isolation_verified = ep3.patterns_seeded > 0 and ep2.patterns_seeded == 0

        failures: list[str] = []
        if transfer_accuracy_gain < 0.0:
            failures.append(
                f"transfer_accuracy_gain={transfer_accuracy_gain:.3f} < 0.0 "
                f"(ep1={ep1.prediction_accuracy:.3f}, ep3={ep3.prediction_accuracy:.3f})"
            )
        if not context_isolation_verified:
            failures.append(
                f"context_isolation failed: ep2_seeded={ep2.patterns_seeded} "
                f"ep3_seeded={ep3.patterns_seeded} "
                f"(need ep2_seeded==0 and ep3_seeded>0)"
            )
        if ep3.patterns_seeded == 0:
            failures.append("episode 3 received no transferred patterns (patterns_seeded==0)")

        passed = len(failures) == 0
        reason = "success" if passed else "; ".join(failures)

        trace["summary"] = {
            "transfer_accuracy_gain": transfer_accuracy_gain,
            "context_isolation_verified": context_isolation_verified,
            "ep1_accuracy": ep1.prediction_accuracy,
            "ep3_accuracy": ep3.prediction_accuracy,
            "ep2_seeded": ep2.patterns_seeded,
            "ep3_seeded": ep3.patterns_seeded,
            "archive_size_final": self.manager.stats()["archive_size"],
        }

        return MELBResult(
            result="success" if passed else "failure",
            reason=reason,
            episodes=episodes,
            transfer_accuracy_gain=transfer_accuracy_gain,
            context_isolation_verified=context_isolation_verified,
            trace=trace,
        )
