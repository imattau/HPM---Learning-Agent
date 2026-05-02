"""Symbolic Generalisation Benchmark — tests cross-scale generalisation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Sequence

from ..adapter import AdapterPacket, AdapterRegistry
from ..adapter.feature_adapters import NormalisationAdapter, NumericAdapter, PrefixBufferAdapter, SymbolicAdapter
from ..preprocessors.prefix_buffer import _to_float_tuple


# ---------------------------------------------------------------------------
# SymbolicPrefixAdapter
# ---------------------------------------------------------------------------

class _SymbolicPrefixAdapter:
    """Reads the last symbolic state (integer) from packet.states and runs
    prefix-buffer logic on those integers for prediction."""

    name: str = "symbolic_prefix"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, buffer_size: int = 2) -> None:
        self.buffer_size = buffer_size
        self.history: deque[int] = deque(maxlen=buffer_size)
        self.transition_memory: defaultdict[tuple[float, ...], Counter] = defaultdict(Counter)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # Read last symbolic state value (integer symbol)
        symbol = int(packet.states[-1].value) if packet.states else 0
        self.history.append(symbol)
        padded = [-1] * (self.buffer_size - len(self.history)) + list(self.history)
        history_window = tuple(float(v) for v in padded)

        from ..core import State
        context = dict(packet.context or {})
        context.update({
            "domain": "symbolic_prefix",
            "history_window": history_window,
        })
        packet.context = context
        packet.states.append(State(value=tuple(padded), context=context))
        packet.log(self.name, {"symbol": symbol, "history_window": history_window}, role="adapter")
        return packet

    def record_transition(self, history_window: Any, next_value: Any) -> None:
        key = _to_float_tuple(history_window)
        if key is None:
            return
        self.transition_memory[key][next_value] += 1

    def predict_transition(self, history_window: Any) -> Any | None:
        key = _to_float_tuple(history_window)
        if key is None:
            return None
        counter = self.transition_memory.get(key)
        if not counter:
            return None
        return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SGBTaskResult:
    name: str
    family: str
    pipeline_accuracies: dict[str, float]
    winner: str
    symbolic_wins: bool


@dataclass(frozen=True, slots=True)
class SGBResult:
    result: str
    reason: str
    task_results: tuple[SGBTaskResult, ...]
    cross_scale_symbolic_accuracy: float
    cross_scale_numeric_accuracy: float
    trace: dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _numeric_pipeline() -> tuple[AdapterRegistry, Any]:
    registry = AdapterRegistry()
    adapter = NumericAdapter()
    registry.register(adapter)
    return registry, adapter


def _prefix_pipeline() -> tuple[AdapterRegistry, Any]:
    registry = AdapterRegistry()
    adapter = PrefixBufferAdapter(buffer_size=2, value_mode="tuple", include_history_context=True)
    registry.register(adapter)
    return registry, adapter


def _symbolic_pipeline() -> tuple[AdapterRegistry, _SymbolicPrefixAdapter]:
    registry = AdapterRegistry()
    norm = NormalisationAdapter(mode="zscore", window_size=8)
    disc = SymbolicAdapter(mode="quantile", n_symbols=4, window_size=8)
    sym_prefix = _SymbolicPrefixAdapter(buffer_size=2)
    registry.register(norm)
    registry.register(disc)
    registry.register(sym_prefix)
    return registry, sym_prefix


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _run_pipeline_on_stream(
    stream: Sequence[float],
    registry: AdapterRegistry,
    terminal_adapter: Any,
    terminal_name: str,
) -> float:
    """Return prediction accuracy for the given pipeline and stream."""
    correct = 0
    total = 0
    previous_prediction: Any = None
    previous_history_window: Any = None

    for index, raw in enumerate(stream):
        packet = AdapterPacket(raw=raw, context={"step": index})
        # Run all adapters in registration order
        for adapter in registry.adapters.values():
            packet = adapter.run(packet)

        state = packet.states[-1]
        history_window = (packet.context or {}).get("history_window")

        if previous_prediction is not None:
            actual = state.value
            correct_flag = previous_prediction == actual
            correct += int(correct_flag)
            total += 1

        if (
            terminal_adapter is not None
            and hasattr(terminal_adapter, "record_transition")
            and previous_history_window is not None
        ):
            terminal_adapter.record_transition(previous_history_window, state.value)

        predicted: Any | None = None
        if (
            terminal_adapter is not None
            and hasattr(terminal_adapter, "predict_transition")
            and history_window is not None
        ):
            predicted = terminal_adapter.predict_transition(history_window)

        previous_prediction = predicted
        previous_history_window = history_window

    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class SymbolicGeneralisationBenchmark:
    """Test cross-scale generalisation: does the symbolic pipeline outperform
    numeric/prefix pipelines when the same structure appears at different scales?
    """

    def run(self) -> SGBResult:
        # Task streams
        same_scale_train = tuple(float(v) for v in ([1, 2] * 6))
        same_scale_test = tuple(float(v) for v in ([1, 2] * 6))
        cross_scale_train = tuple(float(v) for v in ([1, 2] * 6))
        cross_scale_test = tuple(float(v) for v in ([100, 200] * 6))
        cross_struct_train = tuple(float(v) for v in ([1, 2] * 6))
        cross_struct_test = tuple(float(v) for v in ([1, 2, 3] * 4))

        tasks = [
            ("same_scale", "same_scale", same_scale_train, same_scale_test),
            ("cross_scale", "cross_scale", cross_scale_train, cross_scale_test),
            ("cross_structure", "cross_structure", cross_struct_train, cross_struct_test),
        ]

        pipeline_names = ["numeric", "prefix_buffer", "symbolic"]
        task_results: list[SGBTaskResult] = []
        cross_scale_symbolic_accuracy = 0.0
        cross_scale_numeric_accuracy = 0.0
        trace: dict[str, Any] = {"tasks": {}}

        for task_name, family, train_stream, test_stream in tasks:
            accuracies: dict[str, float] = {}

            # --- numeric ---
            num_registry, num_adapter = _numeric_pipeline()
            _run_pipeline_on_stream(train_stream, num_registry, num_adapter, "numeric")
            num_acc = _run_pipeline_on_stream(test_stream, num_registry, num_adapter, "numeric")
            accuracies["numeric"] = num_acc

            # --- prefix_buffer ---
            pfx_registry, pfx_adapter = _prefix_pipeline()
            _run_pipeline_on_stream(train_stream, pfx_registry, pfx_adapter, "prefix_buffer")
            pfx_acc = _run_pipeline_on_stream(test_stream, pfx_registry, pfx_adapter, "prefix_buffer")
            accuracies["prefix_buffer"] = pfx_acc

            # --- symbolic ---
            # Train phase: learn transition memory from symbols
            sym_registry, sym_adapter = _symbolic_pipeline()
            _run_pipeline_on_stream(train_stream, sym_registry, sym_adapter, "symbolic_prefix")
            # Reset preprocessing windows (norm + disc) so test scale is learned fresh,
            # but preserve sym_adapter.transition_memory (the symbolic pattern).
            norm_adapter = sym_registry.adapters.get("normalisation")
            disc_adapter = sym_registry.adapters.get("symbolic")
            sp_adapter = sym_registry.adapters.get("symbolic_prefix")
            if norm_adapter is not None and hasattr(norm_adapter, "window"):
                norm_adapter.window.clear()
            if disc_adapter is not None and hasattr(disc_adapter, "window"):
                disc_adapter.window.clear()
            if sp_adapter is not None and hasattr(sp_adapter, "history"):
                sp_adapter.history.clear()
            sym_acc = _run_pipeline_on_stream(test_stream, sym_registry, sym_adapter, "symbolic_prefix")
            accuracies["symbolic"] = sym_acc

            winner = max(accuracies, key=lambda k: accuracies[k])
            symbolic_wins = winner == "symbolic"

            if family == "cross_scale":
                cross_scale_symbolic_accuracy = sym_acc
                cross_scale_numeric_accuracy = num_acc

            trace["tasks"][task_name] = {
                "accuracies": accuracies,
                "winner": winner,
            }

            task_results.append(SGBTaskResult(
                name=task_name,
                family=family,
                pipeline_accuracies=accuracies,
                winner=winner,
                symbolic_wins=symbolic_wins,
            ))

        passed = (
            cross_scale_symbolic_accuracy >= 0.7
            and cross_scale_numeric_accuracy < cross_scale_symbolic_accuracy
        )
        reason = (
            "success"
            if passed
            else (
                f"cross_scale_symbolic={cross_scale_symbolic_accuracy:.2f} "
                f"cross_scale_numeric={cross_scale_numeric_accuracy:.2f}: "
                + ("symbolic accuracy below 0.7" if cross_scale_symbolic_accuracy < 0.7
                   else "numeric not beaten by symbolic")
            )
        )

        return SGBResult(
            result="success" if passed else "failure",
            reason=reason,
            task_results=tuple(task_results),
            cross_scale_symbolic_accuracy=cross_scale_symbolic_accuracy,
            cross_scale_numeric_accuracy=cross_scale_numeric_accuracy,
            trace=trace,
        )
