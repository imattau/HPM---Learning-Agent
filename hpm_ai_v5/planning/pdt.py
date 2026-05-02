"""Prefix Disambiguation Task benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core import CoreConfig, PatternEngine, State
from ..preprocessors import PrefixBufferPreprocessor


Symbol = float


@dataclass(frozen=True, slots=True)
class PDTStepResult:
    index: int
    prefix: list[str]
    predicted: str | None
    actual: str | None
    correct: bool
    confidence: float
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PDTResult:
    result: str
    reason: str
    accuracy: float
    critical_accuracy: float
    step_results: tuple[PDTStepResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class PrefixDisambiguationTask:
    """Expose the lack of short-term memory for ambiguous prefixes."""

    a: Symbol = 0.0
    b: Symbol = 1.0
    c: Symbol = 2.0
    critical_threshold: float = 0.7
    use_prefix_buffer: bool = False
    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=6))
    engine: PatternEngine = field(init=False)
    prefix_buffer: PrefixBufferPreprocessor | None = field(init=False, default=None)
    prefix_engines: dict[tuple[Any, ...], PatternEngine] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.engine = PatternEngine(config=self.config)
        self.prefix_buffer = PrefixBufferPreprocessor(
            buffer_size=3,
            value_mode="tuple",
            include_history_context=True,
        ) if self.use_prefix_buffer else None

    def _symbol(self, value: Symbol) -> str:
        if value == self.a:
            return "A"
        if value == self.b:
            return "B"
        if value == self.c:
            return "C"
        return f"{value:g}"

    def _stream_blocks(self) -> tuple[tuple[Symbol, ...], tuple[Symbol, ...]]:
        # Both blocks share the same visible A -> B prefix at the decision point.
        # Only the symbol two steps back differs, which the current core does not
        # represent in selection context.
        return (
            (self.c, self.a, self.b, self.c),
            (self.a, self.a, self.b, self.a),
        )

    def _reset_short_term_memory(self) -> None:
        if self.prefix_buffer is not None:
            self.prefix_buffer.history = []

    def _engine_for_state(self, state: State) -> PatternEngine:
        if self.prefix_buffer is None:
            return self.engine
        return self.engine

    def _sequence(self, blocks: Iterable[Sequence[Symbol]], repeats: int) -> list[Symbol]:
        stream: list[Symbol] = []
        for _ in range(max(1, repeats)):
            for block in blocks:
                stream.extend(float(item) for item in block)
        return stream

    def _episodes(self, blocks: Iterable[Sequence[Symbol]], repeats: int) -> list[tuple[Symbol, ...]]:
        episodes: list[tuple[Symbol, ...]] = []
        for _ in range(max(1, repeats)):
            for block in blocks:
                episodes.append(tuple(float(item) for item in block))
        return episodes

    def _encode_state(self, symbol: Symbol, step: int) -> State:
        if self.prefix_buffer is None:
            return State(value=float(symbol), step=step, context={"mode": "pdt"})
        processed = self.prefix_buffer.preprocess(float(symbol), context={"mode": "pdt", "step": step})
        return State(value=processed.state.value, step=step, context=processed.context)

    def _decode_prediction(self, forecast_value: Any) -> Symbol | None:
        if isinstance(forecast_value, Sequence) and forecast_value:
            numeric_tail = [item for item in forecast_value if isinstance(item, (int, float))]
            if numeric_tail:
                return float(numeric_tail[-1])
        if not isinstance(forecast_value, (int, float)):
            return None
        numeric = int(round(float(forecast_value)))
        if self.prefix_buffer is not None and self.prefix_buffer.value_mode == "code":
            width = max(1, self.prefix_buffer.buffer_size)
            return float((numeric // (self.prefix_buffer.code_base ** (width - 1))) % self.prefix_buffer.code_base)
        return float(forecast_value)

    def _predict_next(self, current: State) -> tuple[Symbol | None, float, dict[str, Any]]:
        action = self.engine.act(goal={"utility": 0.0, "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25, "min_confidence": 0.3}, horizon=1)
        predicted = None if action.forecast is None else self._decode_prediction(action.forecast.value)
        return predicted, action.confidence, action.reasoning_trace.to_dict() if action.reasoning_trace is not None else {}

    def _predict_next_with_buffer(self, state: State) -> Symbol | None:
        if self.prefix_buffer is None:
            return None
        history_window = state.context.get("history_window")
        if history_window is None:
            return None
        predicted = self.prefix_buffer.predict_transition(history_window)
        return None if predicted is None else float(predicted)

    def _run_stream(self, stream: Sequence[Symbol], *, train: bool) -> tuple[list[PDTStepResult], float, int]:
        step_results: list[PDTStepResult] = []
        correct = 0
        total = 0
        for index, symbol in enumerate(stream):
            state = self._encode_state(symbol, index)
            engine = self._engine_for_state(state)
            engine.observe(state)
            if index + 1 >= len(stream):
                continue
            if symbol != self.b:
                continue
            buffer_prediction = self._predict_next_with_buffer(state)
            predicted, confidence, reasoning_trace = self._predict_next_with_engine(engine, state)
            if buffer_prediction is not None:
                predicted = buffer_prediction
            actual = stream[index + 1]
            is_correct = predicted == actual
            if not train:
                total += 1
                if is_correct:
                    correct += 1
                step_results.append(
                    PDTStepResult(
                        index=index,
                        prefix=[self._symbol(stream[index - 2]), self._symbol(stream[index - 1]), self._symbol(symbol)] if index >= 2 else [self._symbol(symbol)],
                        predicted=None if predicted is None else self._symbol(predicted),
                        actual=self._symbol(actual),
                        correct=is_correct,
                        confidence=confidence,
                        reasoning_trace={**reasoning_trace, "buffer_prediction": None if buffer_prediction is None else self._symbol(buffer_prediction)},
                    )
                )
            elif self.prefix_buffer is not None:
                history_window = state.context.get("history_window")
                if history_window is not None:
                    self.prefix_buffer.record_transition(history_window, actual)
        accuracy = (correct / total) if total else 0.0
        return step_results, accuracy, total

    def _predict_next_with_engine(self, engine: PatternEngine, current: State) -> tuple[Symbol | None, float, dict[str, Any]]:
        action = engine.act(goal={"utility": 0.0, "alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25, "min_confidence": 0.3}, horizon=1)
        predicted = None if action.forecast is None else self._decode_prediction(action.forecast.value)
        return predicted, action.confidence, action.reasoning_trace.to_dict() if action.reasoning_trace is not None else {}

    def run(self, *, training_repeats: int = 12, evaluation_repeats: int = 6) -> PDTResult:
        train_blocks = self._stream_blocks()
        train_episodes = self._episodes(train_blocks, training_repeats)
        eval_episodes = self._episodes(train_blocks, evaluation_repeats)

        for episode in train_episodes:
            self._reset_short_term_memory()
            self.engine.current_state = None
            self.engine.history = []
            self.engine.last_match = None
            self._run_stream(episode, train=True)

        self._reset_short_term_memory()
        step_results: list[PDTStepResult] = []
        accuracy_total = 0
        total = 0
        for episode in eval_episodes:
            self.engine.current_state = None
            self.engine.history = []
            self.engine.last_match = None
            episode_results, accuracy, episode_total = self._run_stream(episode, train=False)
            step_results.extend(episode_results)
            accuracy_total += int(round(accuracy * episode_total))
            total += episode_total
        accuracy = (accuracy_total / total) if total else 0.0
        critical_accuracy = accuracy
        passed = critical_accuracy >= self.critical_threshold
        reason = "success" if passed else "prefix disambiguation failed"
        trace = {
            "critical_positions": total,
            "training_repeats": training_repeats,
            "evaluation_repeats": evaluation_repeats,
            "evaluation_accuracy": accuracy,
            "threshold": self.critical_threshold,
            "prefix_buffer": self.use_prefix_buffer,
            "episodes": len(eval_episodes),
        }
        return PDTResult(
            result="success" if passed else "failure",
            reason=reason,
            accuracy=accuracy,
            critical_accuracy=critical_accuracy,
            step_results=tuple(step_results),
            trace=trace,
        )
