"""Delta extraction for the v5 core."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if _is_sequence(value):
        return tuple(value)
    return (value,)


def _numeric_distance(before: Sequence[Any], after: Sequence[Any]) -> tuple[tuple[float, ...], float]:
    limit = min(len(before), len(after))
    deltas = tuple(float(after[i]) - float(before[i]) for i in range(limit))
    penalty = abs(len(after) - len(before))
    magnitude = (sum(abs(delta) for delta in deltas) / limit) if limit else 0.0
    return deltas, magnitude + penalty


@dataclass(frozen=True, slots=True)
class Delta:
    """Difference between two states."""

    before: Any
    after: Any
    value: Any
    magnitude: float
    level: str = "state"

    @classmethod
    def between(cls, before: Any, after: Any, *, level: str = "state") -> "Delta":
        """Build a delta from two observations."""

        if isinstance(before, Real) and isinstance(after, Real):
            value = float(after) - float(before)
            return cls(before=before, after=after, value=value, magnitude=abs(value), level=level)

        before_seq = _as_tuple(before)
        after_seq = _as_tuple(after)
        if before_seq and after_seq and all(isinstance(item, Real) for item in before_seq + after_seq):
            value, magnitude = _numeric_distance(before_seq, after_seq)
            return cls(before=before, after=after, value=value, magnitude=magnitude, level=level)

        value = after if before != after else ()
        magnitude = 0.0 if before == after else 1.0
        return cls(before=before, after=after, value=value, magnitude=magnitude, level=level)
