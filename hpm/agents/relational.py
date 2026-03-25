from __future__ import annotations

from dataclasses import dataclass, field
import uuid

import numpy as np


@dataclass(frozen=True)
class RelationalEdge:
    """Small typed relation token used for inspectable structural messages."""

    source: str
    relation: str
    target: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "confidence": float(self.confidence),
        }


@dataclass
class RelationalBundle:
    """Dense belief state paired with a compact structural summary."""

    agent_id: str
    mu: np.ndarray
    weight: float
    epistemic_loss: float
    strategic_confidence: float = 1.0
    relations: tuple[RelationalEdge, ...] = ()


@dataclass
class StructuralMessage:
    """Constrained inter-agent message carrying inspectable structural hypotheses."""

    source_agent_id: str
    relations: tuple[RelationalEdge, ...]
    confidence: float = 1.0
    provenance: tuple[str, ...] = ()
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "source_agent_id": self.source_agent_id,
            "confidence": float(self.confidence),
            "provenance": list(self.provenance),
            "relations": [edge.to_dict() for edge in self.relations],
        }


def mean_relation_confidence(relations: tuple[RelationalEdge, ...]) -> float:
    if not relations:
        return 0.0
    return float(np.mean([edge.confidence for edge in relations]))
