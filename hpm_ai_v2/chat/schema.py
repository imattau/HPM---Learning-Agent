"""Chat types: a response is a grounded plan first and text second."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

Act = Literal["answer", "abstain", "clarify"]
ClaimKind = Literal["source", "inference"]


@dataclass(frozen=True)
class Citation:
    source_id: str
    start: int
    end: int
    text: str
    title: str = ""


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citations: tuple[Citation, ...]
    kind: ClaimKind = "source"


@dataclass(frozen=True)
class ResponsePlan:
    act: Act
    task_family: str
    claims: tuple[Claim, ...]
    confidence: float
    fmt: str = "extractive"
    procedure_id: Optional[str] = None
    effective_question: str = ""
    reason: str = ""
    format_style: Optional[str] = None
    # (source_id, label) candidates offered when act == "clarify".
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Reply:
    turn_id: str
    text: str
    plan: ResponsePlan
    realizer: str
    verified: bool
    sentence_map: tuple["RealizedSentence", ...] = ()

    @property
    def abstained(self) -> bool:
        return self.plan.act == "abstain"

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(c for claim in self.plan.claims for c in claim.citations)


@dataclass(frozen=True)
class RealizedSentence:
    """One emitted factual sentence and the source claims that support it."""

    text: str
    claim_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class Realization:
    text: str
    sentences: tuple[RealizedSentence, ...]


@dataclass(frozen=True)
class TurnEvent:
    event: str
    turn_id: str
    user_id: str
    session_id: str
    user_text: str
    effective_question: str
    act: str
    reply_text: str
    confidence: float
    citations: tuple[Citation, ...]
    verified: bool
    abstained: bool
    scope_source_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
