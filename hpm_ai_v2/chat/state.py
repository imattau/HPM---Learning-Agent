"""Small, serializable conversation state for reference resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_REFERENCE = re.compile(r"\b(?:it|they|them|this|that|these|those|there)\b", re.I)


@dataclass
class ConversationState:
    user_id: str = "default"
    session_id: str = ""
    source_ids: tuple[str, ...] = ()
    turns: list[dict] = field(default_factory=list)

    def resolve_reference(self, question: str) -> tuple[str, bool]:
        """Append the last resolved topic when the new turn uses a reference word."""
        if not self.turns or not _REFERENCE.search(question):
            return question, False
        prior = self.turns[-1].get("effective_question") or self.turns[-1].get("question")
        if not prior:
            return question, False
        return f"{question} Context: {prior}", True

    def remember(
        self,
        *,
        turn_id: str,
        question: str,
        effective_question: str,
        source_id: str | None,
        answer: str | None,
        citations: list[dict] | None = None,
    ) -> None:
        self.turns.append({
            "turn_id": turn_id,
            "question": question,
            "effective_question": effective_question,
            "source_id": source_id,
            "answer": answer,
            "citations": citations or [],
            "feedback": [],
        })
