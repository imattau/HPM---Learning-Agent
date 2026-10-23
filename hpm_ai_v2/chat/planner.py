"""Build an HPM-native response plan from the supervised study retriever."""
from __future__ import annotations

import re

from hpm_ai_v2.training.study import StudyEpisode
from hpm_ai_v2.chat.schema import Citation, Claim, ResponsePlan


_PROCEDURES = (
    ("flashcards", re.compile(r"\bflash[ -]?cards?\b", re.I)),
    ("summarize", re.compile(r"\b(?:summari[sz]e|summary|overview)\b", re.I)),
    ("outline", re.compile(r"\boutline\b", re.I)),
    ("define", re.compile(r"\b(?:define|definition|what is|what are)\b", re.I)),
)


def classify_task(question: str) -> str:
    for task_type, pattern in _PROCEDURES:
        if pattern.search(question):
            return task_type
    return "grounded_qa"


class GroundedPlanner:
    def __init__(self, trainer):
        self.trainer = trainer

    def plan(self, turn_id: str, question: str) -> ResponsePlan:
        # The synthetic episode is only a query carrier; predict() reads no labels.
        task_type = classify_task(question)
        prediction = self.trainer.predict_question(
            question, episode_id=f"chat-{turn_id}", task_type=task_type
        )
        act = "answer" if prediction.selected_evidence_id and prediction.answer else "abstain"
        citation_rows = prediction.citations or ((prediction.citation,) if prediction.citation else ())
        claims = tuple(
            Claim(
                claim_id=f"claim-{turn_id}-{index}",
                text=row["text"],
                citations=(Citation(**row),),
            )
            for index, row in enumerate(citation_rows, start=1)
        ) if act == "answer" else ()
        return ResponsePlan(
            act=act,
            task_family=task_type,
            claims=claims,
            confidence=prediction.confidence,
            fmt=task_type,
            effective_question=question,
            reason="supported_by_source" if claims else "no_retrievable_supported_answer",
        )
