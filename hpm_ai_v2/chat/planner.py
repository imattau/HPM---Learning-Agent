"""Build an HPM-native response plan from the supervised study retriever."""
from __future__ import annotations

import re

from hpm_ai_v2.chat.schema import Citation, Claim, ResponsePlan


_PROCEDURES = (
    ("compare", re.compile(r"\b(?:compare|contrast|differences? between|similarities between)\b", re.I)),
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
    def __init__(self, trainer, ambiguity_margin: float = 0.02):
        self.trainer = trainer
        # Ask for clarification when the runner-up source scores within this
        # relative margin of the best one.
        self.ambiguity_margin = ambiguity_margin

    def _ambiguous_sources(self, prediction) -> tuple[tuple[str, str], ...]:
        ranked = prediction.ranked_evidence
        if len(ranked) < 2 or ranked[0][1] <= 0:
            return ()
        best = ranked[0][1]
        close = [sid for sid, score in ranked if best - score <= best * self.ambiguity_margin]
        if len(close) < 2:
            return ()
        sources = self.trainer.sources
        return tuple((sid, sources[sid].title or sid) for sid in close)

    def plan(
        self,
        turn_id: str,
        question: str,
        *,
        user_id: str = "default",
        scope_source_id: str | None = None,
    ) -> ResponsePlan:
        task_type = classify_task(question)
        learned_workflow = self.trainer.predict_workflow(question, user_id=user_id)
        if learned_workflow is not None:
            task_type = learned_workflow[0]
        if task_type == "compare":
            rows = self.trainer.predict_comparison_sources(question, user_id=user_id)
            if len(rows) < 2:
                return ResponsePlan(
                    act="abstain",
                    task_family="compare",
                    claims=(),
                    confidence=min((row["confidence"] for row in rows), default=0.0),
                    fmt="compare",
                    effective_question=question,
                    reason="comparison_requires_two_supported_sources",
                )
            claims = tuple(
                Claim(
                    claim_id=f"claim-{turn_id}-{index}",
                    text=row["text"],
                    citations=(Citation(**row["citation"]),),
                )
                for index, row in enumerate(rows, start=1)
            )
            return ResponsePlan(
                act="answer",
                task_family="compare",
                claims=claims,
                confidence=min(row["confidence"] for row in rows),
                fmt="compare",
                effective_question=question,
                reason="two_source_comparison",
            )

        # The synthetic episode is only a query carrier; predict() reads no labels.
        prediction = self.trainer.predict_question(
            question, episode_id=f"chat-{turn_id}", task_type=task_type, user_id=user_id, scope_source_id=scope_source_id
        )
        format_style = None
        if task_type in {"summarize", "outline"}:
            style = self.trainer.get_preference(user_id, "summary_format")
            if style in {"bullets", "numbered"}:
                format_style = style
        act = "answer" if prediction.selected_evidence_id and prediction.answer else "abstain"
        options = (
            self._ambiguous_sources(prediction)
            if act == "answer" and scope_source_id is None else ()
        )
        if options:
            return ResponsePlan(
                act="clarify",
                task_family=task_type,
                claims=(),
                confidence=prediction.confidence,
                fmt=task_type,
                effective_question=question,
                reason="ambiguous_between_sources",
                format_style=format_style,
                procedure_id=learned_workflow[2] if learned_workflow else None,
                options=options,
            )
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
            format_style=format_style,
            procedure_id=learned_workflow[2] if learned_workflow else None,
        )
