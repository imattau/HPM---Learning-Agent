"""Grounded, replayable chat session over a fitted study trainer."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

from hpm_ai_v2.chat.controller import GroundedActController
from hpm_ai_v2.chat.planner import GroundedPlanner
from hpm_ai_v2.chat.realizers import ExtractiveRealizer
from hpm_ai_v2.chat.schema import Reply, TurnEvent
from hpm_ai_v2.chat.state import ConversationState
from hpm_ai_v2.chat.verifier import GroundingVerifier


class ChatSession:
    """A minimal Python API for single- and multi-turn grounded study QA."""

    def __init__(
        self,
        trainer,
        *,
        user_id: str = "default",
        session_id: str | None = None,
        event_log: str | Path | None = None,
        state: ConversationState | None = None,
    ):
        self.trainer = trainer
        self.state = state or ConversationState(
            user_id=user_id,
            session_id=session_id or uuid.uuid4().hex,
            source_ids=tuple(trainer.sources),
        )
        self.event_log = Path(event_log) if event_log else None
        self.controller = GroundedActController()
        self.planner = GroundedPlanner(trainer)
        self.realizer = ExtractiveRealizer()
        self.verifier = GroundingVerifier(trainer.sources)

    def send(self, user_text: str) -> Reply:
        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        turn_id = uuid.uuid4().hex
        effective_question, _resolved = self.state.resolve_reference(user_text.strip())
        plan = self.planner.plan(turn_id, effective_question)
        plan = replace(plan, act=self.controller.choose(plan))
        verified = self.verifier.verify(plan)
        if not verified:
            plan = replace(plan, act="abstain", claims=(), reason="grounding_verification_failed")
        reply = Reply(
            turn_id=turn_id,
            text=self.realizer.realize(plan),
            plan=plan,
            realizer=self.realizer.name,
            verified=verified,
        )
        citations = tuple(citation for claim in plan.claims for citation in claim.citations)
        source_id = citations[0].source_id if citations else None
        self.state.remember(
            turn_id=turn_id,
            question=user_text.strip(),
            effective_question=effective_question,
            source_id=source_id,
            answer=plan.claims[0].text if plan.claims else None,
        )
        self._log(TurnEvent(
            event="chat_turn",
            turn_id=turn_id,
            user_id=self.state.user_id,
            session_id=self.state.session_id,
            user_text=user_text.strip(),
            effective_question=effective_question,
            act=plan.act,
            reply_text=reply.text,
            confidence=plan.confidence,
            citations=citations,
            verified=reply.verified,
            abstained=reply.abstained,
        ))
        return reply

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "user_id": self.state.user_id,
            "session_id": self.state.session_id,
            "source_ids": list(self.state.source_ids),
            "turns": self.state.turns,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, trainer, path: str | Path, *, event_log: str | Path | None = None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        state = ConversationState(
            user_id=data["user_id"],
            session_id=data["session_id"],
            source_ids=tuple(data["source_ids"]),
            turns=list(data["turns"]),
        )
        return cls(trainer, event_log=event_log, state=state)

    def _log(self, event: TurnEvent) -> None:
        if self.event_log is None:
            return
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        payload = {"time": time.time(), **event.to_dict()}
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
