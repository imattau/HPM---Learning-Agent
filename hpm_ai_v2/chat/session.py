"""Grounded, replayable chat session over a fitted study trainer."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from hpm_ai_v2.chat.controller import GroundedActController
from hpm_ai_v2.chat.evidence import ExpandingPlanner
from hpm_ai_v2.chat.feedback import (
    THUMBS, Ledger, apply_entry, correction_entry, find_answer_node, preference_entry,
)
from hpm_ai_v2.chat.planner import GroundedPlanner
from hpm_ai_v2.chat.realizers import PlanBoundRealizer
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
        ledger: Ledger | None = None,
        planner=None,
        expand_evidence: bool = True,
    ):
        self.trainer = trainer
        self.ledger = ledger
        self.state = state or ConversationState(
            user_id=user_id,
            session_id=session_id or uuid.uuid4().hex,
            source_ids=tuple(trainer.sources),
        )
        self.event_log = Path(event_log) if event_log else None
        self.controller = GroundedActController()
        if planner is None:
            # Evidence expansion passed its pre-registered gate (see the evidence-access design doc);
            # expand_evidence=False gives the trainer's own single-sentence or top-three evidence.
            planner = ExpandingPlanner(trainer) if expand_evidence else GroundedPlanner(trainer)
        self.planner = planner
        self.realizer = PlanBoundRealizer()
        self.verifier = GroundingVerifier(trainer.sources)

    def send(
        self,
        user_text: str,
        *,
        scope_source_id: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> Reply:
        """Run one turn. ``on_stage`` is told when each real step starts.

        Stages: interpret, plan, verify, realize. They report progress only.
        """
        stage = on_stage or (lambda _name: None)
        if not user_text.strip():
            raise ValueError("user_text must not be empty")
        if scope_source_id is not None and scope_source_id not in self.trainer.sources:
            raise ValueError(f"unknown scope_source_id: {scope_source_id}")
        turn_id = uuid.uuid4().hex
        stage("interpret")
        # A scoped send answers an earlier clarification, so its text is already resolved.
        if scope_source_id is None:
            effective_question, _resolved = self.state.resolve_reference(user_text.strip())
        else:
            effective_question = user_text.strip()
        stage("plan")
        plan = self.planner.plan(
            turn_id, effective_question,
            user_id=self.state.user_id, scope_source_id=scope_source_id,
        )
        plan = replace(plan, act=self.controller.choose(plan))
        stage("verify")
        verified = self.verifier.verify(plan)
        if not verified:
            plan = replace(plan, act="abstain", claims=(), reason="grounding_verification_failed")
        stage("realize")
        realization = self.realizer.realize_with_map(plan)
        verified = verified and self.verifier.verify_realization(plan, realization)
        if not verified and plan.act == "answer":
            plan = replace(plan, act="abstain", claims=(), reason="realization_verification_failed")
            realization = self.realizer.realize_with_map(plan)
        reply = Reply(
            turn_id=turn_id,
            text=realization.text,
            plan=plan,
            realizer=self.realizer.name,
            verified=verified,
            sentence_map=realization.sentences,
        )
        citations = tuple(citation for claim in plan.claims for citation in claim.citations)
        source_id = citations[0].source_id if citations else None
        self.state.remember(
            turn_id=turn_id,
            question=user_text.strip(),
            effective_question=effective_question,
            source_id=source_id,
            answer=plan.claims[0].text if plan.claims else None,
            citations=[asdict(citation) for citation in citations],
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
            scope_source_id=scope_source_id,
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

    # -- learning from feedback ---------------------------------------------------
    def _turn(self, turn_id: str) -> dict:
        for turn in self.state.turns:
            if turn["turn_id"] == turn_id:
                return turn
        raise ValueError(f"unknown turn: {turn_id}")

    def correction_candidates(self, turn_id: str, max_sources: int = 3) -> list[dict]:
        """Source sentences a user may pick as the right answer to this turn.

        Candidates come from the best-ranked sources for the turn's question
        (all sources when nothing ranked), so a correction always names an
        exact span of the user's own material.
        """
        turn = self._turn(turn_id)
        prediction = self.trainer.predict_question(
            turn["effective_question"], episode_id=f"candidates-{turn_id}", user_id=self.state.user_id
        )
        source_ids = [sid for sid, _score in prediction.ranked_evidence][:max_sources]
        for citation in turn.get("citations", []):
            if citation["source_id"] not in source_ids:
                source_ids.append(citation["source_id"])
        if not source_ids:
            source_ids = list(self.trainer.sources)[:max_sources + 2]
        shown = {(c["source_id"], c["start"], c["end"]) for c in turn.get("citations", [])}
        rows = []
        for sid in source_ids:
            source = self.trainer.sources[sid]
            for node_id in self.trainer.source_to_answer_nodes.get(sid, ()):
                data = self.trainer.answer_node_data[node_id]
                rows.append({
                    "source_id": sid, "title": source.title or sid,
                    "start": data["start"], "end": data["end"], "text": data["text"],
                    "current": (sid, data["start"], data["end"]) in shown,
                })
        return rows

    def feedback(self, turn_id: str, signal: str, *, correction: dict | None = None) -> dict:
        """Record a thumb ('up'/'down') or a correction ('correct') for a past turn.

        A correction must name the exact source span that answers the turn
        (``{"source_id", "start", "end"}``). The result says what was learned
        and, for a correction, whether re-asking the question now picks it.
        """
        turn = self._turn(turn_id)
        done = turn.setdefault("feedback", [])
        user_id = self.state.user_id
        if signal in THUMBS:
            if any(s in THUMBS for s in done):
                raise ValueError("a thumb was already recorded for this turn")
            nodes = [
                node for node in (
                    find_answer_node(self.trainer, c["source_id"], c["start"], c["end"])
                    for c in turn.get("citations", [])
                ) if node is not None
            ]
            if nodes:
                entry = {"type": "thumb", "id": uuid.uuid4().hex, "turn_id": turn_id,
                         "user_id": user_id, "signal": signal, "answer_node_ids": nodes}
                apply_entry(self.trainer, entry)
                if self.ledger is not None:
                    self.ledger.append(entry)
            result = {
                "turn_id": turn_id, "signal": signal, "learned": bool(nodes),
                "message": ("Noted. This nudges how the cited sentence ranks for later questions."
                            if nodes else "Noted. This reply cited nothing, so there is nothing to adjust."),
            }
        elif signal == "correct":
            if "correct" in done:
                raise ValueError("a correction was already recorded for this turn")
            if not isinstance(correction, dict):
                raise ValueError("correction must name the source span that answers the question")
            try:
                source_id, start, end = correction["source_id"], int(correction["start"]), int(correction["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("correction needs source_id, start and end") from None
            node_id = find_answer_node(self.trainer, source_id, start, end)
            if node_id is None:
                raise ValueError("correction must be one of the source's sentences")
            text = self.trainer.answer_node_data[node_id]["text"]
            question = turn["effective_question"]
            entry = correction_entry(
                entry_id=uuid.uuid4().hex, turn_id=turn_id, user_id=user_id,
                question=question, source_id=source_id, text=text,
            )
            apply_entry(self.trainer, entry)
            if self.ledger is not None:
                self.ledger.append(entry)
            again = self.trainer.predict_question(
                question, episode_id=f"reask-{turn_id}", user_id=user_id
            )
            cited = again.citation or {}
            effective = (cited.get("source_id"), cited.get("start"), cited.get("end")) == (source_id, start, end)
            result = {
                "turn_id": turn_id, "signal": signal, "learned": True, "effective": effective,
                "now_answers": cited or None,
                "message": ("Saved. Asking the same question now gives your span." if effective else
                            "Saved. Asking with exactly the same wording still gives the earlier answer: "
                            "a correction applies to your close paraphrases that also share wording "
                            "with the span you chose."),
            }
        else:
            raise ValueError("signal must be 'up', 'down' or 'correct'")
        done.append(signal)
        self._log_raw({"event": "chat_feedback", "turn_id": turn_id, "user_id": user_id,
                       "session_id": self.state.session_id, "signal": signal,
                       "learned": result["learned"], "effective": result.get("effective")})
        return result

    def set_preference(self, key: str, value: str) -> dict:
        entry = preference_entry(entry_id=uuid.uuid4().hex, user_id=self.state.user_id, key=key, value=value)
        apply_entry(self.trainer, entry)
        if self.ledger is not None:
            self.ledger.append(entry)
        self._log_raw({"event": "chat_preference", "user_id": self.state.user_id,
                       "session_id": self.state.session_id, "key": key, "value": value})
        return {"key": key, "value": value}

    def _log(self, event: TurnEvent) -> None:
        self._log_raw(event.to_dict())

    def _log_raw(self, fields: dict) -> None:
        if self.event_log is None:
            return
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        payload = {"time": time.time(), **fields}
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
