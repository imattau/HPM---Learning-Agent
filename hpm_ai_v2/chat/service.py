"""Framework-free chat service: conversations, persistence and message JSON.

The HTTP server and any future client are thin wrappers over this class. It
owns one shared trainer, one ChatSession per conversation, and a lock around
``send`` because the trainer is not thread-safe.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from hpm_ai_v2.chat.presentation import UngroundedClaim, to_blocks
from hpm_ai_v2.chat.schema import Reply
from hpm_ai_v2.chat.session import ChatSession
from hpm_ai_v2.chat.state import ConversationState

_ID = re.compile(r"^[0-9a-f]{32}$")
_TITLE_CHARS = 60


class NotFound(KeyError):
    pass


class ChatService:
    def __init__(self, trainer, data_dir: str | Path, *, user_id: str = "default", info: dict | None = None):
        self.trainer = trainer
        self.user_id = user_id
        self.info = dict(info or {})
        self.data_dir = Path(data_dir)
        self._dir = self.data_dir / "conversations"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.event_log = self.data_dir / "events.jsonl"
        self._lock = threading.Lock()
        self._sessions: dict[str, ChatSession] = {}

    # -- read-only configuration and sources ---------------------------------
    def config(self) -> dict:
        return {
            "user_id": self.user_id,
            "source_count": len(self.trainer.sources),
            "realizer": "schema-extractive",
            **self.info,
        }

    def sources(self) -> list[dict]:
        return [
            {"source_id": s.source_id, "title": s.title or s.source_id, "length": len(s.text)}
            for s in self.trainer.sources.values()
        ]

    def source(self, source_id: str) -> dict:
        source = self.trainer.sources.get(source_id)
        if source is None:
            raise NotFound(source_id)
        return {"source_id": source.source_id, "title": source.title or source.source_id, "text": source.text}

    # -- conversations ----------------------------------------------------------
    def create_conversation(self) -> dict:
        conv_id = uuid.uuid4().hex
        now = time.time()
        record = {
            "id": conv_id, "title": "New chat", "created_at": now, "updated_at": now,
            "session": {
                "user_id": self.user_id, "session_id": uuid.uuid4().hex,
                "source_ids": list(self.trainer.sources), "turns": [],
            },
            "messages": [],
        }
        with self._lock:
            self._write(record)
        return self._summary(record)

    def list_conversations(self) -> list[dict]:
        rows = []
        for path in self._dir.glob("*.json"):
            try:
                rows.append(self._summary(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda row: row["updated_at"], reverse=True)

    def get_conversation(self, conv_id: str) -> dict:
        record = self._read(conv_id)
        return {**self._summary(record), "messages": record["messages"]}

    def rename(self, conv_id: str, title: str) -> dict:
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty")
        with self._lock:
            record = self._read(conv_id)
            record["title"] = title[:_TITLE_CHARS * 2]
            self._write(record)
        return self._summary(record)

    def delete(self, conv_id: str) -> None:
        with self._lock:
            path = self._path(conv_id)
            if not path.exists():
                raise NotFound(conv_id)
            path.unlink()
            self._sessions.pop(conv_id, None)

    # -- turns --------------------------------------------------------------------
    def send(self, conv_id: str, text: str, *, scope_source_id: str | None = None) -> dict:
        with self._lock:
            record = self._read(conv_id)
            return self._send_locked(record, text, scope_source_id)

    def edit(self, conv_id: str, turn_id: str, text: str) -> dict:
        """Replace a user turn: drop it and everything after it, then re-send."""
        with self._lock:
            record = self._read(conv_id)
            index = next(
                (i for i, m in enumerate(record["messages"]) if m["role"] == "user" and m["turn_id"] == turn_id),
                None,
            )
            if index is None:
                raise NotFound(turn_id)
            dropped = [m["turn_id"] for m in record["messages"][index:] if m["role"] == "user"]
            record["messages"] = record["messages"][:index]
            turns = record["session"]["turns"]
            keep = next((i for i, t in enumerate(turns) if t["turn_id"] == turn_id), len(turns))
            record["session"]["turns"] = turns[:keep]
            self._sessions.pop(conv_id, None)
            self._append_event({"event": "chat_turns_edited", "session_id": record["session"]["session_id"],
                                "user_id": self.user_id, "dropped_turn_ids": dropped})
            return self._send_locked(record, text, None)

    # -- internals ----------------------------------------------------------------
    def _send_locked(self, record: dict, text: str, scope_source_id: str | None) -> dict:
        session = self._session_for(record)
        reply = session.send(text, scope_source_id=scope_source_id)
        now = time.time()
        user_message = {
            "role": "user", "turn_id": reply.turn_id, "text": text.strip(),
            "scope_source_id": scope_source_id, "created_at": now,
        }
        assistant = self._assistant_message(reply)
        assistant["created_at"] = now
        record["messages"] += [user_message, assistant]
        record["session"]["turns"] = session.state.turns
        if record["title"] == "New chat":
            record["title"] = text.strip()[:_TITLE_CHARS]
        record["updated_at"] = now
        self._write(record)
        return assistant

    def _assistant_message(self, reply: Reply) -> dict:
        plan = reply.plan
        try:
            blocks, citations = to_blocks(plan)
        except UngroundedClaim as exc:
            blocks, citations = [{"type": "error", "text": f"Reply withheld: {exc}"}], []
        return {
            "role": "assistant", "turn_id": reply.turn_id, "act": plan.act,
            "task_family": plan.task_family, "confidence": plan.confidence,
            "verified": reply.verified, "blocks": blocks, "citations": citations,
            "why": {
                "reason": plan.reason, "procedure_id": plan.procedure_id,
                "effective_question": plan.effective_question,
            },
            "text": reply.text,
        }

    def _session_for(self, record: dict) -> ChatSession:
        conv_id = record["id"]
        session = self._sessions.get(conv_id)
        if session is None:
            data = record["session"]
            state = ConversationState(
                user_id=data["user_id"], session_id=data["session_id"],
                source_ids=tuple(data["source_ids"]), turns=list(data["turns"]),
            )
            session = ChatSession(self.trainer, event_log=self.event_log, state=state)
            self._sessions[conv_id] = session
        return session

    def _path(self, conv_id: str) -> Path:
        if not _ID.match(conv_id or ""):
            raise NotFound(conv_id)
        return self._dir / f"{conv_id}.json"

    def _read(self, conv_id: str) -> dict:
        path = self._path(conv_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise NotFound(conv_id) from None

    def _write(self, record: dict) -> None:
        path = self._path(record["id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _append_event(self, payload: dict) -> None:
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False) + "\n")

    @staticmethod
    def _summary(record: dict) -> dict:
        return {k: record[k] for k in ("id", "title", "created_at", "updated_at")}
