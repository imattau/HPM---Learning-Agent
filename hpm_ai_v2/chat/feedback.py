"""Learning from chat feedback: thumbs, scoped corrections and preferences.

Every learning step is a ledger entry applied through ``apply_entry``. The same
function runs live and when a restarted server replays the ledger on top of the
unchanged base checkpoint, so the base checkpoint is never rewritten and the
learned state is fully auditable.

Learning is deliberately narrow:

* a thumb moves the Observer weight of the cited answer sentences only (never
  a whole source), and does nothing for a reply that cited nothing;
* a correction names an exact source span and becomes a user-scoped
  correction episode in the study trainer, which the trainer applies only to
  that user's close paraphrases;
* a preference is an explicit key and value the planner already honours.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hpm_ai_v2.training import StudyEpisode

SUPPORTED_PREFERENCES: dict[str, tuple[str, ...]] = {"summary_format": ("bullets", "numbered")}
THUMBS = ("up", "down")


class LedgerMismatch(RuntimeError):
    """The ledger was written against a different base checkpoint."""


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_answer_node(trainer, source_id: str, start: int, end: int) -> str | None:
    """Return the answer-sentence node covering exactly this source span."""
    for node_id in trainer.source_to_answer_nodes.get(source_id, ()):
        data = trainer.answer_node_data[node_id]
        if data["start"] == start and data["end"] == end:
            return node_id
    return None


def apply_entry(trainer, entry: dict[str, Any]) -> None:
    kind = entry["type"]
    if kind == "thumb":
        for node_id in entry["answer_node_ids"]:
            if entry["signal"] == "up":
                trainer.agent.observer.boost_id(node_id, gain=trainer.feedback_gain)
            else:
                trainer.agent.observer.penalize_id(node_id, penalty=trainer.feedback_penalty)
    elif kind == "correction":
        trainer.fit([], [StudyEpisode(**entry["episode"])])
    elif kind == "preference":
        trainer.fit([], [StudyEpisode(**entry["episode"])])
    else:
        raise ValueError(f"unknown ledger entry type: {kind}")


def correction_entry(*, entry_id: str, turn_id: str, user_id: str, question: str,
                     source_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "correction", "id": entry_id, "turn_id": turn_id, "user_id": user_id,
        "episode": {
            "episode_id": f"chat-correction-{entry_id}", "split": "train",
            "task_type": "correction", "question": question, "evidence_id": source_id,
            "expected_answer": text, "user_id": user_id, "correction_of": turn_id,
        },
    }


def preference_entry(*, entry_id: str, user_id: str, key: str, value: str) -> dict[str, Any]:
    if key not in SUPPORTED_PREFERENCES:
        raise ValueError(f"unsupported preference: {key}")
    if value not in SUPPORTED_PREFERENCES[key]:
        raise ValueError(f"{key} must be one of {list(SUPPORTED_PREFERENCES[key])}")
    return {
        "type": "preference", "id": entry_id, "user_id": user_id, "key": key, "value": value,
        "episode": {
            "episode_id": f"chat-preference-{entry_id}", "split": "train",
            "task_type": "preference", "question": f"Use {value} for summaries.",
            "evidence_id": None, "expected_answer": value, "user_id": user_id,
            "preference_key": key,
        },
    }


class Ledger:
    """Append-only JSONL of learning entries, bound to one base checkpoint."""

    def __init__(self, path: str | Path, base_digest: str | None = None):
        self.path = Path(path)
        self.base_digest = base_digest
        self._entries: list[dict[str, Any]] = []
        if self.path.exists():
            lines = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
            header, self._entries = (lines[0], lines[1:]) if lines else ({}, [])
            if header and header.get("type") != "base":
                raise LedgerMismatch("learning ledger has no base header")
            recorded = header.get("digest") if header else None
            if self._entries and recorded != base_digest:
                raise LedgerMismatch(
                    "the learning ledger was written against a different checkpoint "
                    f"({recorded!r} vs {base_digest!r}); use a new --data-dir or the original checkpoint"
                )

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8") as stream:
            if new_file:
                stream.write(json.dumps({"type": "base", "digest": self.base_digest}) + "\n")
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._entries.append(entry)

    def replay(self, trainer) -> int:
        """Apply every recorded entry to a freshly loaded trainer, in order."""
        for entry in self._entries:
            apply_entry(trainer, entry)
        return len(self._entries)
